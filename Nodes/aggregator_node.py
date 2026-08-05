import json
from typing import Optional, List
from pydantic import BaseModel, Field, ValidationError
from Schema.aggregator_output_schema import AggregatorOutput
from State.supply_chain_state import SupplyChainState
from Utils.logger import get_logger
from langchain_core.prompts import ChatPromptTemplate
from Config.llm_config import synthesis_llm
from langchain_core.messages import AIMessage

# 🌟 IMPORT FOR NBA HELPER
from Utils.next_best_action_helper import generate_next_best_actions
from Utils.caculate_accuracy_of_chat import calculate_accuracy_score
logger = get_logger("AGGREGATOR")


# ==========================================
# 2. SYSTEM PROMPT WITH JSON INSTRUCTIONS
# ==========================================
# ==========================================
# 2. SYSTEM PROMPT WITH JSON INSTRUCTIONS
# ==========================================
AGGREGATOR_SYSTEM_PROMPT = """You are the final response generator for a Supply Chain and Logistics Assistant.
Your system has completed several internal tasks to gather information for the user.

Your job is to synthesize these internal worker results into a single, cohesive, and user-friendly response that directly answers the user's original query.

GENERAL RULES:
1. Do NOT mention "workers", "tasks", or internal processes to the user. Present the information as if you knew it instantly.
2. Aggregate the data logically. Use Markdown tables, lists, or bold text for readability.
3. If any worker reported an error or failed to find information, politely apologize and explain what specific information could not be retrieved.

CITATION RULES (MANDATORY):
4. Every major fact, status update, policy reference, or news event MUST include an inline citation showing exactly where it came from.
5. Apply citations using the exact source tags or URLs provided in the internal data.
   - For Internal DB/Tools: "...vehicle speed is 65 km/h [Source: Secure Logistics Database]."
   - For RAG Docs: "...claims must be filed within 24 hours [Source: operations_guide.pdf, Page 12]."
   - For Web/News: "...flooding reported on NH-48 ([Times of India](https://url_here))."
6. Do NOT invent, hallucinate, or guess sources. Only cite sources explicitly present in the provided internal data.

CHART GENERATION RULES:
7. If the internal data contains numeric metrics, route comparisons, delay times, or inventory counts, set 'is_chartable' to true and generate a 'chart_payload'.
8. If the data is purely text (e.g., policies, weather summaries, news), set 'is_chartable' to false and 'chart_payload' to null.

OUTPUT FORMAT (STRICT JSON REQUIRED):
You MUST output your response as a valid JSON object. Do not include any conversational text before or after the JSON.

Required JSON Schema Example:
{{
  "final_answer": "The delivery to Mumbai is delayed by 45 minutes [Source: Logistics DB].",
  "is_chartable": true,
  "chart_payload": {{
    "chart_type": "bar",
    "title": "Delivery Delays (Minutes)",
    "labels": ["Mumbai"],
    "datasets": [
      {{
        "label": "Delay",
        "data": [45]
      }}
    ]
  }}
}}
"""


def aggregator_node(state: SupplyChainState):
    logger.info("--- 📊 RUNNING AGGREGATOR NODE ---")
    question = state.get("question", "")
    worker_responses_list = state.get("worker_responses", [])
    state_messages = state.get("messages", [])
    
    # 🌟 FIX: Check if the Orchestrator already provided a direct answer
    existing_generation = state.get("generation", "")
    
    if not worker_responses_list:
        # If Orchestrator bypassed workers but gave a valid explanation (e.g., "I don't have that tool")
        if existing_generation:
            logger.info("ℹ️ Orchestrator provided a direct answer. Bypassing synthesis.")
            
            # Optionally add the NBA and confidence score to the Orchestrator's direct answer
            nba_list = generate_next_best_actions(user_query=question, final_response=existing_generation)
            if nba_list:
                existing_generation += "\n\n**💡 Suggested Next Actions:**\n" + "\n".join([f"- {action}" for action in nba_list])
                
            return {
                "generation": existing_generation,
                "next_best_actions": nba_list,
                "messages": [AIMessage(content=existing_generation)]
            }
            
        # Only throw the generic error if BOTH workers failed AND Orchestrator gave no explanation
        else:
            logger.warning("⚠️ No worker responses found. Returning default error to user.")
            error_msg = "I'm sorry, but I was unable to retrieve the requested information at this time."
            
            if state.get("loop_count", 0) >= 3:
                error_msg += " (System Request Timeout)"
                
            logger.info(f"📤 [AGGREGATOR OUTPUT] -> {error_msg}")
            return {
                "generation": error_msg,
                "messages": [AIMessage(content=error_msg)]
            }

        
    combined_worker_data = "\n\n".join(worker_responses_list)
    logger.info(f"📥 [AGGREGATOR INPUT] Aggregating {len(worker_responses_list)} worker result(s).")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", AGGREGATOR_SYSTEM_PROMPT),
        ("human", "Original User Query: {question}\n\nInternal Data Gathered:\n{worker_data}")
    ])
    
    # 🌟 Removed structured output, using standard LLM call
    chain = prompt | synthesis_llm
    
    try:
        response = chain.invoke({
            "question": question,
            "worker_data": combined_worker_data
        })
        
        # 🌟 MANUAL JSON PARSING (Same as Orchestrator)
        raw_content = response.content.strip()
        
        if raw_content.startswith("```"):
            lines = raw_content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_content = "\n".join(lines).strip()

        data = json.loads(raw_content)
        parsed_response = AggregatorOutput.model_validate(data)
        
        # Extract the text
        final_answer = parsed_response.final_answer
        
        # Extract the chart data (convert to dict if it exists)
        chart_data = parsed_response.chart_payload.model_dump() if (parsed_response.is_chartable and parsed_response.chart_payload) else None
        
        if chart_data:
            logger.info(f"📊 Chart Payload Generated: {chart_data['title']}")
        
        if state.get("loop_count", 0) >= 3:
            final_answer += "\n\n> ⚠️ **System Notice:** _Processing timed out before all tasks could complete. The information above may be partial._"

        logger.info("📊 Calculating deterministic confidence score...")
        accuracy_text = calculate_accuracy_score(state_messages)
        final_answer += f"\n\n---\n**📊 System Confidence:** {accuracy_text}"
        
        logger.info("🧠 Generating Next Best Actions...")
        nba_list = generate_next_best_actions(user_query=question, final_response=final_answer)
        
        if nba_list:
            nba_text = "\n\n**💡 Suggested Next Actions:**\n" + "\n".join([f"- {action}" for action in nba_list])
            final_answer += nba_text
        
        logger.info("✅ Aggregator successfully generated response, accuracy score, NBA, and Chart Payload.")
        logger.info(f"📤 [AGGREGATOR OUTPUT TEXT] -> \n{final_answer}")
        
        return {
            "generation": final_answer,
            "next_best_actions": nba_list,
            "chart_payload": chart_data,
            "messages": [AIMessage(content=final_answer)]
        }
        
    except (json.JSONDecodeError, ValidationError, Exception) as e:
        logger.error(f"❌ Aggregator failed during synthesis/parsing: {e}")
        fallback = "An error occurred while formatting your final response. Please try again."
        logger.info(f"📤 [AGGREGATOR FALLBACK OUTPUT] -> {fallback}")
        
        return {
            "generation": fallback,
            "next_best_actions": [], 
            "chart_payload": None,
            "messages": [AIMessage(content=fallback)]
        }
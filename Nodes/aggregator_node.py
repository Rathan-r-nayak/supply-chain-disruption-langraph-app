from State.supply_chain_state import SupplyChainState
from Utils.logger import get_logger
from langchain_core.prompts import ChatPromptTemplate
from Config.llm_config import fast_llm
from langchain_core.messages import AIMessage

logger = get_logger("AGGREGATOR")

AGGREGATOR_SYSTEM_PROMPT = """You are the final response generator for a secure Banking Assistant.
Your system has completed several internal tasks to gather information for the user.

Your job is to synthesize these internal worker results into a single, cohesive, and user-friendly response that directly answers the user's original query.

Rules:
1. Do NOT mention "workers", "tasks", or internal processes to the user. Present the information as if you knew it instantly.
2. Aggregate the data logically. Use Markdown tables, lists, or bold text for readability.
3. If any worker reported an error or failed to find information, politely apologize and explain what specific information could not be retrieved.
"""

def aggregator_node(state: SupplyChainState):
    question = state.get("question", "")
    worker_responses_list = state.get("worker_responses", [])
    
    combined_worker_data = "\n\n".join(worker_responses_list)
    
    if not combined_worker_data:
        logger.warning("⚠️ No worker responses found. Returning default error to user.")
        error_msg = "I'm sorry, but I was unable to process your request at this time."
        
        # Log the exact error output provided to user
        logger.info(f"📤 [AGGREGATOR OUTPUT] -> {error_msg}")
        return {
            "generation": error_msg,
            "messages": [AIMessage(content=error_msg)]
        }

    logger.info(f"📥 [AGGREGATOR INPUT] Aggregating {len(worker_responses_list)} worker result(s).")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", AGGREGATOR_SYSTEM_PROMPT),
        ("human", "Original User Query: {question}\n\nInternal Data Gathered:\n{worker_data}")
    ])
    
    chain = prompt | fast_llm
    
    try:
        response = chain.invoke({
            "question": question,
            "worker_data": combined_worker_data
        })
        
        final_answer = response.content
        
        # 🌟 Log the exact final output being delivered to the state
        logger.info("✅ Aggregator successfully generated the final response.")
        logger.info(f"📤 [AGGREGATOR OUTPUT] -> \n{final_answer}")
        
        return {
            "generation": final_answer,
            "messages": [AIMessage(content=final_answer)]
        }
    except Exception as e:
        logger.error(f"❌ Aggregator failed during synthesis: {e}")
        fallback = "An error occurred while formatting your final response."
        logger.info(f"📤 [AGGREGATOR FALLBACK OUTPUT] -> {fallback}")
        
        return {
            "generation": fallback,
            "messages": [AIMessage(content=fallback)]
        }
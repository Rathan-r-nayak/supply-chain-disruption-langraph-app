import json
from typing import Any
from pydantic import ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from State.supply_chain_state import SupplyChainState
from Schema.task import OrchestratorPlan
from Utils.helpers import format_chat_history, get_short_term_memory
from Utils.logger import get_logger
from Config.llm_config import brain_llm, test_brain_llm

logger = get_logger("ORCHESTRATOR")

ORCHESTRATOR_SYSTEM_PROMPT = """You are the Lead Orchestrator for a secure Supply Chain & Logistics Assistant.
Your job is to break the user's request into actionable tasks based ONLY on the tools you have available.

System Context:
The current active user ID is: '{user_id}'. 
- If this ID is a number, the user is a Driver. 
- If this ID is 'admin' or 'system_admin', the user is the Administrator.
NEVER ask the user for their ID. Use this system-provided ID for all internal context and tool calls.

User Profile / Long-Term Memory:
{user_memories}

Worker Results so far:
{worker_responses}

AVAILABLE TOOLS:
{tool_descriptions}

Available Tool Types (You MUST choose one of these exact values for 'tool_type'):
- "safe": Use for routine, low-risk data fetching (e.g., tracking routes, fetching weather).
- "sensitive": Use for high-risk actions (e.g., flagging disruptions, rerouting) that require human approval.
- "rag": Use specifically when searching internal logistics documentation or policies.

CRITICAL RULES:
1. DEPENDENCIES & EXECUTION: 
   - PARALLEL: If multiple tasks are independent, output them together in the 'tasks' list.
   - SEQUENTIAL: If Task B depends on the output of Task A, **ONLY output Task A right now**. Wait for the 'Worker Results' to populate in the next loop before generating Task B.
2. NEVER assign a task for a tool that does not exist in your "AVAILABLE TOOLS" list.
3. 🛑 FAILURE RULE: If the "Worker Results" indicate a tool failed or data is missing, DO NOT retry the exact same task. Set is_workflow_complete to True and explain the limitation in your final_answer.
4. If the chat history or worker results already contain the answers you need, set is_workflow_complete to True and provide the final_answer.
5. Output STRICTLY valid JSON with no conversational text before or after.

Required JSON Schema Example:
{{
  "is_workflow_complete": false,
  "tasks": [
    {{
      "task_id": "task_1",
      "description": "Fetch weather for Bengaluru using fetch_weather_tool",
      "assigned_worker": "LogisticsService",
      "tool_type": "safe"
    }}
  ],
  "final_answer": ""
}}
"""

def get_orchestrator_node(all_mcp_tools: list[Any]):
    """
    Factory function to inject available tools into the orchestrator prompt.
    """
    # Dynamically build a string of available tools and their descriptions
    tool_info = "\n".join([f"- {getattr(tool, 'name', 'Unknown')}: {getattr(tool, 'description', 'No description provided')}" for tool in all_mcp_tools])

    def orchestrator_node(state: SupplyChainState, config: RunnableConfig):
        # Read and increment the loop count
        current_count = state.get("loop_count", 0)
        new_count = current_count + 1
        
        logger.info(f"--- 🧠 RUNNING ORCHESTRATOR (Iteration: {new_count}) ---")
        
        memories = state.get("memories", "No known facts.")
        user_id = config.get("configurable", {}).get("user_id", "Unknown")
        
        worker_responses = state.get("worker_responses", [])
        formatted_responses = "\n".join(worker_responses) if worker_responses else "None yet."
            
        question = state.get("question", "")
        
        # 🌟 CHANGED: Build the combined history context
        summary = state.get("conversation_summary", "")
        raw_messages = state.get("messages", [])
        immediate_context_msgs = raw_messages[-2:] if len(raw_messages) >= 2 else raw_messages
        immediate_context_str = format_chat_history(immediate_context_msgs)
        
        chat_history_text = f"Summary of older conversation:\n{summary}\n\nImmediate Context:\n{immediate_context_str}"
        
        # Inject tool_info into the prompt template formatting
        prompt = ChatPromptTemplate.from_messages([
            ("system", ORCHESTRATOR_SYSTEM_PROMPT),
            ("human", "Conversation Context:\n{chat_history}\n\nRequest: {question}") # Updated label to "Conversation Context"
        ])

        chain = prompt | test_brain_llm

        try:
            response = chain.invoke({
                "user_id": user_id,
                "user_memories": memories,
                "worker_responses": formatted_responses,
                "tool_descriptions": tool_info, # 🌟 INJECTED TOOLS HERE
                "chat_history": chat_history_text,
                "question": question,
            })
            
            raw_content = response.content.strip()
            
            if raw_content.startswith("```"):
                lines = raw_content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_content = "\n".join(lines).strip()

            data = json.loads(raw_content)
            plan = OrchestratorPlan.model_validate(data)
            
            if plan.is_workflow_complete:
                logger.info("✅ Workflow complete.")
                return {
                    "is_workflow_complete": True, 
                    "tasks": [], 
                    "generation": plan.final_answer,
                    "loop_count": 0  
                }
            else:
                logger.info(f"👷 Created {len(plan.tasks)} tasks based on available tools.")
                return {
                    "is_workflow_complete": False, 
                    "tasks": plan.tasks,
                    "loop_count": new_count 
                }

        except (json.JSONDecodeError, ValidationError, Exception) as e:
            logger.error(f"❌ Orchestrator plan failed: {e}")
            return {
                "is_workflow_complete": True, 
                "generation": "I encountered an error while planning tasks.",
                "loop_count": 0
            }
            
    return orchestrator_node
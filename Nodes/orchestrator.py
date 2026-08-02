import json
from pydantic import ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from State.supply_chain_state import SupplyChainState
from Schema.task import OrchestratorPlan
from Utils.helpers import format_chat_history, get_short_term_memory
from Utils.logger import get_logger
from Config.llm_config import primary_llm

logger = get_logger("ORCHESTRATOR")

ORCHESTRATOR_SYSTEM_PROMPT = """You are the Lead Orchestrator for a secure Supply Chain & Logistics Assistant.
Your job is to break the user's request into a list of parallel tasks.

System Context:
The current active user ID is: '{user_id}'. 
- If this ID is a number, the user is a Driver. 
- If this ID is 'admin' or 'system_admin', the user is the Administrator.
NEVER ask the user for their ID. Use this system-provided ID for all internal context and tool calls. If the user is an admin running a global scan, you do not need a driver ID.

User Profile / Long-Term Memory:
{user_memories}

Worker Results so far:
{worker_responses}

Available Tool Types (You MUST choose one of these exact values for 'tool_type'):
- "safe": Use for routine, low-risk data fetching (e.g., tracking routes, fetching weather, reading driver details).
- "sensitive": Use for high-risk actions (e.g., flagging disruptions, rerouting) that require human-in-the-loop approval.
- "rag": Use specifically when searching internal logistics documentation, knowledge bases, or policy PDFs.

CRITICAL RULES:
1. If you need data, set is_workflow_complete to False and generate a list of Tasks.
2. 🛑 FAILURE RULE: If the "Worker Results" indicate a tool is missing, an error occurred, or a task cannot be completed, DO NOT retry the task. You MUST set is_workflow_complete to True and explain the limitation in your final_answer.
3. Break independent requests into separate tasks (e.g., Task 1: Check weather, Task 2: Check traffic).
4. If the chat history or worker results already contain the answers you need, set is_workflow_complete to True and provide the final_answer.
5. Output STRICTLY valid JSON with no conversational text before or after. Do NOT include comments.

Required JSON Schema Example:
{{
  "is_workflow_complete": false,
  "tasks": [
    {{
      "task_id": "task_1",
      "description": "Retrieve the active trip details for driver ID {user_id}.",
      "assigned_worker": "LogisticsService",
      "tool_type": "safe"
    }}
  ],
  "final_answer": ""
}}
"""

def orchestrator_node(state: SupplyChainState, config: RunnableConfig):
    logger.info("--- 🧠 RUNNING ORCHESTRATOR: PLANNING TASKS ---")
    memories = state.get("memories", "No known facts.")
    
    # Extract user_id from config
    user_id = config.get("configurable", {}).get("user_id", "Unknown")
    
    worker_responses = state.get("worker_responses", [])
    formatted_responses = "\n".join(worker_responses) if worker_responses else "None yet."
        
    question = state.get("question", "")
    recent_messages = get_short_term_memory(state.get("messages", []), k=6)
    chat_history_text = format_chat_history(recent_messages)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", ORCHESTRATOR_SYSTEM_PROMPT),
        ("human", "Conversation History:\n{chat_history}\n\nRequest: {question}")
    ])

    chain = prompt | primary_llm

    try:
        response = chain.invoke({
            "user_id": user_id,
            "user_memories": memories,
            "worker_responses": formatted_responses,
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
                "generation": plan.final_answer
            }
        else:
            logger.info(f"👷 Created {len(plan.tasks)} parallel tasks.")
            return {
                "is_workflow_complete": False, 
                "tasks": plan.tasks
            }

    except (json.JSONDecodeError, ValidationError, Exception) as e:
        logger.error(f"❌ Orchestrator plan failed: {e}")
        return {"is_workflow_complete": True, "generation": "I encountered an error while planning tasks."}
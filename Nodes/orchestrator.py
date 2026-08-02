import json
from pydantic import ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from State.banking_state import BankingState
from Schema.task import OrchestratorPlan
from Utils.helpers import format_chat_history, get_short_term_memory
from Utils.Logger import get_logger
from Config.llm_config import primary_llm

logger = get_logger("ORCHESTRATOR")

# 🌟 1. Updated Prompt with the "Worker Results" variable and strict Failure Rules
ORCHESTRATOR_SYSTEM_PROMPT = """You are the Lead Orchestrator for a secure Banking Assistant.
Your job is to break the user's request into a list of parallel tasks.

User Profile / Long-Term Memory:
{user_memories}

Worker Results so far:
{worker_responses}

Available Tool Types (You MUST choose one of these exact values for 'tool_type'):
- "safe": Use for routine, low-risk data fetching (e.g., checking balances, reading public policies, viewing account details).
- "sensitive": Use for high-risk, write, or financial transfer actions that require human-in-the-loop approval.
- "rag": Use specifically when searching internal documentation, knowledge bases, or policy PDFs.

CRITICAL RULES:
1. If you need data, set is_workflow_complete to False and generate a list of Tasks.
2. 🛑 FAILURE RULE: If the "Worker Results" indicate a tool is missing, an error occurred, or a task cannot be completed, DO NOT retry the task. You MUST set is_workflow_complete to True and explain the limitation in your final_answer.
3. Break independent requests into separate tasks (e.g., Task 1: Check balance, Task 2: Check loan rates).
4. If the chat history or worker results already contain the answers you need, set is_workflow_complete to True and provide the final_answer.
5. Output STRICTLY valid JSON with no conversational text before or after. Do NOT include comments.

Required JSON Schema Example (NOTICE THE DOUBLE CURLY BRACES TO ESCAPE PYTHON FORMATTING):
{{
  "is_workflow_complete": false,
  "tasks": [
    {{
      "task_id": "task_1",
      "description": "Retrieve the balance for account number 1001.",
      "assigned_worker": "AccountService",
      "tool_type": "safe"
    }}
  ],
  "final_answer": ""
}}
"""

def orchestrator_node(state: BankingState):
    logger.info("--- 🧠 RUNNING ORCHESTRATOR: PLANNING TASKS ---")
    memories = state.get("memories", "No known facts.")
    
    # 🌟 2. Extract and format the worker responses
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
        # 🌟 3. Pass the formatted responses to the LLM
        response = chain.invoke({
            "user_memories": memories,
            "worker_responses": formatted_responses,
            "chat_history": chat_history_text,
            "question": question,
        })
        
        raw_content = response.content.strip()
        
        # Clean markdown formatting if present
        if raw_content.startswith("```"):
            lines = raw_content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_content = "\n".join(lines).strip()

        # Parse and validate
        data = json.loads(raw_content)
        plan = OrchestratorPlan.model_validate(data)
        
        if plan.is_workflow_complete:
            logger.info("✅ Workflow complete.")
            return {
                "is_workflow_complete": True, 
                "tasks": [], # 🌟 CLEAR TASKS
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
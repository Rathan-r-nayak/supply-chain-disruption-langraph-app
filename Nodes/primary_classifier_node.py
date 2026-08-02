import json
from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from State.banking_state import BankingState
from Config.llm_config import fast_llm
from Utils.Logger import get_logger
from Utils.helpers import format_chat_history, get_short_term_memory

logger = get_logger("Primary Classifier")

class TriageDecision(BaseModel):
    is_workflow_required: bool = Field(
        description="True if the request needs banking tools or complex logic. False for simple greetings or chit-chat."
    )
    direct_response: str = Field(
        description="If workflow is not required, provide the direct conversational response here.", 
        default="How can I help you?"
    )

TRIAGE_SYSTEM_PROMPT = """You are the first line of defense for a secure Banking Assistant.
Your job is to decide if the user's request requires the full banking workflow (tools, orchestration) or if it can be answered directly (e.g., greetings, pleasantries).

User Profile / Long-Term Memory:
{user_memories}

Rules:
1. If the user greets you, use their name or profile details from Long-Term Memory to personalize the 'direct_response'.
2. If the user asks about banking data, policies, or transactions, set 'is_workflow_required' to True.
3. Do not attempt to answer banking questions directly. Always route them to the workflow.
4. Output STRICTLY valid JSON with no conversational text before or after.
Example format:
{{"is_workflow_required": false, "direct_response": "Hello Rathan, how can I help you today?"}}
"""

def triage_router(state: BankingState):
    question = state.get("question", "")
    memories = state.get("memories", "No known facts.")
    
    logger.info(f"🗣️ USER REQ : {question}")
    logger.info("--- 🛡️ RUNNING INTENT ROUTER & GATEKEEPER CHECK ---")

    if not question:
        return {
            "requires_workflow": False,
            "generation": "How can I help you today?",
            "worker_responses": [],
        }

    recent_messages = get_short_term_memory(state.get("messages", []), k=4)
    chat_history = format_chat_history(recent_messages)

    prompt = ChatPromptTemplate.from_messages([
        ("system", TRIAGE_SYSTEM_PROMPT),
        ("human", "Chat History:\n{chat_history}\n\nUser Request: {question}"),
    ])

    chain = prompt | fast_llm

    try:
        # 🌟 FIX 2: Use standard .invoke() instead of await .ainvoke()
        response = chain.invoke({
            "question": question,
            "user_memories": memories,
            "chat_history": chat_history
        })
        
        raw_content = response.content.strip()
        logger.info(f"Raw LLM Response: {raw_content}")

        # Clean markdown formatting if present
        if raw_content.startswith("```"):
            lines = raw_content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:] 
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1] 
            raw_content = "\n".join(lines).strip()

        data = json.loads(raw_content)
        decision = TriageDecision.model_validate(data)
        
        logger.info(f"✅ Successfully validated model: {decision}")

        if decision.is_workflow_required:
            return {
                "requires_workflow": True,
                "worker_responses": [],
            }

        return {
            "requires_workflow": False,
            "messages": [AIMessage(content=decision.direct_response)],
            "generation": decision.direct_response,
            "worker_responses": [],
        }

    except (json.JSONDecodeError, ValidationError, Exception) as e:
        logger.error(f"❌ Primary classifier parsing/validation failed: {e}")
        logger.warning("Failsafe triggered: Defaulting to Action pipeline.")

        return {
            "requires_workflow": True,
            "worker_responses": [],
        }
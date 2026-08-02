import json
from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from State.supply_chain_state import SupplyChainState
from Config.llm_config import fast_llm
from Utils.logger import get_logger
from Utils.helpers import format_chat_history, get_short_term_memory

logger = get_logger("Primary Classifier")

class TriageDecision(BaseModel):
    is_workflow_required: bool = Field(
        description="True if the request needs supply chain tools, news fetching, or complex logic. False for simple greetings."
    )
    direct_response: str = Field(
        description="If workflow is not required, provide the direct conversational response here.", 
        default="How can I help you?"
    )

TRIAGE_SYSTEM_PROMPT = """You are the first line of defense for a secure Supply Chain & Logistics Assistant.
Your job is to decide if the user's request requires the full orchestration workflow (tools, database checks) or if it can be answered directly.

System Context:
The current active user ID is: '{user_id}'. 
- If this ID is a number, the user is a Driver. 
- If this ID is 'admin' or 'system_admin', the user is the Administrator.
NEVER ask the user for their ID or role. Use this system-provided ID for all internal context and tool calls.

User Profile / Long-Term Memory:
{user_memories}

Rules:
1. If the user greets you, use their ID or profile details to personalize the 'direct_response'.
2. If the user asks about trips, weather, disruptions, or route details, set 'is_workflow_required' to True.
3. Do not attempt to answer logistics questions directly. Always route them to the workflow.
4. Output STRICTLY valid JSON with no conversational text before or after.
"""

def triage_router(state: SupplyChainState, config: RunnableConfig):
    question = state.get("question", "")
    memories = state.get("memories", "No known facts.")
    
    # Extract the user/driver ID passed from server.py config
    user_id = config.get("configurable", {}).get("user_id", "Unknown")
    
    logger.info(f"🗣️ USER REQ : {question} | User ID: {user_id}")

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
        response = chain.invoke({
            "question": question,
            "user_memories": memories,
            "chat_history": chat_history,
            "user_id": user_id
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
        decision = TriageDecision.model_validate(data)

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

    except Exception as e:
        logger.error(f"❌ Primary classifier parsing/validation failed: {e}")
        return {
            "requires_workflow": True,
            "worker_responses": [],
        }
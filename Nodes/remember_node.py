# Nodes/remember_node.py
import uuid
import json
from pydantic import BaseModel, Field, ValidationError
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from State.banking_state import BankingState
from Config.llm_config import fast_llm
from Utils.Logger import get_logger

logger = get_logger("REMEMBER_NODE")

class MemoryFact(BaseModel):
    fact: str = Field(description="A distinct, standalone fact about the user's finances or preferences.")

class MemoryExtraction(BaseModel):
    facts: List[MemoryFact] = Field(
        description="List of extracted facts. Empty if nothing new is learned.", 
        default_factory=list
    )

def remember_node(state: BankingState, config: RunnableConfig, store: BaseStore):
    logger.info("--- 💾 RUNNING REMEMBER NODE (LTM EXTRACTION) ---")
    
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    namespace = ("user", user_id, "facts")
    
    messages = state.get("messages", [])
    if not messages:
        return {}
        
    # Grab the latest human message to look for direct introductions
    latest_human_msg = next((msg.content for msg in reversed(messages) if msg.type == 'human'), "")
    
    # Force-extract obvious statements like "I am [Name]" or "My name is [Name]" instantly
    import re
    name_match = re.search(r"(?:i am|i'm|my name is)\s+([a-zA-Z]+)", latest_human_msg, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).capitalize()
        fact_text = f"User's name is {name}"
        store.put(namespace, str(uuid.uuid4()), {"fact": fact_text})
        logger.info(f"💾 [LTM Direct Saved] -> {fact_text}")
        return {}

    # Otherwise, run standard LLM extraction on transcript history...
    transcript = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in messages[-4:]])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an advanced Long-Term Memory (LTM) extraction agent. 
    Extract permanent, durable user facts (identity, account preferences, explicit rules). 
    If the user states their name, occupation, or preferences, extract them into the list.
    Output STRICTLY as a JSON object:
    {{
        "memories": ["User's name is Rathan"]
    }}
    """),
            ("human", "Conversation:\n{chat_history}")
    ])
    
    chain = prompt | fast_llm
    
    try:
        response = chain.invoke({"chat_history": transcript})
        raw_content = response.content.strip()
        
        if raw_content.startswith("```"):
            lines = raw_content.splitlines()
            if lines[0].startswith("```"): lines = lines[1:]
            if lines and lines[-1].startswith("```"): lines = lines[:-1]
            raw_content = "\n".join(lines).strip()

        data = json.loads(raw_content)
        extraction = MemoryExtraction.model_validate(data)
        
        for item in extraction.facts:
            store.put(namespace, str(uuid.uuid4()), {"fact": item.fact})
            logger.info(f"💾 [LTM Saved] -> {item.fact}")
            
    except Exception as e:
        logger.error(f"❌ Failed to extract memory: {e}")
        
    return {}
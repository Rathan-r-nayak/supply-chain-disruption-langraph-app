# Nodes/remember_node.py
import uuid
import json
import re
from pydantic import BaseModel, Field, ValidationError
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from State.supply_chain_state import SupplyChainState
from Config.llm_config import fast_llm
from Utils.logger import get_logger

logger = get_logger("REMEMBER_NODE")

class MemoryFact(BaseModel):
    fact: str = Field(description="A distinct, standalone fact about the user's finances or preferences.")

class MemoryExtraction(BaseModel):
    facts: List[MemoryFact] = Field(
        description="List of extracted facts. Empty if nothing new is learned.", 
        default_factory=list
    )

def remember_node(state: SupplyChainState, config: RunnableConfig, store: BaseStore):
    logger.info("--- 💾 RUNNING REMEMBER NODE (LTM EXTRACTION) ---")
    
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    namespace = ("user", user_id, "facts")
    
    messages = state.get("messages", [])
    if not messages:
        return {}
        
    # 1. Fetch existing memories to prevent duplication
    existing_items = store.search(namespace)
    existing_facts_raw = [item.value.get("fact", "") for item in existing_items]
    existing_facts_lower = [f.lower() for f in existing_facts_raw]
    
    # Grab the latest human message to look for direct introductions
    latest_human_msg = next((msg.content for msg in reversed(messages) if msg.type == 'human'), "")
    
    # 2. Force-extract obvious statements with deduplication check
    name_match = re.search(r"(?:i am|i'm|my name is)\s+([a-zA-Z]+)", latest_human_msg, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).capitalize()
        fact_text = f"User's name is {name}"
        
        if fact_text.lower() not in existing_facts_lower:
            store.put(namespace, str(uuid.uuid4()), {"fact": fact_text})
            logger.info(f"💾 [LTM Direct Saved] -> {fact_text}")
        else:
            logger.info(f"⏭️ [LTM Skipped] Fact already exists: {fact_text}")
        return {}

    # 3. Standard LLM extraction with existing memory context
    transcript = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in messages[-4:]])
    existing_memories_str = "\n".join([f"- {f}" for f in existing_facts_raw]) if existing_facts_raw else "None"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an advanced Long-Term Memory (LTM) extraction agent. 
    Extract permanent, durable user facts (identity, account preferences, explicit rules). 
    
    EXISTING MEMORIES FOR THIS USER:
    {existing_memories}
    
    CRITICAL RULE: DO NOT extract or output facts that are already covered by the EXISTING MEMORIES. Only extract completely NEW information.
    
    Output STRICTLY as a JSON object matching this schema:
    {{
        "facts": [
            {{"fact": "New extracted fact here"}}
        ]
    }}
    """),
        ("human", "Conversation:\n{chat_history}")
    ])
    
    chain = prompt | fast_llm
    
    try:
        response = chain.invoke({
            "existing_memories": existing_memories_str,
            "chat_history": transcript
        })
        raw_content = response.content.strip()
        
        if raw_content.startswith("```"):
            lines = raw_content.splitlines()
            if lines[0].startswith("```"): lines = lines[1:]
            if lines and lines[-1].startswith("```"): lines = lines[:-1]
            raw_content = "\n".join(lines).strip()

        data = json.loads(raw_content)
        extraction = MemoryExtraction.model_validate(data)
        
        for item in extraction.facts:
            # Final safety check in case the LLM ignored instructions
            if item.fact.lower() not in existing_facts_lower:
                store.put(namespace, str(uuid.uuid4()), {"fact": item.fact})
                logger.info(f"💾 [LTM LLM Saved] -> {item.fact}")
            else:
                logger.info(f"⏭️ [LTM LLM Skipped] Fact already exists: {item.fact}")
                
    except Exception as e:
        logger.error(f"❌ Failed to extract memory: {e}")
        
    return {}
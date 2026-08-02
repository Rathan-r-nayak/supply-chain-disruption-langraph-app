# Nodes/recall_node.py
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from State.banking_state import BankingState
from Utils.Logger import get_logger

logger = get_logger("RECALL_NODE")

def recall_node(state: BankingState, config: RunnableConfig, store: BaseStore):
    """Fetches memories from the Store and injects them into the State."""
    logger.info("--- 🧠 RECALLING USER MEMORIES ---")
    
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    namespace = ("user", user_id, "facts")
    
    try:
        saved_items = store.search(namespace)
        
        if not saved_items:
            logger.info(f"ℹ️ No previous memories found for user '{user_id}'.")
            return {"memories": "No known facts."}
        
        logger.info(f"✅ Fetched {len(saved_items)} memory fact(s) for user '{user_id}'.")
        
        memory_text = "\n".join([f"- {item.value['fact']}" for item in saved_items])
        
        # 🌟 Passes this string directly into the global state
        return {"memories": memory_text}
        
    except Exception as e:
        logger.error(f"❌ Error searching memories for user '{user_id}': {e}")
        return {"memories": "No known facts."}
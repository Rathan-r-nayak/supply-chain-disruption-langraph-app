from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from State.supply_chain_state import SupplyChainState
from Utils.logger import get_logger

logger = get_logger("RECALL_NODE")

def recall_node(state: SupplyChainState, config: RunnableConfig, store: BaseStore):
    """Fetches memories from the Store and injects them into the State."""
    logger.info("--- 🧠 RUNNING RECALL NODE (LTM RETRIEVAL) ---")
    
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    namespace = ("user", user_id, "facts")
    logger.info(f"🔍 Searching long-term memory for User ID: '{user_id}' (Namespace: {namespace})")
    
    try:
        saved_items = store.search(namespace)
        
        if not saved_items:
            logger.info(f"ℹ️ No prior LTM facts found for user '{user_id}'.")
            return {"memories": "No known facts."}
        
        memory_text = "\n".join([f"- {item.value['fact']}" for item in saved_items])
        logger.info(f"✅ Retrieved {len(saved_items)} LTM fact(s) for user '{user_id}'.")
        return {"memories": memory_text}
        
    except Exception as e:
        logger.error(f"❌ Failed to recall LTM facts for user '{user_id}': {e}")
        return {"memories": "No known facts."}
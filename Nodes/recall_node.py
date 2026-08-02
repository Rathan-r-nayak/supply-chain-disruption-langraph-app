from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from State.supply_chain_state import SupplyChainState
from Utils.logger import get_logger

logger = get_logger("RECALL_NODE")

def recall_node(state: SupplyChainState, config: RunnableConfig, store: BaseStore):
    """Fetches memories from the Store and injects them into the State."""
    
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    namespace = ("user", user_id, "facts")
    
    try:
        saved_items = store.search(namespace)
        
        if not saved_items:
            return {"memories": "No known facts."}
        
        memory_text = "\n".join([f"- {item.value['fact']}" for item in saved_items])
        return {"memories": memory_text}
        
    except Exception as e:
        return {"memories": "No known facts."}
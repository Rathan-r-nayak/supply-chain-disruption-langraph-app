# Nodes/semantic_cache_nodes.py
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from State.supply_chain_state import SupplyChainState
from cache_config.cache_config import app_semantic_cache
from Utils.logger import get_logger

logger = get_logger("SEMANTIC_CACHE")

# 🌟 Define exactly which tools access private user data
PRIVATE_TOOLS = {
    "get_driver_trip_details", 
    "start_trip", 
    "advance_trip", 
    "flag_disruptions"
}

def check_semantic_cache_node(state: SupplyChainState, config: RunnableConfig):
    """Runs before triage. Checks if we already have an answer for this question."""
    logger.info("--- 🔍 CHECKING SEMANTIC CACHE ---")
    
    question = state.get("question", "")
    user_id = config.get("configurable", {}).get("user_id", "Unknown")
    
    cached_response = app_semantic_cache.get(query=question, current_user_id=user_id)
    
    if cached_response:
        msg = f"{cached_response}\n\n*(This was a lightning-fast cached response ⚡)*"
        return {
            "is_cache_hit": True,
            "generation": msg,
            "messages": [AIMessage(content=msg)]
        }
    
    return {"is_cache_hit": False}

def save_semantic_cache_node(state: SupplyChainState, config: RunnableConfig):
    """Runs at the end of the graph. Dynamically assigns scope based on tool usage."""
    if state.get("is_cache_hit", False):
        return {}

    logger.info("--- 💾 SAVING TO SEMANTIC CACHE ---")
    question = state.get("question", "")
    generation = state.get("generation", "")
    user_id = config.get("configurable", {}).get("user_id", "Unknown")
    
    # DYNAMIC SCOPE DETECTION
    # Default to global cache (so everyone shares the answer)
    detected_scope = "global"
    
    # Look back through the conversation messages to see what tools the AI used
    for msg in state.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                # If ANY private tool was used to generate this answer, lock it to the user
                if tc["name"] in PRIVATE_TOOLS:
                    detected_scope = "user"
                    break
        if detected_scope == "user":
            break
            
    if question and generation:
        app_semantic_cache.set(
            query=question, 
            response=generation, 
            scope=detected_scope, 
            user_id=user_id
        )
        
    return {}
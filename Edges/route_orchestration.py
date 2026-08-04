from langgraph.constants import Send
from State.supply_chain_state import SupplyChainState
from Utils.logger import get_logger

logger = get_logger("ROUTER")

# 🌟 Set your maximum allowed loop limit here
MAX_ORCHESTRATOR_LOOPS = 3

def orchestrator_router(state: SupplyChainState):
    """
    If max loops exceeded, force route to aggregator.
    If complete, route to aggregator.
    If tasks exist, map them to parallel workers using the Send API.
    """
    # 🌟 1. Prevent infinite loops by checking the counter first
    loop_count = state.get("loop_count", 0)
    if loop_count >= MAX_ORCHESTRATOR_LOOPS:
        logger.error(f"⚠️ FORCE BAILOUT: Orchestrator hit max loops ({MAX_ORCHESTRATOR_LOOPS}). Bypassing workers.")
        return "aggregator"

    # 2. Normal completion check
    if state.get("is_workflow_complete", False):
        return "aggregator"
        
    tasks = state.get("tasks", [])
    
    # 🌟 3. Map to parallel workers with the WORKER CLEAN SLATE fix
    return [
        Send("worker_subgraph", {
            "task": task,
            "messages": state.get("messages", []), # Pass context so the worker isn't blind
            "worker_responses": [],                # Reset tool outputs for this specific run
            "executed_tools": []                   # Reset tool tracking for this specific run
        }) 
        for task in tasks
    ]
from langgraph.constants import Send
from State.banking_state import BankingState

def orchestrator_router(state: BankingState):
    """
    If complete, route to aggregator.
    If tasks exist, map them to parallel workers using the Send API.
    """
    if state.get("is_workflow_complete", False):
        return "aggregator"
        
    tasks = state.get("tasks", [])
    
    # 🌟 FIX: Inject an empty messages list so the worker's reducer doesn't crash!
    return [
        Send("worker_subgraph", {
            "task": task,
            "messages": []  # <--- THIS IS THE MAGIC FIX
        }) 
        for task in tasks
    ]
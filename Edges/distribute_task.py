from langgraph.constants import Send
from State.supply_chain_state import SupplyChainState
from Utils.Logger import get_logger

logger = get_logger("DISTRIBUTE_TASK")

def distribute_tasks(state: SupplyChainState):
    """
    Reads the tasks from the state and spins up a worker node for each one.
    """
    tasks = state.get("tasks", [])
    logger.info(f"🔀 DISTRIBUTING: Spinning up {len(tasks)} parallel worker(s).")
    
    # Send API maps the target node name ("worker_node") to the specific input for that node
    # Note: The worker_node must accept a state schema that matches this payload
    return [Send("worker_node", {"task": task}) for task in tasks]
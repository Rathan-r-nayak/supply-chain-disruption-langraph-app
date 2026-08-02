from Utils.Logger import get_logger

logger = get_logger("DISTRIBUTE_TASKS_EDGE")

def distribute_tasks(state):
    tasks = state.get("tasks", [])
    if not tasks:
        logger.info("🔀 ROUTING: No tasks found in state. Routing to aggregator.")
        return "aggregator" # Or END
    
    # Grab the current task
    task = tasks[0]
    
    # Check for 'type' first (since your Task schema uses 'type'), then 'agent'
    if isinstance(task, dict):
        assigned_agent = task.get("type") or task.get("agent", "worker_node")
    else:
        assigned_agent = getattr(task, "type", None) or getattr(task, "agent", "worker_node")
    
    logger.info(f"🔀 DISTRIBUTING TASK: Routing to target agent node '{assigned_agent}' for task: {task}")

    # Map the agent name directly to your graph node names
    if assigned_agent == "account_agent":
        return "account_agent"
    elif assigned_agent == "transaction_agent":
        return "transaction_agent"
    elif assigned_agent == "knowledge_agent":
        return "knowledge_agent"
    else:
        # Fallback if the name doesn't match
        logger.warning(f"⚠️ Unknown agent '{assigned_agent}', falling back to worker_node.")
        return "worker_node"
from Utils.logger import get_logger

logger = get_logger("ROUTE_WORKER_TOOLS")

def route_worker_tools(state: dict):
    messages = state.get("messages", [])
    
    if not messages:
        logger.warning("⚠️ No messages found in worker state! Routing to END.")
        return "__end__"
        
    last_message = messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        task = state.get("task")
        
        if task and hasattr(task, "tool_type"):
            if task.tool_type == "safe":
                return "safe_tools_node"
            elif task.tool_type == "sensitive":
                return "sensitive_tools_node"
            elif task.tool_type == "rag":
                return "rag_subgraph" # 🌟 Updated to exactly match the node name in worker_sub_graph.py
                
        logger.warning("⚠️ Task tool_type missing. Defaulting to safe_tools_node.")
        return "safe_tools_node"
        
    return "__end__"
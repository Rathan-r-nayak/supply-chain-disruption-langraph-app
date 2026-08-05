from Tools.tool_config import RAG_TOOL_NAMES, SENSITIVE_TOOL_NAMES
from Utils.logger import get_logger
# 🌟 Import your tool configs here (adjust the import path to match your project)
# from Config.tool_config import SENSITIVE_TOOL_NAMES

logger = get_logger("ROUTE_WORKER_TOOLS")

def route_worker_tools(state: dict):
    messages = state.get("messages", [])
    
    if not messages:
        logger.warning("⚠️ No messages found in worker state! Routing to END.")
        return "__end__"
        
    last_message = messages[-1]
    
    # Check if the AI decided to call a tool
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # Extract the name of the tool the LLM wants to execute
        # (Since parallel_tool_calls=False, we only need to check the first one)
        tool_name = last_message.tool_calls[0]["name"]
        
        logger.info(f"🔀 Evaluating routing for tool: '{tool_name}'")
        
        # 🌟 1. Check if it requires human approval (Sensitive)
        if tool_name in SENSITIVE_TOOL_NAMES:
            logger.info(f"🚨 '{tool_name}' is a SENSITIVE tool. Routing to sensitive_tools_node.")
            return "sensitive_tools_node"
            
        # 🌟 2. Check if it requires the RAG Subgraph
        elif tool_name in RAG_TOOL_NAMES:
            logger.info(f"📚 '{tool_name}' is a RAG tool. Routing to rag_subgraph.")
            return "rag_subgraph"
            
        # 🌟 3. Default to Safe Tools
        else:
            logger.info(f"✅ '{tool_name}' is a SAFE tool. Routing to safe_tools_node.")
            return "safe_tools_node"
            
    # If the LLM just returned a text summary instead of a tool call, end the worker loop
    return "__end__"
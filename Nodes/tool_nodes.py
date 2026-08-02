from langgraph.prebuilt import ToolNode
from Utils.logger import get_logger

logger = get_logger("TOOL_NODES")

# Define which tools require Human-in-the-Loop approval. 
# (e.g., flagging disruptions could go here if you want LangGraph to pause)
SENSITIVE_TOOL_NAMES = {}

def get_tool_nodes(all_mcp_tools):
    safe_tools = [t for t in all_mcp_tools if t.name not in SENSITIVE_TOOL_NAMES]
    sensitive_tools = [t for t in all_mcp_tools if t.name in SENSITIVE_TOOL_NAMES]
    
    logger.info(f"🔧 Loaded {len(safe_tools)} safe tools and {len(sensitive_tools)} sensitive tools.")
    
    safe_tools_node = ToolNode(safe_tools)
    sensitive_tools_node = ToolNode(sensitive_tools)
    
    return safe_tools_node, sensitive_tools_node
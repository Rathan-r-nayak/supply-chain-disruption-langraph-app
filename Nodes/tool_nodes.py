# Nodes/tool_nodes.py
from langgraph.prebuilt import ToolNode
from Utils.Logger import get_logger

logger = get_logger("TOOL_NODES")

# Define which tools require Human-in-the-Loop approval
SENSITIVE_TOOL_NAMES = {"transfer_money", "pay_bill", "update_password"}

def get_tool_nodes(all_mcp_tools):
    """
    Splits the real MCP tools into safe and sensitive buckets and 
    wraps them in LangGraph ToolNodes.
    """
    # Filter tools based on names
    safe_tools = [t for t in all_mcp_tools if t.name not in SENSITIVE_TOOL_NAMES]
    sensitive_tools = [t for t in all_mcp_tools if t.name in SENSITIVE_TOOL_NAMES]
    
    logger.info(f"🔧 Loaded {len(safe_tools)} safe tools and {len(sensitive_tools)} sensitive tools.")
    
    # Wrap them in LangGraph's executable ToolNode
    safe_tools_node = ToolNode(safe_tools)
    sensitive_tools_node = ToolNode(sensitive_tools)
    
    return safe_tools_node, sensitive_tools_node
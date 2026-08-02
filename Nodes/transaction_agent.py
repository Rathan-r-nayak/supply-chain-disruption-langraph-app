from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from Config.llm_config import fast_llm
from State.supply_chain_state import SupplyChainState
from Utils.Logger import get_logger

logger = get_logger("TRANSACTION_AGENT")

def get_transaction_agent_nodes(all_mcp_tools: list):
    
    # 1. Split the tools by security clearance
    safe_tools = [t for t in all_mcp_tools if t.name == "verify_account"]
    sensitive_tools = [t for t in all_mcp_tools if t.name == "execute_transfer"]
    logger.info(f"💸 Transaction Agent initialized with {len(safe_tools)} safe tool(s) and {len(sensitive_tools)} sensitive tool(s).")
    
    # 2. Bind ALL tools to the LLM so it knows they exist
    llm_with_tools = fast_llm.bind_tools(safe_tools + sensitive_tools)
    
    def transaction_agent_node(state: SupplyChainState):
        logger.info("--- 💸 RUNNING TRANSACTION AGENT ---")
        try:
            response = llm_with_tools.invoke(state["messages"])
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_names = [tc["name"] for tc in response.tool_calls]
                logger.info(f"🛠️ Transaction Agent requested tool call(s): {tool_names}")
            else:
                snippet = response.content[:100] if response.content else ""
                logger.info(f"💬 Transaction Agent generated response: '{snippet}...'")
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"❌ Transaction Agent execution failed: {e}")
            raise e

    # 3. Create TWO separate ToolNodes
    safe_tools_node = ToolNode(safe_tools)
    sensitive_tools_node = ToolNode(sensitive_tools)
    
    return transaction_agent_node, safe_tools_node, sensitive_tools_node
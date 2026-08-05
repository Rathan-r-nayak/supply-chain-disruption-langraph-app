# Nodes/rag_formatter_node.py
from langchain_core.messages import ToolMessage
from State.supply_chain_state import SupplyChainState
from Utils.logger import get_logger

logger = get_logger("RAG_FORMATTER")

def rag_format_response_node(state: SupplyChainState):
    """Packages the verified RAG answer as a tool response."""
    logger.info("--- 📦 RUNNING RAG FORMATTER NODE ---")
    messages = state.get("messages", [])
    
    # Find the AI message that triggered the tool call
    for msg in reversed(messages):
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] in ["search_bank_policies", "search_logistics_policies"]:
                    tool_call_id = tc["id"]
                    
                    # Your final verified RAG text (e.g., from your generation node)
                    final_rag_answer = state.get("generation", "No policies found.")
                    logger.info(f"✅ Packaging RAG answer for tool '{tc['name']}' (ID: {tool_call_id})")
                    
                    # Package it as a ToolMessage so the Master Agent accepts it
                    tool_msg = ToolMessage(
                        content=final_rag_answer,
                        tool_call_id=tool_call_id,
                        name=tc["name"]
                    )
                    return {"messages": [tool_msg]}
                    
    logger.warning("⚠️ No matching RAG tool call found in message history.")
    return {"messages": []} # Fallback
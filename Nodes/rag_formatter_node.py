# Nodes/rag_formatter_node.py
from langchain_core.messages import ToolMessage
from State.banking_state import BankingState

def rag_format_response_node(state: BankingState):
    """Packages the verified RAG answer as a tool response."""
    messages = state.get("messages", [])
    
    # Find the AI message that triggered the tool call
    for msg in reversed(messages):
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "search_bank_policies":
                    tool_call_id = tc["id"]
                    
                    # Your final verified RAG text (e.g., from your generation node)
                    final_rag_answer = state.get("generation", "No policies found.")
                    
                    # Package it as a ToolMessage so the Master Agent accepts it
                    tool_msg = ToolMessage(
                        content=final_rag_answer,
                        tool_call_id=tool_call_id,
                        name="search_bank_policies"
                    )
                    return {"messages": [tool_msg]}
                    
    return {"messages": []} # Fallback
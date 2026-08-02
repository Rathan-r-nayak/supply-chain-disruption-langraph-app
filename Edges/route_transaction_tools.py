from State.banking_state import BankingState
from Utils.Logger import get_logger

logger = get_logger("ROUTE_TRANSACTION_TOOLS")

def route_transaction_tools(state: BankingState):
    """Routes tool calls to either the safe zone or the sensitive (paused) zone."""
    last_message = state["messages"][-1]
    
    # If the LLM didn't call a tool, it's done. Route to aggregator.
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        logger.info("➡️ ROUTING: No tool call detected from Transaction Agent. Routing to aggregator.")
        return "aggregator"
        
    # Check WHICH tool the LLM is trying to call
    tool_name = last_message.tool_calls[0]["name"]
    
    if tool_name == "execute_transfer":
        logger.warning(f"🔒 ROUTING: Sensitive tool '{tool_name}' requested. Routing to sensitive_tools (Human-in-the-Loop check).")
        return "sensitive_tools" # This path will be paused!
    else:
        logger.info(f"⚡ ROUTING: Safe tool '{tool_name}' requested. Routing to safe_tools.")
        return "safe_tools"      # This path runs instantlysensitive_tools
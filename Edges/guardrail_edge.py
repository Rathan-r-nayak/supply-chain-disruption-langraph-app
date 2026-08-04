from langgraph.graph import END
from State.supply_chain_state import SupplyChainState

def guardrail_edge(state: SupplyChainState):
    """Routes to semantic cache if safe, otherwise terminates the graph."""
    if state.get("is_safe") is False:
        return END
    
    # Corrected return value to match main.py routing dictionary
    return "check_semantic_cache"
from langgraph.graph import END

from State.supply_chain_state import SupplyChainState


def guardrail_edge(state: SupplyChainState):
    """Routes to triage if safe, otherwise terminates the graph."""
    if state.get("is_safe") is False:
        return END
    return "triage_router"
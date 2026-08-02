from langgraph.graph import END

from State.banking_state import BankingState


def guardrail_edge(state: BankingState):
    """Routes to triage if safe, otherwise terminates the graph."""
    if state.get("is_safe") is False:
        return END
    return "triage_router"
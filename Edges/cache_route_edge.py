from State.supply_chain_state import SupplyChainState


def route_after_cache(state: SupplyChainState):
    if state.get("is_cache_hit"):
        return "output_guardrail" # Skip straight to the end!
    return "triage_router"
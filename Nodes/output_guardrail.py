import re
from State.supply_chain_state import SupplyChainState
from Utils.logger import get_logger

logger = get_logger("OUTPUT_GUARDRAIL")

PII_OUTPUT_PATTERNS = [
    (r"\b(?:\d[ -]*?){13,16}\b", "[REDACTED_CARD]"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]"),
]

def output_guardrail_node(state: SupplyChainState):
    """Post-execution validation: Checks for output compliance and data leakage."""
    raw_generation = state.get("generation", "")
    if not raw_generation:
        return {}

    try:
        generation = str(raw_generation)
        
        # 1. Check for Accidental PII Leakage
        for pattern, replacement in PII_OUTPUT_PATTERNS:
            if re.search(pattern, generation):
                generation = re.sub(pattern, replacement, generation)

        # 2. Logistics Safety Disclaimer Injection
        safety_keywords = ["eta", "weather", "route", "time", "arrival", "disruption"]
        if any(kw in generation.lower() for kw in safety_keywords):
            disclaimer = "\n\n*Disclaimer: Route ETAs and weather conditions are estimates and subject to real-world changes. Always prioritize road safety.*"
            if disclaimer not in generation:
                generation += disclaimer

        return {"generation": generation}

    except Exception as e:
        fallback_message = "An internal error occurred while formatting your response. Please try asking your question again."
        return {"generation": fallback_message}
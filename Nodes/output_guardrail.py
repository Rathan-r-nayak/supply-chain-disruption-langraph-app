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
    logger.info("--- 🛡️ RUNNING OUTPUT GUARDRAIL NODE ---")
    raw_generation = state.get("generation", "")
    if not raw_generation:
        logger.warning("⚠️ No generation content found in state to validate. Skipping output guardrail.")
        return {}

    try:
        generation = str(raw_generation)
        
        # 1. Check for Accidental PII Leakage
        pii_leaked = False
        for pattern, replacement in PII_OUTPUT_PATTERNS:
            if re.search(pattern, generation):
                pii_leaked = True
                generation = re.sub(pattern, replacement, generation)
        if pii_leaked:
            logger.warning("🚨 OUTPUT PII LEAKAGE PREVENTED: Masked sensitive pattern in output.")

        # 2. Logistics Safety Disclaimer Injection
        safety_keywords = ["eta", "weather", "route", "time", "arrival", "disruption"]
        if any(kw in generation.lower() for kw in safety_keywords):
            disclaimer = "\n\n*Disclaimer: Route ETAs and weather conditions are estimates and subject to real-world changes. Always prioritize road safety.*"
            if disclaimer not in generation:
                generation += disclaimer
                logger.info("ℹ️ Injected mandatory logistics safety disclaimer into output.")

        logger.info("✅ Output guardrail validation complete.")
        return {"generation": generation}

    except Exception as e:
        logger.error(f"❌ Output guardrail validation error: {e}")
        fallback_message = "An internal error occurred while formatting your response. Please try asking your question again."
        return {"generation": fallback_message}
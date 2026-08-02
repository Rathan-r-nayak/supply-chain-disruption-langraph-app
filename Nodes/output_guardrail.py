import re
from State.banking_state import BankingState
from Utils.Logger import get_logger

logger = get_logger("OUTPUT_GUARDRAIL")

# Dictionary of PII patterns and their redacted replacements
# We use capturing groups (\1) for IDs and Accounts to keep the label intact but mask the value
PII_OUTPUT_PATTERNS = [
    # 1. Absolute patterns (Redact whenever seen)
    (r"\b(?:\d[ -]*?){13,16}\b", "[REDACTED_CARD]"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]"),
    
    # 2. Context-aware patterns (Redact only when accompanied by identifying keywords)
    # Matches: "Customer ID: 123456XYZ" or "Client-ID: 98765"
    (r"(?i)(customer\s*id|cust\s*id|client\s*id|customer\s*number)[\s:#=-]+([A-Za-z0-9]{5,15})\b", r"\1: [REDACTED_CUSTOMER_ID]"),
    
    # Matches: "Account: 123456789" or "ACC123456789"
    (r"(?i)(account\s*number|acct\s*no|account|acc)[\s:#=-]*(\d{4,18})\b", r"\1: [REDACTED_ACCOUNT]"),
    
    # Matches: "PIN: 1234" or "Password: mypass"
    (r"(?i)(password|pin|security\s*code|cvv)[\s:#=-]+([A-Za-z0-9@#$%^&+=]{3,20})\b", r"\1: [REDACTED_SECURE_DATA]")
]

def output_guardrail_node(state: BankingState):
    """Post-execution validation: Checks for output compliance and data leakage."""
    logger.info("🛑 [OUTPUT GUARDRAIL] Verifying generated response...")
    
    raw_generation = state.get("generation", "")
    if not raw_generation:
        return {}

    try:
        # Ensure it is a string to prevent TypeError if the state holds an object
        generation = str(raw_generation)
        found_leak = False
        
        # 1. Check for Accidental PII/Sensitive Leakage in Output
        for pattern, replacement in PII_OUTPUT_PATTERNS:
            if re.search(pattern, generation):
                found_leak = True
                generation = re.sub(pattern, replacement, generation)
                
        if found_leak:
            logger.warning("🚨 [OUTPUT GUARDRAIL] Sensitive information leak detected in AI response! Redacting...")

        # 2. Financial Disclaimer Injection for Investment/Advice queries
        advice_keywords = ["invest", "stocks", "mutual funds", "returns", "portfolio advice", "yield"]
        if any(kw in generation.lower() for kw in advice_keywords):
            disclaimer = "\n\n*Disclaimer: This information is for informational purposes only and does not constitute formal financial advice.*"
            if disclaimer not in generation:
                generation += disclaimer
                logger.info("⚖️ [OUTPUT GUARDRAIL] Financial advice disclaimer appended.")

        logger.info("✅ [OUTPUT GUARDRAIL] Output cleared and formatted safely.")
        return {"generation": generation}

    except Exception as e:
        # 🌟 THE FAIL-SAFE: If anything breaks in the guardrail, block the output entirely.
        logger.error(f"❌ [OUTPUT GUARDRAIL] Critical error during output verification: {e}")
        
        fallback_message = (
            "An internal error occurred while securely formatting your response. "
            "For your protection, the output has been blocked. Please try asking your question again."
        )
        return {"generation": fallback_message}
# Nodes/guardrail_node.py
import re
from State.supply_chain_state import SupplyChainState
from Utils.logger import get_logger

logger = get_logger("GUARDRAIL")

def guardrail_node(state: SupplyChainState):
    """
    Acts as a zero-trust entry point. 
    Masks PII and blocks prompt injections before LLM processing.
    """
    question = state.get("question", "")
    logger.info("--- 🛑 RUNNING SECURITY GUARDRAILS ---")
    
    # 1. Prompt Injection & Abuse Check
    blocked_terms = ["ignore previous", "system prompt", "bypass", "jailbreak", "override"]
    if any(term in question.lower() for term in blocked_terms):
        logger.warning(f"🚨 SECURITY ALERT: Malicious input detected. Blocking request.")
        return {
            "is_safe": False,
            "requires_workflow": False,
            "generation": "Security Alert: This request violates bank security policies and has been blocked."
        }
    
    # 2. PII Masking (Example: Masking 16-digit credit card numbers)
    # This ensures the LLM in the triage_router never sees the actual card number
    safe_question = re.sub(r'\b\d{16}\b', '[CARD_NUMBER_MASKED]', question)
    
    if safe_question != question:
        logger.info("🛡️ PII Masked: Sensitive data scrubbed from input.")
        
    logger.info("✅ Security guardrail passed.")
    # Proceed to triage with the sanitized question
    return {
        "question": safe_question,
        "is_safe": True
    }
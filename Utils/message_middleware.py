import os
import re
import uuid
from langchain_core.messages import ToolMessage
from Utils.logger import get_logger

logger = get_logger("PII_MIDDLEWARE")

SCRATCHPAD_DIR = "data/scratchpad"
OFFLOAD_THRESHOLD_CHARS = 4000  # ~1,000 tokens



# Pre-compile regex patterns for performance
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_REGEX = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
CREDIT_CARD_REGEX = re.compile(r'\b(?:\d[ -]*?){13,16}\b')

def sanitize_pii_tool_message(msg: ToolMessage) -> ToolMessage:
    """
    Scubs sensitive PII (emails, phone numbers, credit card numbers) 
    from ToolMessage content before it is exposed to the LLM context.
    """
    if not isinstance(msg.content, str) or not msg.content:
        return msg

    sanitized_content = msg.content
    
    # 1. Mask Emails
    if EMAIL_REGEX.search(sanitized_content):
        sanitized_content = EMAIL_REGEX.sub("[REDACTED_EMAIL]", sanitized_content)
        
    # 2. Mask Phone Numbers
    if PHONE_REGEX.search(sanitized_content):
        sanitized_content = PHONE_REGEX.sub("[REDACTED_PHONE]", sanitized_content)

    # 3. Mask Credit Cards
    if CREDIT_CARD_REGEX.search(sanitized_content):
        sanitized_content = CREDIT_CARD_REGEX.sub("[REDACTED_CARD]", sanitized_content)

    if sanitized_content != msg.content:
        logger.info(f"🛡️ [PII MIDDLEWARE] Sanitized sensitive data in ToolMessage {msg.tool_call_id}")
        
        # Return new ToolMessage preserving tool metadata
        return ToolMessage(
            content=sanitized_content,
            tool_call_id=msg.tool_call_id,
            name=getattr(msg, "name", None)
        )

    return msg


def offload_large_tool_message(msg: ToolMessage) -> ToolMessage:
    """
    If a ToolMessage content exceeds the threshold, save full content 
    to disk and return a new ToolMessage with a pointer and preview.
    """
    if not isinstance(msg.content, str) or len(msg.content) <= OFFLOAD_THRESHOLD_CHARS:
        return msg

    os.makedirs(SCRATCHPAD_DIR, exist_ok=True)
    file_id = f"tool_out_{uuid.uuid4().hex[:8]}.json"
    filepath = os.path.join(SCRATCHPAD_DIR, file_id)

    # 1. Save full payload to temporary file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(msg.content)

    logger.info(f"📁 [OFFLOADED TO DISK] Saved {len(msg.content)} chars to '{filepath}'")

    # 2. Build concise preview for LLM context window
    lines = msg.content.splitlines()
    preview = "\n".join(lines[:6]) if len(lines) >= 6 else msg.content[:500]

    offloaded_content = (
        f"⚠️ [DATA OFFLOADED TO DISK - EXCEEDED TOKEN LIMIT]\n"
        f"Full data saved to file: '{filepath}'\n"
        f"Data size: {len(msg.content)} characters (~{len(msg.content)//4} tokens)\n"
        f"--- PREVIEW (First few lines) ---\n"
        f"{preview}\n"
        f"--- END PREVIEW ---"
    )

    # 3. Return updated ToolMessage preserving tool_call_id
    return ToolMessage(content=offloaded_content, tool_call_id=msg.tool_call_id, name=getattr(msg, "name", None))


def process_tool_message_pipeline(msg: ToolMessage) -> ToolMessage:
    """Chains multiple tool message middleware steps sequentially."""
    # Step 1: Offload to disk if too large
    msg = offload_large_tool_message(msg)
    
    # Step 2: Mask sensitive PII
    msg = sanitize_pii_tool_message(msg)
    
    return msg
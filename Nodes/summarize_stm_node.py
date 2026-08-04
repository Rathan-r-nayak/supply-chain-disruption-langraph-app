from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from State.supply_chain_state import SupplyChainState
from Config.llm_config import fast_llm
from Utils.logger import get_logger

logger = get_logger("SUMMARIZE_NODE")

def summarize_conversation_node(state: SupplyChainState):
    """Summarizes older conversation history while ignoring internal tool messages and offload pointers."""
    messages = state.get("messages", [])
    current_summary = state.get("conversation_summary", "")
    
    # 🌟 1. Filter ONLY human and assistant messages (ignore ToolMessage, SystemMessage, etc.)
    valid_dialogue = [
        m for m in messages 
        if isinstance(m, (HumanMessage, AIMessage)) and getattr(m, 'content', '').strip()
    ]
    
    # Trigger summarization only when we have more than 6 conversational turns
    if len(valid_dialogue) <= 6:
        return {}
        
    logger.info("--- 🗜️ COMPRESSING SHORT-TERM MEMORY ---")
    
    # Summarize everything except the last 2 turns to maintain immediate context
    messages_to_summarize = valid_dialogue[:-2]
    
    # Format cleanly into User / Assistant dialogue string
    formatted_lines = []
    for m in messages_to_summarize:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        # Clean out any automated confidence badges or system notices before summarizing
        clean_content = m.content.split("---")[0].strip() 
        formatted_lines.append(f"{role}: {clean_content}")
        
    chat_history_str = "\n".join(formatted_lines)
    
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a short-term memory compressor for a logistics assistant.\n"
            "Distill the conversation into a concise summary.\n"
            "Retain key factual details: locations, driver IDs, trip statuses, and unresolved questions.\n"
            "Do NOT include system notices, confidence scores, or raw file paths.\n\n"
            "Existing Summary to extend:\n{current_summary}"
        )),
        ("human", "New Dialogue to summarize:\n{chat_history}")
    ])
    
    response = fast_llm.invoke(summary_prompt.format(
        current_summary=current_summary,
        chat_history=chat_history_str
    ))
    
    logger.info("✅ Short-Term Memory compressed successfully.")
    
    # Return updated summary string without modifying the raw message objects in SQLite
    return {"conversation_summary": response.content}
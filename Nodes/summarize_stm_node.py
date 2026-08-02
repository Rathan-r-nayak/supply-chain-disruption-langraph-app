from langchain_core.messages import SystemMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate
from State.supply_chain_state import SupplyChainState
from Config.llm_config import fast_llm
from Utils.Logger import get_logger

logger = get_logger("SUMMARIZE_NODE")

def summarize_conversation_node(state: SupplyChainState):
    """Summarizes older messages to prevent context window bloat."""
    messages = state.get("messages", [])
    
    # Trigger summarization only when the conversation gets long (e.g., > 6 messages)
    if len(messages) <= 6:
        return {}
        
    logger.info("--- 🗜️ COMPRESSING SHORT-TERM MEMORY ---")
    
    # We summarize everything EXCEPT the last 2 messages to maintain immediate context
    messages_to_summarize = messages[:-2]
    
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Distill the following conversation into a concise summary. Retain all key logistics details, locations, driver IDs, and unresolved issues. Do not lose factual data."),
        ("human", "Conversation to summarize:\n{chat_history}")
    ])
    
    # Format the messages to summarize into text
    chat_history_str = "\n".join([f"{m.type}: {m.content}" for m in messages_to_summarize if m.content])
    
    # Generate the summary using your faster, cheaper model
    response = fast_llm.invoke(summary_prompt.format_messages(chat_history=chat_history_str))
    new_summary = f"Previous Conversation Summary: {response.content}"
    
    # 🌟 CRITICAL LANGGRAPH LOGIC: 
    # Use RemoveMessage to explicitly delete the old, raw messages from the state.
    delete_ops = [RemoveMessage(id=m.id) for m in messages_to_summarize if m.id]
    
    # Create the new summary message
    new_memory = [SystemMessage(content=new_summary)]
    
    logger.info("✅ Short-Term Memory compressed successfully.")
    
    # Return the deletion operations followed by the new summary.
    # LangGraph's 'add_messages' reducer will handle this sequence cleanly.
    return {"messages": delete_ops + new_memory}
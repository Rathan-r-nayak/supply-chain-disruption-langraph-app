# Utils/HistoryLoader.py
from Utils.Logger import get_logger

logger = get_logger("HISTORY_LOADER")

def load_chat_history(app_instance, thread_id: str):
    """
    Fetches past conversations from the LangGraph Postgres checkpointer.
    Reconstructs the chat log by finding completed graph runs.
    """
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Fetch all state snapshots for this specific thread
        history = list(app_instance.get_state_history(config))
        history.reverse() # Order chronologically (oldest to newest)
        
        chat_messages = []
        
        for snapshot in history:
            # We only extract data from states where the graph has finished its run 
            # (meaning there are no pending 'next' nodes to execute)
            if not snapshot.next:
                val = snapshot.values
                
                # Match these exactly to your app.invoke() and state schema keys
                q = val.get("question")
                a = val.get("generation")
                
                if q and a:
                    # Prevent adding duplicate consecutive turns if the graph 
                    # looped or was interrupted multiple times with the same state.
                    if not chat_messages or chat_messages[-1]["content"] != a:
                        chat_messages.append({"role": "user", "content": q})
                        chat_messages.append({"role": "assistant", "content": a})
        
        return chat_messages
        
    except Exception as e:
        logger.error(f"Failed to load chat history: {e}")
        return []
import streamlit as st
import requests

# 🌟 Point to the new FastAPI port (8080)
FASTAPI_URL = "http://127.0.0.1:8080"

st.set_page_config(page_title="Secure Banking AI", page_icon="🏦", layout="centered")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "dev_session_001"
if "user_id" not in st.session_state:
    st.session_state.user_id = "rathan"

st.title("🏦 Secure Banking Orchestrator")

# --- 1. Fetch History from FastAPI ---
def fetch_history():
    try:
        res = requests.get(f"{FASTAPI_URL}/chat/history?thread_id={st.session_state.thread_id}")
        if res.status_code == 200:
            return res.json()
    except requests.exceptions.ConnectionError:
        st.error("🔌 Cannot connect to FastAPI backend. Is it running?")
    return {"messages": [], "is_paused": False}

history_data = fetch_history()
messages = history_data.get("messages", [])
is_paused = history_data.get("is_paused", False)

# --- Render Chat History ---
for msg in messages:
    if msg["role"] == "thought":
        # Render historical thoughts in a collapsed expander so they don't clutter the screen
        with st.expander("🧠 Agent Thoughts", expanded=False):
            st.markdown(msg["content"])
    else:
        st.chat_message(msg["role"]).write(msg["content"])

# --- 2. Evaluate Graph State (The Gatekeeper) ---
if is_paused:
    st.error("🔒 **Action Pending Approval**")
    st.write("A sensitive transaction or web search requires your authorization.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve", use_container_width=True):
            requests.post(
                f"{FASTAPI_URL}/chat/stream", 
                json={"thread_id": st.session_state.thread_id, "user_id": st.session_state.user_id, "action": "approve"}
            )
            st.rerun()
    with col2:
        if st.button("❌ Reject", use_container_width=True):
            requests.post(
                f"{FASTAPI_URL}/chat/stream", 
                json={"thread_id": st.session_state.thread_id, "user_id": st.session_state.user_id, "action": "reject"}
            )
            st.rerun()

# --- 3. Standard Chat Input (Streams the SSE response) ---
else:
    if prompt := st.chat_input("Ask about policies or request a transaction..."):
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            
            # 🌟 Create separate UI containers for Thoughts and Final Messages
            thought_expander = st.expander("🧠 Agent Thoughts", expanded=True)
            thought_placeholder = thought_expander.empty()
            
            message_placeholder = st.empty()
            
            full_thought_text = ""
            full_message_text = ""
            current_event_type = "message"  # Default event type
            
            payload = {
                "thread_id": st.session_state.thread_id,
                "user_id": st.session_state.user_id,
                "message": prompt
            }
            
            try:
                with requests.post(f"{FASTAPI_URL}/chat/stream", json=payload, stream=True) as r:
                    for line in r.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            
                            # 🌟 1. Detect which event we are currently receiving
                            if decoded_line.startswith("event: "):
                                current_event_type = decoded_line[7:].strip()
                            
                            # 🌟 2. Parse the data payload for that event
                            elif decoded_line.startswith("data: "):
                                data = decoded_line[6:]
                                
                                if data == "__INTERRUPT__":
                                    st.rerun()
                                    break
                                    
                                data = data.replace('\\n', '\n')
                                
                                # 🌟 3. Route the text to the correct UI box
                                if current_event_type == "thought":
                                    full_thought_text += data
                                    thought_placeholder.markdown(full_thought_text + "▌")
                                elif current_event_type == "message":
                                    full_message_text += data
                                    message_placeholder.markdown(full_message_text + "▌")
                                    
                # Final cleanup (remove the "▌" blinking cursors)
                if full_thought_text:
                    thought_placeholder.markdown(full_thought_text)
                else:
                    thought_placeholder.markdown("_Direct response generated without background tools._")
                    
                message_placeholder.markdown(full_message_text)
                
                # Refresh page to log the state to history properly
                st.rerun()
                
            except Exception as e:
                st.error(f"Stream error: {e}")
"""
Standalone Streamlit frontend for the Banking Agent Chatbot.
"""

import uuid
import requests
import streamlit as st
from datetime import datetime

# =============================================================================
# CONFIG
# =============================================================================
API_BASE_URL = "http://localhost:8080"

st.set_page_config(page_title="Banking Agent Assistant", page_icon="🏦", layout="wide")

def format_timestamp(iso_string: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%b %d, %Y · %-I:%M %p")
    except (ValueError, TypeError):
        return iso_string
    

# =============================================================================
# API CLIENT HELPERS
# =============================================================================

def api_login(user_id: str, password: str) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE_URL}/auth/login",
            json={"user_id": user_id, "password": password},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach backend: {e}")
        return None

def api_fetch_threads(user_id: str) -> list[dict]:
    try:
        resp = requests.get(f"{API_BASE_URL}/chat/threads", params={"user_id": user_id}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load conversations: {e}")
        return []

def api_fetch_history(thread_id: str) -> dict:
    try:
        resp = requests.get(f"{API_BASE_URL}/chat/history", params={"thread_id": thread_id}, timeout=10)
        resp.raise_for_status()
        return resp.json()  
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load conversation history: {e}")
        return {"messages": [], "is_paused": False}

def api_upload_document(user_id: str, uploaded_file) -> dict | None:
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        data = {"user_id": user_id}
        resp = requests.post(f"{API_BASE_URL}/admin/upload", data=data, files=files, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Upload failed: {e}")
        return None

def api_fetch_documents() -> list[dict]:
    try:
        resp = requests.get(f"{API_BASE_URL}/admin/documents", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load documents: {e}")
        return []

# 🌟 UPDATED: Now supports multipart/form-data for image uploads
# 🌟 UPDATED: Preserves spaces for streaming tokens!
def stream_chat(thread_id: str, user_id: str, message: str | None = None, action: str | None = None, image_bytes: bytes | None = None, image_name: str | None = None):
    data = {"thread_id": thread_id, "user_id": user_id}
    if message:
        data["message"] = message
    if action:
        data["action"] = action
        
    files = {}
    if image_bytes and image_name:
        files["image"] = (image_name, image_bytes, "image/jpeg")

    try:
        with requests.post(f"{API_BASE_URL}/chat/stream", data=data, files=files if files else None, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            event_type = None
            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line is None or raw_line == "":
                    continue
                if raw_line.startswith("event:"):
                    event_type = raw_line[len("event:"):].strip()
                elif raw_line.startswith("data:"):
                    
                    # 🌟 THE FIX: Slice off "data:" safely.
                    data_str = raw_line[len("data:"):]
                    
                    # SSE protocol usually puts one space after the colon (e.g., "data: Hello").
                    # If that single protocol space exists, remove it, but keep all other spaces!
                    if data_str.startswith(" "):
                        data_str = data_str[1:]
                        
                    data_str = data_str.replace("\\n", "\n") 
                    yield event_type, data_str
                    event_type = None
                    
    except requests.exceptions.RequestException as e:
        yield "error", f"Connection error: {e}"

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
defaults = {
    "authenticated": False,
    "user_id": None,
    "role": None,               
    "current_thread_id": None,
    "messages": [],             
    "is_paused": False,         
    "threads_cache": [],
    "documents_cache": [],
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# =============================================================================
# VIEW: LOGIN
# =============================================================================
def render_login():
    st.title("🏦 Banking Agent Assistant")
    st.subheader("Sign in")

    with st.form("login_form"):
        user_id = st.text_input("User ID")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", use_container_width=True)

    if submitted:
        if not user_id or not password:
            st.warning("Please enter both a User ID and a password.")
            return
        result = api_login(user_id, password)
        if result is None:
            st.error("Invalid User ID or password.")
        else:
            st.session_state.authenticated = True
            st.session_state.user_id = result["user_id"]
            st.session_state.role = result["role"]
            st.rerun()

    st.caption("Demo accounts — admin1/admin123 (ADMIN), cust1/cust123 (CUSTOMER)")

# =============================================================================
# VIEW: SIDEBARS
# =============================================================================
def load_thread(thread_id: str):
    st.session_state.current_thread_id = thread_id
    history = api_fetch_history(thread_id)
    st.session_state.messages = history["messages"]
    st.session_state.is_paused = history["is_paused"]

def start_new_conversation():
    st.session_state.current_thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.is_paused = False

def render_account_header():
    st.markdown(f"**{st.session_state.user_id}**  \n`{st.session_state.role}`")
    if st.button("Log out", use_container_width=True):
        for key in defaults:
            st.session_state[key] = defaults[key]
        st.rerun()
    st.divider()

def render_admin_sidebar():
    with st.sidebar:
        render_account_header()
        st.subheader("📄 Upload document")
        uploaded_file = st.file_uploader("Choose a file", label_visibility="collapsed")
        if uploaded_file is not None and st.button("Upload", use_container_width=True):
            with st.spinner("Uploading..."):
                result = api_upload_document(st.session_state.user_id, uploaded_file)
            if result:
                st.success(f"'{result['filename']}' — {result['status']}")
            st.rerun() 

def render_customer_sidebar():
    with st.sidebar:
        render_account_header()
        st.subheader("💬 Conversations")
        if st.button("+ New conversation", use_container_width=True):
            start_new_conversation()
            st.rerun()

        st.session_state.threads_cache = api_fetch_threads(st.session_state.user_id)

        if not st.session_state.threads_cache:
            st.caption("No conversations yet.")
            return

        st.markdown(
            """
            <style>
            div[class*="st-key-thread_list"] button {
                border: none !important;
                background: transparent !important;
                box-shadow: none !important;
                text-align: left !important;
                justify-content: flex-start !important;
                padding: 0.4rem 0.6rem !important;
                font-weight: 400 !important;
            }
            div[class*="st-key-thread_list"] button:hover {
                background: var(--secondary-background-color) !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        with st.container(key="thread_list"):
            for t in st.session_state.threads_cache:
                is_active = t["thread_id"] == st.session_state.current_thread_id
                if is_active:
                    # 🌟 ENHANCED HIGHLIGHT: Added a left accent border and cleaner padding
                    st.markdown(
                        f'<div style="background-color: var(--secondary-background-color); '
                        f'border-left: 4px solid #ff4b4b; '
                        f'border-radius: 4px; padding: 0.5rem 0.75rem; font-weight: 600; '
                        f'margin-bottom: 0.25rem; font-size: 0.9rem;">'
                        f'{t["title"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    if st.button(t["title"], key=f"thread_{t['thread_id']}", use_container_width=True):
                        load_thread(t["thread_id"])
                        st.rerun()

# =============================================================================
# VIEW: ADMIN DOCUMENT LIBRARY
# =============================================================================
def render_admin_documents():
    st.title("🏦 Banking Agent Assistant — Admin")
    st.subheader("📚 Uploaded documents")

    st.session_state.documents_cache = api_fetch_documents()

    if not st.session_state.documents_cache:
        st.info("No documents have been uploaded yet. Use the sidebar to upload one.")
        return

    display_rows = [
        {**doc, "uploaded_at": format_timestamp(doc["uploaded_at"])}
        for doc in st.session_state.documents_cache
    ]

    st.dataframe(
        display_rows,
        column_config={
            "filename": "Filename",
            "uploaded_by": "Uploaded by",
            "uploaded_at": "Uploaded at",
        },
        use_container_width=True,
        hide_index=True,
    )

# =============================================================================
# VIEW: CHAT (CUSTOMER only)
# =============================================================================
# =============================================================================
# VIEW: CHAT (CUSTOMER only)
# =============================================================================
def render_message_history():
    for msg in st.session_state.messages:
        if msg["role"] == "thought":
            with st.chat_message("assistant", avatar="🧠"):
                with st.expander("Agent reasoning", expanded=False):
                    st.write(msg["content"])
        else:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("has_image"):
                    st.caption("📎 _Image Attached_")

def run_stream_and_render(message: str | None, action: str | None, image_bytes: bytes | None = None, image_name: str | None = None):
    thought_placeholder = None
    message_placeholder = None
    thought_buffer = ""
    message_buffer = ""
    hit_interrupt = False

    for event_type, data in stream_chat(
        st.session_state.current_thread_id, 
        st.session_state.user_id, 
        message=message, 
        action=action,
        image_bytes=image_bytes,
        image_name=image_name
    ):
        if event_type == "thought":
            if thought_placeholder is None:
                with st.chat_message("assistant", avatar="🧠"):
                    with st.expander("Agent reasoning", expanded=True):
                        thought_placeholder = st.empty()
            thought_buffer += data
            thought_placeholder.write(thought_buffer)

        elif event_type == "message":
            if data == "__INTERRUPT__":
                hit_interrupt = True
                continue
            if message_placeholder is None:
                cm = st.chat_message("assistant")
                message_placeholder = cm.empty()
            message_buffer += data
            message_placeholder.write(message_buffer)

        elif event_type == "error":
            st.error(data)

    if thought_buffer:
        st.session_state.messages.append({"role": "thought", "content": thought_buffer})
    if message_buffer:
        st.session_state.messages.append({"role": "assistant", "content": message_buffer})
        
    st.session_state.is_paused = hit_interrupt

def render_chat():
    st.title("🏦 Banking Agent Assistant")

    if st.session_state.current_thread_id is None:
        start_new_conversation()

    # Lock messages above input controls
    chat_container = st.container()
    
    with chat_container:
        render_message_history()

    if st.session_state.is_paused:
        st.warning("⚠️ This action needs your approval before it continues.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Approve", use_container_width=True):
                run_stream_and_render(message=None, action="approve")
                st.rerun()
        with col2:
            if st.button("❌ Reject", use_container_width=True):
                run_stream_and_render(message=None, action="reject")
                st.rerun()
        return 

    # 🌟 NEW: The integrated Chat Input with native file attachment!
    # Streamlit natively renders a [+] icon inside the chat box when accept_file=True
    if user_input := st.chat_input("Type your message or attach an image...", accept_file=True, file_type=["png", "jpg", "jpeg"]):
        
        prompt_text = user_input.text
        image_bytes = None
        image_name = None
        
        # If the user clicked the + icon and attached files
        if user_input.files:
            uploaded_file = user_input.files[0]
            image_bytes = uploaded_file.getvalue()
            image_name = uploaded_file.name

        # Log message to UI
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt_text, 
            "has_image": bool(user_input.files)
        })
        
        with chat_container:
            with st.chat_message("user"):
                st.write(prompt_text)
                if user_input.files:
                    st.caption(f"📎 Attached: {image_name}")

        # Send payload to stream
        run_stream_and_render(message=prompt_text, action=None, image_bytes=image_bytes, image_name=image_name)
        st.rerun()

# =============================================================================
# MAIN
# =============================================================================
def main():
    if not st.session_state.authenticated:
        render_login()
        return

    if st.session_state.role == "ADMIN":
        render_admin_sidebar()
        render_admin_documents()
    else:
        render_customer_sidebar()
        render_chat()

if __name__ == "__main__":
    main()
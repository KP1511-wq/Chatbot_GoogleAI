import streamlit as st
import requests

# Point this to where your FastAPI app is running
API_URL = "http://localhost:8000/chat" 

st.set_page_config(page_title="Heart Disease Assistant", page_icon="❤️")

# ---------- Sidebar: Config & Reset ----------
st.sidebar.title("Session Settings")

config_id = st.sidebar.number_input(
    "Config ID (thread id)",
    min_value=1,
    value=1,
    step=1,
    help="Used as thread_id for the backend agent to remember history."
)

if "current_config_id" not in st.session_state:
    st.session_state.current_config_id = config_id

# Reset chat if config changes
if config_id != st.session_state.current_config_id:
    st.session_state.current_config_id = config_id
    st.session_state.messages = []

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []

# ---------- Main Layout ----------
st.title("❤️ Heart Disease Data Assistant")
st.write("Ask questions about the heart disease database (e.g., 'Show me the average age of patients' or 'What does cp mean?').")

# Initialise history
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}


def call_api(message: str, cfg_id: int) -> str:
    """Send message to FastAPI and return the chatbot response as text."""
    try:
        resp = requests.post(
            API_URL,
            json={"message": message, "config_id": cfg_id},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "(No 'response' field in API reply)")
    except Exception as e:
        return f"⚠️ Error talking to backend: {e}"


# ---------- Handle new input FIRST ----------
user_input = st.chat_input("Type your question here...")

if user_input:
    # 1) store user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2) call backend
    with st.spinner("Thinking..."):
        reply = call_api(user_input, st.session_state.current_config_id)

    # 3) store assistant reply
    st.session_state.messages.append({"role": "assistant", "content": reply})


# ---------- Render chat history ONCE ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
import streamlit as st
from google import genai
from datetime import datetime
import pytz

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖")
st.title("🤖 Gemini AI Chatbot")
st.caption("Powered by Google AI Studio")

# ------------------ GEMINI CLIENT ------------------
from google import genai

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents="Explain how AI works in a few words",
)

print(response.text)

# ------------------ TIME FUNCTION ------------------
def get_current_datetime():
    """
    Returns the current date & time in IST
    """
    tz = pytz.timezone("Asia/Kolkata")
    now = datetime.now(tz)
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timezone": "Asia/Kolkata"
    }

# ------------------ HELPERS ------------------
def is_datetime_query(text):
    keywords = ["time", "date", "day", "today", "now"]
    return any(word in text.lower() for word in keywords)

def build_prompt(messages):
    return "\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
        for m in messages
    )

# ------------------ SESSION STATE ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------ DISPLAY CHAT ------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ USER INPUT ------------------
user_input = st.chat_input("Ask me anything...")

# ------------------ RESPONSE LOGIC ------------------
if user_input:
    # Store & display user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # ⏰ HANDLE DATE / TIME LOCALLY
    if is_datetime_query(user_input):
        now = get_current_datetime()
        bot_reply = (
            f"📅 Date: {now['date']}\n"
            f"⏰ Time: {now['time']}\n"
            f"🌍 Timezone: {now['timezone']}"
        )

    # 🤖 NORMAL CHAT → GEMINI
    else:
        prompt = build_prompt(st.session_state.messages)
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt
            )
            bot_reply = response.text
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                bot_reply = "🚫 API quota exhausted. Please enable billing or try later."
            else:
                bot_reply = f"❌ Error: {e}"

    # Store & display assistant reply
    st.session_state.messages.append(
        {"role": "assistant", "content": bot_reply}
    )
    with st.chat_message("assistant"):
        st.markdown(bot_reply)

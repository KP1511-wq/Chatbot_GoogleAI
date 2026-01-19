# streamlit_app.py
import streamlit as st
import time

# --- 1. FORCE UI TO LOAD FIRST ---
st.set_page_config(page_title="Debug Mode", page_icon="🐞")
st.title("🐞 Debug Mode: Active")
st.write("Step 1: UI has loaded successfully.")

# Now we try to import the heavy stuff
st.write("Step 2: Importing app logic... (This might freeze)")
try:
    from app import load_db, get_sql_agent, get_joke_chain, get_router_chain
    st.write("Step 2: Imports complete! ✅")
except Exception as e:
    st.error(f"CRITICAL ERROR during imports: {e}")
    st.stop()

# --- 2. LOAD RESOURCES ---
@st.cache_resource
def initialize_bot():
    st.write("Step 3: Connecting to Database... ⏳")
    # 1. Load DB
    db_path = 'heart.db'
    db = load_db(db_path)
    table_info = db.get_table_info()
    st.write(f"Step 3: DB Connected! Found tables: {table_info} ✅")
    
    st.write("Step 4: Loading AI Agents... (This is the slow part) ⏳")
    # 2. Load Agents
    sql_agent = get_sql_agent(db)
    joke_chain = get_joke_chain(table_info)
    router = get_router_chain(table_info)
    
    return sql_agent, joke_chain, router, table_info

# Initialize
try:
    sql_agent, joke_chain, router, table_info = initialize_bot()
    st.success("Step 5: All Systems Go! The app is ready.", icon="🚀")
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

# --- 3. CHAT INTERFACE ---
st.divider()
st.header("❤️ Heart Disease Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am ready."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            intent = router.invoke({"question": prompt}).strip().upper()
            if "DATA" in intent:
                response = sql_agent.invoke({"input": prompt})['output']
            else:
                response = joke_chain.invoke({"question": prompt})
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error: {e}")
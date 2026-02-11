import streamlit as st
import requests
import json
import pandas as pd

# --- CONFIGURATION ---
AGENT_API_URL = "http://localhost:8001/chat"
st.set_page_config(page_title="Housing Agent", page_icon="🏠", layout="wide")

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.header("🏠 Housing Data Agent")
    st.markdown("I can help you analyze housing prices, find specific properties, and visualize trends.")
    st.markdown("---")
    st.markdown("**Try asking:**")
    st.markdown("- *Show me the cheapest houses near the bay*")
    st.markdown("- *Plot average house price by ocean proximity*")
    st.markdown("- *Find houses under $200k with > 3 bedrooms*")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("🤖 AI Real Estate Analyst")

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Check if content is JSON (Chart) or Text
        if isinstance(message["content"], dict):
            # It's a Chart!
            st.vega_lite_chart(message["content"].get("data", {}).get("values", []), 
                               message["content"], 
                               use_container_width=True)
        else:
            # It's Text
            st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about housing data..."):
    # 1. Display User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Call Agent API
    try:
        with st.spinner("Analyzing data..."):
            response = requests.post(AGENT_API_URL, json={"message": prompt})
            
            if response.status_code == 200:
                result = response.json().get("response")
                
                # Check if result is a Chart (JSON Dict) or Text
                with st.chat_message("assistant"):
                    if isinstance(result, dict) and "$schema" in result:
                        st.success("📊 Chart Generated!")
                        st.vega_lite_chart(result.get("data", {}).get("values", []), 
                                           result, 
                                           use_container_width=True)
                        # Save chart to history as a dict
                        st.session_state.messages.append({"role": "assistant", "content": result})
                    else:
                        st.markdown(result)
                        # Save text to history
                        st.session_state.messages.append({"role": "assistant", "content": result})
            else:
                st.error(f"Error: {response.status_code} - {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to Agent. Is 'chatbot_agent.py' running on Port 8001?")
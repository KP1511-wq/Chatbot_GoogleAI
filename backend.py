# backend.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import sqlite3
from typing import Dict, List, Literal

# LangChain / LangGraph Imports
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

# Import your LLM and DB Tools
from config import llm
from tools_db import execute_read_query, get_columns, get_sample_rows, DB_PATH

# ==========================================
# 1. SETUP & CONTEXT
# ==========================================
TABLE_NAME = "heart_disease_info"

# Get context to teach the AI about the database
try:
    columns = get_columns(DB_PATH, TABLE_NAME)
    sample_rows = get_sample_rows(DB_PATH, TABLE_NAME, n=2)
except Exception as e:
    print(f"⚠️ Warning: Could not load DB context. Make sure heart.db exists. Error: {e}")
    columns = []
    sample_rows = []

# Define Metadata Tool (Data Dictionary)
HEART_DISEASE_METADATA = {
    "Personal Info": ["age", "sex (1=male, 0=female)"],
    "Vitals": ["trestbps (Resting Blood Pressure)", "chol (Cholesterol)", "fbs (Fasting Blood Sugar > 120)"],
    "Heart Health": ["cp (Chest Pain Type)", "restecg (Resting ECG)", "thalach (Max Heart Rate)", "exang (Exercise Angina)"],
    "Target": ["target (1=Disease, 0=No Disease)"]
}

@tool
def get_data_dictionary() -> Dict[str, List[str]]:
    """
    Returns the dictionary explaining what the specific medical columns mean.
    Useful when the user asks about the dataset
    """
    print("Hi I am triggered!!")
    return HEART_DISEASE_METADATA

# ==========================================
# 2. AGENT PROMPT
# ==========================================
sql_agent_prompt = f"""
You are an intelligent agent helping users analyze Heart Disease data.

**Database Context:**
- Table Name: {TABLE_NAME}
- Columns: {columns}
- Sample Data: {sample_rows}

**Tools Available:**
1. `execute_read_query`: Use this to fetch actual data (counts, averages, patient lists).
2. `get_data_dictionary`: Use this if the user asks what a column means (e.g., "What is cp?").

**Rules:**
1. If the user asks for data, write a valid SQL query and use `execute_read_query`.
2. If the query fails, analyze the error and try again.
3. If the user asks for definitions, use `get_data_dictionary`.
4. Be professional and concise. Do not expose internal code or SQL unless asked.

**Goal:** Provide accurate answers based strictly on the database content.
"""

# Bind tools to the model
tools = [execute_read_query, get_data_dictionary]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = llm.bind_tools(tools)

# ==========================================
# 3. LANGGRAPH NODE DEFINITIONS
# ==========================================

def llm_call(state: MessagesState):
    """LLM Node: Decides whether to call a tool or reply to user."""
    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content=sql_agent_prompt)] + state["messages"]
            )
        ]
    }

def tool_node(state: MessagesState):
    """Tool Node: Executes the tools requested by the LLM."""
    result = []
    last_message = state["messages"][-1]
    
    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
        
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Conditional Edge: Tool Loop vs End."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END

# ==========================================
# 4. BUILD THE GRAPH
# ==========================================
agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

checkpointer = InMemorySaver()
agent = agent_builder.compile(checkpointer=checkpointer)

# ==========================================
# 5. FASTAPI APPLICATION
# ==========================================
app = FastAPI(title="Heart Disease Chatbot API")

class ChatRequest(BaseModel):
    message: str
    config_id: int = 1

class ChatResponse(BaseModel):
    response: str

def query_to_chatbot(user_message: str, config_id: int):
    # Unique thread for conversation history
    config: RunnableConfig = {"configurable": {"thread_id": str(config_id)}}
    
    # Invoke the graph
    messages = [HumanMessage(content=user_message)]
    responses = agent.invoke({"messages": messages}, config)
    
    # Extract final text
    return responses['messages'][-1].content

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    bot_reply = query_to_chatbot(req.message, req.config_id)
    return ChatResponse(response=bot_reply)

if __name__ == "__main__":
    # Run server
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
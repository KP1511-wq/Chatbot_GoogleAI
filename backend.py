# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import sqlite3
import json
from typing import Dict, List, Literal

# LangChain / LangGraph Imports
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

# Import your LLM
from config import llm

# ==========================================
# 1. DATABASE TOOLS & SETUP
# ==========================================
DB_PATH = "heart.db"
TABLE_NAME = "heart_disease_info"

# Define column metadata (Acting as your "Context")
HEART_DISEASE_METADATA = {
    "Personal Info": ["age", "sex (1=male, 0=female)"],
    "Vitals": ["trestbps (Resting Blood Pressure)", "chol (Cholesterol)", "fbs (Fasting Blood Sugar > 120)"],
    "Heart Health": ["cp (Chest Pain Type)", "restecg (Resting ECG)", "thalach (Max Heart Rate)", "exang (Exercise Angina)"],
    "Target": ["target (1=Disease, 0=No Disease)"]
}

# --- Helper Functions ---
def get_db_connection():
    return sqlite3.connect(DB_PATH)

def get_columns(table_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    cols = [row[1] for row in cursor.fetchall()]
    conn.close()
    return cols

def get_sample_rows(table_name, n=2):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name} LIMIT {n}")
    rows = cursor.fetchall()
    conn.close()
    return rows

# --- Tools Definition ---

@tool
def execute_read_query(query: str) -> str:
    """Executes a read-only SQL query against the database."""
    clean_query = query.strip().upper()
    if not clean_query.startswith("SELECT"):
        return "Error: Only SELECT queries are allowed."
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        data = cursor.fetchall()
        headers = [desc[0] for desc in cursor.description]
        conn.close()
        return f"Headers: {headers}\nData: {data}"
    except Exception as e:
        return f"Database Error: {e}"

@tool
def get_data_dictionary() -> Dict[str, List[str]]:
    """Returns the dictionary explaining column meanings."""
    return HEART_DISEASE_METADATA

# ==========================================
# 2. AGENT PROMPT & CONFIGURATION
# ==========================================
columns = get_columns(TABLE_NAME)
sample_rows = get_sample_rows(TABLE_NAME)

sql_agent_prompt = f"""
You are an intelligent agent helping users analyze Heart Disease data.

**Database Context:**
- Table Name: {TABLE_NAME}
- Columns: {columns}
- Sample Data: {sample_rows}

**Tools Available:**
1. `execute_read_query`: Use this to fetch actual data (counts, averages, patient lists).
2. `get_data_dictionary`: Use this if the user asks what a column means.

**Rules:**
1. If the user asks for data, write a valid SQL query and use `execute_read_query`.
2. If the user asks for definitions, use `get_data_dictionary`.
3. Be professional and concise.
"""

# Bind tools to the model
tools = [execute_read_query, get_data_dictionary]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = llm.bind_tools(tools)

# ==========================================
# 3. LANGGRAPH NODE DEFINITIONS
# ==========================================
def llm_call(state: MessagesState):
    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content=sql_agent_prompt)] + state["messages"]
            )
        ]
    }

def tool_node(state: MessagesState):
    result = []
    last_message = state["messages"][-1]
    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
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
# 5. FASTAPI APPLICATION (The Crucial Part)
# ==========================================
# This variable 'app' is what uvicorn looks for!
app = FastAPI(title="Heart Disease Chatbot API")

class ChatRequest(BaseModel):
    message: str
    config_id: int = 1

class ChatResponse(BaseModel):
    response: str

def query_to_chatbot(user_message: str, config_id: int):
    config: RunnableConfig = {"configurable": {"thread_id": str(config_id)}}
    messages = [HumanMessage(content=user_message)]
    responses = agent.invoke({"messages": messages}, config)
    return responses['messages'][-1].content

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    bot_reply = query_to_chatbot(req.message, req.config_id)
    return ChatResponse(response=bot_reply)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
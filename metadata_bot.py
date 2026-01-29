import sqlite3
import pandas as pd
import json
import sys
import uvicorn
from typing import Dict, List, Literal, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# --- 1. SETUP LOGGER ---
from logger_config import setup_logger
logger = setup_logger("API_Agent")

# --- LANGCHAIN IMPORTS ---
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

# --- IMPORT MODEL ---
try:
    from config import model
    logger.info("LLM Model imported successfully.")
except ImportError:
    logger.critical(" ERROR: Could not find 'config.py'")
    sys.exit()

DB_PATH = "housing.db"
TABLE_NAME = "housing_data"

# --- 2. HELPER FUNCTIONS ---
def clean_response_content(content: Any) -> str:
    """
    Fixes the issue where the bot returns [{'type': 'text', 'text': ...}]
    instead of a simple string.
    """
    # Case A: It's a list (The issue you saw in the screenshot)
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and 'text' in part:
                text_parts.append(part['text'])
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts)
    
    # Case B: It's already a string
    return str(content)

def load_metric_groups():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT json_data FROM ai_groups WHERE key='main_grouping'", conn)
        conn.close()
        if not df.empty:
            return json.loads(df.iloc[0]['json_data'])
    except Exception as e:
        logger.warning(f"Could not load groupings: {e}")
    return {"General": []}

# --- 3. DEFINE TOOLS ---
metric_groups = load_metric_groups()

@tool
def get_housing_context() -> Dict[str, List[str]]:
    """Returns the grouping mapping of columns."""
    logger.info("Tool Triggered: get_housing_context")
    return metric_groups

@tool
def execute_read_query(query: str):
    """Executes a SELECT query on 'housing_data'."""
    logger.info(f"Tool Triggered: execute_read_query | SQL: {query}")
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        row_count = len(df)
        logger.info(f"SQL Success. Retrieved {row_count} rows.")
        
        if row_count > 20:
            return f"Result (First 20 of {row_count} rows):\n{df.head(20).to_string(index=False)}"
        return df.to_string(index=False)
    except Exception as e:
        logger.error(f"SQL Error: {e}")
        return f"SQL Error: {e}"

# --- 4. GRAPH SETUP ---
def get_columns(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    cols = [row[1] for row in cursor.fetchall()]
    conn.close()
    return cols

columns = get_columns(DB_PATH, TABLE_NAME)

housing_agent_prompt = f"""
You are a California Housing Data Assistant.
Table: {TABLE_NAME}
Columns: {columns}
Groups: {list(metric_groups.keys())}

RULES:
1. Use 'execute_read_query' to get actual data.
2. Use 'get_housing_context' if asked about categories (like 'Financials').
3. Answer concisely in plain text.
"""

tools = [execute_read_query, get_housing_context]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = model.bind_tools(tools)

def llm_call(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([SystemMessage(content=housing_agent_prompt)] + state["messages"])]}

def tool_node(state: MessagesState):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    if state["messages"][-1].tool_calls: return "tool_node"
    return END

agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

checkpointer = InMemorySaver()
agent = agent_builder.compile(checkpointer=checkpointer)

# --- 5. FASTAPI APP ---
app = FastAPI(title="Housing Data Agent API")

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "1"

class ChatResponse(BaseModel):
    response: str

# Middleware to log requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"API Request: {request.method} {request.url}")
    response = await call_next(request)
    return response

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    logger.info(f"Processing User Query: '{req.message}' (Thread: {req.thread_id})")
    
    try:
        # 1. SETUP CONFIG
        config: RunnableConfig = {"configurable": {"thread_id": req.thread_id}}
        
        # 2. RUN AGENT (This was commented out before!)
        messages = [HumanMessage(content=req.message)]
        result = agent.invoke({"messages": messages}, config)
        
        # 3. CLEAN OUTPUT
        raw_content = result['messages'][-1].content
        clean_text = clean_response_content(raw_content)

        logger.info("Response generated successfully.")
        return ChatResponse(response=clean_text)
        
    except Exception as e:
        logger.error(f"Critical API Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("--- SERVER STARTING ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)
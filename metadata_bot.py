import sqlite3
import pandas as pd
import json
import sys
from typing import Dict, List, Literal

# --- LANGCHAIN / LANGGRAPH IMPORTS ---
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

# --- IMPORT YOUR MODEL ---
try:
    from config import model
except ImportError:
    print("❌ ERROR: Could not find 'config.py'")
    sys.exit()

# Force output buffering
sys.stdout.reconfigure(line_buffering=True)

# --- CONFIGURATION ---
DB_PATH = "housing.db"
TABLE_NAME = "housing_data"

# --- 1. HELPER FUNCTIONS (Simulating db_tools) ---
def get_columns(db_path, table_name):
    """Fetches column names from SQLite."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    cols = [row[1] for row in cursor.fetchall()]
    conn.close()
    return cols

def get_sample_rows(db_path, table_name):
    """Fetches 3 sample rows for context."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 3", conn)
    conn.close()
    return df.to_dict(orient='records')

def load_metric_groups():
    """Fetches the AI-generated grouping map from the DB."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT json_data FROM ai_groups WHERE key='main_grouping'", conn)
        conn.close()
        if not df.empty:
            return json.loads(df.iloc[0]['json_data'])
    except Exception:
        pass
    return {"General": []} # Fallback

# --- 2. PREPARE STATIC DATA ---
columns = get_columns(DB_PATH, TABLE_NAME)
sample_rows = get_sample_rows(DB_PATH, TABLE_NAME)
metric_groups = load_metric_groups()

# --- 3. DEFINE TOOLS ---

@tool
def get_housing_context() -> Dict[str, List[str]]:
    """
    Returns the full mapping of Housing Metric groups to their corresponding
    individual column names.
    
    Structure: { "<Group Name>": ["<col1>", "<col2>"] }
    
    Useful for:
      - Understanding which columns belong to 'Financials', 'Location', etc.
      - Resolving vague user queries like "Show me money-related data".
    """
    return metric_groups

@tool
def execute_read_query(query: str):
    """
    Executes a SELECT query on the 'housing_data' table and returns the results.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        if len(df) > 20:
            return f"Result (First 20 of {len(df)} rows):\n{df.head(20).to_string(index=False)}"
        return df.to_string(index=False)
    except Exception as e:
        return f"SQL Error: {e}"

# --- 4. PROMPT ENGINEERING ---

housing_agent_prompt = f"""
You are an agent that helps users with questions about the California Housing Dataset.
Provide clear, simple, and professional answers.

You have access to:
- The table name: {TABLE_NAME}
- The available columns: {columns}
- Sample rows: {sample_rows}
- Metric Groups: {list(metric_groups.keys())}
- A context tool to fetch column groupings.

-----------------------------
HOW TO ANSWER
-----------------------------
1. If the user asks for specific data (count, average, list):
   - Write a valid SQL query for SQLite.
   - Use 'execute_read_query'.

2. If the user asks about a general category (e.g., "Financials", "Location"):
   - Call 'get_housing_context' to see which columns belong to that category.
   - Then query those specific columns.

3. If the user asks for definitions:
   - Explain using the column names.

-----------------------------
RULES
-----------------------------
- Do NOT assume columns exist if they are not in the list.
- Do NOT use Markdown for the final answer, just plain text.
- Be concise.
"""

# --- 5. GRAPH CONSTRUCTION ---

tools = [execute_read_query, get_housing_context]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = model.bind_tools(tools)

def llm_call(state: MessagesState):
    """Node: LLM decides action."""
    return {
        "messages": [
            llm_with_tools.invoke(
                [SystemMessage(content=housing_agent_prompt)] + state["messages"]
            )
        ]
    }

def tool_node(state: MessagesState):
    """Node: Execute tools."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        print(f"   ⚙️ Executing Tool: {tool_call['name']}...", flush=True)
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Edge: Check for tool calls."""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tool_node"
    return END

# Build the Graph
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

# Compile
checkpointer = InMemorySaver()
agent = agent_builder.compile(checkpointer=checkpointer)

# --- 6. RUN LOOP (Replaces FastAPI) ---

# --- REPLACEMENT MAIN LOOP ---

def main():
    print("--- 🏠 HOUSING AGENT (LangGraph) ---", flush=True)
    print("System initialized. Type 'exit' to quit.\n")
    
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    while True:
        try:
            print("You > ", end="", flush=True)
            user_input = sys.stdin.readline().strip()
            
            if not user_input: continue
            if user_input.lower() in ['exit', 'quit']: break
            
            # Run the Graph
            messages = [HumanMessage(content=user_input)]
            responses = agent.invoke({"messages": messages}, config)
            
            # --- THE FIX IS HERE ---
            last_msg = responses['messages'][-1]
            content = last_msg.content
            
            # 1. Handle List Output (The messy JSON you saw)
            if isinstance(content, list):
                # Extract just the 'text' parts and join them
                bot_reply = "".join([block['text'] for block in content if 'text' in block])
            
            # 2. Handle Simple String Output
            else:
                bot_reply = str(content)
            
            print(f"\nBot > {bot_reply}\n")
            print("-" * 40)
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
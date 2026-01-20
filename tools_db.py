import sqlite3
from typing import List, Any
from langchain.tools import tool

# ✅ Single Source of Truth for Database Path
DB_PATH = "heart.db"

def get_db_connection() -> sqlite3.Connection:
    """Creates and returns a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)

def get_all_tables(db_path: str = DB_PATH) -> List[str]:
    """Returns a list of all table names in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables

def get_columns(db_path: str, table_name: str) -> List[str]:
    """Returns a list of column names for a specific table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    cols = [row[1] for row in cursor.fetchall()]
    conn.close()
    return cols

def get_sample_rows(db_path: str, table_name: str, n: int = 2) -> List[tuple]:
    """Returns the first N rows of data to help the AI understand values."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name} LIMIT {n};")
    rows = cursor.fetchall()
    conn.close()
    return rows

# ==========================================
# ACTUAL AI TOOLS (Decorated with @tool)
# ==========================================

@tool
def execute_read_query(query: str) -> str:
    """
    Executes a read-only SQL query against the heart.db database.
    Input must be a valid SQL SELECT statement.
    """
    clean_query = query.strip().upper()
    
    # Security: Prevent deletions or drops
    if not clean_query.startswith("SELECT"):
        return "Error: Only SELECT queries are allowed for safety."
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        data = cursor.fetchall()
        
        # Get headers to make the output readable for the LLM
        headers = [desc[0] for desc in cursor.description] if cursor.description else []
        
        conn.close()
        print("hola , Hi I am triggered!!")

        return f"Headers: {headers}\nData: {data}"

    except Exception as e:
        return f"Database Error: {e}"
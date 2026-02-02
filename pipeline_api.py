import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import sqlite3
import json
import os
import uuid
import shutil
from typing import Optional, List, Dict, Any
from langchain_core.messages import HumanMessage

# --- 1. CONFIGURATION & SETUP ---
WORKING_DIR = "pipeline_workspace"
PENDING_JSON = os.path.join(WORKING_DIR, "pending_contexts.json")
FINAL_DB_JSON = os.path.join(WORKING_DIR, "final_records.json")

# Ensure directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
for path in [PENDING_JSON, FINAL_DB_JSON]:
    if not os.path.exists(path):
        with open(path, "w") as f: json.dump({}, f)

# Try importing your specific LLM wrapper
try:
    from config import model
    print("✅ LLM Model imported successfully.")
except ImportError:
    print("⚠️ 'config.py' not found. Using Mock AI mode.")
    model = None

# Try importing PyMongo for NoSQL support
try:
    from pymongo import MongoClient
    HAS_MONGO = True
except ImportError:
    HAS_MONGO = False
    print("⚠️ 'pymongo' not installed. NoSQL support disabled.")

app = FastAPI(title="GenAI Universal Data Pipeline")


# --- 2. DATA MODELS ---

class DbIngestRequest(BaseModel):
    connection_string: str  # e.g., "housing.db" OR "mongodb://localhost:27017/"
    db_type: str = "sqlite" # Options: 'sqlite', 'mongodb'
    target_name: Optional[str] = None # Table Name (SQL) or Collection Name (NoSQL)
    custom_query: Optional[str] = None # Optional custom query

class ContextUpdate(BaseModel):
    """Payload for the 'Review and Edit' step"""
    summary: Optional[str] = None
    business_tags: Optional[List[str]] = None
    user_notes: Optional[str] = None

# --- 3. HELPER FUNCTIONS ---

def load_json(filepath):
    with open(filepath, "r") as f: return json.load(f)

def save_json(filepath, data):
    with open(filepath, "w") as f: json.dump(data, f, indent=4)

def fetch_data(conn_str: str, db_type: str, target: str = None, query: str = None) -> pd.DataFrame:
    """Universal Adapter: Connects to SQL or NoSQL."""
    
    # A. SQLITE Logic
    if db_type == "sqlite":
        conn = sqlite3.connect(conn_str)
        if not target and not query:
            # Auto-detect first table
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            if not tables: raise ValueError("Database is empty.")
            target = tables[0][0]
        
        sql = query if query else f"SELECT * FROM {target} LIMIT 5"
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df

    # B. MONGODB Logic
    elif db_type == "mongodb":
        if not HAS_MONGO: raise ImportError("Install 'pymongo' to use NoSQL.")
        
        client = MongoClient(conn_str)
        db_name = "test_db" # Default DB name (In prod, extract from conn_str)
        
        if not target:
            target = client[db_name].list_collection_names()[0]
            
        # Fetch 5 docs, exclude ID
        data = list(client[db_name][target].find({}, {"_id": 0}).limit(5))
        client.close()
        return pd.DataFrame(data)

    else:
        raise ValueError(f"Unsupported DB type: {db_type}")

def run_ai_analysis(df: pd.DataFrame, source_info: str) -> Dict:
    """The 'Gen AI generates context' Node."""
    data_preview = df.head(5).to_string(index=False)
    
   
    # ... inside run_ai_analysis function ...

    prompt = f"""
    ROLE: 
    You are the 'Ingestion Agent' for an Enterprise Data Pipeline. 
    Your job is to validate and profile incoming datasets before they are committed to the database.

    CONTEXT OF THIS OPERATION:
    - Action: Ingesting and profiling a new dataset.
    - Fetch Source: {source_info} 
    
    DATA SAMPLE (First 5 Rows):
    {data_preview}
    
    TASK:
    1. Context Statement: Start by explicitly stating what you are analyzing and where it was fetched from (e.g., "I am analyzing the 'housing' table fetched from the SQLite database...").
    2. Business Summary: Write a concise description of the data's value.
    3. Tags: Suggest 3 technical or business keyword tags.
    """
    
    # ... rest of the function ...
    
    if model:
        try:
            ai_response = model.invoke([HumanMessage(content=prompt)]).content.strip()
        except Exception as e:
            ai_response = f"AI Error: {str(e)}"
    else:
        ai_response = "Mock Summary: Data loaded successfully (AI Model not found)."

    context_id = str(uuid.uuid4())[:8]
    return {
        "id": context_id,
        "source": source_info,
        "columns": list(df.columns),
        "ai_summary": ai_response,
        "business_tags": ["pending_review"],
        "status": "pending_review",
        "timestamp": pd.Timestamp.now().isoformat()
    }

# --- 4. API ENDPOINTS ---

# NODE 1: Ingest Database (SQL/NoSQL)
@app.post("/ingest/db")
async def ingest_db(request: DbIngestRequest):
    try:
        df = fetch_data(request.connection_string, request.db_type, request.target_name, request.custom_query)
        if df.empty: raise HTTPException(400, "No data found.")
        
        context = run_ai_analysis(df, f"{request.db_type.upper()}: {request.target_name or 'Auto'}")
        
        # Save to Pending JSON
        pending = load_json(PENDING_JSON)
        pending[context["id"]] = context
        save_json(PENDING_JSON, pending)
        
        return {"message": "Analyzed", "review_id": context["id"], "preview": context}
    except Exception as e:
        raise HTTPException(500, f"Ingest Error: {str(e)}")

# NODE 1 (Alt): Ingest File (Excel/CSV)
@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(WORKING_DIR, file.filename)
        with open(file_path, "wb") as f: shutil.copyfileobj(file.file, f)
        
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
            
        context = run_ai_analysis(df, f"File: {file.filename}")
        
        pending = load_json(PENDING_JSON)
        pending[context["id"]] = context
        save_json(PENDING_JSON, pending)
        
        return {"message": "Analyzed", "review_id": context["id"], "preview": context}
    except Exception as e:
        raise HTTPException(500, f"File Error: {str(e)}")

# NODE 2: Review & Edit (Get Context)
@app.get("/context/{review_id}")
async def get_context(review_id: str):
    pending = load_json(PENDING_JSON)
    if review_id not in pending: raise HTTPException(404, "ID not found")
    return pending[review_id]

# NODE 2: Review & Edit (Update Context)
@app.post("/context/{review_id}/update")
async def update_context(review_id: str, updates: ContextUpdate):
    pending = load_json(PENDING_JSON)
    if review_id not in pending: raise HTTPException(404, "ID not found")
    
    current = pending[review_id]
    if updates.summary: current["ai_summary"] = str(updates.summary)
    if updates.business_tags: current["business_tags"] = updates.business_tags
    if updates.user_notes: current["user_notes"] = updates.user_notes
    
    save_json(PENDING_JSON, pending)
    return {"message": "Updated", "data": current}

# NODE 3: Commit (Save to Final JSON)
@app.post("/context/{review_id}/commit")
async def commit_context(review_id: str):
    pending = load_json(PENDING_JSON)
    final_db = load_json(FINAL_DB_JSON)
    
    if review_id not in pending: raise HTTPException(404, "ID not found")
    
    # Move record
    record = pending.pop(review_id)
    record["status"] = "committed"
    final_db[review_id] = record
    
    save_json(PENDING_JSON, pending)
    save_json(FINAL_DB_JSON, final_db)
    
    return {"message": "Committed to Database", "record": record}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
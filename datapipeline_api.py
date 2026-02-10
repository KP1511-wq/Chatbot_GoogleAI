import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sqlite3
import json
import os
import uuid
import datetime
from typing import Optional, List, Dict
from langchain_core.messages import HumanMessage

# --- 1. CONFIGURATION ---
WORKING_DIR = "pipeline_workspace"
KNOWLEDGE_BASE_FILE = os.path.join(WORKING_DIR, "final_records.json")
DB_FILE = os.path.join(WORKING_DIR, "housing.db")

# Create workspace
if not os.path.exists(WORKING_DIR): os.makedirs(WORKING_DIR)
if not os.path.exists(KNOWLEDGE_BASE_FILE): 
    with open(KNOWLEDGE_BASE_FILE, "w") as f: json.dump({}, f)

# Try importing LLM (or use mock)
try:
    from config import model
except ImportError:
    model = None
    print("⚠️ LLM not found. Using Mock mode for summaries.")

app = FastAPI(title="Backend: Data Pipeline & Tools")

# --- 2. AUTO-FIX DATABASE (Self-Healing) ---
def initialize_database():
    if os.path.exists(DB_FILE):
        print(f"✅ Database found at: {DB_FILE}")
        return

    csv_source = "housing.csv"
    if os.path.exists(csv_source):
        print(f"⚠️ Database missing. Converting '{csv_source}' to SQL...")
        try:
            df = pd.read_csv(csv_source)
            conn = sqlite3.connect(DB_FILE)
            df.to_sql("housing", conn, if_exists="replace", index=False)
            conn.close()
            print(f"🎉 Success! Created '{DB_FILE}' with table 'housing'.")
        except Exception as e:
            print(f"❌ Error converting CSV: {e}")
    else:
        print("ℹ️ No data found. Waiting for manual ingest.")

initialize_database()

# --- 3. PART A: CONTEXT GENERATION API ---
class DbIngestRequest(BaseModel):
    connection_string: str = DB_FILE
    target_name: str = "housing"

@app.post("/ingest/generate_context")
async def ingest_and_analyze(request: DbIngestRequest):
    """
    Reads the DB, asks AI to summarize it, and saves metadata to JSON.
    """
    try:
        # 1. Read Data
        conn = sqlite3.connect(request.connection_string)
        df = pd.read_sql_query(f"SELECT * FROM {request.target_name} LIMIT 5", conn)
        total_rows = pd.read_sql_query(f"SELECT COUNT(*) FROM {request.target_name}", conn).iloc[0,0]
        conn.close()

        # 2. Generate Context with AI
        data_preview = df.to_string(index=False)
        columns = ", ".join(df.columns)
        
        prompt = f"""
        ANALYZE THIS DATASET:
        - Table: {request.target_name}
        - Total Rows: {total_rows}
        - Columns: {columns}
        - Sample: {data_preview}
        
        TASK: Write a 1-sentence business summary and 3 tags.
        """
        
        if model:
            ai_response = model.invoke([HumanMessage(content=prompt)]).content
        else:
            ai_response = f"Dataset with {total_rows} rows. Columns: {columns}"

        # 3. Save to Knowledge Base
        record_id = str(uuid.uuid4())[:8]
        record = {
            "source": request.target_name,
            "columns": df.columns.tolist(),
            "ai_summary": str(ai_response).strip(),
            "ingested_at": str(datetime.datetime.now())
        }

        with open(KNOWLEDGE_BASE_FILE, "r+") as f:
            kb = json.load(f)
            kb[record_id] = record
            f.seek(0)
            json.dump(kb, f, indent=4)

        return {"status": "Context Generated", "id": record_id, "summary": record["ai_summary"]}

    except Exception as e:
        raise HTTPException(500, detail=str(e))

# --- 4. PART B: Q&A TOOL API (The Researcher) ---
class HousingQuery(BaseModel):
    ocean_proximity: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    limit: int = 5
    sort_by: Optional[str] = None
    sort_order: Optional[str] = "ASC"

@app.post("/tools/housing_query")
async def query_housing_data(params: HousingQuery):
    """
    Safe API for the Agent to query specific rows.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        query = "SELECT * FROM housing WHERE 1=1"
        args = []

        if params.ocean_proximity:
            query += " AND ocean_proximity = ?"
            args.append(params.ocean_proximity)
        if params.min_price:
            query += " AND median_house_value >= ?"
            args.append(params.min_price)
        if params.max_price:
            query += " AND median_house_value <= ?"
            args.append(params.max_price)
            
        # Safe Sorting
        if params.sort_by in ["median_house_value", "median_income", "housing_median_age"]:
            order = "DESC" if params.sort_order == "DESC" else "ASC"
            query += f" ORDER BY {params.sort_by} {order}"
        
        query += f" LIMIT {params.limit}"
        
        df = pd.read_sql_query(query, conn, params=args)
        conn.close()
        
        if df.empty: return {"result": "No houses found."}
        return {"result": df.to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
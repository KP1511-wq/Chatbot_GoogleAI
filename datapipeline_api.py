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

# Ensure workspace exists
if not os.path.exists(WORKING_DIR): os.makedirs(WORKING_DIR)
if not os.path.exists(KNOWLEDGE_BASE_FILE): 
    with open(KNOWLEDGE_BASE_FILE, "w") as f: json.dump({}, f)

# Try importing LLM (Graceful fallback if missing)
try:
    from config import model
except ImportError:
    model = None
    print("⚠️ LLM not found in pipeline. Context summaries will be mocked.")

app = FastAPI(title="Backend: Data Pipeline & Tools")

# --- 2. AUTO-FIX DATABASE ---
def initialize_database():
    """Converts CSV to SQLite on startup if DB is missing."""
    if os.path.exists(DB_FILE): return
    csv_source = "housing.csv"
    if os.path.exists(csv_source):
        try:
            df = pd.read_csv(csv_source)
            conn = sqlite3.connect(DB_FILE)
            df.to_sql("housing", conn, if_exists="replace", index=False)
            conn.close()
            print(f"🎉 Created '{DB_FILE}' from housing.csv")
        except Exception as e:
            print(f"❌ Error converting CSV: {e}")

initialize_database()

# --- 3. CONTEXT API (Ingest) ---
class DbIngestRequest(BaseModel):
    connection_string: str = DB_FILE
    target_name: str = "housing"

@app.post("/ingest/generate_context")
async def ingest_and_analyze(request: DbIngestRequest):
    """Reads a sample of the DB and saves a summary to JSON."""
    try:
        conn = sqlite3.connect(request.connection_string)
        df = pd.read_sql_query(f"SELECT * FROM {request.target_name} LIMIT 5", conn)
        total_rows = pd.read_sql_query(f"SELECT COUNT(*) FROM {request.target_name}", conn).iloc[0,0]
        conn.close()

        data_preview = df.to_string(index=False)
        columns = ", ".join(df.columns)
        
        # Ask LLM to summarize
        prompt = f"ANALYZE THIS DATA:\nTable: {request.target_name}\nRows: {total_rows}\nCols: {columns}\nSample: {data_preview}\nTASK: Write a 1-sentence summary."
        ai_response = model.invoke([HumanMessage(content=prompt)]).content if model else f"Dataset with {total_rows} rows."

        # Save to Knowledge Base
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

        return {"status": "Context Generated", "id": record_id}

    except Exception as e:
        raise HTTPException(500, detail=str(e))

# --- 4. TOOL A: RAW DATA SEARCH (For "Find me a house...") ---
class HousingQuery(BaseModel):
    ocean_proximity: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    limit: int = 5
    sort_by: Optional[str] = None
    sort_order: Optional[str] = "ASC"

@app.post("/tools/housing_query")
async def query_housing_data(params: HousingQuery):
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
            
        if params.sort_by in ["median_house_value", "median_income", "housing_median_age"]:
            order = "DESC" if params.sort_order == "DESC" else "ASC"
            query += f" ORDER BY {params.sort_by} {order}"
        
        query += f" LIMIT {params.limit}"
        
        df = pd.read_sql_query(query, conn, params=args)
        conn.close()
        return {"result": df.to_dict(orient="records")} if not df.empty else {"result": "No houses found."}

    except Exception as e:
        raise HTTPException(500, detail=str(e))

# --- 5. TOOL B: STATISTICS (For Charts/Graphs) ---
class StatsQuery(BaseModel):
    group_by: str       # e.g. "ocean_proximity"
    target_col: str     # e.g. "median_house_value"
    agg_type: str = "AVG" # AVG, COUNT, SUM

@app.post("/tools/housing_stats")
async def query_housing_stats(params: StatsQuery):
    try:
        conn = sqlite3.connect(DB_FILE)
        
        # Security: Whitelist columns
        safe_cols = ["ocean_proximity", "housing_median_age", "median_house_value", "median_income", "total_rooms"]
        if params.group_by not in safe_cols or params.target_col not in safe_cols:
            return {"error": "Invalid column name."}

        # Build Aggregation Query
        query = f"""
            SELECT {params.group_by}, {params.agg_type}({params.target_col}) as value
            FROM housing
            GROUP BY {params.group_by}
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return {"result": df.to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
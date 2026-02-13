import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sqlite3
import json
import os
import uuid
import datetime
from typing import Optional, List, Any
from langchain_core.messages import HumanMessage

# --- CONFIGURATION ---
WORKING_DIR = "pipeline_workspace"
KNOWLEDGE_BASE_FILE = os.path.join(WORKING_DIR, "final_records.json")
DB_FILE = os.path.join(WORKING_DIR, "housing.db")

if not os.path.exists(WORKING_DIR): os.makedirs(WORKING_DIR)
if not os.path.exists(KNOWLEDGE_BASE_FILE): 
    with open(KNOWLEDGE_BASE_FILE, "w") as f: json.dump({}, f)

try:
    from config import model
except ImportError:
    model = None

app = FastAPI(title="Backend: Advanced Data Pipeline")

# --- DATABASE INIT ---
def initialize_database():
    if os.path.exists(DB_FILE): return
    csv_source = "housing.csv"
    if os.path.exists(csv_source):
        try:
            df = pd.read_csv(csv_source)
            conn = sqlite3.connect(DB_FILE)
            df.to_sql("housing", conn, if_exists="replace", index=False)
            conn.close()
            print(f"🎉 Created '{DB_FILE}' from CSV.")
        except Exception as e:
            print(f"❌ Error converting CSV: {e}")

initialize_database()

# --- AUTO-GENERATE CONTEXT ON STARTUP ---
def auto_generate_context():
    """Automatically generate context if knowledge base is empty"""
    if os.path.exists(KNOWLEDGE_BASE_FILE):
        with open(KNOWLEDGE_BASE_FILE, "r") as f:
            kb = json.load(f)
            if kb:  # If knowledge base already has data, skip
                print("✅ Knowledge base already populated.")
                return
    
    # Generate context
    print("📚 Generating database context...")
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT * FROM housing", conn)
        
        # Get column information
        column_info = {}
        for col in df.columns:
            col_data = {
                "type": str(df[col].dtype),
                "sample_values": df[col].dropna().head(3).tolist()
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_data["min"] = float(df[col].min())
                col_data["max"] = float(df[col].max())
                col_data["mean"] = float(df[col].mean())
            
            if df[col].dtype == 'object' or df[col].nunique() < 20:
                col_data["unique_values"] = df[col].unique().tolist()
                col_data["unique_count"] = int(df[col].nunique())
            
            column_info[col] = col_data
        
        conn.close()
        
        record = {
            "source": "housing",
            "total_rows": len(df),
            "columns": list(df.columns),
            "column_details": column_info,
            "sample_data": df.head(5).to_dict(orient="records"),
            "description": "California Housing dataset with location, property features, and median house values.",
            "ingested_at": str(datetime.datetime.now())
        }
        
        with open(KNOWLEDGE_BASE_FILE, "w") as f:
            json.dump({"auto_generated": record}, f, indent=4)
        
        print("✅ Context generated successfully!")
    except Exception as e:
        print(f"❌ Error generating context: {e}")

auto_generate_context()

# --- CONTEXT GENERATION ---
class DbIngestRequest(BaseModel):
    connection_string: str = DB_FILE
    target_name: str = "housing"

@app.post("/ingest/generate_context")
async def ingest_and_analyze(request: DbIngestRequest):
    try:
        conn = sqlite3.connect(request.connection_string)
        
        # Get total rows
        total_rows = pd.read_sql_query(f"SELECT COUNT(*) FROM {request.target_name}", conn).iloc[0,0]
        
        # Get full dataframe for analysis
        df = pd.read_sql_query(f"SELECT * FROM {request.target_name}", conn)
        
        # Get column information with types and sample values
        column_info = {}
        for col in df.columns:
            col_data = {
                "type": str(df[col].dtype),
                "sample_values": df[col].dropna().head(3).tolist()
            }
            
            # Add statistics for numeric columns
            if df[col].dtype in ['int64', 'float64']:
                col_data["min"] = float(df[col].min())
                col_data["max"] = float(df[col].max())
                col_data["mean"] = float(df[col].mean())
            
            # Add unique values for categorical columns (if reasonable count)
            if df[col].dtype == 'object' or df[col].nunique() < 20:
                col_data["unique_values"] = df[col].unique().tolist()
                col_data["unique_count"] = int(df[col].nunique())
            
            column_info[col] = col_data
        
        conn.close()

        record_id = str(uuid.uuid4())[:8]
        record = {
            "source": request.target_name,
            "total_rows": int(total_rows),
            "columns": list(df.columns),
            "column_details": column_info,
            "sample_data": df.head(5).to_dict(orient="records"),
            "description": "California Housing dataset with location, property features, and median house values.",
            "ingested_at": str(datetime.datetime.now())
        }

        with open(KNOWLEDGE_BASE_FILE, "r+") as f:
            kb = json.load(f)
            kb[record_id] = record
            f.seek(0)
            json.dump(kb, f, indent=4)
            f.truncate()

        return {"status": "Context Generated", "id": record_id, "record": record}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

# --- TOOL 1: SEARCH (Fully Optional Params) ---
class HousingQuery(BaseModel):
    ocean_proximity: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_bedrooms: Optional[float] = None
    max_bedrooms: Optional[float] = None
    limit: Optional[int] = 5
    sort_by: Optional[str] = "median_house_value"
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
        if params.min_bedrooms:
            query += " AND total_bedrooms >= ?"
            args.append(params.min_bedrooms)
        if params.max_bedrooms:
            query += " AND total_bedrooms <= ?"
            args.append(params.max_bedrooms)

        sort_col = params.sort_by if params.sort_by else "median_house_value"
        order = "DESC" if params.sort_order and params.sort_order.upper() == "DESC" else "ASC"
        limit = params.limit if params.limit else 5
        
        query += f" ORDER BY {sort_col} {order} LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn, params=args)
        conn.close()
        
        result = df.to_dict(orient="records")
        return {
            "result": result,
            "count": len(result),
            "query_params": params.dict()
        }
    except Exception as e:
        return {"result": [], "error": str(e)}

# --- TOOL 2: STATISTICS (Fully Optional Params) ---
class StatsQuery(BaseModel):
    group_by: Optional[str] = "ocean_proximity"
    target_col: Optional[str] = "median_house_value"
    agg_type: Optional[str] = "AVG"

@app.post("/tools/housing_stats")
async def query_housing_stats(params: StatsQuery):
    try:
        conn = sqlite3.connect(DB_FILE)
        
        # Defaults if missing
        g_by = params.group_by if params.group_by else "ocean_proximity"
        t_col = params.target_col if params.target_col else "median_house_value"
        agg = params.agg_type if params.agg_type else "AVG"

        # Map 'average' -> 'AVG', handle case-insensitive
        agg_map = {
            "average": "AVG", 
            "mean": "AVG", 
            "avg": "AVG", 
            "sum": "SUM", 
            "count": "COUNT", 
            "min": "MIN", 
            "max": "MAX"
        }
        sql_agg = agg_map.get(agg.lower(), "AVG")

        # Construct query
        query = f"SELECT {g_by}, {sql_agg}({t_col}) as value FROM housing GROUP BY {g_by} ORDER BY value DESC"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        result = df.to_dict(orient="records")
        
        return {
            "result": result,
            "query_params": {
                "group_by": g_by,
                "target_col": t_col,
                "agg_type": sql_agg
            },
            "count": len(result)
        }
    except Exception as e:
        return {"result": [], "error": str(e)}

# --- ADDITIONAL ENDPOINT: Get Database Schema ---
@app.get("/schema")
async def get_schema():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(housing)")
        columns = cursor.fetchall()
        conn.close()
        
        schema = [
            {
                "name": col[1],
                "type": col[2],
                "nullable": not col[3],
                "primary_key": bool(col[5])
            }
            for col in columns
        ]
        
        return {"table": "housing", "columns": schema}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

# --- HEALTH CHECK ---
@app.get("/health")
async def health_check():
    db_exists = os.path.exists(DB_FILE)
    kb_exists = os.path.exists(KNOWLEDGE_BASE_FILE)
    
    return {
        "status": "healthy" if db_exists and kb_exists else "degraded",
        "database": "connected" if db_exists else "missing",
        "knowledge_base": "loaded" if kb_exists else "missing"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
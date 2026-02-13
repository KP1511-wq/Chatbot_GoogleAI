import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
import requests
import re
import ast
from typing import Union
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from config import model
except ImportError:
    model = None

WORKING_DIR = "pipeline_workspace"
KNOWLEDGE_BASE_FILE = os.path.join(WORKING_DIR, "final_records.json")

# URLS
SEARCH_API_URL = "http://127.0.0.1:8000/tools/housing_query"
STATS_API_URL = "http://127.0.0.1:8000/tools/housing_stats"

app = FastAPI(title="Agent Interface")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: Union[dict, str]

# --- ROBUST JSON PARSER ---
def parse_llm_output(text: str):
    try:
        # Remove markdown code blocks if present
        text = re.sub(r'```json\s*|\s*```', '', text)
        
        # Find JSON block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match: return None
        blob = match.group(0)
        
        # Clean up escaped newlines and extra whitespace
        blob = blob.replace('\\n', ' ').replace('\n', ' ')
        blob = re.sub(r'\s+', ' ', blob)
        
        # Try Standard JSON
        try:
            return json.loads(blob)
        except Exception as e:
            # Try Python Dict (Single Quotes)
            try:
                return ast.literal_eval(blob)
            except:
                print(f"JSON Parse Error: {e}")
                print(f"Attempted to parse: {blob[:200]}")
                return None
    except Exception as e:
        print(f"Outer Parse Error: {e}")
        return None

def validate_and_fix_vegalite(spec: dict) -> dict:
    """Validate and fix common Vega-Lite errors"""
    
    # Ensure required top-level keys
    if "data" not in spec:
        return None
    if "mark" not in spec:
        return None
    if "encoding" not in spec:
        return None
    
    # Fix common nesting issues
    if "config" in spec:
        if "view" in spec["config"]:
            view = spec["config"].pop("view")
            if "width" in view:
                spec["width"] = view["width"]
            if "height" in view:
                spec["height"] = view["height"]
        
        # Remove empty config
        if not spec["config"]:
            del spec["config"]
    
    # Move misplaced axis settings
    if "axis" in spec:
        # This shouldn't be at top level
        del spec["axis"]
    
    # Ensure data.values exists
    if "values" not in spec["data"]:
        return None
    
    # Ensure encoding channels have required fields
    for channel in ["x", "y", "theta", "color"]:
        if channel in spec["encoding"]:
            enc = spec["encoding"][channel]
            if "field" not in enc:
                return None
    
    return spec

def get_context_summary():
    if not os.path.exists(KNOWLEDGE_BASE_FILE): return "No data."
    with open(KNOWLEDGE_BASE_FILE, "r") as f:
        data = json.load(f)
    return str(data)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not model: return ChatResponse(response="Error: No AI model.")

    system_prompt = f"""
    ROLE: Intelligent Data Agent for California Housing Dataset.
    
    DATABASE CONTEXT:
    {get_context_summary()}
    
    AVAILABLE TOOLS:
    
    1. housing_query - For retrieving specific house records
       Parameters (all optional):
       - ocean_proximity: Filter by location (e.g., "NEAR OCEAN", "INLAND", "<1H OCEAN", "NEAR BAY", "ISLAND")
       - min_price / max_price: Filter by median_house_value
       - min_bedrooms / max_bedrooms: Filter by total_bedrooms
       - limit: Number of results (default 5)
       - sort_by: Column to sort by (default "median_house_value")
       - sort_order: "ASC" or "DESC" (default "ASC")
    
    2. housing_stats - For aggregated statistics and charts
       Parameters (all optional):
       - group_by: Column to group by (default "ocean_proximity")
       - target_col: Column to aggregate (default "median_house_value")
       - agg_type: "AVG", "SUM", "COUNT", "MIN", "MAX" (default "AVG")
    
    QUERY INTERPRETATION RULES:
    
    - "costliest" / "most expensive" / "highest price" → sort_by="median_house_value", sort_order="DESC"
    - "cheapest" / "lowest price" / "least expensive" → sort_by="median_house_value", sort_order="ASC"
    - "largest" / "biggest" / "most rooms" → sort_by="total_rooms" or "total_bedrooms", sort_order="DESC"
    - "plot" / "chart" / "graph" / "visualize" → Use housing_stats tool
    - "average" / "mean" → Use housing_stats with agg_type="AVG"
    - "total" / "sum" → Use housing_stats with agg_type="SUM"
    - "find" / "show" / "get" / "list" → Use housing_query tool
    
    EXAMPLES:
    
    User: "Find the 5 most expensive houses"
    Response: {{"tool": "housing_query", "parameters": {{"sort_by": "median_house_value", "sort_order": "DESC", "limit": 5}}}}
    
    User: "Show me the cheapest houses near the ocean"
    Response: {{"tool": "housing_query", "parameters": {{"ocean_proximity": "NEAR OCEAN", "sort_by": "median_house_value", "sort_order": "ASC", "limit": 5}}}}
    
    User: "Plot average house price by ocean proximity"
    Response: {{"tool": "housing_stats", "parameters": {{"group_by": "ocean_proximity", "target_col": "median_house_value", "agg_type": "AVG"}}}}
    
    User: "What are the most expensive inland houses?"
    Response: {{"tool": "housing_query", "parameters": {{"ocean_proximity": "INLAND", "sort_by": "median_house_value", "sort_order": "DESC", "limit": 5}}}}
    
    IMPORTANT:
    - Always output valid JSON with "tool" and "parameters" keys
    - For greetings, respond conversationally (no tool call)
    - Default limit is 5 unless user specifies otherwise
    - For visualization requests, ALWAYS use housing_stats
    """

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=request.message)]

    try:
        # Step 1: Think
        ai_msg_1 = model.invoke(messages).content
        content_1 = str(ai_msg_1).strip()
        
        # Step 2: Parse Tool
        tool_call = parse_llm_output(content_1)

        if tool_call:
            # Handle "Lazy" format (missing "tool" key)
            tool_name = None
            params = {}

            if "tool" in tool_call:
                tool_name = tool_call["tool"]
                params = tool_call.get("parameters", {})
            elif "housing_stats" in tool_call:
                tool_name = "housing_stats"
                params = tool_call["housing_stats"]
            elif "housing_query" in tool_call:
                tool_name = "housing_query"
                params = tool_call["housing_query"]

            # EXECUTE TOOL
            if tool_name == "housing_query":
                print(f"⚡ Searching: {params}")
                api_res = requests.post(SEARCH_API_URL, json=params)
                result_data = api_res.json()
                
                # Format the response better
                final_prompt = f"""
                User asked: "{request.message}"
                
                Query executed with parameters: {json.dumps(params, indent=2)}
                
                Database returned {result_data.get('count', 0)} results:
                {json.dumps(result_data.get('result', []), indent=2)}
                
                Instructions:
                - Provide a clear, natural language summary of the results
                - Highlight the most relevant information based on the user's query
                - If showing house prices, format them with $ and commas (e.g., $240,084)
                - If the query was about "costliest" or "cheapest", emphasize the price information
                - Be concise but informative
                """
                ai_msg_2 = model.invoke([HumanMessage(content=final_prompt)]).content
                return ChatResponse(response=str(ai_msg_2))

            elif tool_name == "housing_stats":
                print(f"📊 Charting: {params}")
                api_res = requests.post(STATS_API_URL, json=params)
                api_data = api_res.json()
                
                print(f"API Response: {api_data}")
                
                # Extract the actual data
                data_values = api_data.get("result", [])
                
                if not data_values:
                    return ChatResponse(response="No data returned from database.")
                
                # Determine chart type from user message
                user_msg_lower = request.message.lower()
                
                # Build Vega-Lite spec directly
                if "pie" in user_msg_lower or "distribution" in user_msg_lower or "share" in user_msg_lower:
                    # PIE CHART
                    vega_spec = {
                        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                        "width": 400,
                        "height": 400,
                        "data": {"values": data_values},
                        "mark": {"type": "arc", "outerRadius": 120},
                        "encoding": {
                            "theta": {"field": "value", "type": "quantitative"},
                            "color": {
                                "field": list(data_values[0].keys())[0],  # First field (group_by)
                                "type": "nominal"
                            }
                        }
                    }
                elif "scatter" in user_msg_lower or "correlation" in user_msg_lower:
                    # SCATTER PLOT
                    vega_spec = {
                        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                        "width": 800,
                        "height": 600,
                        "data": {"values": data_values},
                        "mark": "circle",
                        "encoding": {
                            "x": {"field": list(data_values[0].keys())[0], "type": "nominal"},
                            "y": {"field": "value", "type": "quantitative"}
                        }
                    }
                elif "line" in user_msg_lower or "trend" in user_msg_lower:
                    # LINE CHART
                    vega_spec = {
                        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                        "width": 800,
                        "height": 600,
                        "data": {"values": data_values},
                        "mark": "line",
                        "encoding": {
                            "x": {"field": list(data_values[0].keys())[0], "type": "nominal"},
                            "y": {"field": "value", "type": "quantitative"}
                        }
                    }
                else:
                    # BAR CHART (default)
                    group_field = list(data_values[0].keys())[0]  # First field is the group_by
                    vega_spec = {
                        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                        "width": 800,
                        "height": 600,
                        "data": {"values": data_values},
                        "mark": "bar",
                        "encoding": {
                            "x": {
                                "field": group_field,
                                "type": "nominal",
                                "axis": {"labelAngle": 0}
                            },
                            "y": {
                                "field": "value",
                                "type": "quantitative"
                            }
                        }
                    }
                
                print(f"Generated Vega-Lite spec: {json.dumps(vega_spec, indent=2)}")
                return ChatResponse(response=vega_spec)

        # If no tool, return text
        return ChatResponse(response=content_1)

    except Exception as e:
        return ChatResponse(response=f"Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
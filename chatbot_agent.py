import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import requests
import re
from typing import Any, Union
from langchain_core.messages import HumanMessage, SystemMessage

# --- SETUP ---
try:
    from config import model
except ImportError:
    print("❌ LLM not found. Agent will fail.")
    model = None

WORKING_DIR = "pipeline_workspace"
KNOWLEDGE_BASE_FILE = os.path.join(WORKING_DIR, "final_records.json")

# Tool URLs
SEARCH_API_URL = "http://localhost:8000/tools/housing_query"
STATS_API_URL = "http://localhost:8000/tools/housing_stats"

app = FastAPI(title="Agent Interface")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str

# 🚨 CHANGE 1: Allow the response to be a Dictionary (JSON Object)
class ChatResponse(BaseModel):
    response: Union[dict, str]

# 🚨 CHANGE 2: Robust Cleaner Function
def extract_json_from_text(text_response: str):
    """
    Finds the first valid JSON object inside the text, ignores the rest.
    """
    try:
        # Remove common markdown clutter
        clean_text = re.sub(r"```json\s*", "", text_response, flags=re.IGNORECASE)
        clean_text = re.sub(r"```", "", clean_text)
        
        # Hunt for the actual curly braces { ... }
        start_idx = clean_text.find("{")
        end_idx = clean_text.rfind("}")
        
        if start_idx != -1 and end_idx != -1:
            possible_json = clean_text[start_idx : end_idx + 1]
            return json.loads(possible_json)
        else:
            return text_response # Return text if no brackets found
            
    except json.JSONDecodeError:
        return text_response

def clean_text(text: str) -> str:
    return " ".join(text.replace("**", "").replace("* ", "").replace("\n", " ").split())

def get_context_summary():
    if not os.path.exists(KNOWLEDGE_BASE_FILE): return "No data."
    with open(KNOWLEDGE_BASE_FILE, "r") as f:
        data = json.load(f)
    if not data: return "No data."
    
    summary = ""
    for k, v in data.items():
        summary += f"- Table: {v['source']}\n  Summary: {v['ai_summary']}\n  Columns: {', '.join(v['columns'])}\n"
    return summary

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not model: return ChatResponse(response="Error: No AI model.")

    # Simplified System Prompt
    system_prompt = f"""
    ROLE: Data Visualization Assistant.
    CONTEXT: {get_context_summary()}
    TOOLS:
    1. Housing Stats (Charts): {{"tool": "housing_stats", "parameters": {{"group_by": "ocean_proximity", "target_col": "median_house_value", "agg_type": "AVG"}}}}
    INSTRUCTIONS: Use 'housing_stats' for charts. Output ONLY JSON.
    """

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=request.message)]

    try:
        ai_msg_1 = model.invoke(messages).content
        content_1 = str(ai_msg_1).strip()

        if '{"tool":' in content_1:
            json_str = content_1.replace("```json", "").replace("```", "").strip()
            tool_call = json.loads(json_str)
            tool_name = tool_call.get("tool")
            
            if tool_name == "housing_stats":
                print(f"📊 Generating Chart Data: {tool_call}")
                api_res = requests.post(STATS_API_URL, json=tool_call["parameters"])
                tool_result = api_res.json()
                
                # STRICT TEMPLATE PROMPT
                final_prompt = f"""
                DATA: {json.dumps(tool_result)}
                TASK: Generate a Vega-Lite v5 JSON spec.
                STRUCTURE:
                {{
                  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                  "data": {{ "values": <DATA_ARRAY> }},
                  "mark": "bar",
                  "encoding": {{
                    "x": {{ "field": "{tool_call['parameters']['group_by']}", "type": "nominal" }},
                    "y": {{ "field": "value", "type": "quantitative" }}
                  }}
                }}
                RULES: Replace <DATA_ARRAY> with valid JSON. Output ONLY JSON.
                """
                
                ai_msg_2 = model.invoke([HumanMessage(content=final_prompt)]).content
                
                # 🚨 CHANGE 3: Clean and Parse before returning
                structured_response = extract_json_from_text(str(ai_msg_2))
                return ChatResponse(response=structured_response)

        return ChatResponse(response=clean_text(content_1))

    except Exception as e:
        return ChatResponse(response=f"Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import requests
from langchain_core.messages import HumanMessage, SystemMessage

# --- SETUP ---
try:
    from config import model
except ImportError:
    print("❌ LLM not found. Agent will fail.")
    model = None

WORKING_DIR = "pipeline_workspace"
KNOWLEDGE_BASE_FILE = os.path.join(WORKING_DIR, "final_records.json")
TOOL_API_URL = "http://localhost:8000/tools/housing_query"

app = FastAPI(title="Agent Interface")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# --- HELPER: TEXT CLEANER ---
def clean_text(text: str) -> str:
    """Removes markdown and newlines for clean output."""
    t = text.replace("**", "").replace("* ", "").replace("\n", " ")
    return " ".join(t.split())

# --- HELPER: LOAD CONTEXT ---
def get_context_summary():
    if not os.path.exists(KNOWLEDGE_BASE_FILE): return "No data."
    with open(KNOWLEDGE_BASE_FILE, "r") as f:
        data = json.load(f)
    if not data: return "No data."
    
    summary = ""
    for k, v in data.items():
        summary += f"- Table: {v['source']}\n"
        summary += f"  Summary: {v['ai_summary']}\n"
        summary += f"  Columns: {', '.join(v['columns'])}\n"
    return summary

# --- MAIN CHAT LOGIC ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not model: return ChatResponse(response="Error: No AI model.")

    # 1. System Prompt (The Brain)
    system_prompt = f"""
    ROLE: Data Assistant.
    
    CONTEXT (What you know):
    {get_context_summary()}
    
    TOOL (What you can do):
    - Name: Housing Search
    - Trigger: Requests for specific houses, prices, or locations.
    - Format: Output JSON ONLY: 
      {{"tool": "housing_query", "parameters": {{"ocean_proximity": "NEAR BAY", "max_price": 500000, "sort_by": "median_house_value", "sort_order": "ASC"}}}}
    
    INSTRUCTIONS:
    - If answer is in CONTEXT, answer directly.
    - If specific rows are needed, OUTPUT JSON.
    """

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=request.message)]

    try:
        # Step 1: Think
        ai_msg_1 = model.invoke(messages).content
        content_1 = str(ai_msg_1).strip()

        # Step 2: Check for Tool Use
        if '{"tool":' in content_1:
            # Extract JSON
            json_str = content_1.replace("```json", "").replace("```", "").strip()
            tool_call = json.loads(json_str)
            
            # Step 3: Act (Call Backend)
            print(f"⚡ calling Tool: {tool_call}")
            api_res = requests.post(TOOL_API_URL, json=tool_call["parameters"])
            tool_result = api_res.json()
            
            # Step 4: Final Answer
            final_prompt = f"User asked: {request.message}\nTool Found: {tool_result}\nSummarize this for the user."
            ai_msg_2 = model.invoke([HumanMessage(content=final_prompt)]).content
            return ChatResponse(response=clean_text(str(ai_msg_2)))

        return ChatResponse(response=clean_text(content_1))

    except Exception as e:
        return ChatResponse(response=f"Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
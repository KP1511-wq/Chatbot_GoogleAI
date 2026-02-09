import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
import requests  # <--- NEW: Needed to talk to the Pipeline API
from langchain_core.messages import HumanMessage, SystemMessage

# --- 1. SETUP & IMPORTS ---
try:
    from config import model
    print("✅ LLM Model imported successfully.")
except ImportError:
    print("❌ ERROR: Could not find 'config.py'.")
    model = None

# Configuration
WORKING_DIR = "pipeline_workspace"
KNOWLEDGE_BASE_FILE = os.path.join(WORKING_DIR, "final_records.json")

# URL of the Tool we created in pipeline_api.py
TOOL_API_URL = "http://localhost:8000/tools/housing_query" 

app = FastAPI(title="Agent with Tool Use")



class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# --- 2. HELPER: LOAD CONTEXT ---
def load_pipeline_context() -> str:
    """Loads the high-level summary of what data we have."""
    if not os.path.exists(KNOWLEDGE_BASE_FILE):
        return "No datasets available."
    try:
        with open(KNOWLEDGE_BASE_FILE, "r") as f:
            data = json.load(f)
        if not data: return "No datasets found."

        context = "METADATA ON AVAILABLE DATASETS:\n"
        for i, (rid, record) in enumerate(data.items(), 1):
            context += f"{i}. SOURCE: {record.get('source')}\n"
            context += f"   SUMMARY: {record.get('ai_summary')}\n"
            context += f"   COLUMNS: {', '.join(record.get('columns', []))}\n\n"
        return context
    except:
        return "Error loading context."

# --- 3. THE CHAT LOOP WITH TOOLS ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_data(request: ChatRequest):
    if not model:
        return ChatResponse(response="System Error: LLM not initialized.")

    pipeline_context = load_pipeline_context()

    # --- PROMPT ENGINEERING: THE "ROUTER" ---
    system_prompt = f"""
    ### ROLE
    You are an Intelligent Data Assistant. You have two ways to answer:
    1. **Knowledge Base:** Use the summary below for general questions (e.g. "What data do we have?").
    2. **Housing Tool:** YOU MUST use this tool to find SPECIFIC rows, prices, or locations.

    ### KNOWLEDGE BASE
    {pipeline_context}

    ### AVAILABLE TOOLS
    - **Housing Search**: Use this to find specific houses.
      - **Trigger:** User asks for "cheapest", "most expensive", "prices", "near X".
      - **Format:** Output JSON.
      - **Sorting:** - For "cheapest", use: "sort_by": "median_house_value", "sort_order": "ASC"
          - For "most expensive", use: "sort_by": "median_house_value", "sort_order": "DESC"
      - **Example:** {{"tool": "housing_query", "parameters": {{"ocean_proximity": "NEAR BAY", "sort_by": "median_house_value", "sort_order": "DESC", "limit": 1}}}}
    
    ### CRITICAL INSTRUCTIONS
    - **NEVER REFUSE TO ANSWER.** If the user asks for specific data , do not say "I cannot." Instead, OUTPUT THE JSON to use the tool.
    - If selecting a tool, output **ONLY JSON**. No text before or after.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=request.message)
    ]

    try:
        # STEP 1: Ask LLM what to do (Think)
        response_1 = model.invoke(messages)
        content_1 = str(response_1.content).strip()

        # STEP 2: Check if it wants to use a tool (Check)
        # We look for the JSON signal
        if '{"tool":' in content_1:
            print(f"🕵️ AI Decided to use tool: {content_1}")
            
            # Clean up JSON (remove markdown blocks if LLM adds them)
            json_str = content_1.replace("```json", "").replace("```", "").strip()
            
            try:
                tool_call = json.loads(json_str)
                
                # STEP 3: Execute Tool (Act)
                # This sends the search parameters to your Pipeline API
                print(f"⚡ Calling API: {TOOL_API_URL} with {tool_call['parameters']}")
                api_res = requests.post(TOOL_API_URL, json=tool_call["parameters"])
                
                if api_res.status_code == 200:
                    tool_data = api_res.json()
                else:
                    tool_data = f"Error from API: {api_res.text}"

            except Exception as e:
                tool_data = f"Tool Execution Failed: {str(e)}"
            

            # STEP 4: Final Summary (Response)
            # Feed the real data back to the LLM to summarize for the user
            final_prompt = f"""
            User Question: {request.message}
            
            REAL DATA FOUND BY TOOL: 
            {tool_data}
            
            Task: Summarize these specific housing results for the user in plain English. Mention prices and locations found.
            """
            response_2 = model.invoke([HumanMessage(content=final_prompt)])
            return ChatResponse(response=str(response_2.content))

        # If no tool used, just return the first answer
        return ChatResponse(response=content_1)

    except Exception as e:
        return ChatResponse(response=f"Agent Error: {str(e)}")

if __name__ == "__main__":
    # Runs on Port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)
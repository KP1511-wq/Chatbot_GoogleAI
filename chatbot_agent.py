import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
from langchain_core.messages import HumanMessage, SystemMessage

# --- 1. SETUP & IMPORTS ---
try:
    from config import model
    print("✅ LLM Model imported successfully.")
except ImportError:
    print("❌ ERROR: Could not find config.py'. Make sure it exists.")
    model = None

# Configuration: Point to the same folder as your pipeline
WORKING_DIR = "pipeline_workspace"
KNOWLEDGE_BASE_FILE = os.path.join(WORKING_DIR, "final_records.json")

app = FastAPI(title="Pipeline Chatbot Agent")

# --- 2. DATA MODELS ---
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# --- 3. HELPER: LOAD PIPELINE DATA (CLEANER VERSION) ---
def load_pipeline_context() -> str:
    """
    Reads the JSON but formats it strictly for AI readability.
    """
    if not os.path.exists(KNOWLEDGE_BASE_FILE):
        return "No datasets available."

    try:
        with open(KNOWLEDGE_BASE_FILE, "r") as f:
            data = json.load(f)
            
        if not data:
            return "No datasets found."

        # CLEANER FORMATTING:
        # We index them by number (1, 2, 3) instead of UUID (f40927...)
        # This forces the AI to speak in terms of "Source Name" not "ID".
        context_text = "AVAILABLE DATASETS:\n"
        for i, (record_id, record) in enumerate(data.items(), 1):
            source = record.get('source', 'Unknown File')
            summary = record.get('ai_summary', 'No summary provided')
            tags = ", ".join(record.get('business_tags', []))
            
            context_text += f"{i}. FILE: {source}\n"
            context_text += f"   SUMMARY: {summary}\n"
            context_text += f"   TAGS: {tags}\n"
            context_text += f"   (Internal ID: {record_id})\n\n"
            
        return context_text

    except Exception as e:
        return f"Error loading data: {str(e)}"

# --- 4. CHAT ENDPOINT (UPDATED PROMPT) ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_data(request: ChatRequest):
    if not model:
        return ChatResponse(response="LLM is not initialized.")

    pipeline_knowledge = load_pipeline_context()

    # UPDATED PROMPT: Forces clean, human-readable output
    system_prompt = f"""
        ### ROLE
        You are the "Pipeline Data Assistant." Your job is to help users understand what data is currently available in our system.
        
        ### KNOWLEDGE BASE (Available Datasets)
        {pipeline_knowledge}
        
        ### INSTRUCTIONS
        1. **Primary Source:** Answer ONLY based on the datasets listed above. Do not hallucinate data we don't have.
        2. **Query Handling:**
        - **If the user asks for a list:** Provide a clean, bulleted list using the File Name and a 1-sentence summary.
        - **If the user asks a specific question (e.g., "Do we have data on X?"):** Search the 'Summary' and 'Tags' of the available datasets. If a match is found, confirm it and name the file.
        - **If no data is found:** Politely state that no such dataset exists in the pipeline.
        
        ### FORMATTING RULES
        - Use **Plain Text** only (no markdown bolding/italics).
        - Be concise and professional.
        - Refer to datasets by their **File Name** (e.g., 'housing.db'), never their internal UUID.

        ### EXAMPLES
        User: "What do we have?"
        You: Here are the datasets in the pipeline:
            1. housing.db: Real estate data with prices and demographics.
            2. sales_q3.csv: Financial records for the third quarter.

        User: "Do we have any financial info?"
        You: Yes, the 'sales_q3.csv' file contains financial records and Q3 performance data.
        
        User: "Show me weather data."
        You: I do not see any weather-related datasets in the current pipeline.
        """

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.message)
        ]
        
        response = model.invoke(messages)
        
        # Standard cleaning for Gemini 1.5 response
        if isinstance(response.content, list):
            bot_reply = "".join([part.get('text', '') for part in response.content if isinstance(part, dict)])
        else:
            bot_reply = str(response.content)
            
        return ChatResponse(response=bot_reply)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

if __name__ == "__main__":
    # We run this on port 8001 so it doesn't conflict with the Pipeline (port 8000)
    uvicorn.run(app, host="0.0.0.0", port=8001)
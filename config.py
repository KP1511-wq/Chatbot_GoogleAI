import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ GROQ_API_KEY not found in .env. Please set it to use the model.")
    model = None
else:
    print("Connecting to GROQ...")
    try:
        model = ChatGroq(api_key=api_key,model_name="llama-3.1-8b-instant",temperature=0)
        print("✅ Connected to GROQ LLM.")
    except Exception as e:
        print(f"❌ Error connecting to GROQ: {e}")
        model = None
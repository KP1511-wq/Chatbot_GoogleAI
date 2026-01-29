import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Google Gemini
# We use 'gemini-3-flash' as it is fast and cheap for chatbots
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash",
    temperature=0
)

# Alias it as 'model' in case your other scripts look for that name
model = llm

if __name__ == "__main__":
    try:
        response = llm.invoke("Hello, are you working?")
        print("✅ Connection Successful!")
        print(f"🤖 Response: {response.content}")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure the key is set (or hardcode it here for testing)
# os.environ["GOOGLE_API_KEY"] = "AIzaSy..." 

# Initialize Google Gemini
# We use 'gemini-2.5-flash-lite' as it is fast and cheap for chatbots
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
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
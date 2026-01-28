import sqlite3
import pandas as pd
import sys
from langchain_core.messages import SystemMessage, HumanMessage

# --- IMPORT YOUR EXISTING MODEL ---
try:
    from config import model
except ImportError:
    print("❌ ERROR: Could not find 'config.py'")
    sys.exit()

sys.stdout.reconfigure(line_buffering=True)
DB_FILE = "housing.db"

class SmartBot:
    def __init__(self):
        self.system_prompt = self.load_context()
        self.chat_history = [] # Simple memory

    def load_context(self):
        """Reads the AI-generated descriptions from the database."""
        try:
            conn = sqlite3.connect(DB_FILE)
            df = pd.read_sql("SELECT * FROM ai_context", conn)
            conn.close()
            
            # Construct the Meta-Prompt
            prompt = "You are a Housing Data Expert. Here is the dictionary for the dataset you are analyzing:\n"
            for _, row in df.iterrows():
                prompt += f"- {row['column_name']}: {row['ai_description']}\n"
            
            prompt += "\nUse this context to answer user questions accurately. Be concise."
            return prompt
        except Exception as e:
            return f"Error loading context: {e}"

    def ask(self, user_input):
        # Build message list: System Context + Chat History + New Question
        messages = [SystemMessage(content=self.system_prompt)] + \
                   self.chat_history + \
                   [HumanMessage(content=user_input)]
        
        # Invoke your model
        response = model.invoke(messages)
        bot_reply = response.content
        
        # Update History (Keep it short for context window efficiency)
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(response) # response is already an AIMessage
        
        return bot_reply

# --- RUN LOOP ---
def main():
    print("\n--- 🤖 AI BOT ONLINE ---", flush=True)
    try:
        bot = SmartBot()
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        return

    print("I have loaded the context from your pipeline. Ask me anything!\n")

    while True:
        try:
            print("You > ", end="", flush=True)
            user_input = sys.stdin.readline().strip()
            
            if not user_input: continue
            if user_input.lower() in ['exit', 'quit']: break
            
            response = bot.ask(user_input)
            print(f"\nBot > {response}\n")
            print("-" * 40)
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
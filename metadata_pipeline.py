import pandas as pd
import sqlite3
import os
import sys
import time
from langchain_core.messages import HumanMessage

# --- IMPORT YOUR EXISTING MODEL ---
try:
    from config import model
    print("✅ Successfully imported 'model' from config.")
except ImportError:
    print("❌ ERROR: Could not find 'config.py' or 'model' object.")
    sys.exit()

# Force output
sys.stdout.reconfigure(line_buffering=True)

CSV_FILE = "housing.csv"
DB_FILE = "housing.db"

class ContextPipeline:
    def get_stats(self, df, col_name):
        """Extracts hard facts about the data."""
        series = df[col_name]
        return {
            "name": col_name,
            "dtype": str(series.dtype),
            "examples": list(series.dropna().unique()[:5])
        }

    def generate_description(self, stats):
        """Uses YOUR MODEL to write a definition."""
        prompt_text = f"""
        Act as a Data Dictionary Expert.
        Column Name: "{stats['name']}"
        Data Type: {stats['dtype']}
        Example Values: {stats['examples']}
        
        Task: Write a 1-sentence description of what this column represents in a housing dataset.
        """
        
        try:
            # Using LangChain 'invoke' syntax
            response = model.invoke([HumanMessage(content=prompt_text)])
            return response.content.strip()
        except Exception as e:
            return f"Description unavailable (Error: {e})"

    def run(self):
        print("--- 🧠 AI CONTEXT PIPELINE STARTED ---", flush=True)
        
        if not os.path.exists(CSV_FILE):
            print(f"❌ Error: {CSV_FILE} not found.")
            return

        print(f"   📥 Loading {CSV_FILE}...", flush=True)
        df = pd.read_csv(CSV_FILE)
        metadata = []

        print(f"   📡 Analyzing {len(df.columns)} columns using your Model...", flush=True)
        
        for col in df.columns:
            print(f"      👉 Learning '{col}'...", end="", flush=True)
            stats = self.get_stats(df, col)
            desc = self.generate_description(stats)
            
            metadata.append({
                "column_name": col,
                "ai_description": desc
            })
            print(" ✅")
            # Optional: Sleep if you hit rate limits
            # time.sleep(1) 

        # SAVE TO DB
        print(f"   💾 Saving Context to {DB_FILE}...", flush=True)
        conn = sqlite3.connect(DB_FILE)
        pd.DataFrame(metadata).to_sql("ai_context", conn, if_exists="replace", index=False)
        df.to_sql("housing_data", conn, if_exists="replace", index=False)
        conn.close()
        
        print("\n🎉 SUCCESS! The AI has documented your dataset.")

if __name__ == "__main__":
    ContextPipeline().run()
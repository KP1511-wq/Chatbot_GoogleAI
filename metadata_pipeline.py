import pandas as pd
import sqlite3
import os
import sys
import json
import time

# --- LANGCHAIN IMPORTS ---
from langchain_core.messages import HumanMessage

# --- IMPORT YOUR EXISTING MODEL ---
try:
    from config import model
    print("✅ Successfully imported 'model' from config.")
except ImportError:
    print("❌ ERROR: Could not find 'config.py' or 'model' object.")
    sys.exit()

# Force output to appear immediately
sys.stdout.reconfigure(line_buffering=True)

# --- CONFIGURATION ---
CSV_FILE = "housing.csv"
DB_FILE = "housing.db"

class ContextPipeline:
    def get_stats(self, df, col_name):
        """Extracts hard facts about the data to help the AI."""
        series = df[col_name]
        return {
            "name": col_name,
            "dtype": str(series.dtype),
            "examples": list(series.dropna().unique()[:5])
        }

    def generate_description(self, stats):
        """Task 1: Ask the AI to define a single column."""
        prompt = f"""
        Act as a Data Dictionary Expert.
        Column Name: "{stats['name']}"
        Data Type: {stats['dtype']}
        Example Values: {stats['examples']}
        
        Task: Write a concise, 1-sentence description of what this column represents in a housing dataset.
        Output: Just the sentence.
        """
        
        try:
            # Using standard LangChain invoke
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            return f"Description unavailable (Error: {e})"

    def generate_grouping_map(self, columns):
        """
        Task 2: Ask the AI to look at ALL columns and categorize them.
        This creates the 'Metric Groups' for the Agent tool.
        """
        print(f"   🧩 AI is categorizing {len(columns)} columns...", end="", flush=True)
        
        prompt = f"""
        I have these columns in a Housing Dataset: {columns}
        
        Task: Group them into 3-5 logical categories (e.g., 'Location', 'Financials', 'Specs').
        
        Output: A valid JSON object mapping Category Name -> List of Columns.
        Example Format: {{"Financials": ["income", "price"], "Location": ["lat", "long"]}}
        
        IMPORTANT: Return ONLY valid JSON. No Markdown formatting.
        """
        
        try:
            response = model.invoke([HumanMessage(content=prompt)]).content
            
            # Clean up if the AI adds ```json ... ``` wrappers
            clean_json = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except Exception as e:
            print(f" ❌ Error grouping: {e}")
            return {"General": columns} # Fallback

    def run(self):
        print("--- 🧠 AI CONTEXT PIPELINE STARTED ---", flush=True)
        
        # 1. READ CSV
        if not os.path.exists(CSV_FILE):
            print(f"❌ Error: {CSV_FILE} not found.")
            return

        print(f"   📥 Loading {CSV_FILE}...", flush=True)
        df = pd.read_csv(CSV_FILE)
        metadata = []

        # 2. GENERATE DEFINITIONS (Column by Column)
        print(f"   📡 Analyzing {len(df.columns)} columns...", flush=True)
        
        for col in df.columns:
            print(f"      👉 Learning '{col}'...", end="", flush=True)
            stats = self.get_stats(df, col)
            desc = self.generate_description(stats)
            
            metadata.append({
                "column_name": col,
                "ai_description": desc
            })
            print(" ✅")
            # time.sleep(1) # Uncomment if hitting rate limits

        # 3. GENERATE GROUPINGS (All at once)
        grouping_map = self.generate_grouping_map(list(df.columns))
        print(" ✅ Done.")

        # 4. SAVE TO DATABASE
        print(f"   💾 Saving Context & Data to {DB_FILE}...", flush=True)
        conn = sqlite3.connect(DB_FILE)
        
        # A. Save the Raw Data (so the agent can query it)
        df.to_sql("housing_data", conn, if_exists="replace", index=False)
        
        # B. Save the Definitions (for the System Prompt)
        pd.DataFrame(metadata).to_sql("ai_context", conn, if_exists="replace", index=False)
        
        # C. Save the Groupings (for the Tool)
        # We save the JSON as a string in a single row
        df_groups = pd.DataFrame([{"key": "main_grouping", "json_data": json.dumps(grouping_map)}])
        df_groups.to_sql("ai_groups", conn, if_exists="replace", index=False)
        
        conn.close()
        
        print("\n🎉 PIPELINE SUCCESS! The database is fully prepped for the Agent.")

if __name__ == "__main__":
    ContextPipeline().run()
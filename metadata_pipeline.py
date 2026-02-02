import pandas as pd
import sqlite3
import os
import sys
import json
import time

# --- 1. IMPORT LOGGER ---
from logger_config import setup_logger
logger = setup_logger("Pipeline_Builder")

# --- LANGCHAIN IMPORTS ---
from langchain_core.messages import HumanMessage

try:
    from config import model
    logger.info("Model imported successfully.")
except ImportError:
    logger.critical("ERROR: Could not find 'config.py'")
    sys.exit()

# Configuration
CSV_FILE = "housing.csv"
DB_FILE = "housing.db"

class ContextPipeline:
    def get_stats(self, df, col_name):
        series = df[col_name]
        return {
            "name": col_name,
            "dtype": str(series.dtype),
            "examples": list(series.dropna().unique()[:5])
        }

    def generate_description(self, stats):
        prompt = f"""
        Act as a Data Dictionary Expert.
        Column: "{stats['name']}" (Type: {stats['dtype']})
        Examples: {stats['examples']}
        Task: Write a 1-sentence description.
        """
        try:
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.warning(f" Failed to describe '{stats['name']}': {e}")
            return "Description unavailable."

    def generate_grouping_map(self, columns):
        logger.info(f"AI is categorizing {len(columns)} columns...")
        
        prompt = f"""
        Group these columns into 3-5 logical categories (JSON format only): {columns}
        Example: {{"Financials": ["income", "price"]}}
        """
        try:
            response = model.invoke([HumanMessage(content=prompt)]).content
            clean_json = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except Exception as e:
            logger.error(f"Error grouping columns: {e}")
            return {"General": columns}

    def run(self):
        logger.info("---  PIPELINE STARTED ---")
        
        if not os.path.exists(CSV_FILE):
            logger.critical(f"File not found: {CSV_FILE}")
            return

        logger.info(f"Loading {CSV_FILE}...")
        df = pd.read_csv(CSV_FILE)
        
        # --- PHASE 1: GENERATION (In Memory Only) ---
        metadata = []
        logger.info(f"Analyzing {len(df.columns)} columns...")
        
        for col in df.columns:
            stats = self.get_stats(df, col)
            desc = self.generate_description(stats)
            metadata.append({"column_name": col, "ai_description": desc})
            logger.info(f" Defined '{col}'")

        grouping_map = self.generate_grouping_map(list(df.columns))

        # --- PHASE 2: THE "CATCH" (Review Step) ---
        print("\n" + "="*50)
        print("🕵️  REVIEW GENERATED CONTEXT BEFORE SAVING")
        print("="*50)
        
        # 1. Show Groups
        print(f"\n🧩 GROUPS DETECTED:\n{json.dumps(grouping_map, indent=2)}")
        
        # 2. Show First 3 Definitions
        print(f"\n📖 SAMPLE DEFINITIONS (First 3):")
        for item in metadata[:3]:
            print(f"   - {item['column_name']}: {item['ai_description']}")
            
        print("\n" + "="*50)
        
        # 3. Ask for Permission
        user_input = input("Do you want to save this to 'housing.db'? (yes/no): ").strip().lower()
        
        if user_input != 'yes':
            logger.warning("Operation Cancelled by User. Nothing was saved.")
            return

        # --- PHASE 3: SAVING (Only if 'yes') ---
        logger.info(f"Saving to {DB_FILE}...")
        conn = sqlite3.connect(DB_FILE)
        
        df.to_sql("housing_data", conn, if_exists="replace", index=False)
        pd.DataFrame(metadata).to_sql("ai_context", conn, if_exists="replace", index=False)
        
        df_groups = pd.DataFrame([{"key": "main_grouping", "json_data": json.dumps(grouping_map)}])
        df_groups.to_sql("ai_groups", conn, if_exists="replace", index=False)
        
        conn.close()
        logger.info("PIPELINE COMPLETE successfully.")

if __name__ == "__main__":
    ContextPipeline().run()
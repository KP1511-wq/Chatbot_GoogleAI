import pandas as pd
import sqlite3
import os

# CONFIGURATION
CSV_FILE = "housing.csv"
DB_FILE = "housing.db"

def create_database():
    print("--- 🛠 SETUP: DATABASE MIGRATION ---")
    
    # 1. Check for CSV
    if not os.path.exists(CSV_FILE):
        print(f"❌ ERROR: '{CSV_FILE}' not found!")
        print("   Please paste the 'housing.csv' file into this folder.")
        return

    # 2. Convert CSV to SQL
    try:
        print(f"⏳ Reading {CSV_FILE}...")
        df = pd.read_csv(CSV_FILE)
        
        print(f"⏳ Creating {DB_FILE}...")
        conn = sqlite3.connect(DB_FILE)
        df.to_sql('housing_table', conn, if_exists='replace', index=False)
        conn.close()
        
        print(f"✅ SUCCESS! Database created with {len(df)} rows.")
        print("   You can now run 'housing_bot.py'.")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")

if __name__ == "__main__":
    create_database()
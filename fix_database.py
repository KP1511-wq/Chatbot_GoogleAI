import sqlite3
import pandas as pd
import os

# 1. Config
csv_file = "housing.csv"
db_file = "housing.db"

# 2. Check if CSV exists
if not os.path.exists(csv_file):
    print(f"❌ Error: Could not find '{csv_file}'. Make sure it is in this folder.")
else:
    # 3. Read CSV
    print(f"📖 Reading {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"   Found {len(df)} rows.")

    # 4. Save to SQLite
    conn = sqlite3.connect(db_file)
    df.to_sql("housing", conn, if_exists="replace", index=False)
    conn.close()
    
    print(f"✅ Success! Copied data from CSV -> '{db_file}' (Table: 'housing')")
    print("👉 You can now run the API request.")
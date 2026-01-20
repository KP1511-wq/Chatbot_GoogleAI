# setup_db.py
import pandas as pd
import sqlite3
import os

# ==========================================
# 1. FIND THE CSV FILE
# ==========================================
# We check multiple common names just in case
possible_files = ['Heart_Disease_Prediction.csv', 'heart.csv', 'heart_disease.csv']
csv_file = None

print("--- STEP 1: Looking for CSV file ---")
for f in possible_files:
    if os.path.exists(f):
        csv_file = f
        print(f"✅ Found CSV file: {csv_file}")
        break

if not csv_file:
    print("❌ ERROR: Could not find any CSV file!")
    print(f"   Please make sure one of these files is in this folder: {possible_files}")
    exit()

# ==========================================
# 2. CREATE THE DATABASE
# ==========================================
db_name = "heart.db"
print(f"\n--- STEP 2: Creating {db_name} ---")

try:
    # Load Data
    df = pd.read_csv(csv_file)
    print(f"   Loaded {len(df)} rows from CSV.")
    
    # Connect (Creates file if missing)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Write Data to Table 'heart_disease_info'
    table_name = "heart_disease_info"
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"✅ Data written to table '{table_name}'")
    
    # ==========================================
    # 3. VERIFY IT WORKED
    # ==========================================
    print(f"\n--- STEP 3: Verifying Database ---")
    
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    if not tables:
        print("❌ CRITICAL ERROR: Database created but NO tables found.")
    else:
        print(f"✅ Success! Tables found in DB: {tables}")
        
        # Check row count
        cursor.execute(f"SELECT count(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"📊 Total Rows: {count}")

    conn.close()

except Exception as e:
    print(f"❌ Error during setup: {e}")
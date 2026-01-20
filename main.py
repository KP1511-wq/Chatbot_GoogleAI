import os
from backend import get_sql_agent, get_joke_chain, get_router_chain
from tools_db import get_all_tables, get_columns, get_sample_rows

db_path = "heart.db"

# 1. Build Schema Context
print("--- LOADING SYSTEM ---")
if not os.path.exists(db_path):
    print(f"❌ Error: {db_path} missing.")
    exit()

tables = get_all_tables(db_path)
schema_info = []

for table in tables:
    cols = get_columns(db_path, table)
    samples = get_sample_rows(db_path, table, n=1)
    schema_info.append(f"Table: {table}\nColumns: {cols}\nSample Row: {samples}")

schema_context = "\n".join(schema_info)
print(f"✅ Schema Loaded: {tables}")

# 2. Init Agents
sql_agent = get_sql_agent(schema_context)
router = get_router_chain(schema_context)
joke_chain = get_joke_chain(schema_context)

# 3. Loop
while True:
    user_input = input("\nUser: ")
    if user_input.lower() == "exit": break
    
    try:
        intent = router.invoke({"question": user_input}).strip().upper()
        if "DATA" in intent:
            res = sql_agent.invoke({"input": user_input})
            print(f"🤖 Bot: {res['output']}")
        else:
            res = joke_chain.invoke({"question": user_input})
            print(f"🤖 Bot: {res}")
    except Exception as e:
        print(f"Error: {e}")
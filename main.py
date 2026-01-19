# main.py
from app import load_db, get_sql_agent, get_joke_chain, get_router_chain

# 1. Setup Data (Use the .db file now)
db_path = 'heart.db'
db = load_db(db_path)

# Get table names for the router's context
table_info = db.get_table_info()

# 2. Initialize Brains
sql_agent = get_sql_agent(db)
joke_chain = get_joke_chain(table_info)
router = get_router_chain(table_info)

# 3. Run Loop
def chat_loop():
    print(f"\n🤖 SYSTEM: Connected to Database. Tables found: {table_info}")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        try:
            # Check Intent
            intent = router.invoke({"question": user_input}).strip().upper()
            
            if "DATA" in intent:
                print(f"DEBUG: Routing to SQL Agent...")
                # The SQL agent takes the input directly
                response = sql_agent.invoke({"input": user_input})
                print(f"🤖 Bot: {response['output']}")
            else:
                print(f"DEBUG: Routing to Joke Generator...")
                response = joke_chain.invoke({"question": user_input})
                print(f"🤖 Bot: {response}")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_loop()
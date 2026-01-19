from app import load_dataset, get_data_agent, get_joke_chain, get_router_chain

# 1. Setup Data
csv_path = 'Heart_Disease_Prediction.csv'
df = load_dataset(csv_path)
column_names = ", ".join(df.columns.tolist())

# 2. Initialize Brains
data_agent = get_data_agent(df)
joke_chain = get_joke_chain(column_names)
router = get_router_chain(column_names)

# 3. Run Loop
def chat_loop():
    print(f"\n🤖 SYSTEM: Ready. Analyzing columns: {column_names}")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        try:
            # Check Intent (Only needs question now)
            intent = router.invoke({"question": user_input}).strip().upper()
            
            if "DATA" in intent:
                print(f"DEBUG: Routing to Data Agent...")
                response = data_agent.invoke({"input": user_input})
                print(f"🤖 Bot: {response['output']}")
            else:
                print(f"DEBUG: Routing to Joke Generator...")
                # Only needs question now, columns are baked in
                response = joke_chain.invoke({"question": user_input})
                print(f"🤖 Bot: {response}")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_loop()
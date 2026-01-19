
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import llm

def load_dataset(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(f"--- Loaded {csv_path} Successfully ---")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find file at {csv_path}.")
        exit()

def get_data_agent(df):
    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )

def get_joke_chain(column_names):
    joke_template = """
    The user asked a question that is NOT related to the dataset.
    User Question: {question}

    Your Task:
    1. Make a witty joke about the user's random topic.
    2. Then, firmly but politely tell them: "I only answer questions about the {columns} dataset."
    """
    
    # FIX: Use .from_template() to avoid variable conflicts
    joke_prompt = PromptTemplate.from_template(
        joke_template, 
        partial_variables={"columns": column_names}
    )
    
    return joke_prompt | llm | StrOutputParser()

def get_router_chain(column_names):
    router_template = """
    You are a classifier. You must decide if a user's question is related to a dataframe 
    with the following columns: [{columns}].

    If the question requires reading, calculating, or analyzing this data, return "DATA".
    If the question is unrelated (e.g., general knowledge, weather, greetings, sports), return "JOKE".

    Question: {question}
    Classification:
    """
    
    # FIX: Use .from_template() here as well
    router_prompt = PromptTemplate.from_template(
        router_template,
        partial_variables={"columns": column_names}
    )
    
    return router_prompt | llm | StrOutputParser()

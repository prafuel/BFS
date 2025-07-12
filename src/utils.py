import pandas as pd
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


# ============= LOAD GROQ MODEL ================================
def load_groq(model_name: str = "mistral-saba-24b", **kwargs):
    llm = ChatGroq(
        model=model_name,
        **kwargs
    )
    return llm

# CSV FILE AND MODEL LOADING
df = pd.read_csv("data/BFS_Share_Price.csv")
llm = load_groq("llama-3.1-8b-instant")

def load_vector_store(documents, embeddings):
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def pandas_agent(query : str):
    try: 
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            agent_type="openai-tools",
            allow_dangerous_code=True,
            return_intermediate_steps=True,
            # max_iterations=5
        )
        response = agent.invoke(query)['output']
        return None if len(response.strip()) == 0 else response
    
    except Exception as e:
        return f"Error : {e}"
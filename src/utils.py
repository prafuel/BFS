# src/utils.py
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.config.config import DF_PATH
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# Load GROQ model
def load_groq(model_name: str = "mistral-saba-24b", **kwargs):
    llm = ChatGroq(
        model=model_name,
        **kwargs
    )
    return llm

# CSV file and model loading
df = pd.read_csv(DF_PATH)
llm = load_groq("llama-3.1-8b-instant")

def load_vector_store(documents, embeddings):
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def pandas_agent(query: str):
    try: 
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            agent_type="openai-tools",
            allow_dangerous_code=True,
            return_intermediate_steps=True,
        )
        response = agent.invoke(query)['output']
        return None if len(response.strip()) == 0 else response
    
    except Exception as e:
        return f"Error : {e}"

# RAG components
def load_documents(directory="data"):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf") and "Earnings Call Transcript" in filename:
            filepath = os.path.join(directory, filename)
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
    return documents

def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def create_vector_store(chunks, embeddings):
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def get_rag_response(query: str, vector_store, llm):
    retriever = vector_store.as_retriever()
    docs = retriever.invoke(query)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
    You are an AI assistant for Bajaj Finserv. Use the following context and the provided tools to answer the user's question.
    If the question is about historical stock prices (highest, lowest, average, comparison), use the pandas_agent tool.
    If the question is about company information from the earnings call transcripts, use the provided context.
    If the question is about drafting a commentary as a CFO, use your general knowledge and the provided context.

    Context from earnings call transcripts:
    {context}

    User question: {query}

    Provide a comprehensive answer based on the available information.
    """
    
    response = llm.invoke(prompt)
    return response.content

def initialize_rag():
    # Initialize RAG components
    documents = load_documents()
    chunks = chunk_documents(documents)
    embeddings = get_embeddings()
    vector_store = create_vector_store(chunks, embeddings)

    return vector_store

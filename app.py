# app.py
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from src.utils import pandas_agent, load_groq, vector_store, get_rag_response

load_dotenv()

llm = load_groq("llama-3.1-8b-instant")

st.set_page_config(page_title="Bajaj Finserv Chatbot", layout="wide")

st.title("📈 Bajaj Finserv Investor Chatbot")
st.markdown("""
Welcome! I can help you with questions about Bajaj Finserv's stock prices and company performance based on their earnings call transcripts.
Try asking:
- "What was the highest/average/lowest stock price across Jan-22 to Apr-22?"
- "Compare Bajaj Finserv stock price from Jan-23 to Mar-23 with Apr-23 to Jun-23."
- "Tell me something on organic traffic of Bajaj Markets."
- "Why is BAGIC facing headwinds in Motor insurance business?"
- "What's the rationale of Hero partnership?"
- "Give me a table with dates explaining discussions regarding Allianz stake sale."
- "Act as a CFO of BAGIC and help me draft commentary for upcoming investor call."
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me anything about Bajaj Finserv..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ""
            # Check if the query is for the pandas agent
            stock_price_keywords = ["highest", "average", "lowest", "compare", "stock price", "share price"]
            if any(keyword in prompt.lower() for keyword in stock_price_keywords):
                response = pandas_agent(prompt)
                if response is None:
                    response = "I couldn't find a specific answer for that stock price query."
            elif "allianz stake sale" in prompt.lower():
                allianz_info = get_rag_response(prompt, vector_store, llm)
                
                matches = re.findall(r'(\w{3} \d{1,2}, \d{4}):\s*(.*?)(?=\w{3} \d{1,2}, \d{4}:|\Z)', allianz_info, re.DOTALL)
                
                if matches:
                    table_data = []
                    for date_str, discussion in matches:
                        table_data.append({"Date": date_str.strip(), "Discussion": discussion.strip()})
                    
                    if table_data:
                        df_allianz = pd.DataFrame(table_data)
                        response = "Here's a summary of discussions regarding Allianz stake sale:\n\n" + df_allianz.to_markdown(index=False)
                    else:
                        response = "I found information about Allianz stake sale, but couldn't format it into a table. Here's the raw info:\n" + allianz_info
                else:
                    response = allianz_info
            elif "cfo of bagic" in prompt.lower() or "draft commentary" in prompt.lower():
                cfo_prompt = f"""
                You are acting as the CFO of BAGIC. Draft a commentary for an upcoming investor call based on the provided context from recent earnings call transcripts.
                Focus on key financial highlights, growth drivers, challenges, and future outlook for BAGIC.
                Ensure the tone is professional and addresses common investor concerns.

                Context from earnings call transcripts:
                {get_rag_response("Summarize BAGIC's performance, growth, profitability, challenges, and outlook from recent earnings calls.", vector_store, llm)}

                Draft the commentary:
                """
                response = llm.invoke(cfo_prompt).content
            else:
                response = get_rag_response(prompt, vector_store, llm)
            
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

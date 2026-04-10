import streamlit as st
import requests
import uuid
import os

# Set page config
st.set_page_config(page_title="Enterprise Routing Agent", page_icon="🤖")
st.title("🤖 Llama 3.1 Routing Agent")
st.caption("Powered by LangGraph, Weaviate RAG, and Tavily Web Search")

# API Endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000/chat")

# Initialize chat history and a unique thread ID for LangGraph memory
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4()) # Creates a unique session ID

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask about the portfolio, web, or just say hi..."):
    # 1. Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Call the FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking (and routing)..."):
            try:
                # Notice we are now passing the thread_id for memory!
                payload = {
                    "query": prompt,
                    "thread_id": st.session_state.thread_id
                }
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]  # We only grab the answer now
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Backend API Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend. Is FastAPI running on port 8000?")
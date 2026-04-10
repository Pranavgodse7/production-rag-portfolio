import os
import torch
import weaviate
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from weaviate.classes.init import Auth

# --- MODERNIZED IMPORTS ---
from langchain_huggingface import HuggingFaceEmbeddings # Fixes deprecation warning
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_groq import ChatGroq

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever

# --- NEW LANGGRAPH & TOOL IMPORTS ---
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict

load_dotenv()

# Global variables to hold our graph and database client
app_graph = None
weaviate_client = None

# 1. Define the Graph State for Memory
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Pydantic models for structured API requests/responses
class QueryRequest(BaseModel):
    query: str
    thread_id: str = "default_user"  # Added thread_id for memory tracking

class QueryResponse(BaseModel):
    answer: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC: Load models and connect to DB ---
    global app_graph, weaviate_client
    print("Initializing Enterprise LangGraph Routing Backend...")
    
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )

    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key)
    )

    vector_store = WeaviateVectorStore(
        client=weaviate_client,
        index_name="Portfolio_RAG_Docs_v2",
        text_key="text",
        embedding=embedding_model
    )

    cross_encoder = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-base", 
        model_kwargs={'device': device}
    )
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)
    
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10, "alpha": 0.5})
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )

    # --- LANGGRAPH SETUP ---
    
    # 1. Define Tool 1: Your Internal RAG
    weaviate_tool = create_retriever_tool(
        compression_retriever,
        "search_portfolio_docs",
        "Use this tool FIRST to search for technical documents, project details, or specific portfolio information."
    )

    # 2. Define Tool 2: The Web Search
    web_search_tool = TavilySearch(
        max_results=2,
        description="Use this tool to search the internet for current events, weather, or general knowledge NOT found in the portfolio."
    )

    # 3. Give the LLM its Toolbelt
    tools = [weaviate_tool, web_search_tool]
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)

    # 4. The Agent Node
    def chatbot(state: State):
        sys_prompt = SystemMessage(content=(
            "You are a brilliant AI routing assistant. "
            "1. If the user says hello or asks a casual question, just chat normally (DO NOT call tools). "
            "2. If the user asks about technical portfolio documents, call the 'search_portfolio_docs' tool. "
            "3. If the user asks about current events or general knowledge, call the 'tavily_search_results_json' tool."
        ))
        messages_to_pass = [sys_prompt] + state["messages"]
        
        response = llm_with_tools.invoke(messages_to_pass)
        return {"messages": [response]}

    # 5. Build the Routing Graph
    graph_builder = StateGraph(State)
    
    graph_builder.add_node("agent", chatbot)
    graph_builder.add_node("tools", ToolNode(tools=tools)) 
    
    graph_builder.add_edge(START, "agent")
    
    # The tools_condition automatically checks if the LLM called a tool
    graph_builder.add_conditional_edges(
        "agent",
        tools_condition, 
    )
    
    # After a tool runs, it ALWAYS loops back to the agent to read the results
    graph_builder.add_edge("tools", "agent")

    # Compile with MemorySaver
    memory = MemorySaver()
    app_graph = graph_builder.compile(checkpointer=memory)
    
    print("✅ LangGraph Routing API Ready!")
    yield 
    
    # --- SHUTDOWN LOGIC: Clean up connections ---
    if weaviate_client:
        weaviate_client.close()
        print("Weaviate connection closed.")

# Initialize FastAPI app
app = FastAPI(title="Portfolio RAG API", lifespan=lifespan)

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not app_graph:
        raise HTTPException(status_code=503, detail="Graph not initialized yet.")
    
    try:
        # Pass the thread_id to the config so MemorySaver knows whose chat this is
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # Invoke the graph
        events = app_graph.invoke(
            {"messages": [("user", request.query)]}, 
            config=config
        )
        
        # The final answer is the very last message in the state
        final_message = events["messages"][-1].content
        
        return QueryResponse(answer=final_message)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
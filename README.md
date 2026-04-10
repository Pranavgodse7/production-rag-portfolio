# 🚀 Enterprise Multi-Agent RAG System

A production-ready Retrieval-Augmented Generation (RAG) microservice architecture. This project features a dynamic LangGraph routing agent capable of intelligently deciding between searching an internal Weaviate vector database, conducting live web searches via Tavily, or engaging in standard conversation.

## 🧠 Architecture Overview

This project is decoupled into a high-performance **FastAPI backend** and an interactive **Streamlit frontend**, fully containerized via **Docker**.

1. **The Orchestrator:** LangGraph state machine with memory tracking (`MemorySaver`).
2. **The Brain:** Llama-3.1-8b (via Groq) acting as a Tool-Calling Agent.
3. **Internal Memory (RAG):** Weaviate Cloud vector database using a Two-Stage Retrieval pipeline:
   - *Stage 1:* Hybrid Search (BM25 + Semantic via `all-MiniLM-L6-v2`).
   - *Stage 2:* Cross-Encoder Re-ranking (`BAAI/bge-reranker-base`) to filter hallucinations.
4. **External Memory (Web):** Tavily API for real-time web search grounding.
5. **Observability:** Full tracing of latency, token usage, and graph execution via LangSmith.

## ✨ Key Features
- **Dynamic Tool Routing:** The LLM autonomously chooses tools based on user intent.
- **Cross-Encoder Re-ranking:** Drastically improves accuracy by mathematically re-scoring chunk relevance before generation.
- **Stateful Memory:** Maintains conversational context across multi-turn interactions using Thread IDs.
- **Dockerized Microservices:** `docker-compose` setup for instant, reproducible deployments.

## 🛠️ Tech Stack
- **Frameworks:** LangChain, LangGraph, FastAPI, Streamlit
- **Models:** Llama 3.1 (LLM), HuggingFace (Embeddings & Re-ranking)
- **Database:** Weaviate Cloud (WCD)
- **APIs:** Groq (Inference), Tavily (Search), LangSmith (Observability)

## 🚦 Quickstart (Docker)

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/pranavgodse7/production-rag-portfolio.git](https://github.com/pranavgodse7/production-rag-portfolio.git)
   cd production-rag-portfolio
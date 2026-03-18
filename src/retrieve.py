import os
import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
CHROMA_PATH = "./chroma_db"

def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not found. Please add it to your .env file.")
        return

    if not os.path.exists(CHROMA_PATH):
        print(f"Error: Database not found at '{CHROMA_PATH}'. Please run ingest.py first.")
        return

    # --- DEVICE CHECK ADDED HERE ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing embedding model on: {device.upper()}")

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )

    print("Connecting to local ChromaDB...")
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
        collection_name="portfolio_rag_phase1"
    )

    print("Initializing Groq LLM...")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,        
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Keep the answer clear and concise."
        "\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n✅ RAG System Ready!")
    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ")
        if query.lower() in ['exit', 'quit']:
            break

        print("Thinking...")
        response = rag_chain.invoke({"input": query})

        print("\n--- Answer ---")
        print(response["answer"])
        
        print("\n--- Sources Used ---")
        for doc in response["context"]:
            print(f"- {doc.metadata.get('source', 'Unknown Document')}")
        print("-" * 40)

if __name__ == "__main__":
    main()
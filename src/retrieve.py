import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_PATH = "./chroma_db"

def main():
    # 1. Check if the database exists
    if not os.path.exists(CHROMA_PATH):
        print(f"Error: Database not found at '{CHROMA_PATH}'. Please run ingest.py first.")
        return

    # 2. Initialize the SAME embedding model used for ingestion
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Connect to the existing local Chroma database
    print("Connecting to local ChromaDB...")
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
        collection_name="portfolio_rag_phase1"
    )

    # 4. Define your query
    query = input("\nEnter your search query: ")

    # 5. Perform the semantic search
    # k=3 means it will return the top 3 most relevant chunks
    print("\nSearching...")
    results = vector_store.similarity_search_with_score(query, k=3)

    if not results:
        print("No matching results found.")
        return

    # 6. Display the results
    print("\n--- Top Results ---")
    for i, (doc, score) in enumerate(results):
        # A lower score in ChromaDB generally indicates closer distance/higher similarity
        print(f"\nResult {i+1} (Distance Score: {score:.4f}):")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content:\n{doc.page_content}")
        print("-" * 40)

if __name__ == "__main__":
    main()
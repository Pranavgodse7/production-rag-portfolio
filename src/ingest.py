import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define paths relative to the root directory
DATA_DIR = "./data"
CHROMA_PATH = "./chroma_db"

def main():
    # 1. Check if data directory exists and has files
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"Error: Please add some .pdf files to the '{DATA_DIR}' directory.")
        return

    print(f"Loading PDF documents from {DATA_DIR}...")
    # UPDATED: Now uses PyPDFLoader and looks specifically for .pdf files
    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    # 2. Chunk the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    # 3. Initialize the embedding model
    print("Initializing embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Create and persist the Chroma database
    print("Embedding chunks and saving to ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH,
        collection_name="portfolio_rag_phase1"
    )
    
    print(f"✅ Successfully ingested PDFs and saved vector database to '{CHROMA_PATH}'")

if __name__ == "__main__":
    main()
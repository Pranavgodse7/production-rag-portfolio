import os
import torch
import weaviate
from dotenv import load_dotenv
from weaviate.classes.init import Auth
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

DATA_DIR = "./data"


def main():
    # 1. Grab credentials from .env
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    if not weaviate_url or not weaviate_api_key:
        print("Error: Weaviate credentials missing in .env file.")
        return

    # 2. Load and Chunk PDFs
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"Error: Please add some .pdf files to the '{DATA_DIR}' directory.")
        return

    print(f"Loading PDF documents from {DATA_DIR}...")
    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # --- UPDATED: LARGER CHUNK SIZE & OVERLAP ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    # 3. Metadata Cleaning Step
    print("Cleaning metadata for Weaviate compatibility...")
    for chunk in chunks:
        cleaned_metadata = {}
        for key, value in chunk.metadata.items():
            clean_key = key.replace('.', '_').replace('-', '_')
            cleaned_metadata[clean_key] = value
        chunk.metadata = cleaned_metadata

    # 4. Initialize Embedding Model (GPU Accelerated)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing embedding model on: {device.upper()}")
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )

    # 5. Connect to Weaviate Cloud
    print("Connecting to Weaviate Cloud...")
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key)
    )

    try:
        # 6. Push data to Weaviate
        print("Uploading chunks and embeddings to Weaviate...")
        # --- UPDATED: NEW INDEX NAME ---
        index_name = "Portfolio_RAG_Docs_v2"

        vector_store = WeaviateVectorStore(
            client=weaviate_client,
            index_name=index_name,
            text_key="text",
            embedding=embedding_model
        )

        vector_store.add_documents(chunks)
        print(f"✅ Successfully ingested {len(chunks)} chunks into Weaviate Cloud (v2)!")

    finally:
        weaviate_client.close()


if __name__ == "__main__":
    main()
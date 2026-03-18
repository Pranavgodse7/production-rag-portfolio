import os
import torch
import weaviate
from dotenv import load_dotenv
from weaviate.classes.init import Auth
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_groq import ChatGroq

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever

load_dotenv()


def main():
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not all([weaviate_url, weaviate_api_key, groq_api_key]):
        print("Error: Missing API keys in .env file.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )

    print("Connecting to Weaviate Cloud...")
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key)
    )

    try:
        # --- UPDATED: CONNECT TO NEW INDEX ---
        index_name = "Portfolio_RAG_Docs_v2"
        vector_store = WeaviateVectorStore(
            client=weaviate_client,
            index_name=index_name,
            text_key="text",
            embedding=embedding_model
        )

        # --- UPDATED: NEW LLAMA 3.1 MODEL ---
        print("Initializing Groq LLM...")
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.0,
        )

        print(f"Loading Cross-Encoder Re-ranker on: {device.upper()}...")
        cross_encoder = HuggingFaceCrossEncoder(
            model_name="BAAI/bge-reranker-base",
            model_kwargs={'device': device}
        )

        # --- UPDATED: KEEP TOP 5 ---
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)

        # --- UPDATED: FETCH TOP 10 ---
        base_retriever = vector_store.as_retriever(
            search_kwargs={"k": 10, "alpha": 0.5}
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

        system_prompt = (
            "You are an AI engineering assistant. "
            "Use the following pieces of retrieved context to answer the question. "
            "If the answer is not in the context, say 'I don't know based on the provided documents.' "
            "Keep the answer highly technical, clear, and concise."
            "\n\n"
            "Context: {context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(compression_retriever, question_answer_chain)

        print("\n✅ Advanced RAG System Ready! (Llama 3.1 + Top 5 Re-ranking)")

        while True:
            query = input("\nEnter your question (or type 'exit' to quit): ")
            if query.lower() in ['exit', 'quit']:
                break

            print("Retrieving, re-ranking, and thinking...")
            response = rag_chain.invoke({"input": query})

            print("\n--- Answer ---")
            print(response["answer"])

            print("\n--- Sources Used (Top 5 Re-ranked) ---")
            for doc in response["context"]:
                source = doc.metadata.get('source', 'Unknown Document')
                print(f"- {source}")
            print("-" * 50)

    finally:
        weaviate_client.close()


if __name__ == "__main__":
    main()
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

load_dotenv()


def main():
    # 1. Grab credentials
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not all([weaviate_url, weaviate_api_key, groq_api_key]):
        print("Error: Missing API keys in .env file.")
        return

    # 2. Initialize Embeddings (GPU Accelerated)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )

    # 3. Connect to Weaviate Cloud
    print("Connecting to Weaviate Cloud...")
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key)
    )

    try:
        index_name = "Portfolio_RAG_Docs"
        vector_store = WeaviateVectorStore(
            client=weaviate_client,
            index_name=index_name,
            text_key="text",
            embedding=embedding_model
        )

        # 4. Initialize Groq LLM
        print("Initializing Groq LLM...")
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.0,
        )

        # 5. Set up the Prompt
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

        # 6. Build the HYBRID Retrieval Chain
        # alpha=0.5 means a 50/50 split between keyword (BM25) and semantic (Vector) search
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3, "alpha": 0.5}
        )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        print("\n✅ Hybrid RAG System Ready! (Powered by Weaviate & Groq)")

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
                # Print the source file name (handling the cleaned metadata keys if necessary)
                source = doc.metadata.get('source', 'Unknown Document')
                print(f"- {source}")
            print("-" * 50)

    finally:
        # Always close the connection when exiting the script
        weaviate_client.close()


if __name__ == "__main__":
    main()
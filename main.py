import logging
import os
from typing import Optional, Tuple
import dagshub
import mlflow
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from src.helper import ask_and_speak, setup_hybrid_rag_chain, get_audio_input

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

LLM_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


def main() -> None:
    api_keys = ["HUGGINGFACEHUB_API_TOKEN", "TAVILY_API_KEY", "MLFLOW_TRACKING_URI"]
    if not all(os.getenv(key) for key in api_keys):
        logging.error("One or more required API keys/URIs are not set in the .env file.")
        return

    dagshub.init(repo_owner='satyajeetrai007', repo_name='Buildathon', mlflow=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    llm_endpoint = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.0,
    )
    chat_model = ChatHuggingFace(llm=llm_endpoint)


    search_tool = TavilySearchResults(max_results=3)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.load_local(
        "agri_faiss_index",  
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    rag_chain = setup_hybrid_rag_chain(chat_model, search_tool, vectorstore_retriever=retriever)
    logging.info("Hybrid RAG chain (DB + Web) is ready for farmer queries.")

    audio_input = get_audio_input()

    if audio_input:
        user_query, audio_path = audio_input
        ask_and_speak(user_query, audio_path, rag_chain)
    else:
        logging.error("Could not get a valid query from the microphone. Exiting.")


if __name__ == "__main__":
    main()

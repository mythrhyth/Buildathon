import logging
import os
import textwrap
from typing import Optional, Tuple
import dagshub
import mlflow
import speech_recognition as sr
import torch
from dotenv import load_dotenv
from gtts import gTTS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable, RunnablePassthrough
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langdetect import detect, LangDetectException
import platform
from mlflow.utils.git_utils import get_git_commit, get_git_branch
from src.helper import ask_and_speak, setup_web_rag_chain, log_environment, get_audio_input

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

    rag_chain = setup_web_rag_chain(chat_model, search_tool)
    logging.info("Web-searching RAG chain is ready for questions.")

    audio_input = get_audio_input()

    if audio_input:
        user_query, audio_path = audio_input
        ask_and_speak(user_query, audio_path, rag_chain)
    else:
        logging.error("Could not get a valid query from the microphone. Exiting.")

if __name__ == "__main__":
    main()

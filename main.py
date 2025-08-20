import logging
import os
import textwrap
from typing import List, Optional

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

# --- 1. CONFIGURATION & INITIALIZATION ---

# Configure logging to provide informative output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# --- Constants ---
LLM_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# --- 2. WEB SEARCH RAG PIPELINE SETUP ---

def setup_web_rag_chain(llm: ChatHuggingFace, search_tool: TavilySearchResults) -> Runnable:
    """
    Sets up a LangChain RAG chain that uses web search as its retriever.

    Args:
        llm: The language model component of the chain.
        search_tool: The Tavily search tool.

    Returns:
        A runnable LangChain RAG chain.
    """
    template = """
    You are a helpful assistant. Answer the user's question based on the following web search results.
    Provide a concise, synthesized answer in the same language as the question.

    Context from web search:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    # This chain will first run the search tool, then pass the results and the original
    # question to the prompt and the LLM.
    return (
        {"context": search_tool, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

# --- 3. CORE APPLICATION LOGIC ---

def get_audio_input() -> Optional[str]:
    """
    Captures audio from the microphone and converts it to text.

    Returns:
        The recognized text as a string, or None if an error occurs.
    """
    recognizer = sr.Recognizer()
    # Increase the pause_threshold to wait 2 seconds after speech ends
    # before considering the phrase complete. The default is 0.8.
    recognizer.pause_threshold = 2.0

    with sr.Microphone() as source:
        logging.info("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        logging.info("Listening... Please ask your question.")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15) # Increased time limit
            logging.info("Recognizing speech...")
            # Use Google's free web speech API
            query = recognizer.recognize_google(audio, language='hi-IN') # Prioritize Hindi
            logging.info(f"User said: {query}")
            return query
        except sr.WaitTimeoutError:
            logging.error("Listening timed out while waiting for phrase to start.")
        except sr.UnknownValueError:
            logging.error("Google Speech Recognition could not understand audio.")
        except sr.RequestError as e:
            logging.error(f"Could not request results from Google Speech Recognition service; {e}")
    return None


def ask_and_speak(query: str, rag_chain: Runnable) -> None:
    """
    Takes a user query, gets an answer from the RAG chain,
    and synthesizes it to speech using gTTS.

    Args:
        query: The user's question.
        rag_chain: The RAG chain to use for generating the answer.
    """
    # Start an MLflow run to log this interaction
    with mlflow.start_run():
        logging.info(f"Received query: '{query}'")
        mlflow.log_param("user_query", query)

        try:
            detected_lang = detect(query)
            logging.info(f"Detected language: {detected_lang}")
            mlflow.log_param("detected_language", detected_lang)
        except LangDetectException as e:
            logging.warning(f"Could not detect language. Defaulting to English. Error: {e}")
            detected_lang = 'en'
            mlflow.log_param("detected_language", "en (defaulted)")

        logging.info("Searching the web and generating answer...")
        ai_message_response = rag_chain.invoke(query)
        text_answer = ai_message_response.content
        
        if not text_answer:
            logging.warning("LLM returned an empty answer.")
            mlflow.log_param("status", "failed_empty_response")
            return

        # Wrap the text for cleaner printing in the terminal
        wrapped_text = textwrap.fill(text_answer, width=100)
        logging.info(f"Generated Text Answer:\n{wrapped_text}")
        # Log the full answer as a text artifact in MLflow
        mlflow.log_text(text_answer, "llm_response.txt")

        logging.info(f"Synthesizing speech in '{detected_lang}' with gTTS...")
        try:
            tts_obj = gTTS(text=text_answer, lang=detected_lang)
            audio_file_path = "answer.mp3"
            tts_obj.save(audio_file_path)
            logging.info(f"SUCCESS! Audio answer saved to {audio_file_path}")
            # Log the generated audio file as an artifact
            mlflow.log_artifact(audio_file_path)
            mlflow.log_param("status", "success")
            
        except Exception as e:
            logging.error(f"An error occurred during speech synthesis: {e}")
            mlflow.log_param("status", "failed_tts")

# --- 4. MAIN EXECUTION BLOCK ---

def main() -> None:
    """
    Main function to set up and run the web-searching RAG application.
    """
    # Check for API keys
    api_keys = ["HUGGINGFACEHUB_API_TOKEN", "TAVILY_API_KEY", "MLFLOW_TRACKING_URI"]
    if not all(os.getenv(key) for key in api_keys):
        logging.error("One or more required API keys/URIs are not set in the .env file.")
        return

    # Initialize DagsHub and MLflow
    dagshub.init(repo_owner='YourUsername', repo_name='YourRepoName', mlflow=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    # Initialize the LLM
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.7,
    )
    chat_model = ChatHuggingFace(llm=llm_endpoint)

    # Setup the web search tool
    search_tool = TavilySearchResults(max_results=3)

    # Setup the RAG chain
    rag_chain = setup_web_rag_chain(chat_model, search_tool)
    logging.info("Web-searching RAG chain is ready for questions.")

    # Get user input from the microphone
    user_query = get_audio_input()

    # If we got a query, process it
    if user_query:
        ask_and_speak(user_query, rag_chain)
    else:
        logging.error("Could not get a valid query from the microphone. Exiting.")

if __name__ == "__main__":
    main()

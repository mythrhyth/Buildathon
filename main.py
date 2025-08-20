import logging
import os
from typing import List, Optional

import speech_recognition as sr
import torch
from dotenv import load_dotenv
from gtts import gTTS
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
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

# --- 2. AGENT AND TOOLS SETUP ---

def setup_agent_executor(llm: ChatHuggingFace) -> AgentExecutor:
    """
    Sets up a LangChain agent with a web search tool.

    Args:
        llm: The language model to power the agent.

    Returns:
        An initialized AgentExecutor.
    """
    # Define the tool(s) the agent can use. Here, we only use Tavily Search.
    tools = [TavilySearchResults(max_results=3)]

    # Get the prompt template for the ReAct agent from LangChain Hub
    prompt = hub.pull("hwchase17/react")

    # Create the agent
    agent = create_react_agent(llm, tools, prompt)

    # Create the agent executor, which runs the agent's reasoning loop
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 3. CORE APPLICATION LOGIC ---

def get_audio_input() -> Optional[str]:
    """
    Captures audio from the microphone and converts it to text.

    Returns:
        The recognized text as a string, or None if an error occurs.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        logging.info("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        logging.info("Listening... Please ask your question.")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            logging.info("Recognizing speech...")
            # Use Google's free web speech API
            query = recognizer.recognize_google(audio, language='en-IN')
            logging.info(f"User said: {query}")
            return query
        except sr.WaitTimeoutError:
            logging.error("Listening timed out while waiting for phrase to start.")
        except sr.UnknownValueError:
            logging.error("Google Speech Recognition could not understand audio.")
        except sr.RequestError as e:
            logging.error(f"Could not request results from Google Speech Recognition service; {e}")
    return None


def ask_and_speak(query: str, agent_executor: AgentExecutor) -> None:
    """
    Takes a user query, gets an answer from the agent,
    and synthesizes it to speech using gTTS.

    Args:
        query: The user's question.
        agent_executor: The agent executor to run.
    """
    logging.info(f"Received query: '{query}'")

    try:
        detected_lang = detect(query)
        logging.info(f"Detected language: {detected_lang}")
    except LangDetectException as e:
        logging.warning(f"Could not detect language. Defaulting to English. Error: {e}")
        detected_lang = 'en'

    logging.info("Agent is thinking and searching the web...")
    # The agent executor takes a dictionary as input
    response = agent_executor.invoke({"input": query})
    text_answer = response.get("output")
    
    if not text_answer:
        logging.warning("Agent returned an empty answer.")
        return

    logging.info(f"Generated Text Answer: {text_answer}")
    logging.info(f"Synthesizing speech in '{detected_lang}' with gTTS...")
    try:
        tts_obj = gTTS(text=text_answer, lang=detected_lang)
        audio_file_path = "answer.mp3"
        tts_obj.save(audio_file_path)
        logging.info(f"SUCCESS! Audio answer saved to {audio_file_path}")
        
    except Exception as e:
        logging.error(f"An error occurred during speech synthesis: {e}")

# --- 4. MAIN EXECUTION BLOCK ---

def main() -> None:
    """
    Main function to set up and run the web-searching agent.
    """
    # Check for API keys
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN") or not os.getenv("TAVILY_API_KEY"):
        logging.error("HUGGINGFACEHUB_API_TOKEN or TAVILY_API_KEY not found in .env file.")
        return

    # Initialize the LLM
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.7,
    )
    chat_model = ChatHuggingFace(llm=llm_endpoint)

    # Setup the agent executor
    agent_executor = setup_agent_executor(chat_model)
    logging.info("Web-searching agent is ready for questions.")

    # Get user input from the microphone
    user_query = get_audio_input()

    # If we got a query, process it
    if user_query:
        ask_and_speak(user_query, agent_executor)
    else:
        logging.error("Could not get a valid query from the microphone. Exiting.")

if __name__ == "__main__":
    main()

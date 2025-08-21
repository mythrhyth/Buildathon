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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

LLM_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

def setup_web_rag_chain(llm: ChatHuggingFace, search_tool: TavilySearchResults) -> Runnable:
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

    return (
        {"context": search_tool, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

def log_environment():
    mlflow.log_param("python_version", platform.python_version())
    mlflow.log_param("torch_version", torch.__version__)
    mlflow.log_param("cuda_available", torch.cuda.is_available())
    mlflow.log_param("system", platform.system())
    mlflow.log_param("machine", platform.machine())

    try:
        mlflow.log_param("git_commit", get_git_commit("."))
        mlflow.log_param("git_branch", get_git_branch("."))
    except Exception as e:
        mlflow.log_param("git_info_error", str(e))

    if os.path.exists("requirements.txt"):
        mlflow.log_artifact("requirements.txt")

def get_audio_input() -> Optional[Tuple[str, str]]:
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 2.0
    audio_file_path = "user_audio.wav"

    with sr.Microphone() as source:
        logging.info("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        logging.info("Listening... Please ask your question.")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
            with open(audio_file_path, "wb") as f:
                f.write(audio.get_wav_data())

            logging.info("Recognizing speech...")
            query = recognizer.recognize_google(audio)
            logging.info(f"User said: {query}")
            return query, audio_file_path
        except Exception as e:
            logging.error(f"Audio input failed: {e}")
    return None

def ask_and_speak(query: str, audio_path: str, rag_chain: Runnable) -> None:
    with mlflow.start_run():
        log_environment()
        
        logging.info(f"Received query: '{query}'")
        mlflow.log_param("user_query", query)
        mlflow.log_param("llm_repo_id", LLM_REPO_ID)
        mlflow.log_artifact(audio_path, "input_audio")
        mlflow.log_artifact(__file__, "code")

        try:
            detected_lang = detect(query)
            mlflow.log_param("detected_language", detected_lang)
        except LangDetectException as e:
            detected_lang = 'en'
            mlflow.log_param("detected_language", "default_en")
            logging.warning(f"Language detection failed: {e}")

        logging.info("Running RAG chain...")
        try:
            ai_message_response = rag_chain.invoke(query)
            text_answer = ai_message_response.content
            with open("final_output.txt", "w", encoding="utf-8") as f:
                f.write(text_answer)
            mlflow.log_artifact("final_output.txt", "final_output_text")
        except Exception as e:
            mlflow.log_param("rag_error", str(e))
            logging.error(f"RAG failed: {e}")
            return

        if not text_answer:
            mlflow.log_param("status", "failed_empty_response")
            return
        
        wrapped_text = textwrap.fill(text_answer, width=100)
        logging.info(f"Generated Text Answer:\n{wrapped_text}")

        logging.info("Synthesizing with gTTS...")
        try:
            tts_obj = gTTS(text=text_answer, lang=detected_lang)
            output_audio_path = "answer.mp3"
            tts_obj.save(output_audio_path)
            mlflow.log_artifact(output_audio_path, "final_output_audio")
            mlflow.log_param("status", "success")
            logging.info(f"SUCCESS! Audio answer saved to {output_audio_path}")
        except Exception as e:
            mlflow.log_param("tts_error", str(e))
            logging.error(f"TTS failed: {e}")

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
        temperature=0.7,
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

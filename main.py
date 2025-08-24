import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from src.helper import ask_and_speak, setup_hybrid_rag_chain, get_audio_input

load_dotenv()

LLM_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

def main() -> None:
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        task="text-generation",
        max_new_tokens=1024,
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

    audio_input = get_audio_input()

    if audio_input:
        user_query, audio_path = audio_input
        ask_and_speak(user_query, audio_path, rag_chain)

if __name__ == "__main__":
    main()

import streamlit as st
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="YouTube Video Q&A",
    page_icon="🎥",
    layout="wide",
)

load_dotenv()

# ------------------ MODELS ------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()

vector_store_cache = {}

# ------------------ HELPERS ------------------
def extract_video_id(url):
    pattern = r"(?:v=|\/|be\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_vector_store(video_id,lang):
    if video_id in vector_store_cache:
        st.sidebar.success("Using cached transcript")
        return vector_store_cache[video_id]

    with st.spinner("Fetching transcript & building index..."):
        yt_api = YouTubeTranscriptApi()
        try:
            transcript_list = yt_api.fetch(video_id=video_id, languages=[lang])
            transcript = " ".join(chunk.text for chunk in transcript_list)
        
        except:
            st.sidebar.error("Transcript not available for this video in the selected language.")
            st.stop()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = splitter.create_documents([transcript])

        vector_store = FAISS.from_documents(chunks, embedding_model)
        vector_store_cache[video_id] = vector_store
        return vector_store

# ------------------ SIDEBAR ------------------
st.sidebar.title("🎥 YouTube Q&A")
st.sidebar.markdown(
    """
    **Ask questions directly from a YouTube video transcript.**

    - Paste a YouTube link  
    - Wait for indexing  
    - Ask anything from the video  
    """
)

url = st.sidebar.text_input("YouTube Video URL")

lang_map={
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te"
}

lang_input = st.sidebar.selectbox(
    "Transcript Language",
    options=["English", "Hindi", "Tamil", "Telugu"],
    index=0,
)

lang=lang_map[lang_input]

# ------------------ MAIN UI ------------------
st.title("🎬 YouTube Video Question Answering")
st.caption("Powered by LangChain, Gemini & FAISS")

if not url:
    st.info("👈 Enter a YouTube URL from the sidebar to begin")
    st.stop()

video_id = extract_video_id(url)
if not video_id:
    st.error("❌ Invalid YouTube URL")
    st.stop()

vector_store = get_vector_store(video_id,lang)
retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
)

template = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say "I don't know".

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"],
)

parallel_chain = RunnableParallel(
    {
        "context": RunnableSequence(retriever, RunnableLambda(format_docs)),
        "question": RunnablePassthrough(),
    }
)

chain = RunnableSequence(
    parallel_chain, template, llm_model, StrOutputParser()
)

# ------------------ CHAT UI ------------------
if "yt_messages" not in st.session_state:
    st.session_state.yt_messages = []

for msg in st.session_state.yt_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about the video...")

if query:
    st.session_state.yt_messages.append(
        {"role": "user", "content": query}
    )

    st.chat_message("user").markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chain.invoke(query)
            st.markdown(result)

    st.session_state.yt_messages.append(
        {"role": "assistant", "content": result}
    )

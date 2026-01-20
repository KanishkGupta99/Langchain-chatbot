import streamlit as st

from backend.yt_chatbot_logic import chatbot
from langchain_core.messages import HumanMessage

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="YouTube Video Q&A",
    page_icon="🎥",
    layout="wide",
)

CONFIG={'configurable':{'thread_id':'1'}}

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
            result=st.write_stream(
                message_chunk.content for message_chunk, metadata in chatbot.stream(
                    {"messages":[HumanMessage(content=query)],"youtube_url":url,"lang":lang}, 
                    config=CONFIG,
                    stream_mode="messages"
                )
            )

    st.text(chatbot.get_state(config=CONFIG))
    st.session_state.yt_messages.append(
        {"role": "assistant", "content": result}
    )

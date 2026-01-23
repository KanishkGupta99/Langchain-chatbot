import re
import uuid
import streamlit as st
from backend.yt_chatbot_logic import chatbot, upsert_thread, fetch_threads
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="YouTube Video Q&A",
    page_icon="🎥",
    layout="wide",
)

# ------------------ LLM SETUP FOR TITLE GENERATION ------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class thread_title(BaseModel):
    title: str = Field(description="title of the thread")

parser = PydanticOutputParser(pydantic_object=thread_title)

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    st.session_state.yt_thread_id = generate_thread_id()
    st.session_state.yt_messages = []
    st.session_state.yt_thread_title = "YouTube Chat"
    st.session_state.yt_url = ""
    st.session_state.yt_previous_url = ""
    st.session_state.is_loading_thread = False

def load_conversation(thread_id):
    """Load a conversation thread. This function modifies session state."""
    thread_id = str(thread_id)
    state = chatbot.get_state({'configurable': {'thread_id': thread_id}})
    if state:
        st.session_state.is_loading_thread = True
        
        st.session_state.yt_thread_id = thread_id
        st.session_state.yt_chat_threads[thread_id] = state.values.get("thread_title", "YouTube Chat")

        temp_messages = []
        for message in state.values.get("messages", []):
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": message.content})
        
        st.session_state['yt_messages'] = temp_messages

        youtube_url = state.values.get("youtube_url", "")
        st.session_state.yt_url = youtube_url
        st.session_state.yt_previous_url = youtube_url # Prevent the "URL changed" trigger
        st.session_state.yt_thread_title = state.values.get("thread_title", "YouTube Chat")
        
        lang_code = state.values.get("lang")
        if lang_code:
            for display, code in lang_map.items():
                if code == lang_code:
                    st.session_state.yt_lang_input = display
                    break

def extract_video_id(url):
    if not url: return None
    patterns = [r"(?:v=|\/)([0-9A-Za-z_-]{11})", r"youtu\.be\/([0-9A-Za-z_-]{11})", r"embed\/([0-9A-Za-z_-]{11})"]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match: return match.group(1)
    return None

def get_yt_thumbnails(url: str):
    video_id = extract_video_id(url)
    if not video_id: return None
    return {"high": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"}

@st.cache_data(ttl=3600)
def get_video_title(video_id: str):
    try:
        # Use the ID to construct a clean URL for title scraping
        response = requests.get(f"https://www.youtube.com/watch?v={video_id}")
        soup = BeautifulSoup(response.text, "html.parser")
        title_tag = soup.find("meta", property="og:title")
        return title_tag["content"] if title_tag else "Title not found"
    except Exception as e:
        return f"Error: {e}"

def generate_thread_title_from_video(video_title: str):
    try:
        template = PromptTemplate(
            template="Write a 3-4 word generalised title for this YouTube video: {video_title} \n {format_instructions}, avoid using punctuation symbols",
            input_variables=['video_title'],
            partial_variables={'format_instructions': parser.get_format_instructions()}
        )
        chain = template | llm | parser
        result = chain.invoke({'video_title': video_title})
        return result.title
    except Exception:
        return " ".join(video_title.split()[:4])

# ------------------ STATES ------------------
lang_map = {"English": "en", "Hindi": "hi", "Tamil": "ta", "Telugu": "te"}

if "yt_thread_id" not in st.session_state:
    st.session_state.yt_thread_id = generate_thread_id()
if "yt_messages" not in st.session_state:
    st.session_state.yt_messages = []
if "yt_url" not in st.session_state:
    st.session_state.yt_url = ""
if "yt_thread_title" not in st.session_state:
    st.session_state.yt_thread_title = "YouTube Chat"
if "yt_chat_threads" not in st.session_state:
    st.session_state.yt_chat_threads = fetch_threads()
if "yt_previous_url" not in st.session_state:
    st.session_state.yt_previous_url = ""
if "is_loading_thread" not in st.session_state:
    st.session_state.is_loading_thread = False

# ------------------ SIDEBAR ------------------
st.sidebar.title("🎥 YouTube Q&A")

url_input = st.sidebar.text_input("YouTube Video URL", key="yt_url", placeholder="https://www.youtube.com/watch?v=...")

lang_input = st.sidebar.selectbox(
    "Transcript Language",
    options=list(lang_map.keys()),
    index=list(lang_map.keys()).index(st.session_state.get("yt_lang_input", "English")),
    key="yt_lang_input"
)
lang = lang_map[lang_input]

if st.sidebar.button("New Chat", on_click=reset_chat):
    pass 

st.sidebar.header("My conversations")
for thread_id, title in list(st.session_state.yt_chat_threads.items())[::-1]:
    st.sidebar.button(
        str(title), 
        key=f"yt-{thread_id}", 
        use_container_width=True,
        on_click=load_conversation,
        args=(thread_id,)
    )

# ------------------ MAIN UI ------------------
st.title("🎬 YouTube Video Question Answering")

current_url = st.session_state.yt_url.strip() if st.session_state.yt_url else ""

if current_url and current_url != st.session_state.get("yt_previous_url", ""):
    if not st.session_state.get("is_loading_thread"):
        st.session_state.yt_thread_id = generate_thread_id()
        st.session_state.yt_messages = []
        st.session_state.yt_thread_title = "YouTube Chat"
    
    st.session_state.yt_previous_url = current_url
    st.session_state.is_loading_thread = False

if not current_url:
    st.info("👈 Enter a YouTube URL from the sidebar to begin")
    st.stop()

video_id = extract_video_id(current_url)
if not video_id:
    st.error("❌ Invalid YouTube URL.")
    st.stop()

thumbs = get_yt_thumbnails(current_url)
if thumbs:
    st.image(thumbs["high"], caption="Video Thumbnail", use_container_width=False)

for msg in st.session_state.yt_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about the video...")

if query:
    if len(st.session_state.yt_messages) == 0:
        video_title = get_video_title(video_id)
        with st.spinner("Generating thread title..."):
            generated_title = generate_thread_title_from_video(video_title)
            st.session_state.yt_thread_title = generated_title
        
        upsert_thread(thread_id=st.session_state.yt_thread_id, title=st.session_state.yt_thread_title)
        st.session_state.yt_chat_threads[st.session_state.yt_thread_id] = st.session_state.yt_thread_title
    
    st.session_state.yt_messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Processing..."):
                placeholder = st.empty()
                full_response = ""
                for message_chunk, metadata in chatbot.stream(
                    {
                        "messages": [HumanMessage(content=query)],
                        "youtube_url": current_url,
                        "lang": lang,
                        "thread_title": st.session_state.yt_thread_title
                    },
                    config={'configurable': {'thread_id': st.session_state.yt_thread_id}},
                    stream_mode="messages"
                ):
                    full_response += message_chunk.content
                    placeholder.markdown(full_response)
            st.session_state.yt_messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Error: {e}")
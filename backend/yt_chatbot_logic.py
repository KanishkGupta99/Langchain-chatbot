import re
import os
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph.message import add_messages

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage

from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from datetime import datetime

from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

cache={}
# ------------------ STATE ------------------

class YTState(TypedDict):
    youtube_url: str
    video_id: str
    lang: str
    context: str
    messages: Annotated[list[BaseMessage], add_messages]
    thread_title: str

# ------------------ NODES ------------------

def extract_video_id(state: YTState) -> dict:
    pattern = r"(?:v=|\/|be\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, state["youtube_url"])
    if not match:
        return {"video_id":""}

    return {"video_id": match.group(1)}

def get_vector_store(state: YTState) -> dict:
    video_id = state["video_id"]
    lang = state["lang"]

    if not video_id:
        return {}

    if video_id in cache:
        return {}

    yt_api=YouTubeTranscriptApi()
    try:
        transcript_list = yt_api.fetch(video_id=video_id,languages=[lang])
        transcript = " ".join(chunk.text for chunk in transcript_list)
    
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.create_documents([transcript])

        vector_store = FAISS.from_documents(chunks, embedding_model)
        cache[video_id] = vector_store

        return {}
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return {}

def get_context(state: YTState) -> dict:
    video_id = state["video_id"]
    
    if not video_id or video_id not in cache:
        return {"context": "Video transcript not available. Please check the YouTube URL and try again."}
    
    vector_store = cache[video_id]
    question = state["messages"][-1].content

    docs = vector_store.similarity_search(question, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)

    return {"context": context}


def answer_question(state: YTState) -> dict:
    messages = state["messages"]
    question= messages[-1].content
    context = state["context"]

    template = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
Providing you all the messages sent by the user so far.

Messages:
{messages}

Question:
{question}

Context:
{context}
""",
        input_variables=["context", "messages","question"],
    )

    prompt = template.format(context=context, messages=messages, question=question)
    result = llm.invoke(prompt)

    return {
        "messages": [result],
        "thread_title": state.get("thread_title","YouTube Chat")
    }

# ------------------ GRAPH ------------------

graph = StateGraph(YTState)

graph.add_node("extract_video_id", extract_video_id)
graph.add_node("get_vector_store", get_vector_store)
graph.add_node("get_context", get_context)
graph.add_node("answer_question", answer_question)

graph.add_edge(START, "extract_video_id")
graph.add_edge("extract_video_id", "get_vector_store")
graph.add_edge("get_vector_store", "get_context")
graph.add_edge("get_context", "answer_question")
graph.add_edge("answer_question", END)

client=MongoClient(os.getenv("MONGO_URI"))
db=client["chatbot_db"]

checkpointer=MongoDBSaver(
    client=client,
    db=db,
    collection_name="simple_chatbot_checkpoints"
)

threads_col=db["yt_threads"]

def upsert_thread(thread_id:str,title:str|None=None)->None:
    threads_col.update_one(
        {'thread_id':str(thread_id)},
        {
            '$set':{
                'title':title or "New Chat",
                'updated_at':datetime.utcnow()
            }
        },
        upsert=True
    )

def fetch_threads()->dict[str,str]:
    docs=threads_col.find(
        {},{"_id":0,"thread_id":1,"title":1}
    ).sort('updated_at',-1)

    return {doc["thread_id"]: doc["title"] for doc in docs}

chatbot = graph.compile(checkpointer=checkpointer)
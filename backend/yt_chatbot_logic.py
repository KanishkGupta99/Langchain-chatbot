import re
import operator
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage

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

    if video_id in cache:
        return {}

    else:
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
        except Exception:
            return {}

def get_context(state: YTState) -> dict:
    vector_store = cache[state["video_id"]]
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
If the context is insufficient, say "I don't know".
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

    return {"messages": [result]}

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

chatbot = graph.compile(checkpointer=InMemorySaver())
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, NotRequired
from dotenv import load_dotenv
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from datetime import datetime
import os
import io

from PIL import Image
import pytesseract
try:
    import easyocr
except Exception:
    easyocr = None

from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

from langgraph.graph.message import add_messages


class chat_state(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_title: str
    image_path: NotRequired[str]
    image_bytes: NotRequired[bytes]


def extract_text_from_image(image_path: str | None = None, image_bytes: bytes | None = None) -> str:
    """Extract text from an image using pytesseract with an easyocr fallback."""
    if image_path:
        img = Image.open(image_path)
    elif image_bytes:
        img = Image.open(io.BytesIO(image_bytes))
    else:
        return ""

    img = img.convert("RGB")

    # Try pytesseract first
    try:
        text = pytesseract.image_to_string(img)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass

    # Fallback to easyocr if available
    if easyocr is not None:
        try:
            import numpy as _np
            arr = _np.array(img)
            reader = easyocr.Reader(["en"], gpu=False)
            results = reader.readtext(arr, detail=0)
            return "\n".join(results)
        except Exception:
            return ""

    return ""


def chat_node(state: chat_state) -> chat_state:
    """Run LLM and carry forward the existing thread title. If an image is provided,
    extract its text and include it as context for answering the user's question.
    """
    messages = state["messages"]

    # Extract question from last user message
    question = messages[-1].content if messages else ""

    ocr_text = ""
    if state.get("image_path") or state.get("image_bytes"):
        ocr_text = extract_text_from_image(state.get("image_path"), state.get("image_bytes"))

    if ocr_text:
        template = PromptTemplate(
            template="""
You are a helpful assistant. Answer ONLY from the provided image OCR text when relevant.

Messages:
{messages}

Question:
{question}

Image OCR Text:
{context}
""",
            input_variables=["context", "messages", "question"],
        )

        prompt = template.format(context=ocr_text, messages=messages, question=question)
        result = llm.invoke(prompt)
    else:
        # No image/context — just invoke the LLM with the messages as-is
        result = llm.invoke(messages)

    return {
        "messages": [result],
        "thread_title": state.get("thread_title", "New Chat"),
    }


client = MongoClient(os.getenv("MONGO_URI"))
db = client["chatbot_db"]

checkpointer = MongoDBSaver(
    client=client,
    db=db,
    collection_name="simple_chatbot_checkpoints",
)

threads_col = db["threads"]


def upsert_thread(thread_id: str, title: str | None = None) -> None:
    threads_col.update_one(
        {"thread_id": str(thread_id)},
        {
            "$set": {
                "title": title or "New Chat",
                "updated_at": datetime.utcnow(),
            }
        },
        upsert=True,
    )


def fetch_threads() -> dict[str, str]:
    docs = threads_col.find({}, {"_id": 0, "thread_id": 1, "title": 1}).sort("updated_at", -1)

    return {doc["thread_id"]: doc["title"] for doc in docs}


graph = StateGraph(chat_state)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
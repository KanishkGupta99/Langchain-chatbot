from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from datetime import datetime
import os

load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

from langgraph.graph.message import add_messages

class chat_state(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    thread_title:str

def chat_node(state:chat_state)->chat_state:
    """Run LLM and carry forward the existing thread title."""
    result=llm.invoke(state['messages'])
    return {
        'messages':[result],
        'thread_title':state.get('thread_title',"New Chat")
    }

client=MongoClient(os.getenv("MONGO_URI"))
db=client["chatbot_db"]

checkpointer=MongoDBSaver(
    client=client,
    db=db,
    collection_name="simple_chatbot_checkpoints"
)

threads_col=db["threads"]

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

graph=StateGraph(chat_state)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot=graph.compile(checkpointer=checkpointer)
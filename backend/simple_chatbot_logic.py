from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

from langgraph.graph.message import add_messages

class chat_state(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    thread_title:str

def chat_node(state:chat_state)->chat_state:
    result=llm.invoke(state['messages'])
    return {'messages':[result]}

conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)
checkpointer=SqliteSaver(conn=conn)

graph=StateGraph(chat_state)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot=graph.compile(checkpointer=checkpointer)
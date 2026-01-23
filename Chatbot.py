import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from backend.simple_chatbot_logic import chatbot, upsert_thread, fetch_threads
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class thread_title(BaseModel):
    title:str=Field(description="title of the thread")

parser=PydanticOutputParser(pydantic_object=thread_title)

# ------------------ UTILITY FUNCTIONS ------------------

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['messages'] = []

def add_thread(thread_id):
    thread_id = str(thread_id)
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'][thread_id]="New Chat"
    upsert_thread(thread_id=thread_id,title=st.session_state['chat_threads'][thread_id])

def load_conversation(thread_id):
    thread_id = str(thread_id)
    state = chatbot.get_state({'configurable': {'thread_id': thread_id}})
    if state:
        st.session_state['chat_threads'][thread_id]=state.values.get('thread_title',"New Chat")
        return state.values['messages']

def get_title(query):
    template=PromptTemplate(
        template="Write a 3-4 word generalised title for this query-{query} \n {format_instructions}, avoid using punctation symbols",
        input_variables=['query'],
        partial_variables={'format_instructions':parser.get_format_instructions}
    )

    chain=template | llm | parser
    result=chain.invoke({'query':query})
    return result.title

def update_thread_title(thread_id,title):
    thread_id = str(thread_id)
    st.session_state['chat_threads'][thread_id]=title
    upsert_thread(thread_id=thread_id,title=title)

# ------------------ SETUP ------------------

st.set_page_config(
    page_title="General Chatbot",
    page_icon="🧠",
    layout="wide",
)

# ------------------ CHAT STATE ------------------

if "messages" not in st.session_state:
    st.session_state['messages'] = []

if "thread_id" not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state['chat_threads'] = fetch_threads()
else:
    # Normalize existing keys to strings, then merge persisted threads
    st.session_state['chat_threads'] = {str(k):v for k,v in st.session_state['chat_threads'].items()}
    st.session_state['chat_threads'] |= fetch_threads()

# ------------------ SIDEBAR ------------------

if st.sidebar.button("New Chat"):
    if len(st.session_state["messages"]) > 0:
        reset_chat()

st.sidebar.header("My conversations")

# Display historical threads
for thread_id,title in list(st.session_state["chat_threads"].items())[::-1]:
    if st.sidebar.button(str(title), key=str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for message in messages:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": message.content})

        st.session_state['messages'] = temp_messages

# ------------------ UI ------------------

st.title("🧠 General Question Answering Chatbot")
st.markdown("Ask anything and get instant answers from Gemini.")
st.divider()

# ------------------ DISPLAY CHAT ------------------

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ INPUT ------------------

query = st.chat_input("Ask your question here...")
CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

if query:
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": query})
    if len(st.session_state["messages"])==1:
        # First message -> create thread entry
        add_thread(st.session_state["thread_id"])
        title=get_title(query)
        update_thread_title(st.session_state["thread_id"],title)
        
    st.chat_message("user").markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Stream the response
            result = st.write_stream(
                message_chunk.content for message_chunk, metadata in chatbot.stream(
                    {
                        "messages": [HumanMessage(content=query)],
                        "thread_title": st.session_state["chat_threads"].get(st.session_state["thread_id"],"New Chat")
                    },
                    config=CONFIG,
                    stream_mode="messages"
                )
            )

    # Save assistant message
    st.session_state["messages"].append({"role": "assistant", "content": result})
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# ------------------ SETUP ------------------
load_dotenv()

st.set_page_config(
    page_title="General Chatbot",
    page_icon="🧠",
    layout="wide",
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()

# ------------------ UI ------------------
st.title("🧠 General Question Answering Chatbot")
st.markdown("Ask anything and get instant answers from Gemini.")
st.divider()

# ------------------ CHAT STATE ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------ DISPLAY CHAT ------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ INPUT ------------------
query = st.chat_input("Ask your question here...")

if query:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    st.chat_message("user").markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model.invoke(query)
            result = parser.parse(response.content)
            st.markdown(result)

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": result}
    )

import streamlit as st
import tempfile

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide",
)

load_dotenv()

# ------------------ MODELS ------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()

# ------------------ HELPERS ------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource(show_spinner=False)
def build_vector_store(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store

# ------------------ SIDEBAR ------------------
st.sidebar.title("📄 PDF Chatbot")
st.sidebar.markdown("Upload a PDF and ask questions from its content.")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"],
)

# ------------------ MAIN UI ------------------
st.title("📄 PDF Question Answering")
st.markdown(
    "Ask natural language questions and get answers **directly from the PDF**."
)

st.divider()

if not uploaded_file:
    st.info("👈 Upload a PDF from the sidebar to begin.")
    st.stop()

# ------------------ SAVE PDF ------------------
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
    tmp_file.write(uploaded_file.read())
    pdf_path = tmp_file.name

# ------------------ BUILD VECTOR STORE ------------------
with st.spinner("Indexing PDF..."):
    vector_store = build_vector_store(pdf_path)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# ------------------ PROMPT ------------------
template = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided document context.
If the answer is not present, say "I don't know".

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
    parallel_chain,
    template,
    llm_model,
    StrOutputParser(),
)

# ------------------ CHAT UI ------------------

if "file_messages" not in st.session_state:
    st.session_state.file_messages = []

for msg in st.session_state.file_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about the PDF...")

if query:
    st.session_state.file_messages.append(
        {"role": "user", "content": query}
    )

    st.chat_message("user").markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching document..."):
            result = chain.invoke(query)
            st.markdown(result)

    st.session_state.file_messages.append(
        {"role": "assistant", "content": result}
    )

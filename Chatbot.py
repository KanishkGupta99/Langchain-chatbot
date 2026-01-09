import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

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
st.markdown("Ask anything or upload an image and ask questions about it.")
st.divider()

# ------------------ CHAT STATE ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------ DISPLAY CHAT ------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ INPUTS ------------------
query = st.chat_input("Ask your question here...")
image = st.file_uploader(
    "Upload an image (optional – OCR will be used)",
    type=["png", "jpg", "jpeg"]
)

# ------------------ OCR ------------------
ocr_text = ""
if image:
    img = Image.open(image)
    ocr_text = pytesseract.image_to_string(img)

# ------------------ PROCESS QUERY ------------------
template=PromptTemplate(
    template="""
        You are given text extracted from an image using OCR.

        OCR TEXT:
        {ocr_text}

        USER QUESTION:
        {query}

        Answer based strictly on the OCR text.
        If no OCR text is there, anwer query as best as possible.
""",
    input_variables=["ocr_text", "query"]
)
if query:
    final_prompt = template.format(
        ocr_text=ocr_text,
        query=query
    )

    # Save user message
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    st.chat_message("user").markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model.invoke(final_prompt)
            result = parser.parse(response.content)
            st.markdown(result)

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": result}
    )

# 🧠 LangChain Chatbot

A multi-purpose chatbot application built with Streamlit and LangChain, featuring general Q&A, YouTube video transcript analysis, and PDF document question-answering capabilities.

## ✨ Features

- **General Chatbot**: Interactive Q&A powered by Google Gemini
- **YouTube Video Q&A**: Ask questions about YouTube videos using their transcripts with multi-language support (English, Hindi, Tamil, Telugu)
- **PDF Document Q&A**: Upload PDF files and get answers directly from document content using RAG (Retrieval-Augmented Generation)

## 🚀 Why This Project is Useful

- **Multi-Modal Support**: Chat with AI, analyze video content, or query documents—all in one application
- **RAG-Powered**: Uses vector embeddings and semantic search for accurate, context-aware responses
- **Multi-Language Support**: YouTube chatbot supports multiple languages for transcript extraction
- **User-Friendly Interface**: Clean Streamlit UI with chat-based interactions
- **Efficient Caching**: YouTube transcripts are cached to improve performance

## 🛠️ Tech Stack

- **Framework**: Streamlit
- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: HuggingFace Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Document Processing**: PyPDFLoader, YouTube Transcript API
- **Orchestration**: LangChain

## 📋 Prerequisites

- Python 3.14 or higher
- Google Gemini API key

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Langchain-chatbot-og
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_gemini_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run Chatbot.py
   ```

   The application will open in your default browser at `http://localhost:8501`

## 📖 Usage

### General Chatbot

Navigate to the main page to chat with the AI assistant powered by Google Gemini.

### YouTube Video Q&A

1. Go to the "YouTube Chatbot" page from the sidebar
2. Paste a YouTube video URL in the sidebar
3. Select the transcript language (English, Hindi, Tamil, or Telugu)
4. Wait for the transcript to be indexed
5. Ask questions about the video content

**Example:**
```
Question: "What are the main topics discussed in this video?"
```

### PDF Document Q&A

1. Go to the "File Chatbot" page from the sidebar
2. Upload a PDF file using the file uploader
3. Wait for the document to be indexed
4. Ask questions about the PDF content

**Example:**
```
Question: "What is the summary of chapter 3?"
```

## 🏗️ Project Structure

```
Langchain-chatbot-og/
├── Chatbot.py                 # Main general chatbot page
├── pages/
│   ├── 1_Youtube_Chatbot.py  # YouTube video Q&A page
│   └── 2_File_Chatbot.py     # PDF document Q&A page
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project configuration
└── README.md               # This file
```

## 🔑 Key Components

- **Vector Store**: FAISS-based vector database for semantic search
- **Retrieval Chain**: LangChain RAG pipeline with similarity search
- **Text Splitting**: Recursive character text splitter with 1000-character chunks and 200-character overlap
- **Embeddings**: Sentence transformers for document and query embeddings

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Getting Help

- **Issues**: Open an issue on GitHub for bug reports or feature requests
- **Documentation**: Check the [LangChain documentation](https://python.langchain.com/) for framework-specific questions
- **Streamlit**: Refer to [Streamlit documentation](https://docs.streamlit.io/) for UI-related queries

## 👤 Maintainer

This project is maintained by the community. Contributions and improvements are always welcome!

---

**Note**: Make sure to keep your API keys secure and never commit them to version control. Use environment variables or a `.env` file (which should be in `.gitignore`).

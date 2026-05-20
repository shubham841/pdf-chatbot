# 📄 PDF Chatbot — RAG Pipeline

A conversational AI chatbot that answers questions from any PDF document using Retrieval-Augmented Generation (RAG). Built with LangChain, FAISS, HuggingFace Embeddings, and Groq LLM.

---

## 🚀 Live Demo

> Upload any PDF → Ask questions → Get accurate, context-grounded answers instantly.

---

## 🧠 What is RAG?

RAG (Retrieval-Augmented Generation) solves a core problem with LLMs — they don't know your private data.

```
PDF Upload → Text Extraction → Chunking → Embeddings → FAISS Vector Store
                                                                ↓
                                          User Query → Similarity Search
                                                                ↓
                                          Retrieved Context + Prompt → LLM → Answer
```

Unlike ChatGPT, this chatbot:
- Only answers from your uploaded document
- Does not hallucinate facts outside the document
- Works with private, domain-specific, or confidential PDFs
- Is fully built from scratch — not a black box

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| PDF Parsing | pdfplumber |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Database | FAISS (Facebook AI Similarity Search) |
| LLM | Groq — `llama-3.1-8b-instant` |
| Orchestration | LangChain |
| Environment | Python 3.14, python-dotenv |

---

## 📁 Project Structure

```
rag-chatbot/
├── app.py              ← Streamlit UI
├── rag_pipeline.py     ← RAG logic (load, chunk, embed, retrieve, answer)
├── requirements.txt    ← All dependencies
├── .env                ← API keys (never commit this)
└── .gitignore
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get your Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up → API Keys → Create API key

### 5. Create `.env` file

```
GROQ_API_KEY=your_groq_api_key_here
```

### 6. Run the app

```bash
python -m streamlit run app.py
```

---

## 📦 Requirements

```
langchain
langchain-community
langchain-core
langchain-text-splitters
langchain-huggingface
langchain-groq
faiss-cpu
pdfplumber
sentence-transformers
streamlit
python-dotenv
```

---

## 💡 How It Works

1. **PDF Upload** — User uploads a PDF via the Streamlit UI
2. **Text Extraction** — `pdfplumber` extracts raw text from each page
3. **Chunking** — Text is split into 500-character overlapping chunks
4. **Embedding** — Each chunk is converted to a vector using HuggingFace `all-MiniLM-L6-v2`
5. **Vector Store** — Vectors are stored in a FAISS index in memory
6. **Query** — User types a question
7. **Similarity Search** — Top 3 most relevant chunks are retrieved from FAISS
8. **LLM Answer** — Retrieved context + question is sent to Groq LLaMA 3.1 for a grounded answer

---

## 🎯 Use Cases

- Resume/CV analysis and Q&A
- Research paper summarization
- Legal document review
- Study notes assistant
- Company document search

---

## 🔑 Why RAG over ChatGPT?

| Feature | ChatGPT | This RAG Chatbot |
|---------|---------|-----------------|
| Uses your private data | ❌ | ✅ |
| Answers grounded in document | ❌ | ✅ |
| No hallucination on facts | ❌ | ✅ |
| Built from scratch | ❌ | ✅ |
| Understand the pipeline | ❌ | ✅ |

---

## 👨‍💻 Author

**Shubham Bisht**  
Built as a GenAI portfolio project demonstrating end-to-end RAG pipeline development.

---

## 📄 License

MIT License

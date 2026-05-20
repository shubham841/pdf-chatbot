import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.documents import Document
import pdfplumber

load_dotenv()

# ── 1. Load PDF ───────────────────────────────────────────
def load_pdf(file_path):
    docs = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"page": i + 1, "source": file_path}
                ))
    if not docs:
        raise ValueError("Could not extract text. PDF may be image-based.")
    return docs

# ── 2. Split into chunks ──────────────────────────────────
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)

# ── 3. Create vector store ────────────────────────────────
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(chunks, embeddings)

# ── LLM — created once, reused every call ─────────────────
llm = ChatGroq(
   model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# ── 4. Answer a question ──────────────────────────────────
def answer_question(vectorstore, query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Answer the question based only on the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {query}
"""
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            return "⚠️ Rate limit hit. Please wait 30 seconds and try again."
        return f"❌ Error: {str(e)}"
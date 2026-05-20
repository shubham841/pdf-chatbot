import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import tempfile
from rag_pipeline import load_pdf, split_docs, create_vectorstore, answer_question

# ── Page config ───────────────────────────────────────────
st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 PDF Chatbot")
st.write("Upload a PDF and ask questions about it.")

# ── Session state ─────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Upload PDF ────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("Processing PDF... this may take a moment."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            temp_path = f.name

        docs = load_pdf(temp_path)
        chunks = split_docs(docs)
        st.session_state.vectorstore = create_vectorstore(chunks)
        os.unlink(temp_path)

    st.success(f"✅ PDF processed! {len(chunks)} chunks created. You can now ask questions.")

# ── Reset button ──────────────────────────────────────────
if st.session_state.vectorstore:
    if st.button("🔄 Upload a new PDF"):
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.rerun()

# ── Chat interface ─────────────────────────────────────────
if st.session_state.vectorstore:
    # Display history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Ask a question about your PDF...")

    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_question(st.session_state.vectorstore, query)
            st.write(answer)

        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

else:
    st.info("👆 Please upload a PDF to get started.")
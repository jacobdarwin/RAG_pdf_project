import os
import streamlit as st

from utils.pdf_loader import load_pdf
from utils.vector_store import create_vector_db
from utils.vector_store import load_vector_db
from utils.rag_chain import create_rag_chain

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🤖"
)

st.title("🤖 RAG PDF Chatbot")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file:

    os.makedirs("data", exist_ok=True)

    pdf_path = os.path.join(
        "data",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF Uploaded")

    chunks = load_pdf(pdf_path)

    vectordb = create_vector_db(chunks)

    st.success("Vector Database Created")

question = st.text_input(
    "Ask a question"
)

if question:

    vectordb = load_vector_db()

    qa_chain = create_rag_chain(vectordb)

    result = qa_chain.invoke(
        {"query": question}
    )

    st.subheader("Answer")

    st.write(result["result"])

    st.subheader("Sources")

    for doc in result["source_documents"]:

        st.write(doc.page_content[:500])
        st.divider()
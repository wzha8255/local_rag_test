import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


# Load Embedding Model
@st.cache_resource
def load_embedding_model():
    """load downloaded embedding model from local folder"""
    model_path = "./models/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_path)


# Load Language Model
@st.cache_resource
def load_llm():
    """ Load language model which has been downloaded to local folder"""
    model_path = "./models/Phi-3"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)


# Process Uploaded PDF
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    reader = PdfReader(tmp_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.create_documents([text])

    return texts

# Build Vector Store
def build_vector_store(docs, embeddings):
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Main Streamlit App
def main():
    st.set_page_config(page_title="Local RAG PDF Chat", layout="wide")
    st.title("📄 Chat with your PDF (Local RAG)")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            docs = process_pdf(uploaded_file)
            embeddings = load_embedding_model()
            vectorstore = build_vector_store(docs, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            llm = load_llm()
            qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

        st.success("PDF processed. You can now ask questions.")

        query = st.text_input("Ask a question about the PDF:")
        if query:
            with st.spinner("Generating answer..."):
                result = qa_chain.run(query)
                st.write(result)

if __name__ == "__main__":
    main()

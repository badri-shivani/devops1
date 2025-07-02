import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline

# Load the question-answering pipeline
@st.cache_resource
def load_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_model = load_model()

# Load the PDF once
@st.cache_data
def load_pdf_notes():
    try:
        with fitz.open("notes.pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            return text
    except Exception as e:
        st.error(f"Could not load PDF: {e}")
        return ""

notes_text = load_pdf_notes()

# Streamlit UI
st.title("🤖 DevOps / DAA & OS Tutor Chatbot from Notes")
st.subheader("Ask a question, and I’ll answer from the uploaded PDF notes.")

user_input = st.text_area("💬 Ask your question here:")

if st.button("Get Answer"):
    if not user_input.strip():
        st.warning("Please enter a question.")
    elif not notes_text.strip():
        st.error("No text loaded from notes.pdf. Please upload a valid PDF.")
    else:
        with st.spinner("Searching your notes..."):
            try:
                result = qa_model({
                    "question": user_input,
                    "context": notes_text
                })
                st.success("Answer:")
                st.write(result["answer"])
            except Exception as e:
                st.error(f"Error: {e}")

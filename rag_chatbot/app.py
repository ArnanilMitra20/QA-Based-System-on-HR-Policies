import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Gemini Model ---
model = genai.GenerativeModel("gemini-2.0-flash")

def gemini_answer(context, question):
    prompt = f"""
You are an HR expert assistant. Use ONLY the following context from HR policies to answer the question accurately. 
Provide an estimated answer if you do not find it in the documents.

-------------------
{context}
-------------------

Question: {question}
Answer:
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# --- Vector DB Setup ---
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="vector_db", embedding_function=embedding)
retriever = db.as_retriever(search_kwargs={"k": 3})

# --- Streamlit UI ---
st.set_page_config(page_title="HR RAG Chatbot (Gemini)", layout="wide")
st.title("🤖 HR RAG Chatbot (Gemini + Vector DB)")
st.write("Ask me anything about HR policies across various sectors. I’ll answer using the policy data we've indexed.")

query = st.text_input("🔎 Ask your question:")

if query:
    with st.spinner("🔍 Retrieving relevant HR policy documents..."):
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        answer = gemini_answer(context, query)

        st.markdown("### ✅ Answer")
        st.markdown(answer)

        st.markdown("---")
        with st.expander("📄 Sources Used (Top 3 Documents)"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Source {i+1}:**\n```\n{doc.page_content[:800]}...\n```")

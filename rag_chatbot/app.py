import asyncio
import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


SECTOR_INDEX_PATHS = {
    path.replace("vector_store_", ""): path
    for path in os.listdir()
    if path.startswith("vector_store_")
}

@st.cache_resource
def load_vector_store(path: str):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


vector_stores = {sector: load_vector_store(path) for sector, path in SECTOR_INDEX_PATHS.items()}


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)


st.set_page_config(page_title="HR RAG Chatbot", page_icon="💬", layout="centered")
st.title("💬 HR Policy Assistant (Multi-Sector)")
st.caption("Ask any HR policy question. Answers are grounded in the sector-specific knowledge base.")

query = st.text_input("🔍 Enter your question:")

if query:
    with st.spinner("🔎 Searching relevant sectors..."):
        
        selected_sectors = [s for s in SECTOR_INDEX_PATHS if s.lower() in query.lower()]
        if not selected_sectors:
            selected_sectors = list(SECTOR_INDEX_PATHS.keys())

        docs = []
        for sector in selected_sectors:
            docs.extend(vector_stores[sector].similarity_search(query, k=3))

        context = "\n\n".join([f"[{doc.metadata.get('sector', 'Unknown')}] {doc.page_content}" for doc in docs])

        prompt = f"""
You are an HR Policy Expert. Use ONLY the information from the provided context to answer.
If the answer is not in the context, say "I couldn't find that in the HR policy knowledge base."

Answer in a clear, concise, and well-structured manner.
Use bullet points or numbered lists when possible.

Context:
{context}

Question: {query}

Answer:
"""
        response = llm.predict(prompt)

    st.markdown("### 📝 Answer")
    st.write(response)

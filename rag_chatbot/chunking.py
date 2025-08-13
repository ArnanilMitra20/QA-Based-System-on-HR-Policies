import os
import json
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# --- CONFIG ---
FOLDER_PATH = "generated_hr_data"  # Folder with your sector JSON files
FAISS_INDEX_PATH = "vector_store"  # Where to save FAISS index
GOOGLE_API_KEY = "AIzaSyDCNh2JK9WOePuq4EXKs9F33hVXvfnmRCA"

# Set API key for LangChain integration
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def load_json_folder(folder_path):
    docs = []
    print(f"📂 Loading JSON files from: {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"❌ Skipping invalid JSON: {filename}")
                    continue
                text_content = json.dumps(data, indent=2)
                docs.append(Document(page_content=text_content, metadata={"source": filename}))
    print(f"✅ Loaded {len(docs)} documents.")
    return docs

def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    print(f"✂️ Chunking documents (size={chunk_size}, overlap={chunk_overlap})...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(docs)
    print(f"✅ Created {len(split_docs)} chunks.")
    return split_docs

def create_vector_store(docs):
    print("🔍 Creating vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_documents(docs, embeddings)
    print("✅ Vector store created.")
    return vector_store

def save_vector_store(vector_store, path):
    vector_store.save_local(path)
    print(f"💾 Saved FAISS vector store to '{path}'")

def main():
    docs = load_json_folder(FOLDER_PATH)
    if not docs:
        print("⚠️ No documents loaded. Exiting.")
        return

    chunks = chunk_documents(docs)
    vector_store = create_vector_store(chunks)
    save_vector_store(vector_store, FAISS_INDEX_PATH)

if __name__ == "__main__":
    main()

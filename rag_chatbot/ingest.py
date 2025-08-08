import os
import json
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Folder with HR JSON files ---
data_folder = r"C:\programs\chatbot\generated_hr_data"  # Use raw string to avoid backslash issues
documents = []

# --- Text splitter for chunking ---
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# --- Load and split documents ---
for filename in os.listdir(data_folder):
    if filename.endswith(".json"):
        filepath = os.path.join(data_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            sector = data.get("sector", filename[:-5])
            content = data.get("hr_policy_data", "")

            # Split into smaller chunks
            chunks = splitter.split_text(content)
            for chunk in chunks:
                documents.append(Document(
                    page_content=chunk,
                    metadata={"sector": sector}
                ))

# --- Load embedding model ---
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Create vector DB with metadata-aware documents ---
Chroma.from_documents(documents, embedding, persist_directory="vector_db")

print("✅ Vector DB created successfully with chunked documents and metadata!")

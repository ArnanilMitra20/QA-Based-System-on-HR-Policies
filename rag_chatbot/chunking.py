import os
import json
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv
load_dotenv()


FOLDER_PATH = "generated_hr_data"  
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def flatten_json(data, parent_key="", sep=": "):
    """Recursively flattens JSON into readable text lines."""
    lines = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{k}" if not parent_key else f"{parent_key} > {k}"
            lines.extend(flatten_json(v, new_key, sep))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            lines.extend(flatten_json(v, f"{parent_key}[{i}]", sep))
    else:
        lines.append(f"{parent_key}{sep}{data}")
    return lines

def chunk_and_save_sector(sector_name, data):
    """Chunk a sector's data and save to its own FAISS index."""
    flat_text = "\n".join(flatten_json(data))
    docs = [Document(page_content=flat_text, metadata={"sector": sector_name})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=GOOGLE_API_KEY
    )
    vector_store = FAISS.from_documents(split_docs, embeddings)

    save_path = f"vector_store_{sector_name}"
    vector_store.save_local(save_path)
    print(f" Saved FAISS index for {sector_name} to '{save_path}'")

def main():
    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith(".json"):
            file_path = os.path.join(FOLDER_PATH, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sector_name = os.path.splitext(filename)[0]
                chunk_and_save_sector(sector_name, data)
            except Exception as e:
                print(f" Error processing {filename}: {e}")

if __name__ == "__main__":
    main()

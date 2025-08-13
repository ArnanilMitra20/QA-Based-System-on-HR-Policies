import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

# --- CONFIG ---
FAISS_INDEX_PATH = "vector_store"  # Path to saved FAISS index
GOOGLE_API_KEY = "AIzaSyDCNh2JK9WOePuq4EXKs9F33hVXvfnmRCA"

# Set API key for LangChain
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def load_vector_store(path):
    print(f"📂 Loading FAISS index from '{path}'...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def create_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

def main():
    vector_store = load_vector_store(FAISS_INDEX_PATH)
    qa_chain = create_qa_chain(vector_store)

    print("\n💬 RAG Chatbot is ready! Type 'exit' to quit.")
    while True:
        query = input("\n❓ Enter your question: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        answer = qa_chain.run(query)
        print(f"\n📝 Answer: {answer}")

if __name__ == "__main__":
    main()

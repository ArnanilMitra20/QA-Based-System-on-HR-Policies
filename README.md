# 🧠 HR Policy Chatbot

The **HR Policy Chatbot** is a Retrieval-Augmented Generation (RAG) powered question-answering system that helps employees and HR teams retrieve accurate answers to HR policy queries. Built with modern NLP and vector search technologies, it combines Large Language Models (LLMs) with high-speed retrieval for precise, context-aware results.

---

## ✨ Features

- 🔍 Retrieve context-aware answers to HR policy questions
- 🗂️ Use sector-wise custom JSON data as a knowledge base
- ⚡ Sub-3s response latency on large datasets
- 🧩 Modular architecture for easy extension to other domains
- 📡 Simple REST API for integration

---

## 🛠️ Tech Stack

- **Backend:** Python
- **Retrieval & Embeddings:** ChromaDB, HuggingFace Transformers
- **LLM:** OpenAI / HuggingFace models
- **Vector Store:** Chroma

---

## 🚀 Installation & Setup

### Prerequisites

Make sure you have the following installed:

- [Python](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installation/)
- (Optional) [Docker](https://www.docker.com/)

---

### Steps to Run Locally

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/hr-policy-chatbot.git
   cd hr-policy-chatbot
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   - Create a `.env` file in the root directory.
   - Add your API keys and configurations.

     Example `.env`:

     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

4. **Add your HR policy data**

   - Place your sector-wise JSON files inside the `data/` folder.

5. **Run the application**

   ```bash
   python main.py
   ```

6. **Test your chatbot**

   - Use CLI prompts or API endpoints to ask questions.

---

## 📡 API Endpoints

### Query

- `POST /api/query`
  - **Request Body:** `{ "question": "Your question here" }`
  - **Response:** `{ "answer": "Generated answer" }`

### Data

- `POST /api/data/upload` — Upload new sector-wise data
- `GET /api/data/list` — List available datasets

---

## 🏗️ Roadmap

- [ ] Add a web UI for end users
- [ ] Support more domains (Finance, Legal)
- [ ] Optimize embedding storage for faster retrieval
- [ ] Enable hybrid retrieval (keyword + vector)

---

## 🤝 Contributing

Contributions are welcome! Follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

---

## 📞 Contact

For any questions or support:

- **GitHub:** [ArnanilMitra20](https://github.com/ArnanilMitra20)
- **Email:** arnanilmitra06@gmail.com

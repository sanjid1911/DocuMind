# 🦙 DocuMind - Llama 3.1 RAG

**DocuMind** is an intelligent document assistant that lets you chat with your PDF files. It uses **Retrieval-Augmented Generation (RAG)** to provide accurate answers based strictly on your document's content.

Powered by **Streamlit** and **Meta Llama 3.1** (via Hugging Face API), this app runs entirely in the cloud—no heavy local GPU required.

---

## 🚀 Features

* **📂 PDF Ingestion:** Upload and process multiple documents instantly.
* **🧠 Llama 3.1 Intelligence:** Uses Meta's latest `Meta-Llama-3.1-8B-Instruct` model for high-quality reasoning.
* **🔍 Vector Search:** Uses `sentence-transformers` to find the exact paragraph needed to answer your question.
* **☁️ Cloud Native:** Runs on the free Hugging Face Inference API.
* **🛡️ Secure:** API keys are managed locally and never exposed.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Orchestration:** LangChain
* **Database:** ChromaDB

---

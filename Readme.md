# 📚 Smart Document Assistant

A Python-based RAG-powered document assistant using FastMCP, FAISS, and Hugging Face. It supports PDF ingestion and semantic search to provide AI-driven answers backed by real documents.

---

## ✅ Features

- **📂 PDF ingestion** from a folder using `PyMuPDF`  
- **🧠 Semantic search** via `sentence-transformers` + `faiss-cpu`  
- **🤖 Chat generation** with Hugging Face `transformers` & `torch`  
- **🔧 Tool interface** built with `fastmcp` for MCP‑compatible agents  
- **⚙️ High‑performance** — powered by `accelerate` and optimized NumPy

---

## 🔧 Requirements

See `requirements.txt`:

```text
sentence-transformers
faiss-cpu
transformers
torch
fastmcp
PyMuPDF
numpy
accelerate

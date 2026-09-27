# AI Research Assistant — Multi-PDF RAG & LLM Evaluation

An AI-powered research assistant that allows users to upload multiple research papers in PDF format, ask natural-language questions, retrieve relevant research content, and generate answers with source references.

The system uses Retrieval-Augmented Generation (RAG), ChromaDB vector search, Sentence Transformers embeddings, and an agentic research workflow.

---

## 🚀 Features

- Multi-PDF research paper upload
- PDF text extraction and intelligent chunking
- Semantic embeddings using Sentence Transformers
- Persistent ChromaDB vector database
- Hybrid retrieval for relevant research chunks
- Source-aware answers with PDF and page references
- Related and unrelated document identification
- Single-PDF removal
- Complete research-set refresh
- Agentic research workflow
- Research and comparison agents
- Retrieval evaluation framework
- FastAPI backend
- Web-based research interface
- Dockerized application

---

## 🧠 System Architecture

```mermaid
flowchart TD
    A[User Uploads PDFs] --> B[PDF Text Extraction]
    B --> C[Text Chunking]
    C --> D[Sentence Transformer Embeddings]
    D --> E[ChromaDB Vector Database]

    F[User Question] --> G[Hybrid Retrieval]
    G --> E
    E --> H[Relevant Research Chunks]

    H --> I[Research / Comparison Agents]
    I --> J[LLM Answer Generation]

    J --> K[Answer + Sources + Pages]

    L[Evaluation Dataset] --> M[Retrieval Evaluation]
    M --> N[Evaluation Metrics]
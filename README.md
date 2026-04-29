# 🤖 RAG Customer Support Assistant

> A production-grade **Retrieval-Augmented Generation (RAG)** system with **LangGraph** workflow orchestration and **Human-in-the-Loop (HITL)** escalation.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-purple)](https://trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📌 Project Overview

This system allows a company to upload a **PDF knowledge base** (product docs, FAQs, policies) and instantly serve a customer support chatbot that:

- 🔍 **Retrieves** relevant passages using semantic similarity (ChromaDB + MiniLM)
- 🧠 **Generates** grounded answers using an LLM (Gemini / GPT-4o)
- 🔀 **Routes** requests through a LangGraph state machine
- 🙋 **Escalates** low-confidence or sensitive queries to human agents (HITL)
- ⚡ **Exposes** a REST API via FastAPI + interactive CLI

---

## 🏗️ System Architecture

```
PDF Knowledge Base
      │
      ▼
[Document Loader] → [Chunker] → [Embedder] → [ChromaDB]
                                                   │
User Query ─────────────────────────────────────► │
      │                                            │
      ▼                                            │
[LangGraph Workflow]                               │
   ├─ input_node     (validate & sanitise)         │
   ├─ intent_node    (classify intent)             │
   ├─ retrieval_node ◄──────────────────────── top-k chunks
   ├─ llm_node       (generate RAG answer)
   ├─ routing_node   (confidence check)
   │      ├─ confidence ≥ 0.6 → output_node → ✅ Answer
   │      └─ confidence < 0.6 → hitl_node   → 🙋 Human Agent
   ▼
REST API (FastAPI) / CLI (Rich)
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-customer-support.git
cd rag-customer-support
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

Get a free Gemini API key at: https://aistudio.google.com

### 4. Ingest Your PDF

```bash
python -m src.ingest --pdf data/your_knowledge_base.pdf
```

### 5. Start the API Server

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/docs

### 6. Or Use the Interactive CLI

```bash
python -m src.cli
```

---

## 📂 Project Structure

```
rag-customer-support/
├── src/
│   ├── __init__.py
│   ├── config.py          # Pydantic settings (env vars)
│   ├── logger.py          # Loguru structured logging
│   ├── ingest.py          # PDF → Chunks → ChromaDB pipeline
│   ├── rag_workflow.py    # LangGraph nodes + state machine
│   ├── hitl.py            # HITL escalation queue (SQLite)
│   ├── app.py             # FastAPI REST API
│   └── cli.py             # Rich interactive CLI
├── tests/
│   └── test_pipeline.py   # Unit + integration tests
├── data/
│   ├── chroma_db/         # ChromaDB persistence (auto-created)
│   └── uploads/           # Uploaded PDFs (auto-created)
├── logs/                  # Application logs (auto-created)
├── .env.example           # Environment variable template
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check |
| `POST` | `/query` | Ask a question (main RAG endpoint) |
| `POST` | `/ingest` | Upload & ingest a PDF |
| `GET` | `/hitl/pending` | List pending escalation tickets |
| `GET` | `/hitl/stats` | Ticket statistics |
| `GET` | `/hitl/{id}` | Get single ticket |
| `PUT` | `/hitl/{id}` | Resolve ticket (human agent) |
| `DELETE` | `/hitl/{id}` | Reject ticket |

### Example Request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is your refund policy?"}'
```

### Example Response

```json
{
  "answer": "Refunds are processed within 7 business days...\n\n📄 *Sources: policy.pdf p.2*",
  "confidence": 0.82,
  "intent": "FAQ",
  "escalated": false,
  "ticket_id": null,
  "latency_ms": 1243.5
}
```

---

## 🔀 LangGraph Workflow

```
START
  │
  ▼
input_node ──────────── Validate, sanitise, initialise state
  │
intent_node ─────────── Classify: FAQ | Technical | Billing | Unknown
  │
retrieval_node ──────── ChromaDB MMR search → top-4 chunks
  │
llm_node ────────────── RAG prompt → LLM → answer + confidence
  │
routing_node ────────── Evaluate escalation triggers
  │
  ├── [confidence ≥ 0.6, no triggers] ──► output_node → END
  │
  └── [any trigger] ──────────────────► hitl_node → END
```

### Escalation Triggers

| Trigger | Condition |
|---------|-----------|
| Low Confidence | `mean(chunk_scores) < 0.60` |
| Missing Context | No chunks retrieved |
| LLM Signal | Response contains `ESCALATE` |
| Sensitive Topic | Legal/medical/financial keywords |
| Complex Query | 3+ question marks or 60+ words |

---

## 🙋 HITL (Human-in-the-Loop)

When a query is escalated:

1. Ticket is created in SQLite with status `PENDING`
2. User receives acknowledgement with ticket ID
3. Agent reviews ticket at `GET /hitl/pending`
4. Agent resolves with `PUT /hitl/{ticket_id}` → `{"response": "..."}`
5. Ticket status becomes `RESOLVED`

---

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## ⚙️ Configuration

All configuration is via `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | — | Gemini API key |
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `RETRIEVAL_TOP_K` | 4 | Chunks retrieved per query |
| `CONFIDENCE_THRESHOLD` | 0.60 | Escalation threshold |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Vector DB path |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 1.5 Flash |
| Embeddings | MiniLM-L6-v2 (local) |
| Vector DB | ChromaDB |
| Workflow | LangGraph |
| RAG Framework | LangChain |
| API | FastAPI |
| HITL Storage | SQLite + SQLAlchemy |
| CLI | Rich |
| Logging | Loguru |
| Testing | Pytest |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

Built as part of the RAG Internship Final Project — Design & Build a RAG-Based Customer Support Assistant.

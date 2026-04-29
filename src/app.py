"""
app.py — FastAPI REST API Server
==================================
Endpoints:
    POST /query           — Ask a question (main RAG endpoint)
    POST /ingest          — Ingest a PDF into the knowledge base
    GET  /hitl/pending    — List pending escalation tickets (agent dashboard)
    GET  /hitl/{id}       — Get a specific ticket
    PUT  /hitl/{id}       — Resolve a ticket (human agent submits answer)
    DELETE /hitl/{id}     — Reject a ticket
    GET  /hitl/stats      — Ticket statistics
    GET  /health          — Health check

Run:
    uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.hitl import hitl_queue
from src.ingest import run_ingestion
from src.logger import logger
from src.rag_workflow import RAGAssistant


# ─────────────────────────────────────────────────────────────────────────────
# Global RAG Assistant (singleton, loaded at startup)
# ─────────────────────────────────────────────────────────────────────────────
_assistant: Optional[RAGAssistant] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG assistant on startup."""
    global _assistant
    logger.info("Starting RAG Customer Support API...")
    try:
        _assistant = RAGAssistant()
        logger.success("RAG Assistant loaded and ready.")
    except Exception as e:
        logger.warning("RAG Assistant could not be fully initialised: {}. "
                       "Run /ingest first.", e)
        _assistant = None
    yield
    logger.info("API shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# App Initialisation
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Customer Support Assistant",
    description=(
        "A production-grade Retrieval-Augmented Generation (RAG) system "
        "with LangGraph workflow orchestration and Human-in-the-Loop (HITL) escalation."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User's question")
    session_id: Optional[str] = Field(default=None, description="Optional session ID for tracking")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is your refund policy for digital products?",
                "session_id": "user-abc-123"
            }
        }


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    intent: Optional[str]
    escalated: bool
    ticket_id: Optional[str]
    latency_ms: float


class ResolveRequest(BaseModel):
    response: str = Field(..., min_length=1, description="Human agent's answer")

    class Config:
        json_schema_extra = {"example": {"response": "Our refund policy allows returns within 30 days."}}


class IngestResponse(BaseModel):
    message: str
    filename: str
    status: str


class HealthResponse(BaseModel):
    status: str
    chroma_db: str
    llm_provider: str
    version: str


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check system health: DB connectivity, LLM config."""
    chroma_status = "ok" if os.path.exists(settings.chroma_persist_dir) else "not_initialised"
    return {
        "status": "ok",
        "chroma_db": chroma_status,
        "llm_provider": settings.llm_provider,
        "version": "1.0.0",
    }


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_endpoint(request: QueryRequest):
    """
    Main RAG endpoint. Accepts a natural language query and returns
    an AI-generated answer grounded in the knowledge base, or escalates
    to HITL if confidence is insufficient.
    """
    global _assistant

    if _assistant is None:
        try:
            _assistant = RAGAssistant()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"RAG Assistant not ready. Please ingest a PDF first. Error: {e}"
            )

    start = time.time()
    result = _assistant.query(
        user_query=request.query,
        session_id=request.session_id or str(uuid.uuid4()),
    )
    latency = round((time.time() - start) * 1000, 2)

    logger.info(
        "Query processed. intent={} escalated={} confidence={:.3f} latency={}ms",
        result.get("intent"),
        result.get("escalated"),
        result.get("confidence", 0),
        latency,
    )

    return {**result, "latency_ms": latency}


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to ingest"),
):
    """
    Upload and ingest a PDF into the knowledge base.
    Runs ingestion in the background. Use /health to check status.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file temporarily
    os.makedirs("./data/uploads", exist_ok=True)
    save_path = f"./data/uploads/{file.filename}"

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # Run ingestion in background
    background_tasks.add_task(_run_ingestion_and_reload, save_path)

    logger.info("PDF '{}' uploaded. Ingestion started in background.", file.filename)
    return {
        "message": "PDF received. Ingestion started in background.",
        "filename": file.filename,
        "status": "processing",
    }


async def _run_ingestion_and_reload(pdf_path: str):
    """Background task: ingest PDF and reload the assistant."""
    global _assistant
    try:
        run_ingestion([pdf_path])
        _assistant = RAGAssistant()
        logger.success("Background ingestion complete. Assistant reloaded.")
    except Exception as e:
        logger.error("Background ingestion failed: {}", e)


# ─────────────────────────────────────────────────────────────────────────────
# HITL Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/hitl/stats", tags=["HITL"])
async def get_hitl_stats():
    """Return escalation ticket statistics."""
    return hitl_queue.stats()


@app.get("/hitl/pending", tags=["HITL"])
async def get_pending_tickets():
    """List all pending escalation tickets for the agent dashboard."""
    tickets = hitl_queue.get_pending()
    return {"count": len(tickets), "tickets": tickets}


@app.get("/hitl/all", tags=["HITL"])
async def get_all_tickets(limit: int = 50, offset: int = 0):
    """List all tickets with pagination."""
    tickets = hitl_queue.get_all(limit=limit, offset=offset)
    return {"count": len(tickets), "tickets": tickets}


@app.get("/hitl/{ticket_id}", tags=["HITL"])
async def get_ticket(ticket_id: str):
    """Get a single escalation ticket by ID."""
    ticket = hitl_queue.get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found.")
    return ticket


@app.put("/hitl/{ticket_id}", tags=["HITL"])
async def resolve_ticket(ticket_id: str, body: ResolveRequest):
    """
    Human agent resolves an escalation ticket.
    The response is stored and the ticket is marked RESOLVED.
    """
    success = hitl_queue.resolve(ticket_id=ticket_id, human_response=body.response)
    if not success:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found.")
    return {"message": "Ticket resolved successfully.", "ticket_id": ticket_id}


@app.delete("/hitl/{ticket_id}", tags=["HITL"])
async def reject_ticket(ticket_id: str, reason: str = ""):
    """Reject a ticket (spam, duplicate, or invalid)."""
    success = hitl_queue.reject(ticket_id=ticket_id, reason=reason)
    if not success:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found.")
    return {"message": "Ticket rejected.", "ticket_id": ticket_id}

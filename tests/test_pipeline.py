"""
tests/test_pipeline.py — Unit & Integration Tests
===================================================
Tests cover:
  - Input validation & sanitisation
  - Confidence scoring logic
  - Routing logic (unit tests, no LLM calls)
  - HITL queue CRUD operations
  - End-to-end graph state flow (mocked LLM + retriever)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.rag_workflow import (
    GraphState,
    compute_confidence,
    input_node,
    routing_node,
    output_node,
    hitl_node,
)
from src.hitl import HITLQueue, EscalationReason, TicketStatus


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def base_state() -> GraphState:
    return {
        "query": "What is your refund policy?",
        "session_id": "test-session",
        "intent": "FAQ",
        "retrieved_chunks": ["Refunds are processed within 7 business days.", "Policy applies to all products."],
        "chunk_scores": [0.82, 0.75],
        "source_pages": ["policy.pdf p.2", "policy.pdf p.3"],
        "llm_response": "Refunds are processed within 7 business days.",
        "confidence": 0.79,
        "should_escalate": False,
        "escalation_reason": None,
        "ticket_id": None,
        "final_answer": None,
    }


@pytest.fixture
def hitl_queue_fixture(tmp_path, monkeypatch):
    """HITL queue backed by a temporary SQLite DB."""
    db_url = f"sqlite:///{tmp_path}/test_hitl.db"
    monkeypatch.setattr("src.hitl.settings.hitl_db_url", db_url)
    from src.hitl import HITLQueue, Base, _get_engine
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)

    queue = HITLQueue()
    # Patch the module-level SessionLocal
    import src.hitl as hitl_module
    hitl_module.SessionLocal = sessionmaker(bind=engine, autoflush=True, autocommit=False)
    return queue


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Confidence Scoring
# ─────────────────────────────────────────────────────────────────────────────
class TestConfidenceScoring:
    def test_high_confidence(self):
        score = compute_confidence([0.85, 0.80], "Here is your answer.")
        assert score == pytest.approx(0.825, abs=1e-3)

    def test_escalate_token_returns_zero(self):
        score = compute_confidence([0.90, 0.88], "ESCALATE")
        assert score == 0.0

    def test_empty_scores_returns_zero(self):
        score = compute_confidence([], "Some answer.")
        assert score == 0.0

    def test_single_score(self):
        score = compute_confidence([0.65], "Answer text.")
        assert score == pytest.approx(0.65)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Input Node
# ─────────────────────────────────────────────────────────────────────────────
class TestInputNode:
    def test_strips_whitespace(self):
        state = input_node({"query": "  hello world  ", "session_id": None})
        assert state["query"] == "hello world"

    def test_removes_null_bytes(self):
        state = input_node({"query": "hello\x00world", "session_id": None})
        assert "\x00" not in state["query"]

    def test_normalises_multi_space(self):
        state = input_node({"query": "what   is   your   policy", "session_id": None})
        assert state["query"] == "what is your policy"

    def test_initialises_state_fields(self):
        state = input_node({"query": "test", "session_id": "s1"})
        assert state["intent"] is None
        assert state["retrieved_chunks"] is None
        assert state["confidence"] is None
        assert state["should_escalate"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Routing Node
# ─────────────────────────────────────────────────────────────────────────────
class TestRoutingNode:
    def test_high_confidence_no_escalation(self, base_state):
        result = routing_node(base_state)
        assert result["should_escalate"] is False
        assert result["escalation_reason"] is None

    def test_low_confidence_triggers_escalation(self, base_state):
        base_state["confidence"] = 0.40
        result = routing_node(base_state)
        assert result["should_escalate"] is True
        assert result["escalation_reason"] == "low_confidence"

    def test_empty_chunks_triggers_escalation(self, base_state):
        base_state["retrieved_chunks"] = []
        base_state["chunk_scores"] = []
        base_state["confidence"] = 0.0
        result = routing_node(base_state)
        assert result["should_escalate"] is True

    def test_llm_escalate_signal(self, base_state):
        base_state["llm_response"] = "ESCALATE — insufficient context"
        base_state["confidence"] = 0.0
        result = routing_node(base_state)
        assert result["should_escalate"] is True
        assert result["escalation_reason"] == "llm_escalation_signal"

    def test_sensitive_keyword_triggers_escalation(self, base_state):
        base_state["query"] = "I want to sue your company"
        result = routing_node(base_state)
        assert result["should_escalate"] is True
        assert result["escalation_reason"] == "sensitive_topic"

    def test_complex_query_triggers_escalation(self, base_state):
        base_state["query"] = "what? and what? and what?" 
        result = routing_node(base_state)
        assert result["should_escalate"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Output Node
# ─────────────────────────────────────────────────────────────────────────────
class TestOutputNode:
    def test_appends_source_footer(self, base_state):
        result = output_node(base_state)
        assert "Sources:" in result["final_answer"]

    def test_includes_llm_response(self, base_state):
        result = output_node(base_state)
        assert "7 business days" in result["final_answer"]

    def test_empty_pages_no_footer(self, base_state):
        base_state["source_pages"] = []
        result = output_node(base_state)
        assert "Sources:" not in result["final_answer"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests: HITL Queue
# ─────────────────────────────────────────────────────────────────────────────
class TestHITLQueue:
    def test_escalate_creates_ticket(self, hitl_queue_fixture):
        tid = hitl_queue_fixture.escalate(
            query="What is the capital of France?",
            reason=EscalationReason.OUT_OF_SCOPE,
        )
        assert isinstance(tid, str) and len(tid) == 36  # UUID

    def test_get_ticket(self, hitl_queue_fixture):
        tid = hitl_queue_fixture.escalate(
            query="Sensitive query",
            reason=EscalationReason.SENSITIVE_TOPIC,
        )
        ticket = hitl_queue_fixture.get_ticket(tid)
        assert ticket is not None
        assert ticket["status"] == TicketStatus.PENDING

    def test_resolve_ticket(self, hitl_queue_fixture):
        tid = hitl_queue_fixture.escalate(
            query="Test query",
            reason=EscalationReason.LOW_CONFIDENCE,
        )
        success = hitl_queue_fixture.resolve(tid, "Here is the human answer.")
        assert success is True
        ticket = hitl_queue_fixture.get_ticket(tid)
        assert ticket["status"] == TicketStatus.RESOLVED
        assert ticket["response"] == "Here is the human answer."

    def test_resolve_nonexistent_ticket(self, hitl_queue_fixture):
        success = hitl_queue_fixture.resolve("nonexistent-id", "answer")
        assert success is False

    def test_pending_list(self, hitl_queue_fixture):
        hitl_queue_fixture.escalate("Q1", EscalationReason.LOW_CONFIDENCE)
        hitl_queue_fixture.escalate("Q2", EscalationReason.MISSING_CONTEXT)
        pending = hitl_queue_fixture.get_pending()
        assert len(pending) >= 2

    def test_stats(self, hitl_queue_fixture):
        stats = hitl_queue_fixture.stats()
        assert "total" in stats
        assert "pending" in stats
        assert "resolved" in stats

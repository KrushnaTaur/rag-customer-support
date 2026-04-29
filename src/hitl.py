"""
hitl.py — Human-in-the-Loop (HITL) Escalation Module
=======================================================
Responsibilities:
  - Persist escalated queries to a SQLite database
  - Allow human agents to resolve tickets
  - Track ticket status: PENDING → RESOLVED / REJECTED
  - Provide query interface for the agent dashboard

Schema:
    escalations(
        id          TEXT PRIMARY KEY,   -- UUID
        query       TEXT,               -- original user question
        context     TEXT,               -- retrieved chunks used
        reason      TEXT,               -- why escalation was triggered
        status      TEXT,               -- PENDING | RESOLVED | REJECTED
        response    TEXT,               -- human agent's answer
        created_at  DATETIME,
        resolved_at DATETIME
    )
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from sqlalchemy import Column, DateTime, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import settings
from src.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────
class TicketStatus(str, Enum):
    PENDING = "PENDING"
    RESOLVED = "RESOLVED"
    REJECTED = "REJECTED"


class EscalationReason(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    MISSING_CONTEXT = "missing_context"
    COMPLEX_QUERY = "complex_query"
    LLM_SIGNAL = "llm_escalation_signal"
    SENSITIVE_TOPIC = "sensitive_topic"
    OUT_OF_SCOPE = "out_of_scope"


# ─────────────────────────────────────────────────────────────────────────────
# ORM Model
# ─────────────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


class EscalationTicket(Base):
    """SQLAlchemy ORM model for escalation tickets."""

    __tablename__ = "escalations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    query = Column(Text, nullable=False)
    context = Column(Text, nullable=True)      # serialised retrieved chunks
    reason = Column(String(64), nullable=False)
    status = Column(String(16), nullable=False, default=TicketStatus.PENDING)
    response = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "context": self.context,
            "reason": self.reason,
            "status": self.status,
            "response": self.response,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Database Setup
# ─────────────────────────────────────────────────────────────────────────────
import os

def _get_engine():
    db_url = settings.hitl_db_url
    # Ensure data directory exists for SQLite
    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "")
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    engine = create_engine(db_url, echo=False, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


_engine = _get_engine()
SessionLocal = sessionmaker(bind=_engine, autoflush=True, autocommit=False)


# ─────────────────────────────────────────────────────────────────────────────
# HITL Queue Interface
# ─────────────────────────────────────────────────────────────────────────────
class HITLQueue:
    """
    Public interface for the HITL escalation queue.
    All database interactions happen through this class.
    """

    # ── Escalation ────────────────────────────────────────────────────────
    def escalate(
        self,
        query: str,
        reason: EscalationReason,
        context: Optional[List[str]] = None,
    ) -> str:
        """
        Create a new escalation ticket.

        Args:
            query:   The user's original question.
            reason:  Why the query is being escalated.
            context: List of retrieved chunk texts (if any).

        Returns:
            ticket_id (UUID string)
        """
        ticket_id = str(uuid.uuid4())
        context_str = "\n---\n".join(context) if context else ""

        with SessionLocal() as session:
            ticket = EscalationTicket(
                id=ticket_id,
                query=query,
                context=context_str,
                reason=reason.value,
                status=TicketStatus.PENDING,
            )
            session.add(ticket)
            session.commit()

        logger.warning(
            "HITL escalation created. ticket_id={} reason={} query='{}'",
            ticket_id,
            reason.value,
            query[:80],
        )
        return ticket_id

    # ── Resolution ────────────────────────────────────────────────────────
    def resolve(self, ticket_id: str, human_response: str) -> bool:
        """
        Resolve an escalation ticket with a human agent's answer.

        Args:
            ticket_id:      UUID of the ticket.
            human_response: The agent's answer to the user's query.

        Returns:
            True if resolved successfully, False if ticket not found.
        """
        with SessionLocal() as session:
            ticket = session.get(EscalationTicket, ticket_id)
            if not ticket:
                logger.error("Ticket not found: {}", ticket_id)
                return False

            ticket.status = TicketStatus.RESOLVED
            ticket.response = human_response
            ticket.resolved_at = datetime.now(timezone.utc)
            session.commit()

        logger.success("Ticket {} resolved by human agent.", ticket_id)
        return True

    def reject(self, ticket_id: str, reason: str = "") -> bool:
        """Mark a ticket as rejected (e.g., spam, duplicate)."""
        with SessionLocal() as session:
            ticket = session.get(EscalationTicket, ticket_id)
            if not ticket:
                return False
            ticket.status = TicketStatus.REJECTED
            ticket.response = reason
            ticket.resolved_at = datetime.now(timezone.utc)
            session.commit()
        logger.info("Ticket {} rejected.", ticket_id)
        return True

    # ── Queries ───────────────────────────────────────────────────────────
    def get_pending(self) -> List[dict]:
        """Return all pending tickets ordered by creation time."""
        with SessionLocal() as session:
            tickets = (
                session.query(EscalationTicket)
                .filter(EscalationTicket.status == TicketStatus.PENDING)
                .order_by(EscalationTicket.created_at.asc())
                .all()
            )
            return [t.to_dict() for t in tickets]

    def get_ticket(self, ticket_id: str) -> Optional[dict]:
        """Retrieve a single ticket by ID."""
        with SessionLocal() as session:
            ticket = session.get(EscalationTicket, ticket_id)
            return ticket.to_dict() if ticket else None

    def get_all(self, limit: int = 100, offset: int = 0) -> List[dict]:
        """Return all tickets with pagination."""
        with SessionLocal() as session:
            tickets = (
                session.query(EscalationTicket)
                .order_by(EscalationTicket.created_at.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )
            return [t.to_dict() for t in tickets]

    def stats(self) -> dict:
        """Return ticket count statistics."""
        with SessionLocal() as session:
            total = session.query(EscalationTicket).count()
            pending = session.query(EscalationTicket).filter_by(status=TicketStatus.PENDING).count()
            resolved = session.query(EscalationTicket).filter_by(status=TicketStatus.RESOLVED).count()
            rejected = session.query(EscalationTicket).filter_by(status=TicketStatus.REJECTED).count()
        return {
            "total": total,
            "pending": pending,
            "resolved": resolved,
            "rejected": rejected,
        }


# Module-level singleton
hitl_queue = HITLQueue()

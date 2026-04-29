"""
rag_workflow.py — LangGraph RAG Workflow Engine
================================================
Implements a stateful graph-based workflow with the following nodes:

    [input] → [intent] → [retrieval] → [llm] → [routing]
                                                    ↓              ↓
                                               [output]       [hitl]

Each node reads from and writes to a shared GraphState TypedDict.
Conditional edges enable intelligent routing to HITL escalation.
"""

from __future__ import annotations

import re
import statistics
from typing import Annotated, List, Optional, TypedDict

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import END, StateGraph

from src.config import settings
from src.hitl import EscalationReason, hitl_queue
from src.ingest import get_embedding_model, load_vector_store
from src.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# Graph State
# ─────────────────────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    """Shared state object passed between every node in the LangGraph."""

    # Input
    query: str
    session_id: Optional[str]

    # Intent classification
    intent: Optional[str]                     # FAQ | Technical | Billing | Unknown

    # Retrieval
    retrieved_chunks: Optional[List[str]]     # chunk texts
    chunk_scores: Optional[List[float]]       # cosine similarity scores
    source_pages: Optional[List[str]]         # source metadata

    # LLM
    llm_response: Optional[str]              # raw LLM output
    confidence: Optional[float]              # computed confidence score

    # Routing
    should_escalate: Optional[bool]
    escalation_reason: Optional[str]

    # Output
    ticket_id: Optional[str]                 # set only on HITL path
    final_answer: Optional[str]              # delivered to user


# ─────────────────────────────────────────────────────────────────────────────
# LLM Factory
# ─────────────────────────────────────────────────────────────────────────────
def _get_llm():
    """Return the configured LLM instance."""
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            google_api_key=settings.google_api_key,
        )
    elif settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            openai_api_key=settings.openai_api_key,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────────────────────────────────────
INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an intent classifier for a customer support system.
Classify the user query into exactly one of these categories:
- FAQ: General questions about products, services, policies
- Technical: Bug reports, how-to questions, setup issues
- Billing: Payments, refunds, subscriptions, pricing
- Unknown: Queries that don't fit the above or are out of scope

Respond with ONLY the category label. Nothing else."""),
    ("human", "{query}"),
])

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful, professional customer support assistant.

INSTRUCTIONS:
1. Answer the user's question using ONLY the provided context below.
2. If the context does not contain sufficient information, respond with exactly: ESCALATE
3. Do NOT invent facts, prices, dates, or policies not mentioned in the context.
4. Be concise and helpful. Use bullet points for multi-step answers.
5. If you cite specific information, mention the source page (e.g., "According to page 3, ...").

CONTEXT:
{context}
"""),
    ("human", "{query}"),
])


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Confidence Scoring
# ─────────────────────────────────────────────────────────────────────────────
def compute_confidence(chunk_scores: List[float], llm_response: str) -> float:
    """
    Compute a confidence score from 0.0 to 1.0.

    Factors:
    - Average cosine similarity of retrieved chunks
    - LLM explicitly signals ESCALATE → 0.0
    - No chunks found → 0.0
    """
    if not chunk_scores:
        return 0.0
    if "ESCALATE" in (llm_response or "").upper():
        return 0.0
    return round(statistics.mean(chunk_scores), 4)


SENSITIVE_KEYWORDS = re.compile(
    r"\b(sue|lawsuit|legal action|attorney|lawyer|court|refund fraud"
    r"|chargeback|police|illegal|death|injury|medical|diagnosis)\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Node Implementations
# ─────────────────────────────────────────────────────────────────────────────

def input_node(state: GraphState) -> GraphState:
    """
    Node 1 — Input Validation & Sanitisation
    Cleans the user query. Initialises optional state fields.
    """
    query = state.get("query", "").strip()
    # Normalise whitespace, strip null bytes
    query = re.sub(r"\s+", " ", query.replace("\x00", ""))

    logger.debug("input_node: query='{}'", query[:80])

    return {
        **state,
        "query": query,
        "intent": None,
        "retrieved_chunks": None,
        "chunk_scores": None,
        "source_pages": None,
        "llm_response": None,
        "confidence": None,
        "should_escalate": False,
        "escalation_reason": None,
        "ticket_id": None,
        "final_answer": None,
    }


def intent_node(state: GraphState) -> GraphState:
    """
    Node 2 — Intent Classification
    Classifies the query into FAQ / Technical / Billing / Unknown.
    """
    query = state["query"]
    if not query:
        return {**state, "intent": "Unknown"}

    try:
        llm = _get_llm()
        chain = INTENT_PROMPT | llm
        result = chain.invoke({"query": query})
        intent = result.content.strip()

        valid_intents = {"FAQ", "Technical", "Billing", "Unknown"}
        if intent not in valid_intents:
            intent = "Unknown"

        logger.debug("intent_node: intent='{}'", intent)
    except Exception as e:
        logger.warning("intent_node failed: {}. Defaulting to Unknown.", e)
        intent = "Unknown"

    return {**state, "intent": intent}


def retrieval_node(state: GraphState) -> GraphState:
    """
    Node 3 — Semantic Retrieval from ChromaDB
    Fetches top-k chunks using MMR for diversity.
    """
    query = state["query"]

    try:
        vector_store = load_vector_store()
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.retrieval_top_k,
                "fetch_k": settings.retrieval_top_k * 3,
                "lambda_mult": 0.7,  # 0 = diversity, 1 = relevance
            },
        )
        docs: List[Document] = retriever.invoke(query)

        # Also get similarity scores for confidence calculation
        scored = vector_store.similarity_search_with_relevance_scores(
            query, k=settings.retrieval_top_k
        )

        chunks = [doc.page_content for doc in docs]
        scores = [score for _, score in scored]
        pages = [
            f"{doc.metadata.get('source', 'unknown')} p.{doc.metadata.get('page', '?')}"
            for doc in docs
        ]

        logger.debug(
            "retrieval_node: {} chunks retrieved. avg_score={:.3f}",
            len(chunks),
            statistics.mean(scores) if scores else 0,
        )
    except Exception as e:
        logger.error("retrieval_node error: {}", e)
        chunks, scores, pages = [], [], []

    return {
        **state,
        "retrieved_chunks": chunks,
        "chunk_scores": scores,
        "source_pages": pages,
    }


def llm_node(state: GraphState) -> GraphState:
    """
    Node 4 — LLM Response Generation
    Builds a RAG prompt from retrieved context and invokes the LLM.
    """
    query = state["query"]
    chunks = state.get("retrieved_chunks") or []
    scores = state.get("chunk_scores") or []

    if not chunks:
        logger.warning("llm_node: No chunks available. Marking for escalation.")
        return {
            **state,
            "llm_response": "ESCALATE",
            "confidence": 0.0,
        }

    context = "\n\n---\n\n".join(
        f"[Source: {state['source_pages'][i]}]\n{chunk}"
        for i, chunk in enumerate(chunks)
    )

    try:
        llm = _get_llm()
        chain = RAG_PROMPT | llm
        result = chain.invoke({"context": context, "query": query})
        llm_response = result.content.strip()
    except Exception as e:
        logger.error("llm_node error: {}", e)
        llm_response = "ESCALATE"

    confidence = compute_confidence(scores, llm_response)

    logger.debug(
        "llm_node: confidence={:.3f}, escalate={}",
        confidence,
        "ESCALATE" in llm_response.upper(),
    )

    return {**state, "llm_response": llm_response, "confidence": confidence}


def routing_node(state: GraphState) -> GraphState:
    """
    Node 5 — Conditional Routing Logic
    Decides whether to deliver the answer or escalate to HITL.

    Escalation triggers:
      1. confidence < threshold
      2. No chunks retrieved (missing context)
      3. LLM response contains ESCALATE token
      4. Query matches sensitive topic regex
      5. Overly complex query (multi-question or very long)
    """
    query = state["query"]
    confidence = state.get("confidence") or 0.0
    chunks = state.get("retrieved_chunks") or []
    llm_response = state.get("llm_response") or ""

    should_escalate = False
    reason = None

    # Trigger 1: Low confidence
    if confidence < settings.confidence_threshold:
        should_escalate = True
        reason = EscalationReason.LOW_CONFIDENCE.value
        logger.info("Routing → HITL: low confidence ({:.3f})", confidence)

    # Trigger 2: Missing context
    elif not chunks:
        should_escalate = True
        reason = EscalationReason.MISSING_CONTEXT.value
        logger.info("Routing → HITL: no relevant chunks found")

    # Trigger 3: LLM signals escalation
    elif "ESCALATE" in llm_response.upper():
        should_escalate = True
        reason = EscalationReason.LLM_SIGNAL.value
        logger.info("Routing → HITL: LLM signalled insufficient context")

    # Trigger 4: Sensitive topic
    elif SENSITIVE_KEYWORDS.search(query):
        should_escalate = True
        reason = EscalationReason.SENSITIVE_TOPIC.value
        logger.info("Routing → HITL: sensitive topic detected")

    # Trigger 5: Complex / out-of-scope query
    elif query.count("?") > 2 or len(query.split()) > 60:
        should_escalate = True
        reason = EscalationReason.COMPLEX_QUERY.value
        logger.info("Routing → HITL: complex query detected")

    else:
        logger.info("Routing → Answer (confidence={:.3f})", confidence)

    return {**state, "should_escalate": should_escalate, "escalation_reason": reason}


def output_node(state: GraphState) -> GraphState:
    """
    Node 6a — Answer Output
    Formats the final answer with source citations.
    """
    llm_response = state.get("llm_response", "")
    pages = state.get("source_pages") or []
    confidence = state.get("confidence", 0.0)

    # Build source footer
    unique_pages = list(dict.fromkeys(pages))  # deduplicate preserving order
    source_footer = ""
    if unique_pages:
        source_footer = f"\n\n📄 *Sources: {', '.join(unique_pages)}*"

    final_answer = llm_response + source_footer

    logger.info("output_node: answer delivered (confidence={:.3f})", confidence)
    return {**state, "final_answer": final_answer}


def hitl_node(state: GraphState) -> GraphState:
    """
    Node 6b — HITL Escalation
    Persists the escalation to the queue and returns an acknowledgement.
    """
    query = state["query"]
    chunks = state.get("retrieved_chunks") or []
    reason_str = state.get("escalation_reason") or EscalationReason.OUT_OF_SCOPE.value

    # Map string reason to enum safely
    try:
        reason = EscalationReason(reason_str)
    except ValueError:
        reason = EscalationReason.OUT_OF_SCOPE

    ticket_id = hitl_queue.escalate(
        query=query,
        reason=reason,
        context=chunks,
    )

    final_answer = (
        f"🙋 Your query has been forwarded to a customer support specialist.\n\n"
        f"**Ticket ID:** `{ticket_id}`\n\n"
        f"You will receive a response shortly. Thank you for your patience!"
    )

    return {**state, "ticket_id": ticket_id, "final_answer": final_answer}


# ─────────────────────────────────────────────────────────────────────────────
# Conditional Edge Router
# ─────────────────────────────────────────────────────────────────────────────
def route_after_routing(state: GraphState) -> str:
    """
    LangGraph conditional edge function.
    Returns the name of the next node to execute.
    """
    if state.get("should_escalate"):
        return "hitl"
    return "output"


# ─────────────────────────────────────────────────────────────────────────────
# Graph Assembly
# ─────────────────────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    """Assemble and compile the LangGraph StateGraph."""

    workflow = StateGraph(GraphState)

    # Register nodes
    workflow.add_node("input", input_node)
    workflow.add_node("intent", intent_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("llm", llm_node)
    workflow.add_node("routing", routing_node)
    workflow.add_node("output", output_node)
    workflow.add_node("hitl", hitl_node)

    # Define linear edges
    workflow.set_entry_point("input")
    workflow.add_edge("input", "intent")
    workflow.add_edge("intent", "retrieval")
    workflow.add_edge("retrieval", "llm")
    workflow.add_edge("llm", "routing")

    # Conditional branching at routing node
    workflow.add_conditional_edges(
        "routing",
        route_after_routing,
        {
            "output": "output",
            "hitl": "hitl",
        },
    )

    # Terminal edges
    workflow.add_edge("output", END)
    workflow.add_edge("hitl", END)

    return workflow.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
class RAGAssistant:
    """
    High-level interface for the RAG Customer Support Assistant.
    Wraps the compiled LangGraph workflow.
    """

    def __init__(self):
        logger.info("Initialising RAG Assistant...")
        self._graph = build_graph()
        logger.success("RAG Assistant ready.")

    def query(self, user_query: str, session_id: Optional[str] = None) -> dict:
        """
        Process a user query through the full RAG pipeline.

        Args:
            user_query:  Natural language question from the user.
            session_id:  Optional session identifier for tracking.

        Returns:
            dict with keys: answer, confidence, intent, escalated, ticket_id
        """
        if not user_query or not user_query.strip():
            return {
                "answer": "Please enter a valid question.",
                "confidence": 0.0,
                "intent": None,
                "escalated": False,
                "ticket_id": None,
            }

        initial_state: GraphState = {
            "query": user_query,
            "session_id": session_id,
            "intent": None,
            "retrieved_chunks": None,
            "chunk_scores": None,
            "source_pages": None,
            "llm_response": None,
            "confidence": None,
            "should_escalate": False,
            "escalation_reason": None,
            "ticket_id": None,
            "final_answer": None,
        }

        final_state = self._graph.invoke(initial_state)

        return {
            "answer": final_state.get("final_answer", "No answer generated."),
            "confidence": final_state.get("confidence", 0.0),
            "intent": final_state.get("intent"),
            "escalated": final_state.get("should_escalate", False),
            "ticket_id": final_state.get("ticket_id"),
        }

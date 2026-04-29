"""
config.py — Centralised application configuration using Pydantic Settings.
All values are loaded from environment variables or .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # ── LLM ───────────────────────────────────────────────────────────────
    google_api_key: str = Field(default="", env="GOOGLE_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    llm_provider: str = Field(default="gemini", env="LLM_PROVIDER")   # gemini | openai
    llm_model: str = Field(default="gemini-1.5-flash", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.0, env="LLM_TEMPERATURE")

    # ── Vector Store ──────────────────────────────────────────────────────
    chroma_persist_dir: str = Field(default="./data/chroma_db", env="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(default="customer_support_kb", env="CHROMA_COLLECTION_NAME")

    # ── Embeddings ────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )

    # ── Chunking ──────────────────────────────────────────────────────────
    chunk_size: int = Field(default=500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, env="CHUNK_OVERLAP")
    retrieval_top_k: int = Field(default=4, env="RETRIEVAL_TOP_K")

    # ── RAG Routing ───────────────────────────────────────────────────────
    confidence_threshold: float = Field(default=0.60, env="CONFIDENCE_THRESHOLD")

    # ── HITL ──────────────────────────────────────────────────────────────
    hitl_db_url: str = Field(default="sqlite:///./data/hitl_queue.db", env="HITL_DB_URL")

    # ── API Server ────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_reload: bool = Field(default=True, env="API_RELOAD")

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./logs/app.log", env="LOG_FILE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance (singleton)."""
    return Settings()


# Module-level singleton for convenience
settings = get_settings()

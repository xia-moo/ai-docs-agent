import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True)
class Settings:
    root: Path
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = "BAAI/bge-reranker-base"
    threads: int = 4
    rerank_batch: int = 2
    llm_key: str = ""
    llm_base: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-flash"
    tavily_key: str = ""
    answer_mode: str = "auto"

    @classmethod
    def read(cls, root: Path | None = None):
        root = (root or Path(os.environ.get("AI_DOCS_ROOT", Path(__file__).resolve().parents[2]))).resolve()
        values = {**dotenv_values(root / ".env"), **os.environ}
        return cls(
            root=root,
            embedding_model=values.get("EMBEDDING_MODEL") or cls.embedding_model,
            reranker_model=values.get("RERANKER_MODEL") or cls.reranker_model,
            threads=max(1, min(8, int(values.get("MODEL_THREADS") or "4"))),
            rerank_batch=max(1, min(4, int(values.get("RERANK_BATCH_SIZE") or "2"))),
            llm_key=values.get("LLM_API_KEY") or "",
            llm_base=(values.get("LLM_BASE_URL") or cls.llm_base).rstrip("/"),
            llm_model=values.get("LLM_MODEL") or cls.llm_model,
            tavily_key=values.get("TAVILY_API_KEY") or "",
            answer_mode=values.get("ANSWER_MODE") or "auto",
        )

    def public(self):
        return {
            "llm_configured": bool(self.llm_key),
            "search_configured": bool(self.tavily_key),
            "llm_base_url": self.llm_base,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "answer_mode": self.answer_mode,
        }

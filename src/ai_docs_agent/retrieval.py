"""BM25, dense cosine retrieval, RRF and bounded-batch cross-encoder reranking."""

import hashlib
import json
import logging
import re
import uuid
from collections import defaultdict
from threading import RLock
from time import perf_counter

import jieba
import numpy as np
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from .documents import parse_document
from .settings import Settings

jieba.setLogLevel(logging.WARNING)
STOPWORDS = {
    "的",
    "了",
    "是",
    "在",
    "和",
    "与",
    "如何",
    "怎么",
    "什么",
    "请",
    "帮",
    "我",
    "how",
    "what",
    "is",
    "the",
    "a",
    "an",
    "to",
    "of",
    "and",
    "does",
}


def tokenize(text):
    # Preserve technical identifiers such as session_state, C++, and bge-m3.
    terms = re.findall(r"[a-zA-Z0-9_]+(?:[.+/-][a-zA-Z0-9_]+)*\+*|[\u3400-\u9fff]+", text.lower())
    result = []
    for term in terms:
        words = jieba.lcut(term) if re.fullmatch(r"[\u3400-\u9fff]+", term) else [term]
        result.extend(word for word in words if word.strip() and word not in STOPWORDS)
    return result


def reciprocal_rank_fusion(rankings, limit=20, constant=60):
    scores, evidence = defaultdict(float), {}
    for ranking in rankings:
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            if hit["id"] in seen:
                continue
            seen.add(hit["id"])
            scores[hit["id"]] += 1 / (constant + rank)
            evidence.setdefault(hit["id"], hit)
    return [
        dict(evidence[key], score=scores[key], retrieval_method="hybrid")
        for key in sorted(scores, key=lambda item: (-scores[item], item))[:limit]
    ]


class RetrievalEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.root
        self.docs_dir = self.root / "data" / "documents"
        self.index_dir = self.root / "data" / "index"
        for directory in (self.docs_dir, self.index_dir, self.root / "data" / "models"):
            directory.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.client = QdrantClient(path=str(self.index_dir / "qdrant"))
        self.embedding = None
        self.reranker = None
        self.bundle = {}
        self.chunks = []
        self.bm25 = None
        self.tokens = []
        saved = self.index_dir / "index.json"
        if saved.exists():
            self._activate(json.loads(saved.read_text(encoding="utf-8")))
        self.last_build = self.bundle.get("built_at")

    def _activate(self, bundle):
        self.bundle = bundle
        self.chunks = bundle.get("chunks", [])
        self.tokens = [tokenize(chunk["section"] + " " + chunk["text"]) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokens) if self.tokens and any(self.tokens) else None

    def _embedder(self):
        if self.embedding is None:
            from fastembed import TextEmbedding

            self.embedding = TextEmbedding(
                self.settings.embedding_model,
                cache_dir=str(self.root / "data" / "models"),
                threads=self.settings.threads,
            )
        return self.embedding

    def build(self, chunk_size=350, overlap=60):
        from datetime import datetime, timezone

        with self.lock:
            sources_file = self.root / "data" / "sources.json"
            sources = json.loads(sources_file.read_text(encoding="utf-8")) if sources_file.exists() else {}
            chunks, documents = [], []
            for path in sorted(self.docs_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}:
                    metadata = sources.get(path.name, {})
                    parsed = parse_document(path, chunk_size, overlap, metadata.get("url"))
                    chunks.extend(chunk.to_dict() for chunk in parsed)
                    documents.append(
                        {
                            "name": path.name,
                            "chunks": len(parsed),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "source_url": metadata.get("url"),
                        }
                    )
            collection = "docs_" + uuid.uuid4().hex
            try:
                if chunks:
                    embedder = self._embedder()
                    vectors = list(embedder.passage_embed([chunk["text"] for chunk in chunks], batch_size=16))
                    self.client.create_collection(
                        collection,
                        vectors_config=models.VectorParams(
                            size=len(vectors[0]), distance=models.Distance.COSINE
                        ),
                    )
                    self.client.upsert(
                        collection,
                        points=[
                            models.PointStruct(
                                id=str(uuid.UUID(chunk["id"])), vector=vector.tolist(), payload=chunk
                            )
                            for chunk, vector in zip(chunks, vectors, strict=True)
                        ],
                    )
                bundle = {
                    "chunks": chunks,
                    "documents": documents,
                    "collection": collection if chunks else None,
                    "embedding_model": self.settings.embedding_model,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "built_at": datetime.now(timezone.utc).isoformat(),
                    "revision": uuid.uuid4().hex,
                }
                destination = self.index_dir / "index.json"
                temporary = destination.with_suffix(".tmp")
                temporary.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(destination)
            except Exception:
                if self.client.collection_exists(collection):
                    self.client.delete_collection(collection)
                raise
            old_collection = self.bundle.get("collection")
            self._activate(bundle)
            self.last_build = bundle["built_at"]
            if old_collection and old_collection != bundle["collection"]:
                self.client.delete_collection(old_collection)
            return self.summary()

    def summary(self):
        return {
            "document_count": len(self.bundle.get("documents", [])),
            "chunk_count": len(self.chunks),
            "documents": self.bundle.get("documents", []),
            "revision": self.bundle.get("revision"),
            "built_at": self.bundle.get("built_at"),
            "embedding_model": self.settings.embedding_model,
            "chunk_size": self.bundle.get("chunk_size", 350),
            "overlap": self.bundle.get("overlap", 60),
        }

    def search(self, question, method="rerank", k=5, candidates=20):
        if method not in {"dense", "bm25", "hybrid", "rerank"} or not 1 <= k <= candidates <= 40:
            raise ValueError("检索参数无效。")
        with self.lock:
            started = perf_counter()
            if not self.chunks:
                return {
                    "hits": [],
                    "latency_ms": 0,
                    "method": method,
                    "revision": self.bundle.get("revision"),
                }
            query_terms = tokenize(question)
            lexical = []
            if method != "dense" and self.bm25 and query_terms:
                scores = self.bm25.get_scores(query_terms)
                for idx in np.argsort(-scores, kind="stable"):
                    # Include matching terms even if this tiny corpus gives a negative IDF score.
                    if set(query_terms).intersection(self.tokens[idx]):
                        lexical.append(
                            dict(self.chunks[idx], score=float(scores[idx]), retrieval_method="bm25")
                        )
                        if len(lexical) >= candidates:
                            break
            dense = []
            if method != "bm25":
                if self.bundle["embedding_model"] != self.settings.embedding_model:
                    raise ValueError("向量模型与现有索引不一致，请重新建立索引。")
                vector = list(self._embedder().query_embed(question))[0]
                points = self.client.query_points(
                    self.bundle["collection"],
                    query=vector.tolist(),
                    limit=candidates,
                    with_payload=True,
                ).points
                dense = [
                    dict(point.payload, score=float(point.score), retrieval_method="dense")
                    for point in points
                ]
            if method == "bm25":
                hits = lexical[:k]
            elif method == "dense":
                hits = dense[:k]
            else:
                hits = reciprocal_rank_fusion([dense, lexical], candidates)
                if method == "rerank" and hits:
                    if self.reranker is None:
                        from fastembed.rerank.cross_encoder import TextCrossEncoder

                        self.reranker = TextCrossEncoder(
                            self.settings.reranker_model,
                            cache_dir=str(self.root / "data" / "models"),
                            threads=self.settings.threads,
                        )
                    reranked = list(
                        self.reranker.rerank(
                            question,
                            [hit["text"] for hit in hits],
                            batch_size=self.settings.rerank_batch,
                        )
                    )
                    for hit, score in zip(hits, reranked, strict=True):
                        hit.update(rerank_score=float(score), retrieval_method="rerank")
                    hits.sort(key=lambda hit: (-hit["rerank_score"], hit["id"]))
                hits = hits[:k]
            return {
                "hits": hits,
                "latency_ms": round((perf_counter() - started) * 1000, 2),
                "method": method,
                "revision": self.bundle.get("revision"),
            }

    def close(self):
        self.client.close()

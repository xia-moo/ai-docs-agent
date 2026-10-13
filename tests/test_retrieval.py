import numpy as np
import pytest
from ai_docs_agent.retrieval import RetrievalEngine, reciprocal_rank_fusion, tokenize
from ai_docs_agent.settings import Settings


class FakeEmbedding:
    def passage_embed(self, texts, batch_size=16):
        return [np.array([1, 1 if "状态" in text else 0, 0.2], dtype=float) for text in texts]

    def query_embed(self, text):
        return [np.array([1, 1, 0.2], dtype=float)]


def test_technical_tokens_and_chinese():
    assert "session_state" in tokenize("如何使用 session_state？")
    assert "bge-m3" in tokenize("BGE-M3 + C++")
    assert "c++" in tokenize("BGE-M3 + C++")
    assert "如何" not in tokenize("如何检索答案？")


def test_rrf_agreement_and_duplicate():
    a, b = {"id": "a"}, {"id": "b"}
    result = reciprocal_rank_fusion([[a, a, b], [b]], 2)
    assert result[0]["id"] == "b"
    single = reciprocal_rank_fusion([[a, a]], 2)
    assert single[0]["score"] == pytest.approx(1 / 61)


def test_atomic_rebuild_and_persistent_index(tmp_path):
    engine = RetrievalEngine(Settings(tmp_path))
    engine.embedding = FakeEmbedding()
    document = engine.docs_dir / "sample.txt"
    document.write_text("状态 checkpoint 支持恢复运行。", encoding="utf-8")
    engine.build()
    original = engine.bundle["revision"]
    assert engine.search("checkpoint", "bm25")["hits"]
    assert not engine.search("量子计算机", "bm25")["hits"]
    assert engine.search("状态", "dense")["hits"][0]["source"] == "sample.txt"
    document.write_text("向量数据库存储 embeddings。", encoding="utf-8")
    engine.build()
    assert engine.bundle["revision"] != original
    assert not engine.search("checkpoint", "bm25")["hits"]
    engine.close()
    reopened = RetrievalEngine(Settings(tmp_path))
    assert reopened.search("embeddings", "bm25")["hits"]
    reopened.close()


def test_failed_build_preserves_previous_index(tmp_path):
    engine = RetrievalEngine(Settings(tmp_path))
    engine.embedding = FakeEmbedding()
    (engine.docs_dir / "valid.txt").write_text("可恢复状态。", encoding="utf-8")
    engine.build()
    previous = engine.bundle["revision"]
    (engine.docs_dir / "broken.txt").write_bytes(b"\xff")
    with pytest.raises(ValueError):
        engine.build()
    assert engine.bundle["revision"] == previous
    assert engine.search("状态", "bm25")["hits"]
    engine.close()

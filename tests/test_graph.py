from pathlib import Path

from ai_docs_agent.graph import choose_route, create_graph
from ai_docs_agent.settings import Settings

HIT = {
    "id": "a",
    "text": "checkpoint 保存状态并支持恢复。",
    "source": "guide.md",
    "section": "状态",
    "page": None,
    "score": 0.7,
    "start": 0,
    "end": 24,
}


class Engine:
    root = Path(".")

    def search(self, *args):
        return {"hits": [HIT], "latency_ms": 1}


def invoke(graph, **overrides):
    return graph.invoke(
        {
            "question": "checkpoint 如何保存状态？",
            "requested_route": "auto",
            "method": "hybrid",
            "k": 3,
            "candidates": 10,
            "answer_mode": "auto",
            "trace": [],
            "warnings": [],
            **overrides,
        }
    )


def test_real_graph_excerpts_and_provenance():
    graph = create_graph(Engine(), lambda: Settings(Path(".")))
    result = invoke(graph)
    assert result["mode"] == "extractive"
    assert result["citations"][0]["source"] == "guide.md"
    assert [step["node"] for step in result["trace"]] == ["路由", "知识库检索", "证据判断", "组织回答"]


def test_live_route_does_not_fabricate_results():
    graph = create_graph(Engine(), lambda: Settings(Path(".")))
    result = invoke(graph, question="PyTorch 最新版本是什么？")
    assert result["status"] == "needs_search_configuration"
    assert result["citations"] == []
    assert choose_route("最新版本", "knowledge") == "knowledge"


def test_model_grade_triggers_web_fallback(monkeypatch):
    from ai_docs_agent import graph as module

    monkeypatch.setattr(module, "grade_evidence", lambda *args: (False, "证据不足"))
    monkeypatch.setattr(
        module, "web_search", lambda *args: [dict(HIT, source="官网", source_url="https://example.org")]
    )
    monkeypatch.setattr(module, "generate_answer", lambda *args: "保存状态。[1]")
    graph = create_graph(Engine(), lambda: Settings(Path("."), llm_key="test", tavily_key="test"))
    result = invoke(graph)
    assert result["route"] == "web"
    assert result["mode"] == "generated"
    assert result["citations"][0]["source_url"] == "https://example.org"


def test_explicit_llm_mode_without_key_is_visible():
    graph = create_graph(Engine(), lambda: Settings(Path(".")))
    result = invoke(graph, answer_mode="llm")
    assert result["status"] == "needs_model_configuration"


def test_insufficient_evidence_cannot_be_forced_into_generated_answer(monkeypatch):
    from ai_docs_agent import graph as module

    monkeypatch.setattr(module, "grade_evidence", lambda *args: (False, "证据不足"))

    def should_not_generate(*args):
        raise AssertionError("Insufficient evidence must not be sent for answer generation")

    monkeypatch.setattr(module, "generate_answer", should_not_generate)
    graph = create_graph(Engine(), lambda: Settings(Path("."), llm_key="test"))
    result = invoke(graph, requested_route="knowledge")
    assert result["status"] == "insufficient_evidence"

"""A real LangGraph: routing -> retrieval -> evidence grading -> optional web -> answer."""

import re
from operator import add
from time import perf_counter
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from .providers import ProviderError, generate_answer, grade_evidence, web_search
from .retrieval import tokenize
from .settings import Settings


class QueryState(TypedDict, total=False):
    question: str
    requested_route: str
    route: str
    method: str
    k: int
    candidates: int
    answer_mode: str
    sources: list[dict]
    answer: str
    citations: list[dict]
    status: str
    mode: str
    sufficient: bool
    retrieval_ms: float
    index_revision: str | None
    trace: Annotated[list[dict], add]
    warnings: Annotated[list[str], add]


def choose_route(question, requested="auto"):
    if requested != "auto":
        return requested
    return (
        "web"
        if re.search(r"最新|今天|实时|新闻|当前版本|latest|today|news|current version", question, re.I)
        else "knowledge"
    )


def _trace(node, detail, started):
    return [{"node": node, "detail": detail, "latency_ms": round((perf_counter() - started) * 1000, 2)}]


def create_graph(engine, settings_reader=None):
    settings_reader = settings_reader or (lambda: Settings.read(engine.root))

    def route(state):
        started = perf_counter()
        destination = choose_route(state["question"], state.get("requested_route", "auto"))
        return {
            "route": destination,
            "trace": _trace("路由", f"选择{'联网搜索' if destination == 'web' else '知识库'}", started),
        }

    def retrieve(state):
        started = perf_counter()
        result = engine.search(state["question"], state["method"], state["k"], state["candidates"])
        return {
            "sources": result["hits"],
            "retrieval_ms": result["latency_ms"],
            "index_revision": result.get("revision"),
            "trace": _trace("知识库检索", f"{state['method']}，召回 {len(result['hits'])} 个片段", started),
        }

    def grade(state):
        started = perf_counter()
        hits, settings = state.get("sources", []), settings_reader()
        if not hits:
            sufficient, reason = False, "知识库尚无候选片段。"
        elif (
            settings.llm_key
            and state.get("answer_mode", "auto") != "extractive"
            and settings.answer_mode != "extractive"
        ):
            try:
                sufficient, reason = grade_evidence(settings, state["question"], hits)
            except ProviderError as exc:
                return {
                    "sufficient": True,
                    "warnings": [str(exc)],
                    "trace": _trace("证据判断", "模型判断不可用，返回候选片段供核对。", started),
                }
        else:
            terms = set(tokenize(state["question"]))
            lexical_match = any(terms.intersection(tokenize(hit["text"])) for hit in hits)
            # This is a transparent candidate filter, not a claim that the answer is supported.
            sufficient = lexical_match or (state["method"] == "dense" and hits[0]["score"] >= 0.65)
            reason = (
                "候选片段存在词汇匹配，充分性尚未由模型验证。"
                if sufficient
                else "没有明显匹配，尝试联网补充。"
            )
        return {"sufficient": sufficient, "trace": _trace("证据判断", reason, started)}

    def after_grade(state):
        if not state["sufficient"] and state.get("requested_route", "auto") == "auto":
            return "web"
        return "answer"

    def search_web(state):
        started = perf_counter()
        settings = settings_reader()
        if not settings.tavily_key:
            return {
                "sources": [],
                "status": "needs_search_configuration",
                "route": "web",
                "warnings": ["请在设置中配置 Tavily API 密钥以使用联网搜索。"],
                "trace": _trace("联网搜索", "缺少 API 配置，未发起外部请求。", started),
            }
        try:
            hits = web_search(settings, state["question"], state["k"])
            return {
                "sources": hits,
                "route": "web",
                "trace": _trace("联网搜索", f"获取 {len(hits)} 个网页来源", started),
            }
        except ProviderError as exc:
            return {
                "sources": [],
                "status": "search_failed",
                "route": "web",
                "warnings": [str(exc)],
                "trace": _trace("联网搜索", "搜索服务不可用。", started),
            }

    def answer(state):
        started = perf_counter()
        hits, settings = state.get("sources", []), settings_reader()
        requested_mode = state.get("answer_mode", "auto")
        if requested_mode == "auto":
            requested_mode = settings.answer_mode
        use_llm = requested_mode == "llm" or (requested_mode == "auto" and bool(settings.llm_key))
        citations = [dict(hit, number=i) for i, hit in enumerate(hits, 1)]
        warnings = []
        status = state.get("status", "ok")
        if not hits:
            text, mode = "没有找到可用证据。请调整问题、补充文档，或配置联网搜索。", "no_evidence"
            if status == "ok":
                status = "no_evidence"
        elif state.get("sufficient") is False and state["route"] != "web":
            text, mode, status = (
                "当前知识库证据不足以回答这个问题。下方保留候选片段供核对。",
                "insufficient_evidence",
                "insufficient_evidence",
            )
        elif use_llm and not settings.llm_key:
            text, mode, status = (
                "模型回答尚未配置。请填写 API 密钥；下方保留检索片段供核对。",
                "needs_configuration",
                "needs_model_configuration",
            )
        elif use_llm:
            try:
                text, mode = generate_answer(settings, state["question"], hits), "generated"
            except ProviderError as exc:
                text, mode, status = excerpts(hits), "extractive_fallback", "generation_failed"
                warnings.append(str(exc))
        else:
            text, mode = excerpts(hits), "extractive"
        return {
            "answer": text,
            "mode": mode,
            "status": status,
            "citations": citations,
            "warnings": warnings,
            "trace": _trace("组织回答", mode, started),
        }

    graph = StateGraph(QueryState)
    graph.add_node("route", route)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("web", search_web)
    graph.add_node("answer", answer)
    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route", lambda state: state["route"], {"knowledge": "retrieve", "web": "web"}
    )
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", after_grade, {"web": "web", "answer": "answer"})
    graph.add_edge("web", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def excerpts(hits):
    return "原文摘录（排名第一的候选片段，尚未由生成模型综合回答）：\n\n" + f"[1] {hits[0]['text'].strip()}"

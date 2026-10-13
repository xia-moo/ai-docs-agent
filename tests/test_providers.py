import json
from pathlib import Path

import httpx
import pytest
from ai_docs_agent import providers
from ai_docs_agent.providers import ProviderError, generate_answer, safe_url
from ai_docs_agent.settings import Settings


def test_generated_citation_validation(monkeypatch):
    hits = [{"source": "doc", "text": "事实", "section": "正文"}]
    monkeypatch.setattr(providers, "chat", lambda *args, **kwargs: "编造引用 [99]")
    with pytest.raises(ProviderError, match="引用"):
        generate_answer(Settings(Path(".")), "问题", hits)
    monkeypatch.setattr(providers, "chat", lambda *args, **kwargs: "没有引用")
    with pytest.raises(ProviderError):
        generate_answer(Settings(Path(".")), "问题", hits)
    assert safe_url("javascript:alert(1)") is None


def test_grade_requires_boolean(monkeypatch):
    monkeypatch.setattr(providers, "chat", lambda *args, **kwargs: '{"sufficient":"true","reason":"x"}')
    with pytest.raises(ProviderError):
        providers.grade_evidence(Settings(Path(".")), "问题", [])


def test_http_generation_contract_and_sanitized_errors(monkeypatch):
    original_client = httpx.Client
    settings = Settings(Path("."), llm_key="dummy-local-test", llm_model="deepseek-flash")

    def handler(request):
        assert request.url.path == "/chat/completions"
        assert request.headers["authorization"] == "Bearer dummy-local-test"
        body = json.loads(request.content)
        assert body["model"] == "deepseek-flash"
        assert body["thinking"]["type"] == "disabled"
        return httpx.Response(200, json={"choices": [{"message": {"content": "根据证据回答。[1]"}}]})

    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(handler))
    )
    answer = generate_answer(settings, "问题", [{"source": "x", "text": "内容"}])
    assert "[1]" in answer

    def failure(request):
        return httpx.Response(401, text="dummy-local-test")

    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(failure))
    )
    with pytest.raises(ProviderError) as error:
        providers.chat(settings, [])
    assert "dummy-local-test" not in str(error.value)


def test_tavily_http_contract_and_safe_urls(monkeypatch):
    original_client = httpx.Client

    def handler(request):
        assert request.url.host == "api.tavily.com"
        assert request.headers["authorization"] == "Bearer dummy-search"
        assert json.loads(request.content)["include_answer"] is False
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "bad", "url": "javascript:alert(1)", "content": "bad"},
                    {"title": "官网", "url": "https://example.org", "content": "资料", "score": 0.9},
                ]
            },
        )

    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(handler))
    )
    hits = providers.web_search(Settings(Path("."), tavily_key="dummy-search"), "今天的消息")
    assert len(hits) == 1
    assert hits[0]["source_url"] == "https://example.org"

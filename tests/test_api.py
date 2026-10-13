from threading import RLock

import pytest
from ai_docs_agent.api import checked_name, create_app
from fastapi import HTTPException
from fastapi.testclient import TestClient


@pytest.mark.parametrize("name", ["NUL.txt", "CON.md", "a?.txt", "../x.txt", "a.txt "])
def test_invalid_windows_filenames(name):
    with pytest.raises(HTTPException):
        checked_name(name)


class FakeEngine:
    def __init__(self, settings):
        self.root = settings.root
        self.docs_dir = self.root / "data" / "documents"
        self.docs_dir.mkdir(parents=True)
        self.lock = RLock()
        self.bundle = {}

    def summary(self):
        return {"documents": [], "document_count": 0, "chunk_count": 0}

    def build(self, *args):
        return self.summary()

    def search(self, *args):
        return {"hits": [], "latency_ms": 0}

    def close(self):
        pass


def test_api_validation_and_upload(tmp_path):
    with TestClient(create_app(tmp_path, FakeEngine)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/health").json()["providers"]["llm_configured"] is False
        assert client.post("/api/query", json={"question": "   "}).status_code == 400
        assert client.post("/api/query", json={"question": "x", "k": 99}).status_code == 422
        assert client.post("/api/ingest", json={"chunk_size": 50, "overlap": 60}).status_code == 400
        escaped = client.post("/api/upload", files={"files": ("../escape.txt", "测试文本")})
        assert escaped.status_code == 400
        result = client.post("/api/upload", files={"files": ("sample.txt", "测试文本，提供检索资料。")})
        assert result.status_code == 200
        assert result.json()["accepted"][0]["name"] == "sample.txt"
        assert client.post("/api/upload", files={"files": ("sample.txt", "不同内容。")}).status_code == 400


def test_settings_do_not_expose_secrets_and_reject_cross_origin(tmp_path):
    with TestClient(create_app(tmp_path, FakeEngine)) as client:
        body = {"llm_key": "never-return-this", "tavily_key": "another-secret"}
        response = client.post("/api/settings", json=body)
        assert response.status_code == 200
        assert response.json()["llm_configured"]
        assert "never-return-this" not in response.text
        assert "another-secret" not in client.get("/api/health").text
        assert (
            client.post("/api/settings", json=body, headers={"Origin": "https://untrusted.test"}).status_code
            == 403
        )
        assert (
            client.post("/api/settings", json={"llm_base_url": "http://external.example"}).status_code == 400
        )

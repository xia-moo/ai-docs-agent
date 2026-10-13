"""Real HTTP integration checks against the locally running application."""

import json
from pathlib import Path
from time import monotonic, sleep

import httpx

ROOT = Path(__file__).resolve().parents[1]


def wait(client, job):
    deadline = monotonic() + 90
    while monotonic() < deadline:
        result = client.get("/api/jobs/" + job["id"]).json()
        if result["status"] == "done":
            return result
        if result["status"] == "failed":
            raise AssertionError(result["message"])
        sleep(0.3)
    raise AssertionError("Index job did not finish")


def verify():
    with httpx.Client(base_url="http://127.0.0.1:8008", timeout=90) as client:
        health = client.get("/api/health").json()
        assert health["index"]["document_count"] >= 7
        for method in ("dense", "bm25", "hybrid", "rerank"):
            result = client.post(
                "/api/query",
                json={
                    "question": "padding=True 如何处理不同长度的句子？",
                    "route": "knowledge",
                    "method": method,
                    "k": 3,
                },
            ).json()
            assert result["status"] == "ok", result
            assert result["citations"]
            assert "preprocessing_zh.md" in {hit["source"] for hit in result["citations"]}
        live = client.post("/api/query", json={"question": "PyTorch 今天发布了什么最新版本？"}).json()
        if not health["providers"]["search_configured"]:
            assert live["status"] == "needs_search_configuration"
            assert not live["citations"]
        empty = client.post("/api/query", json={"question": "  "})
        assert empty.status_code == 400
        upload = client.post(
            "/api/upload",
            files={
                "files": (
                    "api-smoke-fixture.txt",
                    "SMOKE_UPLOAD_CHECK 独立测试标记。上传文档后可以检索并追溯来源。".encode(),
                    "text/plain",
                )
            },
        )
        upload.raise_for_status()
        wait(client, upload.json()["job"])
        result = client.post(
            "/api/query",
            json={"question": "SMOKE_UPLOAD_CHECK", "method": "bm25", "route": "knowledge", "k": 3},
        ).json()
        assert any(hit["source"] == "api-smoke-fixture.txt" for hit in result["citations"])
        removed = client.delete("/api/documents/api-smoke-fixture.txt")
        removed.raise_for_status()
        wait(client, removed.json()["job"])
        final = client.get("/api/health").json()
        report = {
            "checks": [
                "four retrieval methods",
                "PDF/MD corpus",
                "query provenance",
                "live configuration boundary",
                "empty query",
                "upload/ingest/query/delete",
            ],
            "health": final,
            "sample_response": result,
        }
        (ROOT / "evals" / "integration.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            "Real API integration checks passed.",
            final["index"]["document_count"],
            "documents;",
            final["index"]["chunk_count"],
            "chunks",
            flush=True,
        )


if __name__ == "__main__":
    verify()

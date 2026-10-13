"""Fetch a small pinned corpus from official repositories, with licenses and hashes."""

import hashlib
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
RECIPES = [
    ("langchain-ai/langgraph", "main", "README.md", "langgraph.md"),
    ("qdrant/qdrant", "master", "README.md", "qdrant.md"),
    ("FlagOpen/FlagEmbedding", "master", "README.md", "flagembedding.md"),
    ("huggingface/transformers", "main", "docs/source/zh/pipeline_tutorial.md", "pipeline_zh.md"),
    ("huggingface/transformers", "main", "docs/source/zh/preprocessing.md", "preprocessing_zh.md"),
    ("huggingface/transformers", "main", "docs/source/zh/chat_templating.md", "chat_template_zh.md"),
]


def fetch():
    target = ROOT / "data" / "documents"
    target.mkdir(parents=True, exist_ok=True)
    references = ROOT / "references"
    references.mkdir(exist_ok=True)
    manifest_path = ROOT / "data" / "sources.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    commits = {}
    with httpx.Client(
        timeout=90, follow_redirects=True, headers={"User-Agent": "ai-docs-agent-learning"}
    ) as client:
        for repo, branch, source_path, name in RECIPES:
            record = existing.get(name)
            if not record:
                if repo not in commits:
                    response = client.get(f"https://api.github.com/repos/{repo}/commits/{branch}")
                    response.raise_for_status()
                    commits[repo] = response.json()
                commit = commits[repo]
                sha = commit["sha"]
                record = {
                    "repo": repo,
                    "revision": sha,
                    "url": f"https://github.com/{repo}/blob/{sha}/{source_path}",
                    "download_url": f"https://raw.githubusercontent.com/{repo}/{sha}/{source_path}",
                    "commit_date": commit["commit"]["committer"]["date"],
                }
            response = client.get(record["download_url"])
            response.raise_for_status()
            content = response.content
            digest = hashlib.sha256(content).hexdigest()
            if record.get("sha256") and record["sha256"] != digest:
                raise RuntimeError(f"Pinned source hash changed: {name}")
            destination = target / name
            if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Refusing to overwrite a changed local document: {name}")
            destination.write_bytes(content)
            record["sha256"] = digest
            existing[name] = record
            manifest_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
            print(name, len(content), "bytes", record.get("commit_date"), flush=True)
            if repo in commits:
                license_file = client.get(
                    f"https://raw.githubusercontent.com/{repo}/{record['revision']}/LICENSE"
                )
                if license_file.is_success:
                    (references / (repo.replace("/", "-") + "-LICENSE")).write_bytes(license_file.content)
        paper_name = "attention_is_all_you_need.pdf"
        paper = existing.get(paper_name) or {
            "url": "https://arxiv.org/abs/1706.03762",
            "download_url": "https://arxiv.org/pdf/1706.03762v7",
            "note": "Research paper used as local study material; do not republish the PDF as project code.",
        }
        response = client.get(paper["download_url"])
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            raise RuntimeError("Paper download did not return a PDF")
        digest = hashlib.sha256(response.content).hexdigest()
        if paper.get("sha256") and digest != paper["sha256"]:
            raise RuntimeError("Paper content changed; check the pinned version")
        paper["sha256"] = digest
        if (target / paper_name).exists() and hashlib.sha256(
            (target / paper_name).read_bytes()
        ).hexdigest() != digest:
            raise RuntimeError(f"Refusing to overwrite a changed local document: {paper_name}")
        (target / paper_name).write_bytes(response.content)
        existing[paper_name] = paper
        manifest_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        print(paper_name, len(response.content), "bytes", flush=True)


if __name__ == "__main__":
    try:
        fetch()
    except (httpx.HTTPError, RuntimeError) as error:
        print("Download failed:", str(error), file=sys.stderr)
        raise SystemExit(1)

"""Benchmark the running application; persist completed methods and any failures."""

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import httpx
import numpy as np
from ai_docs_agent.metrics import rank_metrics

ROOT = Path(__file__).resolve().parents[1]


def evaluate(url, repeats=2):
    dataset = json.loads((ROOT / "evals" / "dataset.json").read_text(encoding="utf-8"))
    with httpx.Client(base_url=url, timeout=90) as client:
        health = client.get("/api/health").json()
        if dataset["index_revision"] != health["index"]["revision"]:
            raise ValueError("Dataset labels belong to an older index. Rebuild labels before benchmarking.")
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "questions": len(dataset["questions"]),
            "k": 5,
            "candidates": 20,
            "repeats": repeats,
            "index_revision": dataset["index_revision"],
            "embedding_model": health["providers"]["embedding_model"],
            "reranker_model": health["providers"]["reranker_model"],
            "metrics": {},
            "runs": {},
            "errors": [],
            "notes": (
                "人工编写并在检索测量前按原文定位的开发集；未经过独立复核，不代表生产或招聘水平。"
                "延迟为本机 HTTP 检索接口的完整耗时，不含答案生成、模型下载和预热。"
                "候选数 20，返回 Top-5；每题重复测量。PDF 的标注页码按文件页序计数。"
            ),
        }
        for method in ("dense", "bm25", "hybrid", "rerank"):
            runs, timings = [], []
            try:
                warmup = client.post(
                    "/api/retrieve",
                    json={
                        "question": dataset["questions"][0]["question"],
                        "method": method,
                        "route": "knowledge",
                        "k": 5,
                        "candidates": 20,
                    },
                )
                warmup.raise_for_status()
                for i, entry in enumerate(dataset["questions"], 1):
                    print(method, i, "/", len(dataset["questions"]), entry["id"], flush=True)
                    scored = None
                    for _ in range(repeats):
                        started = perf_counter()
                        response = client.post(
                            "/api/retrieve",
                            json={
                                "question": entry["question"],
                                "method": method,
                                "route": "knowledge",
                                "k": 5,
                                "candidates": 20,
                            },
                        )
                        response.raise_for_status()
                        elapsed = (perf_counter() - started) * 1000
                        result = response.json()
                        if result["revision"] != dataset["index_revision"]:
                            raise ValueError("Index changed during evaluation")
                        ids = [hit["id"] for hit in result["hits"]]
                        metrics = rank_metrics(ids, entry["relevant_ids"])
                        timings.append(elapsed)
                        scored = {"question_id": entry["id"], "ranked_ids": ids, **metrics}
                    runs.append(scored)
                report["metrics"][method] = {
                    **{
                        key: statistics.mean(run[key] for run in runs)
                        for key in ("hit", "recall", "mrr", "ndcg")
                    },
                    "p50_ms": float(np.percentile(timings, 50)),
                    "p95_ms": float(np.percentile(timings, 95)),
                    "timing_samples": len(timings),
                }
                report["runs"][method] = runs
                print(method, report["metrics"][method], flush=True)
            except (httpx.HTTPError, ValueError) as exc:
                report["errors"].append(f"{method}: {type(exc).__name__}")
            (ROOT / "evals" / "results.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        lines = [
            "# 检索开发集实测",
            "",
            report["notes"],
            "",
            "| 方法 | Hit@5 | Recall@5 | MRR@5 | nDCG@5 | P50 ms | P95 ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for method, metrics in report["metrics"].items():
            lines.append(
                f"| {method} | {metrics['hit']:.3f} | {metrics['recall']:.3f} | {metrics['mrr']:.3f} | "
                f"{metrics['ndcg']:.3f} | {metrics['p50_ms']:.1f} | {metrics['p95_ms']:.1f} |"
            )
        lines.extend(
            [
                "",
                f"问题数：{report['questions']}；每题测量 {repeats} 次。",
                "",
                f"索引版本：`{report['index_revision']}`",
                "",
                "错误：" + ("；".join(report["errors"]) or "无"),
            ]
        )
        (ROOT / "evals" / "results.md").write_text("\n".join(lines), encoding="utf-8")
        if report["errors"]:
            raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8008")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error("repeats must be 1..10")
    evaluate(args.url, args.repeats)

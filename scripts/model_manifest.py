import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]


def audit():
    cache = ROOT / "data" / "models"
    records = []
    for repo in sorted(cache.glob("models--*")):
        reference = repo / "refs" / "main"
        if not reference.exists():
            continue
        revision = reference.read_text().strip()
        snapshot = repo / "snapshots" / revision
        files = []
        for path in snapshot.rglob("*.onnx"):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while data := stream.read(1024 * 1024):
                    digest.update(data)
            files.append(
                {
                    "name": path.relative_to(snapshot).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
        record = {
            "repository": repo.name.removeprefix("models--").replace("--", "/"),
            "revision": revision,
            "onnx_files": files,
        }
        if "MiniLM" in repo.name:
            tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
            tokenizer.no_truncation()
            bundle = json.loads((ROOT / "data" / "index" / "index.json").read_text(encoding="utf-8"))
            counts = [len(tokenizer.encode(chunk["text"]).ids) for chunk in bundle["chunks"]]
            record["corpus_max_tokens"] = max(counts, default=0)
            record["chunks_over_512_tokens"] = sum(count > 512 for count in counts)
        records.append(record)
    destination = ROOT / "data" / "model-manifest.json"
    destination.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    audit()

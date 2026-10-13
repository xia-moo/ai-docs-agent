import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Literal
from urllib.parse import urlsplit

from dotenv import set_key
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .documents import ALLOWED_SUFFIXES, MAX_FILE_BYTES, parse_document
from .graph import create_graph
from .jobs import IndexJobs
from .providers import safe_url
from .retrieval import RetrievalEngine
from .settings import Settings


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    method: Literal["dense", "bm25", "hybrid", "rerank"] = "rerank"
    route: Literal["auto", "knowledge", "web"] = "auto"
    answer_mode: Literal["auto", "extractive", "llm"] = "auto"
    k: int = Field(default=5, ge=1, le=10)
    candidates: int = Field(default=20, ge=10, le=40)


class BuildRequest(BaseModel):
    chunk_size: int = Field(default=350, ge=50, le=1500)
    overlap: int = Field(default=60, ge=0, le=500)


class ProviderSettings(BaseModel):
    llm_key: str = Field(default="", max_length=500)
    llm_base_url: str = Field(default="https://api.deepseek.com", max_length=500)
    llm_model: str = Field(default="deepseek-flash", min_length=1, max_length=100)
    tavily_key: str = Field(default="", max_length=500)
    answer_mode: Literal["auto", "extractive", "llm"] = "auto"


def checked_name(name):
    if not name or name in {".", ".."} or any(char in name for char in "/\\:\x00") or name.startswith("."):
        raise HTTPException(400, "文件名无效。")
    if (
        name.endswith((".", " "))
        or PureWindowsPath(name).is_reserved()
        or any(char in name for char in '<>"|?*')
        or any(ord(char) < 32 for char in name)
    ):
        raise HTTPException(400, "文件名无效。")
    if len(name) > 150:
        raise HTTPException(400, "文件名过长。")
    return name


def validate_and_save(engine, name, data):
    with TemporaryDirectory() as directory:
        temporary = Path(directory) / name
        temporary.write_bytes(data)
        parse_document(temporary)
    with engine.lock:
        destination = engine.docs_dir / name
        if destination.exists():
            if destination.read_bytes() == data:
                return {"name": name, "status": "already_exists"}
            raise HTTPException(409, "同名文件已存在，请先改名再上传。")
        destination.write_bytes(data)
    return {"name": name, "status": "uploaded"}


def create_app(root: Path | None = None, engine_factory=RetrievalEngine):
    settings = Settings.read(root)

    @asynccontextmanager
    async def lifespan(application):
        engine = engine_factory(settings)
        application.state.engine = engine
        application.state.jobs = IndexJobs(engine)
        application.state.graph = create_graph(engine, lambda: Settings.read(settings.root))
        graph_file = settings.root / "docs" / "graph.mmd"
        graph_file.parent.mkdir(parents=True, exist_ok=True)
        graph_file.write_text(application.state.graph.get_graph().draw_mermaid(), encoding="utf-8")
        yield
        application.state.jobs.close()
        engine.close()

    app = FastAPI(title="AI Docs Agent", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def same_origin(request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            expected = f"{request.url.scheme}://{request.url.netloc}"
            if origin and origin != expected:
                return JSONResponse({"detail": "请从本地应用页面执行操作。"}, status_code=403)
        return await call_next(request)

    @app.get("/api/health")
    def health(request: Request):
        return {
            "app": "ai-docs-agent",
            "status": "ok",
            "version": "1.0.0",
            **request.app.state.jobs.public(),
            "providers": Settings.read(settings.root).public(),
        }

    @app.get("/api/documents")
    def documents(request: Request):
        indexed = {doc["name"]: doc for doc in request.app.state.jobs.public()["index"]["documents"]}
        result = []
        for path in sorted(request.app.state.engine.docs_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES:
                saved = indexed.get(path.name)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                result.append(
                    {
                        "name": path.name,
                        "bytes": path.stat().st_size,
                        "indexed": bool(saved and digest == saved["sha256"]),
                        "chunks": saved["chunks"] if saved else 0,
                        "source_url": saved.get("source_url") if saved else None,
                    }
                )
        return {"documents": result}

    @app.post("/api/upload")
    async def upload(request: Request, files: list[UploadFile] = File(...)):
        if not 1 <= len(files) <= 10:
            raise HTTPException(400, "一次支持上传 1–10 个文件。")
        engine, accepted, errors = request.app.state.engine, [], []
        for file in files:
            try:
                name = checked_name(file.filename or "")
                if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
                    raise HTTPException(400, "仅支持 TXT、Markdown、PDF。")
                data = await file.read(MAX_FILE_BYTES + 1)
                if len(data) > MAX_FILE_BYTES:
                    raise HTTPException(400, "文件超过 10 MB。")
                accepted.append(await run_in_threadpool(validate_and_save, engine, name, data))
            except (HTTPException, ValueError) as exc:
                errors.append(
                    {
                        "name": file.filename or "",
                        "message": exc.detail if isinstance(exc, HTTPException) else str(exc),
                    }
                )
            finally:
                await file.close()
        if not accepted:
            return JSONResponse({"accepted": accepted, "errors": errors}, status_code=400)
        return {"accepted": accepted, "errors": errors, "job": request.app.state.jobs.submit()}

    @app.post("/api/ingest", status_code=202)
    def ingest(body: BuildRequest, request: Request):
        if body.overlap >= body.chunk_size:
            raise HTTPException(400, "重叠字符数必须小于切分长度。")
        return request.app.state.jobs.submit(body.chunk_size, body.overlap)

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, request: Request):
        result = request.app.state.jobs.get(job_id)
        if result is None:
            raise HTTPException(404, "任务不存在。")
        return result

    @app.get("/api/documents/{name}/file")
    def file(name: str, request: Request):
        name = checked_name(name)
        path = request.app.state.engine.docs_dir / name
        if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise HTTPException(404, "文档不存在。")
        media = "application/pdf" if path.suffix.lower() == ".pdf" else "text/plain; charset=utf-8"
        return FileResponse(path, media_type=media, filename=name, content_disposition_type="inline")

    @app.delete("/api/documents/{name}")
    def delete_document(name: str, request: Request):
        name = checked_name(name)
        engine = request.app.state.engine
        with engine.lock:
            path = engine.docs_dir / name
            if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
                raise HTTPException(404, "文档不存在。")
            path.unlink()
        return {"deleted": name, "job": request.app.state.jobs.submit()}

    @app.post("/api/query")
    def query(body: QueryRequest, request: Request):
        if not body.question.strip():
            raise HTTPException(400, "请输入问题。")
        started = perf_counter()
        try:
            state = request.app.state.graph.invoke(
                {
                    "question": body.question.strip(),
                    "requested_route": body.route,
                    "method": body.method,
                    "k": body.k,
                    "candidates": body.candidates,
                    "answer_mode": body.answer_mode,
                    "trace": [],
                    "warnings": [],
                }
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503, "检索暂时不可用，请检查索引状态或本地模型缓存。") from exc
        return {
            **state,
            "latency_ms": round((perf_counter() - started) * 1000, 2),
            "revision": state.get("index_revision") or request.app.state.engine.bundle.get("revision"),
        }

    @app.post("/api/retrieve")
    def retrieve(body: QueryRequest, request: Request):
        if not body.question.strip():
            raise HTTPException(400, "请输入问题。")
        try:
            return request.app.state.engine.search(body.question, body.method, body.k, body.candidates)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503, "检索暂时不可用，请检查索引和模型缓存。") from exc

    @app.get("/api/settings")
    def public_settings():
        return Settings.read(settings.root).public()

    @app.post("/api/settings")
    def configure(body: ProviderSettings):
        parsed = urlsplit(body.llm_base_url)
        if not safe_url(body.llm_base_url) or parsed.query or parsed.fragment or parsed.username is not None:
            raise HTTPException(400, "模型 API 地址无效。")
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1"}:
            raise HTTPException(400, "远端模型 API 请使用 HTTPS。")
        pairs = {
            "LLM_BASE_URL": body.llm_base_url.rstrip("/"),
            "LLM_MODEL": body.llm_model,
            "ANSWER_MODE": body.answer_mode,
        }
        if body.llm_key.strip():
            pairs["LLM_API_KEY"] = body.llm_key.strip()
        if body.tavily_key.strip():
            pairs["TAVILY_API_KEY"] = body.tavily_key.strip()
        for key, value in pairs.items():
            if "\n" in value or "\r" in value:
                raise HTTPException(400, "配置值不能包含换行。")
        env_file = settings.root / ".env"
        for key, value in pairs.items():
            set_key(str(env_file), key, value)
        return Settings.read(settings.root).public()

    @app.get("/api/evaluation")
    def evaluation():
        path = settings.root / "evals" / "results.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "not_run"}

    web = Path(__file__).parent / "web"
    app.mount("/static", StaticFiles(directory=web), name="static")

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(web / "index.html")

    return app


app = create_app()

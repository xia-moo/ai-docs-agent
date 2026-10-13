"""Single-writer index jobs; health remains available while inference is running."""

import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Lock


class IndexJobs:
    def __init__(self, engine):
        self.engine = engine
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="index")
        self.lock = Lock()
        self.current = None
        self.summary = engine.summary()
        self.jobs = {}

    def submit(self, size=350, overlap=60):
        with self.lock:
            if self.current and self.jobs[self.current]["status"] == "queued":
                return deepcopy(self.jobs[self.current])
            job_id = uuid.uuid4().hex
            self.current = job_id
            self.jobs[job_id] = {"id": job_id, "status": "queued", "message": "等待建立索引"}
        self.pool.submit(self._run, job_id, size, overlap)
        return self.get(job_id)

    def _run(self, job_id, size, overlap):
        with self.lock:
            self.jobs[job_id].update(status="running", message="正在解析文档和计算向量")
        try:
            summary = self.engine.build(size, overlap)
            with self.lock:
                self.summary = summary
                self.jobs[job_id].update(status="done", message="索引已更新", result=summary)
        except Exception as exc:
            # File/parser errors are safe; network/model internals can contain request details.
            message = str(exc) if isinstance(exc, ValueError) else "索引建立失败，请查看本地日志和模型缓存。"
            with self.lock:
                self.jobs[job_id].update(status="failed", message=message)

    def get(self, job_id):
        with self.lock:
            return deepcopy(self.jobs.get(job_id))

    def public(self):
        with self.lock:
            return {
                "index": deepcopy(self.summary),
                "job": deepcopy(self.jobs.get(self.current)) if self.current else None,
            }

    def close(self):
        self.pool.shutdown(wait=True)

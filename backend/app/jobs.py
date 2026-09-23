"""Background job queue (one worker thread) with live progress for the UI to poll."""
from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from typing import Any, Callable, Optional


class Job:
    def __init__(self, kind: str, title: str, fn: Callable[["Job"], Any], params: Optional[dict] = None):
        self.id = uuid.uuid4().hex[:10]
        self.kind, self.title, self.fn = kind, title, fn
        self.params = params or {}
        self.status = "queued"
        self.progress = 0.0
        self.message = "Waiting in queue"
        self.created = time.time()
        self.started: Optional[float] = None
        self.finished: Optional[float] = None
        self.result: Any = None
        self.error: Optional[str] = None
        self.logs: list[str] = []
        self.series: list[dict] = []      # training curve points: {seed, step, ep_reward}
        self.evals: list[dict] = []       # validation points: {seed, step, sharpe, ...}
        self.cancel_requested = False
        self._lock = threading.Lock()

    def log(self, msg: str):
        with self._lock:
            self.logs.append(time.strftime("%H:%M:%S ") + msg)
            self.logs = self.logs[-200:]

    def update(self, progress: Optional[float] = None, message: Optional[str] = None):
        if progress is not None:
            self.progress = float(min(max(progress, 0.0), 1.0))
        if message is not None:
            self.message = message

    def to_dict(self, full: bool = True) -> dict:
        d = {"id": self.id, "kind": self.kind, "title": self.title, "status": self.status,
             "progress": round(self.progress, 4), "message": self.message, "created": self.created,
             "started": self.started, "finished": self.finished, "error": self.error, "params": self.params,
             "result": self.result if isinstance(self.result, (dict, list, str, type(None))) else None}
        if full:
            with self._lock:
                d.update(logs=list(self.logs[-60:]), series=list(self.series[-1500:]), evals=list(self.evals))
        return d


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.q: "queue.Queue[Job]" = queue.Queue()
        self.worker = threading.Thread(target=self._loop, daemon=True, name="job-worker")
        self.worker.start()

    def submit(self, job: Job) -> Job:
        self.jobs[job.id] = job
        self.q.put(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def list(self) -> list[dict]:
        return [j.to_dict(full=False) for j in sorted(self.jobs.values(), key=lambda j: -j.created)][:50]

    def cancel(self, job_id: str) -> bool:
        j = self.jobs.get(job_id)
        if not j or j.status in ("done", "failed", "cancelled"):
            return False
        j.cancel_requested = True
        if j.status == "queued":
            j.status = "cancelled"
            j.message = "Cancelled before start"
        return True

    def _loop(self):
        while True:
            job = self.q.get()
            if job.status == "cancelled":
                continue
            job.status, job.started = "running", time.time()
            job.update(0.0, "Starting")
            try:
                job.result = job.fn(job)
                if job.cancel_requested:
                    job.status, job.message = "cancelled", "Cancelled"
                else:
                    job.status = "done"
                    job.update(1.0, "Finished")
            except Exception as e:  # noqa: BLE001
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"
                job.log(traceback.format_exc()[-1500:])
                job.message = "Failed"
            job.finished = time.time()


manager = JobManager()

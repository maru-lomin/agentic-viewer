"""Background KV extraction jobs triggered from uploaded PDFs."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agentic_viewer.inference_client import InferenceError, invoke_inference, wait_for_inference_api


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class InferenceTask:
    filename: str
    status: str = "pending"
    run_id: Optional[str] = None
    error: Optional[str] = None
    n_kv: Optional[int] = None
    seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "status": self.status,
            "run_id": self.run_id,
            "error": self.error,
            "n_kv": self.n_kv,
            "seconds": self.seconds,
        }


@dataclass
class InferenceJob:
    job_id: str
    hooks: str
    tasks: List[InferenceTask]
    status: str = "queued"
    current_index: int = 0
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks if t.status in {"done", "error"})
        failed = sum(1 for t in self.tasks if t.status == "error")
        current = None
        if 0 <= self.current_index < total:
            task = self.tasks[self.current_index]
            if task.status == "running":
                current = {"index": self.current_index, "filename": task.filename}
        progress_pct = round(100.0 * completed / total, 1) if total else None
        run_ids = [t.run_id for t in self.tasks if t.run_id]
        return {
            "job_id": self.job_id,
            "hooks": self.hooks,
            "status": self.status,
            "total": total,
            "completed": completed,
            "failed": failed,
            "current": current,
            "tasks": [t.to_dict() for t in self.tasks],
            "run_ids": run_ids,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "message": self.message,
            "progress_pct": progress_pct,
        }


class InferenceJobManager:
    def __init__(self, inference_api_url: str) -> None:
        self.inference_api_url = inference_api_url
        self._jobs: Dict[str, InferenceJob] = {}
        self._file_bytes: Dict[str, List[bytes]] = {}
        self._lock = threading.Lock()
        self._active_job_id: Optional[str] = None

    def get_job(self, job_id: str) -> Optional[InferenceJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def get_active_job(self) -> Optional[InferenceJob]:
        with self._lock:
            if self._active_job_id:
                return self._jobs.get(self._active_job_id)
            return None

    def start(
        self,
        files: Sequence[Tuple[str, bytes]],
        *,
        hooks: str = "agentic_config",
    ) -> InferenceJob:
        if not files:
            raise ValueError("at least one PDF file is required")
        for name, _ in files:
            if not str(name).lower().endswith(".pdf"):
                raise ValueError(f"only PDF files are supported: {name}")

        with self._lock:
            if self._active_job_id:
                active = self._jobs.get(self._active_job_id)
                if active and active.status in {"queued", "running"}:
                    raise RuntimeError(
                        f"inference job {self._active_job_id} is already running"
                    )

        job_id = f"infer-{uuid.uuid4().hex[:12]}"
        job = InferenceJob(
            job_id=job_id,
            hooks=hooks,
            tasks=[InferenceTask(filename=name) for name, _ in files],
        )
        with self._lock:
            self._jobs[job_id] = job
            self._file_bytes[job_id] = [data for _, data in files]
            self._active_job_id = job_id

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id,),
            name=f"inference-job-{job_id}",
            daemon=True,
        )
        thread.start()
        return job

    def _run_job(self, job_id: str) -> None:
        job = self._jobs[job_id]
        file_bytes_list = self._file_bytes.get(job_id, [])
        job.status = "running"
        job.started_at = _utc_now()

        try:
            wait_for_inference_api(self.inference_api_url, timeout_s=180)
        except InferenceError as exc:
            job.status = "error"
            job.message = str(exc)
            job.finished_at = _utc_now()
            for task in job.tasks:
                if task.status == "pending":
                    task.status = "error"
                    task.error = str(exc)
            self._cleanup_job_files(job_id)
            return

        for index, (task, file_bytes) in enumerate(zip(job.tasks, file_bytes_list)):
            job.current_index = index
            task.status = "running"
            request_id = f"agentic-{uuid.uuid4()}"
            task.run_id = request_id
            try:
                result = invoke_inference(
                    self.inference_api_url,
                    filename=task.filename,
                    file_bytes=file_bytes,
                    hooks=job.hooks,
                    request_id=request_id,
                )
                meta = result.get("meta") if isinstance(result, dict) else {}
                task.run_id = (
                    (meta or {}).get("run_id")
                    or (result or {}).get("run_id")
                    or request_id
                )
                task.n_kv = len((result or {}).get("kv_results") or [])
                task.seconds = (meta or {}).get("seconds")
                task.status = "done"
            except InferenceError as exc:
                task.status = "error"
                task.error = str(exc)
            except Exception as exc:  # noqa: BLE001
                task.status = "error"
                task.error = str(exc)

        job.current_index = len(job.tasks)
        failed = sum(1 for t in job.tasks if t.status == "error")
        job.status = "error" if failed == len(job.tasks) else ("partial" if failed else "done")
        job.finished_at = _utc_now()
        self._cleanup_job_files(job_id)

        with self._lock:
            if self._active_job_id == job_id:
                self._active_job_id = None

    def _cleanup_job_files(self, job_id: str) -> None:
        with self._lock:
            self._file_bytes.pop(job_id, None)


def make_inference_job_manager(inference_api_url: str) -> InferenceJobManager:
    return InferenceJobManager(inference_api_url)

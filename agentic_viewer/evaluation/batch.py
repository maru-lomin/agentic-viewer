"""Background batch agentic-evaluation jobs."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agentic_viewer.evaluation.agentic_client import invoke_agentic_eval_safe
from agentic_viewer.evaluation.baseline import load_or_compute_run_eval
from agentic_viewer.evaluation.summary import read_agentic_evals


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def agentic_key_is_done(data: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(data, dict):
        return False
    status = str(data.get("status") or "")
    if status == "error":
        return False
    if status == "done" or data.get("is_correct_answer"):
        return True
    return False


def plan_batch_tasks(
    run_ids: Sequence[str],
    runs_root: Path,
    *,
    skip_existing: bool = True,
) -> Tuple[List[Tuple[str, str]], int]:
    """
    Return (tasks_to_run, skipped_count).

    Each task is (run_id, key). Validates run directories exist.
    """
    runs_root = runs_root.resolve()
    tasks: List[Tuple[str, str]] = []
    skipped = 0

    for run_id in run_ids:
        run_dir = (runs_root / run_id).resolve()
        if not str(run_dir).startswith(str(runs_root)) or not run_dir.is_dir():
            raise KeyError(f"run not found: {run_id}")

        eval_report = load_or_compute_run_eval(
            run_dir, run_id=run_id, write_cache=True
        )
        if not isinstance(eval_report, dict):
            continue

        by_key = read_agentic_evals(run_dir)
        for row in eval_report.get("per_key") or []:
            if not isinstance(row, dict) or "key" not in row:
                continue
            key = str(row["key"])
            if skip_existing and agentic_key_is_done(by_key.get(key)):
                skipped += 1
                continue
            tasks.append((run_id, key))

    return tasks, skipped


@dataclass
class BatchJob:
    job_id: str
    run_ids: List[str]
    skip_existing: bool
    status: str = "queued"
    total: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    current_run_id: Optional[str] = None
    current_key: Optional[str] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        current = None
        if self.current_run_id and self.current_key:
            current = {"run_id": self.current_run_id, "key": self.current_key}
        progress_pct = None
        if self.total > 0:
            progress_pct = round(100.0 * self.completed / self.total, 1)
        return {
            "job_id": self.job_id,
            "run_ids": self.run_ids,
            "skip_existing": self.skip_existing,
            "status": self.status,
            "total": self.total,
            "completed": self.completed,
            "skipped": self.skipped,
            "failed": self.failed,
            "current": current,
            "errors": list(self.errors),
            "cancel_requested": self.cancel_requested,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "message": self.message,
            "progress_pct": progress_pct,
        }


class BatchJobManager:
    def __init__(
        self,
        *,
        runs_root: Path,
        inference_api_url: str,
        inflight_tracker: Dict[str, str],
    ) -> None:
        self.runs_root = runs_root
        self.inference_api_url = inference_api_url
        self.inflight_tracker = inflight_tracker
        self._jobs: Dict[str, BatchJob] = {}
        self._lock = threading.Lock()
        self._active_job_id: Optional[str] = None

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def get_active_job(self) -> Optional[BatchJob]:
        with self._lock:
            if self._active_job_id:
                return self._jobs.get(self._active_job_id)
            return None

    def _any_run_busy(self, run_ids: Sequence[str]) -> Optional[str]:
        for run_id in run_ids:
            key = self.inflight_tracker.get(run_id)
            if key:
                return f"agentic-evaluation already running for run_id={run_id} key={key!r}"
        with self._lock:
            if self._active_job_id:
                active = self._jobs.get(self._active_job_id)
                if active and active.status == "running":
                    return f"batch job {self._active_job_id} is already running"
        return None

    def start(
        self,
        run_ids: Sequence[str],
        *,
        skip_existing: bool = True,
    ) -> BatchJob:
        ids = [str(x).strip() for x in run_ids if str(x).strip()]
        if not ids:
            raise ValueError("run_ids is required")

        busy = self._any_run_busy(ids)
        if busy:
            raise RuntimeError(busy)

        tasks, skipped = plan_batch_tasks(
            ids, self.runs_root, skip_existing=skip_existing
        )

        job = BatchJob(
            job_id=str(uuid.uuid4()),
            run_ids=ids,
            skip_existing=skip_existing,
            total=len(tasks),
            skipped=skipped,
            status="queued" if tasks else "done",
            message="no pending keys" if not tasks else None,
            finished_at=_utc_now() if not tasks else None,
        )

        with self._lock:
            self._jobs[job.job_id] = job
            if tasks:
                self._active_job_id = job.job_id

        if tasks:
            thread = threading.Thread(
                target=self._run_job,
                args=(job.job_id, tasks),
                daemon=True,
                name=f"agentic-batch-{job.job_id[:8]}",
            )
            thread.start()
        return job

    def cancel(self, job_id: str) -> BatchJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"job not found: {job_id}")
            if job.status in {"done", "cancelled", "error"}:
                return job
            job.cancel_requested = True
            job.message = "cancel requested"
            return job

    def _run_job(self, job_id: str, tasks: List[Tuple[str, str]]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = _utc_now()

        try:
            for run_id, key in tasks:
                with self._lock:
                    job = self._jobs[job_id]
                    if job.cancel_requested:
                        job.status = "cancelled"
                        job.message = "cancelled by user"
                        job.current_run_id = None
                        job.current_key = None
                        job.finished_at = _utc_now()
                        return
                    job.current_run_id = run_id
                    job.current_key = key

                self.inflight_tracker[run_id] = key
                try:
                    ok, payload = invoke_agentic_eval_safe(
                        self.inference_api_url, run_id, key
                    )
                finally:
                    self.inflight_tracker.pop(run_id, None)

                with self._lock:
                    job = self._jobs[job_id]
                    job.completed += 1
                    job.current_run_id = None
                    job.current_key = None
                    if not ok:
                        job.failed += 1
                        job.errors.append(
                            {
                                "run_id": run_id,
                                "key": key,
                                "error": payload.get("error") or "error",
                            }
                        )

            with self._lock:
                job = self._jobs[job_id]
                if job.cancel_requested:
                    job.status = "cancelled"
                    job.message = "cancelled by user"
                else:
                    job.status = "done"
                    job.message = (
                        f"finished: {job.completed - job.failed} ok, "
                        f"{job.failed} failed, {job.skipped} skipped"
                    )
                job.finished_at = _utc_now()
        except Exception as exc:
            with self._lock:
                job = self._jobs.get(job_id)
                if job:
                    job.status = "error"
                    job.message = str(exc)
                    job.finished_at = _utc_now()
                    job.current_run_id = None
                    job.current_key = None
        finally:
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None


def make_batch_manager(
    runs_root: Path,
    inference_api_url: str,
    inflight_tracker: Dict[str, str],
) -> BatchJobManager:
    return BatchJobManager(
        runs_root=runs_root,
        inference_api_url=inference_api_url,
        inflight_tracker=inflight_tracker,
    )

"""Background batch agentic-evaluation jobs."""

from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from agentic_viewer.evaluation.agentic_client import (
    cancel_agentic_eval_safe,
    invoke_agentic_eval_safe,
)
from agentic_viewer.evaluation.baseline import load_or_compute_run_eval
from agentic_viewer.evaluation.live_progress import format_live_progress, read_eval_live_progress
from agentic_viewer.evaluation.status_cleanup import mark_running_eval_status_cancelled
from agentic_viewer.evaluation.summary import read_agentic_evals


def default_max_parallel_evals() -> int:
    return max(1, int(os.environ.get("AGENTIC_EVAL_MAX_PARALLEL", "8") or 8))


def enrich_batch_job_dict(
    job_dict: Dict[str, Any],
    runs_root: Path,
) -> Dict[str, Any]:
    """Attach live EvalMaster / SearchAgent progress for the current batch key."""
    if job_dict.get("status") not in {"running", "queued"}:
        return job_dict
    current = job_dict.get("current")
    if not isinstance(current, dict):
        return job_dict
    run_id = str(current.get("run_id") or "").strip()
    key = str(current.get("key") or "").strip()
    if not run_id or not key:
        return job_dict
    live = read_eval_live_progress(runs_root / run_id, key)
    if not live:
        return job_dict
    enriched = dict(current)
    enriched["live"] = live
    enriched["live_label"] = format_live_progress(live)
    return {**job_dict, "current": enriched}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def agentic_key_is_done(data: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(data, dict):
        return False
    status = str(data.get("status") or "")
    if status == "error":
        return False
    if status == "done" or data.get("is_correct_answer") or data.get("is_valid_gold"):
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
    active: List[Dict[str, str]] = field(default_factory=list)
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
            "active": list(self.active),
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
        inflight_tracker: Dict[str, Set[str]],
        max_parallel_evals: Optional[int] = None,
    ) -> None:
        self.runs_root = runs_root
        self.inference_api_url = inference_api_url
        self.inflight_tracker = inflight_tracker
        self.max_parallel_evals = max(
            1, int(max_parallel_evals or default_max_parallel_evals())
        )
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
            keys = self.inflight_tracker.get(run_id)
            if keys:
                sample = next(iter(keys))
                return (
                    f"agentic-evaluation already running for run_id={run_id} "
                    f"key={sample!r}"
                )
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
            run_ids = list(job.run_ids)
        for run_id in run_ids:
            cancel_agentic_eval_safe(self.inference_api_url, run_id)
            mark_running_eval_status_cancelled(self.runs_root / run_id)
            self.inflight_tracker.pop(run_id, None)
        with self._lock:
            return self._jobs[job_id]

    def _track_inflight(self, run_id: str, key: str) -> None:
        keys = self.inflight_tracker.setdefault(run_id, set())
        keys.add(key)

    def _untrack_inflight(self, run_id: str, key: str) -> None:
        keys = self.inflight_tracker.get(run_id)
        if not keys:
            return
        keys.discard(key)
        if not keys:
            self.inflight_tracker.pop(run_id, None)

    def _set_active_task(
        self,
        job_id: str,
        *,
        run_id: str,
        key: str,
        add: bool,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            entry = {"run_id": run_id, "key": key}
            if add:
                if entry not in job.active:
                    job.active.append(entry)
                job.current_run_id = run_id
                job.current_key = key
            else:
                job.active = [x for x in job.active if x != entry]
                if job.active:
                    last = job.active[-1]
                    job.current_run_id = last["run_id"]
                    job.current_key = last["key"]
                else:
                    job.current_run_id = None
                    job.current_key = None

    def _eval_one(self, job_id: str, run_id: str, key: str) -> Tuple[bool, Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.cancel_requested:
                return False, {"error": "cancelled by user"}

        self._track_inflight(run_id, key)
        self._set_active_task(job_id, run_id=run_id, key=key, add=True)
        try:
            return invoke_agentic_eval_safe(self.inference_api_url, run_id, key)
        finally:
            self._untrack_inflight(run_id, key)
            self._set_active_task(job_id, run_id=run_id, key=key, add=False)

    def _run_job(self, job_id: str, tasks: List[Tuple[str, str]]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = _utc_now()

        try:
            workers = min(self.max_parallel_evals, max(1, len(tasks)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._eval_one, job_id, run_id, key): (run_id, key)
                    for run_id, key in tasks
                }
                for fut in as_completed(futures):
                    with self._lock:
                        job = self._jobs[job_id]
                        if job.cancel_requested:
                            for pending in futures:
                                pending.cancel()
                            job.status = "cancelled"
                            job.message = "cancelled by user"
                            job.current_run_id = None
                            job.current_key = None
                            job.active = []
                            job.finished_at = _utc_now()
                            return

                    run_id, key = futures[fut]
                    try:
                        ok, payload = fut.result()
                    except Exception as exc:
                        ok, payload = False, {"error": str(exc)}

                    with self._lock:
                        job = self._jobs[job_id]
                        job.completed += 1
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
                job.current_run_id = None
                job.current_key = None
                job.active = []
        except Exception as exc:
            with self._lock:
                job = self._jobs.get(job_id)
                if job:
                    job.status = "error"
                    job.message = str(exc)
                    job.finished_at = _utc_now()
                    job.current_run_id = None
                    job.current_key = None
                    job.active = []
        finally:
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None


def make_batch_manager(
    runs_root: Path,
    inference_api_url: str,
    inflight_tracker: Dict[str, Set[str]],
    *,
    max_parallel_evals: Optional[int] = None,
) -> BatchJobManager:
    return BatchJobManager(
        runs_root=runs_root,
        inference_api_url=inference_api_url,
        inflight_tracker=inflight_tracker,
        max_parallel_evals=max_parallel_evals,
    )

"""Lightweight FastAPI viewer for agentic run traces under outputs/runs/."""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from agentic_viewer.timezone import KST, to_kst

# Ensure local timezone is Asia/Seoul (KST)
os.environ.setdefault("TZ", "Asia/Seoul")
if hasattr(time, "tzset"):
    try:
        time.tzset()
    except Exception:
        pass

from agentic_viewer.datasets import DatasetStore
from agentic_viewer.datasets_page import DATASETS_HTML
from agentic_viewer.evaluation.agentic_client import (
    AgenticEvalError,
    invoke_agentic_eval,
)
from agentic_viewer.evaluation.baseline import load_or_compute_run_eval
from agentic_viewer.evaluation.batch import make_batch_manager
from agentic_viewer.evaluation.status_cleanup import cleanup_all_running_eval_statuses
from agentic_viewer.evaluation.summary import (
    agentic_eval_summary,
    read_agentic_evals,
)
from agentic_viewer.evaluation_page import EVALUATION_HTML
from agentic_viewer.ground_truth_page import GROUND_TRUTH_HTML
from agentic_viewer.image_tokens import replace_base64_images
from agentic_viewer.inference_jobs import make_inference_job_manager
from agentic_viewer.prompts_page import PROMPTS_HTML
from agentic_viewer.templates import get_template
from agentic_viewer.wrong_cases_page import WRONG_CASES_HTML


def default_runs_root() -> Path:
    """Prefer shared repo outputs/runs, else legacy inference-pipeline path."""
    env = os.environ.get("AGENTIC_RUNS_DIR")
    if env:
        return Path(env).resolve()
    repo_root = Path(__file__).resolve().parent.parent.parent
    shared = repo_root / "outputs" / "runs"
    legacy = repo_root / "inference-pipeline" / "outputs" / "runs"
    local = repo_root / "agentic-viewer" / "runs"
    if shared.is_dir() or (repo_root / "outputs").is_dir():
        return shared.resolve()
    if legacy.is_dir() or (repo_root / "inference-pipeline").is_dir():
        return legacy.resolve()
    return local.resolve()


RUNS_ROOT = default_runs_root()
INFERENCE_API_URL = os.environ.get("INFERENCE_API_URL", "http://127.0.0.1:8010")

# In-flight per-key background evals triggered from the viewer: run_id -> set(keys)
_AGENTIC_EVAL_INFLIGHT: Dict[str, Set[str]] = {}
_BATCH_MANAGER = make_batch_manager(
    RUNS_ROOT,
    inference_api_url=INFERENCE_API_URL,
    inflight_tracker=_AGENTIC_EVAL_INFLIGHT,
)
_INFERENCE_JOB_MANAGER = make_inference_job_manager(
    runs_root=RUNS_ROOT,
    inference_api_url=INFERENCE_API_URL,
    batch_manager=_BATCH_MANAGER,
)
_DATASET_STORE = DatasetStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    cleaned = cleanup_all_running_eval_statuses(
        RUNS_ROOT, reason="cancelled by system (server startup cleanup)"
    )
    total = sum(cleaned.values())
    if total:
        print(
            f"Startup cleanup: marked {total} stale evaluation(s) cancelled across {len(cleaned)} run(s)"
        )
    yield


app = FastAPI(title="Agentic Run Trace Viewer", version="0.3.0", lifespan=lifespan)


def _list_agentic_evals(run_id: str) -> Dict[str, Any]:
    root = _run_dir(run_id)
    inflight = _AGENTIC_EVAL_INFLIGHT.get(run_id)
    return {
        "by_key": read_agentic_evals(root),
        "inflight": sorted(inflight) if inflight else None,
    }


def _call_inference_agentic_eval(run_id: str, key: str) -> Dict[str, Any]:
    active = _BATCH_MANAGER.get_active_job()
    if active and active.status == "running" and run_id in active.run_ids:
        raise HTTPException(
            status_code=409,
            detail=f"batch agentic-evaluation job {active.job_id} is running for this run",
        )
    try:
        return invoke_agentic_eval(INFERENCE_API_URL, run_id, key)
    except AgenticEvalError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


def _run_dir(run_id: str) -> Path:
    path = (RUNS_ROOT / run_id).resolve()
    if not str(path).startswith(str(RUNS_ROOT)) or not path.is_dir():
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return path


def _assert_run_deletable(run_id: str, force: bool = False) -> None:
    if force:
        _AGENTIC_EVAL_INFLIGHT.pop(run_id, None)
        active_infer = _INFERENCE_JOB_MANAGER.get_active_job()
        if active_infer and active_infer.status in {"queued", "running"}:
            for task in active_infer.tasks:
                if task.run_id == run_id and task.status in {"pending", "running"}:
                    task.status = "cancelled"
                    task.error = "run deleted by user"
        active_batch = _BATCH_MANAGER.get_active_job()
        if (
            active_batch
            and active_batch.status in {"queued", "running"}
            and run_id in active_batch.run_ids
        ):
            active_batch.run_ids = [r for r in active_batch.run_ids if r != run_id]
        return

    inflight = _AGENTIC_EVAL_INFLIGHT.get(run_id)
    if inflight:
        raise HTTPException(
            status_code=409,
            detail=f"agentic-evaluation is in progress for run {run_id}",
        )

    active_batch = _BATCH_MANAGER.get_active_job()
    if (
        active_batch
        and active_batch.status in {"queued", "running"}
        and run_id in active_batch.run_ids
    ):
        raise HTTPException(
            status_code=409,
            detail=f"batch job {active_batch.job_id} is using this run",
        )

    active_infer = _INFERENCE_JOB_MANAGER.get_active_job()
    if active_infer and active_infer.status in {"queued", "running"}:
        for task in active_infer.tasks:
            if task.run_id == run_id and task.status in {"pending", "running"}:
                raise HTTPException(
                    status_code=409,
                    detail="KV extraction is in progress for this run",
                )


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        return json.loads(text)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _eval_summary(report: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(report, dict):
        return None
    summary = report.get("summary")
    if isinstance(summary, dict) and summary.get("evaluated", False):
        return {
            "evaluated": True,
            "exact_match_rate": summary.get("exact_match_rate", 0.0),
            "n_gold": summary.get("n_gold", 0),
            "n_pred": summary.get("n_pred", 0),
            "n_correct": summary.get("n_correct", 0),
        }
    return None


def _compute_run_eval(run_id: str, *, refresh: bool = False) -> Dict[str, Any]:
    """Score 04_result.json against dataset/answer_sheet.json; cache as 05_eval.json."""
    root = _run_dir(run_id)
    data = load_or_compute_run_eval(root, run_id=run_id, refresh=refresh, write_cache=True)
    if data is None:
        return {
            "error": "no result or ground-truth answer sheet available",
            "evaluated": False,
        }
    return data


def _parse_ts(ts_str: Optional[str]) -> Optional[datetime]:
    if not ts_str or not isinstance(ts_str, str):
        return None
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return to_kst(dt)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.strptime(ts_str, fmt)
            return to_kst(dt)
        except ValueError:
            pass
    return None


def _format_display_ts(dt: Optional[datetime]) -> str:
    if not dt:
        return ""
    kst_dt = to_kst(dt)
    return kst_dt.strftime("%Y-%m-%d %H:%M") if kst_dt else ""


def _enrich_run_groups(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign fallback run_group_id / run_group_name based on dataset sessions."""
    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        did = r.get("dataset_id")
        if did and not r.get("run_group_id"):
            by_dataset.setdefault(did, []).append(r)

    if not by_dataset:
        return rows

    SESSION_GAP = 30 * 60

    for did, unassigned in by_dataset.items():
        max_v = 0
        for r in rows:
            if r.get("dataset_id") == did:
                gid = str(r.get("run_group_id") or "")
                m = re.search(r"-run-v(\d+)", gid, re.IGNORECASE)
                if m:
                    max_v = max(max_v, int(m.group(1)))

        timed = []
        for r in unassigned:
            dt = _parse_ts(r.get("started_at")) or _parse_ts(r.get("finished_at"))
            timed.append((dt, r))

        timed.sort(key=lambda item: item[0] or datetime.min.replace(tzinfo=KST))

        sessions: List[List[tuple[Optional[datetime], Dict[str, Any]]]] = []
        curr_session: List[tuple[Optional[datetime], Dict[str, Any]]] = []

        for dt, r in timed:
            if not curr_session:
                curr_session.append((dt, r))
            else:
                prev_dt = curr_session[-1][0]
                if dt and prev_dt and (dt - prev_dt).total_seconds() <= SESSION_GAP:
                    curr_session.append((dt, r))
                elif not dt and not prev_dt:
                    curr_session.append((dt, r))
                else:
                    sessions.append(curr_session)
                    curr_session = [(dt, r)]

        if curr_session:
            sessions.append(curr_session)

        cur_v = max_v
        for sess in sessions:
            cur_v += 1
            dts = [dt for dt, _ in sess if dt]
            earliest = min(dts) if dts else None
            time_str = _format_display_ts(earliest)
            ds_name = sess[0][1].get("dataset_name") or did
            gid = f"{did}-run-v{cur_v}"
            gname = f"{ds_name}-run-v{cur_v} ({time_str})" if time_str else f"{ds_name}-run-v{cur_v}"

            for _, r in sess:
                r["run_group_id"] = gid
                r["run_group_name"] = gname

    return rows


def _next_dataset_run_version(dataset_id: str) -> int:
    max_v = 0
    pattern = re.compile(rf"{re.escape(dataset_id)}-run-v(\d+)", re.IGNORECASE)
    runs = list_runs()
    for r in runs:
        if r.get("dataset_id") != dataset_id:
            continue
        gid = str(r.get("run_group_id") or "")
        m = pattern.search(gid)
        if m:
            try:
                max_v = max(max_v, int(m.group(1)))
            except ValueError:
                pass
    return max_v + 1


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _serve_run_file(root: Path, rel: str):
    """Serve a file under ``root`` (JSON as JSONResponse, text as HTML pre)."""
    rel = rel.lstrip("/")
    target = (root / rel).resolve()
    if not str(target).startswith(str(root.resolve())) or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    if target.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
        return FileResponse(target)
    if target.suffix.lower() in {".json", ".jsonl", ".md", ".txt"}:
        text = target.read_text(encoding="utf-8", errors="replace")
        if target.suffix.lower() == ".json":
            try:
                obj = json.loads(text)
                return JSONResponse(obj)
            except json.JSONDecodeError:
                pass
        if target.suffix.lower() == ".jsonl":
            lines_out = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                        obj["text"] = replace_base64_images(obj["text"])
                    if isinstance(obj, dict) and isinstance(obj.get("content"), str):
                        obj["content"] = replace_base64_images(obj["content"])
                    lines_out.append(json.dumps(obj, ensure_ascii=False))
                except json.JSONDecodeError:
                    lines_out.append(line)
            text = "\n".join(lines_out)
        return HTMLResponse(
            f"<pre style='white-space:pre-wrap;font-family:ui-monospace,monospace'>"
            f"{_escape(text)}</pre>"
        )
    return FileResponse(target)


# --- Import and mount modular APIRouters ---

from agentic_viewer.routers.chat import (
    delete_agentic_eval_chat_api,
    delete_search_chat_api_endpoint,
    delete_vlm_chat_api_endpoint,
    get_agentic_eval_chat_api,
    get_search_chat_api_endpoint,
    get_vlm_chat_api_endpoint,
    post_agentic_eval_chat,
    post_search_chat_api,
    post_vlm_chat_api,
    router as chat_router,
)
from agentic_viewer.routers.datasets import (
    add_dataset_files,
    create_dataset,
    datasets_page,
    delete_dataset,
    delete_dataset_file,
    get_active_inference_job,
    get_dataset,
    get_inference_job,
    list_datasets,
    post_inference_job,
    post_inference_job_from_dataset,
    router as datasets_router,
)
from agentic_viewer.routers.evaluations import (
    cancel_batch_job,
    evaluation_page,
    get_active_batch_job,
    get_agentic_eval_file,
    get_agentic_eval_tree,
    get_agentic_evals,
    get_batch_job,
    get_eval,
    get_evaluation_summary,
    list_agentic_eval_keys_api,
    post_agentic_eval,
    post_batch_agentic_eval,
    post_cleanup_all_stale_eval,
    post_cleanup_stale_eval,
    router as evaluations_router,
)
from agentic_viewer.routers.ground_truth import (
    get_ground_truth_document,
    get_ground_truth_index,
    ground_truth_page,
    put_ground_truth_key,
    router as ground_truth_router,
    upload_ground_truth,
)
from agentic_viewer.routers.prompts import (
    api_diff_prompt,
    api_get_prompt,
    api_get_prompt_backup,
    api_list_prompt_backups,
    api_list_prompts,
    api_restore_prompt,
    api_save_prompt,
    prompts_page,
    router as prompts_router,
)
from agentic_viewer.routers.runs import (
    INDEX_HTML,
    delete_run,
    delete_runs_batch,
    get_agent_tree,
    get_chunk,
    get_chunk_highlights,
    get_conversation,
    get_file,
    get_page_highlights,
    get_pdf,
    get_pdf_info,
    get_run,
    get_timeline,
    get_timing,
    index,
    list_chunks,
    list_pages,
    list_runs,
    list_steps,
    list_steps_detail,
    router as runs_router,
)
from agentic_viewer.routers.wrong_cases import (
    api_batch_create_wrong_cases,
    api_create_wrong_case,
    api_delete_wrong_case,
    api_get_wrong_case_detail,
    api_list_wrong_cases,
    api_update_wrong_case,
    router as wrong_cases_router,
    wrong_cases_page,
)

# Register routers
app.include_router(runs_router)
app.include_router(evaluations_router)
app.include_router(datasets_router)
app.include_router(ground_truth_router)
app.include_router(wrong_cases_router)
app.include_router(prompts_router)
app.include_router(chat_router)

# Directly expose sub-router routes on app.routes for inspection and compatibility
for _r in (
    runs_router,
    evaluations_router,
    datasets_router,
    ground_truth_router,
    wrong_cases_router,
    prompts_router,
    chat_router,
):
    app.routes.extend(_r.routes)


def main() -> None:
    import uvicorn

    host = os.environ.get("TRACE_VIEWER_HOST", "0.0.0.0")
    port = int(os.environ.get("TRACE_VIEWER_PORT", "8099"))
    print(f"Trace viewer on http://{host}:{port}  runs_root={RUNS_ROOT}  datasets={_DATASET_STORE.managed_root}")
    uvicorn.run(
        "agentic_viewer.app:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()

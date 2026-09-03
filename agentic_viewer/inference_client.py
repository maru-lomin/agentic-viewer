"""HTTP client for inference-pipeline KV extraction."""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, Optional


class InferenceError(Exception):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def wait_for_inference_api(
    api_url: str,
    *,
    timeout_s: int = 180,
    poll_s: float = 2.0,
) -> None:
    """Block until the inference API responds on /docs or /openapi.json."""
    health_urls = [api_url.rstrip("/") + "/docs", api_url.rstrip("/") + "/openapi.json"]
    deadline = time.time() + timeout_s
    last_error = "unknown"
    while time.time() < deadline:
        for health_url in health_urls:
            try:
                req = urllib.request.Request(health_url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as response:
                    if response.status < 500:
                        return
                    last_error = f"{health_url} -> {response.status}"
            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    return
                last_error = f"{health_url} -> {exc.code}"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
        time.sleep(poll_s)
    raise InferenceError(
        f"Inference API not ready at {api_url}: {last_error}",
        status_code=502,
    )


def invoke_inference(
    api_url: str,
    *,
    filename: str,
    file_bytes: bytes,
    hooks: str = "agentic_config",
    timeout: float = 7200,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Call POST /inference with a base64-encoded PDF (same as client.py --upload)."""
    req_id = request_id or f"agentic-{uuid.uuid4()}"
    payload = json.dumps(
        {
            "x_request_id": req_id,
            "protocol": "grpc",
            "hooks": hooks,
            "rotation_90n": False,
            "rotation_fine": False,
            "file_path": filename,
            "file": base64.b64encode(file_bytes).decode("utf-8"),
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/inference",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise InferenceError("invalid response from inference API", status_code=502)
            return data
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict) and "detail" in parsed:
                detail = str(parsed["detail"])
        except Exception:
            pass
        raise InferenceError(detail, status_code=exc.code) from exc
    except urllib.error.URLError as exc:
        raise InferenceError(
            f"Cannot reach inference API at {api_url}/inference: {exc.reason}. "
            "Set INFERENCE_API_URL or start the API (inference-pipeline/run_api.sh).",
            status_code=502,
        ) from exc

"""HTTP client for inference-pipeline agentic-evaluation."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Tuple


class AgenticEvalError(Exception):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def invoke_agentic_eval(
    api_url: str,
    run_id: str,
    key: str,
    *,
    timeout: float = 1800,
) -> Dict[str, Any]:
    """Call POST /agentic-eval on the inference API."""
    payload = json.dumps(
        {
            "run_id": run_id,
            "key": key,
            "hooks": "agentic-evaluation_config",
            "protocol": "grpc",
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/agentic-eval",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise AgenticEvalError("invalid response from inference API", status_code=502)
            return data
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict) and "detail" in parsed:
                detail = str(parsed["detail"])
        except Exception:
            pass
        raise AgenticEvalError(detail, status_code=exc.code) from exc
    except urllib.error.URLError as exc:
        raise AgenticEvalError(
            f"Cannot reach inference API at {api_url}/agentic-eval: {exc.reason}. "
            "Set INFERENCE_API_URL or start the API (inference-pipeline/run_api.sh).",
            status_code=502,
        ) from exc


def invoke_agentic_eval_safe(
    api_url: str,
    run_id: str,
    key: str,
) -> Tuple[bool, Dict[str, Any]]:
    """Return (ok, result_or_error_payload)."""
    try:
        return True, invoke_agentic_eval(api_url, run_id, key)
    except AgenticEvalError as exc:
        return False, {
            "key": key,
            "status": "error",
            "error": str(exc),
            "status_code": exc.status_code,
        }


def cancel_agentic_eval(api_url: str, run_id: str, *, timeout: float = 10) -> Dict[str, Any]:
    """Request cancellation of the in-flight eval for one extraction run_id."""
    payload = json.dumps({"run_id": run_id}).encode("utf-8")
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/agentic-eval/cancel",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise AgenticEvalError("invalid cancel response", status_code=502)
            return data
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict) and "detail" in parsed:
                detail = str(parsed["detail"])
        except Exception:
            pass
        raise AgenticEvalError(detail, status_code=exc.code) from exc
    except urllib.error.URLError as exc:
        raise AgenticEvalError(
            f"Cannot reach inference API at {api_url}/agentic-eval/cancel: {exc.reason}",
            status_code=502,
        ) from exc


def cancel_agentic_eval_safe(api_url: str, run_id: str) -> None:
    try:
        cancel_agentic_eval(api_url, run_id)
    except AgenticEvalError:
        pass


def invoke_agentic_eval_chat(
    api_url: str,
    run_id: str,
    key: str,
    message: str,
    *,
    timeout: float = 300,
) -> Dict[str, Any]:
    """Call POST /agentic-eval/chat on the inference API."""
    payload = json.dumps(
        {
            "run_id": run_id,
            "key": key,
            "message": message,
            "hooks": "agentic-evaluation_config",
            "protocol": "grpc",
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/agentic-eval/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise AgenticEvalError("invalid response from inference API", status_code=502)
            return data
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict) and "detail" in parsed:
                detail = str(parsed["detail"])
        except Exception:
            pass
        raise AgenticEvalError(detail, status_code=exc.code) from exc
    except urllib.error.URLError as exc:
        raise AgenticEvalError(
            f"Cannot reach inference API at {api_url}/agentic-eval/chat: {exc.reason}",
            status_code=502,
        ) from exc


def get_agentic_eval_chat(
    api_url: str,
    run_id: str,
    key: str,
    *,
    timeout: float = 30,
) -> Dict[str, Any]:
    """Call GET /agentic-eval/chat on the inference API."""
    import urllib.parse
    qs = urllib.parse.urlencode({"run_id": run_id, "key": key})
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/agentic-eval/chat?{qs}",
        headers={"Content-Type": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise AgenticEvalError("invalid response from inference API", status_code=502)
            return data
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict) and "detail" in parsed:
                detail = str(parsed["detail"])
        except Exception:
            pass
        raise AgenticEvalError(detail, status_code=exc.code) from exc
    except urllib.error.URLError as exc:
        raise AgenticEvalError(
            f"Cannot reach inference API at {api_url}/agentic-eval/chat: {exc.reason}",
            status_code=502,
        ) from exc


def delete_agentic_eval_chat(
    api_url: str,
    run_id: str,
    key: str,
    *,
    timeout: float = 30,
) -> Dict[str, Any]:
    """Call DELETE /agentic-eval/chat on the inference API."""
    import urllib.parse
    qs = urllib.parse.urlencode({"run_id": run_id, "key": key})
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/agentic-eval/chat?{qs}",
        headers={"Content-Type": "application/json"},
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise AgenticEvalError("invalid response from inference API", status_code=502)
            return data
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict) and "detail" in parsed:
                detail = str(parsed["detail"])
        except Exception:
            pass
        raise AgenticEvalError(detail, status_code=exc.code) from exc
    except urllib.error.URLError as exc:
        raise AgenticEvalError(
            f"Cannot reach inference API at {api_url}/agentic-eval/chat: {exc.reason}",
            status_code=502,
        ) from exc


def invoke_vlm_chat(
    api_url: str,
    run_id: str,
    key: str,
    message: str,
    *,
    timeout: float = 300,
) -> Dict[str, Any]:
    """Call POST /vlm-chat on the inference API."""
    payload = json.dumps(
        {
            "run_id": run_id,
            "key": key,
            "message": message,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/vlm-chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise AgenticEvalError("invalid response from inference API", status_code=502)
            return data
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict) and "detail" in parsed:
                detail = str(parsed["detail"])
        except Exception:
            pass
        raise AgenticEvalError(detail, status_code=exc.code) from exc
    except urllib.error.URLError as exc:
        raise AgenticEvalError(
            f"Cannot reach inference API at {api_url}/vlm-chat: {exc.reason}",
            status_code=502,
        ) from exc


def get_vlm_chat(
    api_url: str,
    run_id: str,
    key: str,
    *,
    timeout: float = 30,
) -> Dict[str, Any]:
    """Call GET /vlm-chat on the inference API."""
    import urllib.parse
    qs = urllib.parse.urlencode({"run_id": run_id, "key": key})
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/vlm-chat?{qs}",
        headers={"Content-Type": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise AgenticEvalError("invalid response from inference API", status_code=502)
            return data
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict) and "detail" in parsed:
                detail = str(parsed["detail"])
        except Exception:
            pass
        raise AgenticEvalError(detail, status_code=exc.code) from exc
    except urllib.error.URLError as exc:
        raise AgenticEvalError(
            f"Cannot reach inference API at {api_url}/vlm-chat: {exc.reason}",
            status_code=502,
        ) from exc


def delete_vlm_chat(
    api_url: str,
    run_id: str,
    key: str,
    *,
    timeout: float = 30,
) -> Dict[str, Any]:
    """Call DELETE /vlm-chat on the inference API."""
    import urllib.parse
    qs = urllib.parse.urlencode({"run_id": run_id, "key": key})
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/vlm-chat?{qs}",
        headers={"Content-Type": "application/json"},
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise AgenticEvalError("invalid response from inference API", status_code=502)
            return data
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict) and "detail" in parsed:
                detail = str(parsed["detail"])
        except Exception:
            pass
        raise AgenticEvalError(detail, status_code=exc.code) from exc
    except urllib.error.URLError as exc:
        raise AgenticEvalError(
            f"Cannot reach inference API at {api_url}/vlm-chat: {exc.reason}",
            status_code=502,
        ) from exc


def invoke_search_chat(
    api_url: str,
    run_id: str,
    key: str,
    message: str,
    *,
    timeout: float = 300,
) -> Dict[str, Any]:
    """Call POST /search-chat on the inference API."""
    payload = json.dumps(
        {
            "run_id": run_id,
            "key": key,
            "message": message,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/search-chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise AgenticEvalError("invalid response from inference API", status_code=502)
            return data
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict) and "detail" in parsed:
                detail = str(parsed["detail"])
        except Exception:
            pass
        raise AgenticEvalError(detail, status_code=exc.code) from exc
    except urllib.error.URLError as exc:
        raise AgenticEvalError(
            f"Cannot reach inference API at {api_url}/search-chat: {exc.reason}",
            status_code=502,
        ) from exc


def get_search_chat(
    api_url: str,
    run_id: str,
    key: str,
    *,
    timeout: float = 30,
) -> Dict[str, Any]:
    """Call GET /search-chat on the inference API."""
    import urllib.parse
    qs = urllib.parse.urlencode({"run_id": run_id, "key": key})
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/search-chat?{qs}",
        headers={"Content-Type": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise AgenticEvalError("invalid response from inference API", status_code=502)
            return data
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict) and "detail" in parsed:
                detail = str(parsed["detail"])
        except Exception:
            pass
        raise AgenticEvalError(detail, status_code=exc.code) from exc
    except urllib.error.URLError as exc:
        raise AgenticEvalError(
            f"Cannot reach inference API at {api_url}/search-chat: {exc.reason}",
            status_code=502,
        ) from exc


def delete_search_chat(
    api_url: str,
    run_id: str,
    key: str,
    *,
    timeout: float = 30,
) -> Dict[str, Any]:
    """Call DELETE /search-chat on the inference API."""
    import urllib.parse
    qs = urllib.parse.urlencode({"run_id": run_id, "key": key})
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/search-chat?{qs}",
        headers={"Content-Type": "application/json"},
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            if not isinstance(data, dict):
                raise AgenticEvalError("invalid response from inference API", status_code=502)
            return data
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict) and "detail" in parsed:
                detail = str(parsed["detail"])
        except Exception:
            pass
        raise AgenticEvalError(detail, status_code=exc.code) from exc
    except urllib.error.URLError as exc:
        raise AgenticEvalError(
            f"Cannot reach inference API at {api_url}/search-chat: {exc.reason}",
            status_code=502,
        ) from exc


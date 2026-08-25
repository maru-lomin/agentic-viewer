"""Replace inline base64 images with a compact image token for agent/viewer text."""

from __future__ import annotations

import re
from typing import Match

# Matches project VLM templates (e.g. assets/vlm_templates/template.json).
DEFAULT_IMAGE_TOKEN = "<image>"

# Markdown image: ![alt](url) — URL may be raw base64 or data:image/...;base64,...
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]*)\)", re.DOTALL)

# HTML <img ... src="..." ...>
_HTML_IMG_RE = re.compile(
    r"<img\b[^>]*?\bsrc\s*=\s*([\"'])(.*?)\1[^>]*>",
    re.IGNORECASE | re.DOTALL,
)

# Bare data-URL payloads that may appear outside markdown/img tags.
_DATA_URL_RE = re.compile(
    r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+",
    re.IGNORECASE,
)

_BASE64_MAGIC_PREFIXES = (
    "iVBOR",  # PNG
    "/9j/",  # JPEG
    "R0lGOD",  # GIF
    "UklGR",  # WEBP/RIFF
    "Qk",  # BMP (short; still gated by length)
)


def _strip_data_url_prefix(url: str) -> str:
    u = (url or "").strip()
    marker = ";base64,"
    lower = u.lower()
    if lower.startswith("data:image/") and marker in lower:
        idx = lower.find(marker)
        return u[idx + len(marker) :].strip()
    return u


def looks_like_base64_payload(url: str, *, min_len: int = 200) -> bool:
    """True if markdown/HTML src is an embedded base64 image (not a file/http path)."""
    if not url:
        return False
    raw = url.strip()
    if raw.lower().startswith(("http://", "https://", "file:")):
        return False
    if raw.startswith(("/", "./", "../")) and len(raw) < min_len:
        return False

    payload = _strip_data_url_prefix(raw)
    compact = re.sub(r"\s+", "", payload)
    if len(compact) < min_len:
        return False
    if compact.lower().startswith(("http://", "https://", "file:")):
        return False

    if any(compact.startswith(p) for p in _BASE64_MAGIC_PREFIXES):
        return True

    # data: URL already identified as image base64
    if raw.lower().startswith("data:image/") and ";base64," in raw.lower():
        return True

    alphabet = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
    )
    ok = sum(1 for c in compact if c in alphabet)
    return (ok / len(compact)) >= 0.98


def replace_base64_images(
    text: str,
    *,
    token: str = DEFAULT_IMAGE_TOKEN,
    keep_alt: bool = True,
) -> str:
    """
    Replace embedded base64 images with ``token`` (default ``<image>``).

    Markdown ``![alt](<base64>)`` becomes ``![alt](<image>)`` when keep_alt,
    otherwise just ``<image>``.
    """
    if not text:
        return text

    def _md_repl(match: Match[str]) -> str:
        alt = match.group(1) or ""
        url = match.group(2) or ""
        if not looks_like_base64_payload(url):
            return match.group(0)
        if keep_alt and alt:
            return f"![{alt}]({token})"
        return token

    out = _MD_IMAGE_RE.sub(_md_repl, text)

    def _html_repl(match: Match[str]) -> str:
        src = match.group(2) or ""
        if not looks_like_base64_payload(src):
            return match.group(0)
        return token

    out = _HTML_IMG_RE.sub(_html_repl, out)
    out = _DATA_URL_RE.sub(token, out)
    return out

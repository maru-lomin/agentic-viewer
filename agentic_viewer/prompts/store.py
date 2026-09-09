"""Store and business logic for agent system prompts."""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic_viewer.timezone import KST, kst_now

PROMPT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "master": {
        "id": "master",
        "title": "KV Master Agent",
        "filename": "master_system_prompt.txt",
        "role": "Inference Master",
        "category": "inference",
        "tags": ["Inference", "Master", "Pipeline"],
        "pipeline_file": "inference-pipeline/agentic/pipeline.py",
        "description": "KV 추출 파이프라인(MasterAgent)의 전체 오케스트레이션 및 최종 추출 조합을 담당합니다. 검색은 SearchAgent에 위임하고 VLM 추출 결과를 종합합니다.",
        "contract": {
            "completion_type": "json_or_short_circuit",
            "completion_description": "모든 스키마 키가 extract_kv_vlm으로 추출되면 파이프라인이 자동 조립(short_circuit)하거나, 에이전트가 단일 JSON 객체(kv_results)로 완료합니다.",
            "final_schema_example": {
                "kv_results": [
                    {
                        "key": "Distance between GTG",
                        "value": "20m",
                        "found": True,
                        "evidence": [
                            {
                                "chunk_id": "12-3",
                                "page": 12,
                                "text": "배치도 섹션에서 터빈 인클로저 간 50ft 이격 거리 명시됨."
                            }
                        ],
                        "search_queries": ["Distance between GTG separation"]
                    }
                ]
            },
            "intermediate_schemas": [
                {
                    "name": "extract_kv_vlm (Structured Output)",
                    "description": "VLM 멀티페이지 KV 추출 도구 호출 시 strict=True JSON Schema로 강제되는 응답 구조입니다.",
                    "schema": {
                        "extractions": [
                            {
                                "key": "string (스키마 키)",
                                "value": "string (추출값 또는 'not_found')",
                                "value_reason": "string (추출 근거 사유)"
                            }
                        ]
                    }
                }
            ],
            "available_tools": [
                {
                    "name": "load_kv_schema",
                    "args": "keys?: string[]",
                    "description": "추출 대상 키의 설명, 검색 큐, 허용값 정의를 확인합니다."
                },
                {
                    "name": "search_pages",
                    "args": "keys: string[], note?: string",
                    "description": "SearchAgent에 1개 또는 2~5개 연관 키의 근거 페이지 검색 작업을 비동기 큐잉합니다 (즉시 반환)."
                },
                {
                    "name": "collect_search_results",
                    "args": "policy?: 'any' | 'all'",
                    "description": "SearchAgent가 비동기로 검색 완료한 근거 페이지(pages) 및 사유(page_reasons)를 수집합니다."
                },
                {
                    "name": "extract_kv_vlm",
                    "args": "pages: int[], keys: string[], hints?: string",
                    "description": "선정된 문서 페이지 이미지와 키를 VLM에 전달하여 Structured Output으로 값을 추출합니다."
                }
            ]
        }
    },
    "search": {
        "id": "search",
        "title": "SearchAgent",
        "filename": "search_system_prompt.txt",
        "role": "Evidence Retrieval",
        "category": "retrieval",
        "tags": ["Shared", "Retrieval", "Worker"],
        "pipeline_file": "inference-pipeline/agentic/search_agent.py",
        "description": "BM25 텍스트 청크 검색 및 페이지 텍스트 조회를 통해 각 추출 키에 대한 유력 근거 페이지(1~3장)를 찾아 사유와 함께 제출합니다. 시작 시 문서 개요(Compact TOC)가 기본 제공됩니다.",
        "contract": {
            "completion_type": "tool_call",
            "completion_description": "SearchAgent는 자유 형식 텍스트만으로 종료할 수 없으며, 반드시 submit_pages 또는 no_relevant_pages 도구를 호출하여 세션을 완료해야 합니다.",
            "final_schema_example": {
                "submit_pages_arguments": {
                    "key": "Distance between GTG",
                    "pages": [12, 11],
                    "page_reasons": {
                        "12": "6.3.4절에서 주요 구조물 간 50ft 이격 거리 명시됨.",
                        "11": "배치도 섹션에서 터빈 인클로저 간 적정 이격 거리가 확인됨."
                    },
                    "page_chunk_id": {
                        "12": "12-3",
                        "11": "11-1"
                    }
                },
                "no_relevant_pages_arguments": {
                    "key": "Portable Fire Extinguishers",
                    "reason": "소화 설비 섹션 및 전체 문서 검색 결과 휴대용 소화기 관련 내용을 찾을 수 없음."
                }
            },
            "intermediate_schemas": [],
            "available_tools": [
                {
                    "name": "load_kv_schema",
                    "args": "keys?: string[]",
                    "description": "키 설명 및 검색 큐(search cues), 허용값을 확인합니다."
                },
                {
                    "name": "bm25_search",
                    "args": "queries: string[], top_k?: int",
                    "description": "헤딩/섹션 단위 BM25 청크 검색을 수행하여 chunk_id, pages, heading_path, snippet을 획득합니다."
                },
                {
                    "name": "get_page_text",
                    "args": "page: int",
                    "description": "특정 페이지 전체 텍스트를 조회합니다 (page 0은 상세 목차, 1 이상은 실제 본문 페이지)."
                },
                {
                    "name": "submit_pages",
                    "args": "pages: int[], page_reasons: dict, page_chunk_id: dict, key?: string",
                    "description": "1~3장의 고신뢰도 근거 페이지를 제출하여 키 검색을 완료합니다. page_reasons는 반드시 한국어로 작성해야 합니다."
                },
                {
                    "name": "no_relevant_pages",
                    "args": "reason: string, key?: string",
                    "description": "문서에 관련 정보가 없음을 확인하고 종료합니다. reason은 반드시 한국어로 작성해야 합니다."
                }
            ]
        }
    },
    "eval_master": {
        "id": "eval_master",
        "title": "Eval Master Agent",
        "filename": "eval_master_system_prompt.txt",
        "role": "Agentic Evaluation",
        "category": "evaluation",
        "tags": ["Evaluation", "Master"],
        "pipeline_file": "inference-pipeline/agentic/evaluation_pipeline.py",
        "description": "단일 추출 키에 대해 추론값(pred)을 문서와 교차 검증하여 정답 여부(is_correct_answer)를 판정합니다. 정답지(GT)가 있으면 GT 유효성(is_valid_gold)도 함께 판정하고, 없으면 is_valid_gold는 n/a입니다.",
        "contract": {
            "completion_type": "tool_call",
            "completion_description": "Eval Master Agent는 판정 완료 시 반드시 submit_evaluation 도구를 정확히 1회 호출하여 구조화된 판정 결과를 기록해야 합니다.",
            "final_schema_example": {
                "submit_evaluation_arguments": {
                    "is_correct_answer": "correct | incorrect",
                    "is_valid_gold": "valid | invalid | n/a",
                    "reason_summary": "1줄 한국어 핵심 요약 (예: 추론값 문서 근거와 일치)",
                    "reason_detail": "페이지 인용과 함께 상세한 한국어 판정 근거 서술"
                }
            },
            "intermediate_schemas": [],
            "available_tools": [
                {
                    "name": "get_kv_result",
                    "args": "key?: string",
                    "description": "해당 키의 모델 추론값(pred), 근거 인용구, 검색된 페이지를 조회합니다."
                },
                {
                    "name": "get_gold",
                    "args": "key?: string",
                    "description": "정답지(answer sheet)의 정답값(gold), 근거 인용구 및 페이지를 조회합니다. GT가 없으면 found=false를 반환합니다."
                },
                {
                    "name": "load_kv_schema",
                    "args": "key?: string",
                    "description": "평가 대상 키의 정의 및 허용값 목록을 확인합니다."
                },
                {
                    "name": "search_pages",
                    "args": "keys: string[]",
                    "description": "독립적 문서 확인이 필요한 경우 SearchAgent에 검색 작업을 요청합니다."
                },
                {
                    "name": "collect_search_results",
                    "args": "policy?: 'any' | 'all'",
                    "description": "SearchAgent의 검색 결과 페이지 및 근거 사유를 수집합니다."
                },
                {
                    "name": "page_image_chat_vlm",
                    "args": "pages: int[], prompt: string",
                    "description": "도면, 표 등 시각적 레이아웃 확인이 필요할 때 VLM에 질의하여 텍스트 답변을 받습니다."
                },
                {
                    "name": "submit_evaluation",
                    "args": "is_correct_answer: str, is_valid_gold: str, reason_summary: str, reason_detail: str",
                    "description": "정답 여부(및 GT가 있으면 GT 유효성, 없으면 n/a)와 한국어 사유를 확정 제출하고 평가를 마칩니다."
                }
            ]
        }
    },
    "kv_schema": {
        "id": "kv_schema",
        "title": "KV Extraction Schema",
        "filename": "kv_description.json",
        "role": "Schema & Cues",
        "category": "schema",
        "tags": ["Schema", "Extraction", "Search Cues"],
        "pipeline_file": "dataset/kv_description.json",
        "description": "문서에서 추출할 각 Key의 설명, 검색 큐(Search Cues), 허용값(Allowed values) 및 기본값을 정의합니다. SearchAgent의 BM25 검색 큐 생성 및 MasterAgent의 추출 정확도에 직접적인 영향을 미칩니다.",
        "contract": {
            "completion_type": "json_schema_definition",
            "completion_description": "각 항목은 key(추출 대상 명칭)와 description(추출 지침, 검색 큐, 허용값)으로 구성된 JSON 배열입니다.",
            "final_schema_example": [
                {
                    "key": "Distance between GTG",
                    "description": "Extract the separation distance between gas turbine (GTG/CTG/CT) enclosures. Search cues: separation distance, turbine enclosures, CT, GTG, 90', feet. Convert feet to meters if needed and map to the nearest allowed choice. Allowed values (choose exactly one): '25m 초과' / '20~25m' / '20m' / '15~20m' / '15m 이내'. If the document has no usable distance information, output '20m' (default when missing)."
                }
            ],
            "intermediate_schemas": [],
            "available_tools": [
                {
                    "name": "load_kv_schema",
                    "args": "keys?: string[]",
                    "description": "MasterAgent 및 SearchAgent가 런타임에 이 스키마를 로드하여 키별 검색 큐와 허용값을 파악합니다."
                }
            ]
        }
    },
    "extract_kv_vlm": {
        "id": "extract_kv_vlm",
        "title": "Extract KV VLM Tool",
        "filename": "extract_kv_vlm_prompt.txt",
        "role": "VLM Extraction",
        "category": "extraction",
        "tags": ["VLM", "Extraction", "Tool"],
        "pipeline_file": "inference-pipeline/agentic/tools.py",
        "description": "선정된 문서 페이지(이미지+텍스트)와 대상 키 정의를 VLM에 전달하여 Structured Output으로 값(value)과 한국어 근거 사유(value_reason)를 추출하는 프롬프트 템플릿입니다. 부재 시 결측 기본값 처리 지침을 포함합니다.",
        "contract": {
            "completion_type": "structured_output_json",
            "completion_description": "OpenAI/vLLM response_format(json_schema strict=True)으로 각 키별 key, value, value_reason 객체 배열이 반환됩니다.",
            "final_schema_example": {
                "extractions": [
                    {
                        "key": "Distance between GTG",
                        "value": "20m",
                        "value_reason": "도면에서 가스 터빈 인클로저 간 거리가 20m로 확인됨."
                    },
                    {
                        "key": "Trend Analysis",
                        "value": "미수행",
                        "value_reason": "문서 전체에 운영 데이터 통계/추세 분석에 대한 언급이 없어 결측 기본값 '미수행'을 적용함."
                    }
                ]
            },
            "intermediate_schemas": [],
            "available_tools": [
                {
                    "name": "{guidance_block}",
                    "args": "placeholder",
                    "description": "SearchAgent가 수집한 페이지별 선정 사유(page_reasons) 및 BM25 chunk_id가 런타임에 동적으로 주입되는 위치입니다."
                },
                {
                    "name": "{schema_json}",
                    "args": "placeholder",
                    "description": "추출 대상 키들의 스키마 정의(설명, 검색 큐, 허용값)가 JSON 형식으로 주입되는 위치입니다."
                }
            ]
        }
    }
}


def repo_root() -> Path:
    here = Path(__file__).resolve()
    # Path is .../agentic-viewer/agentic_viewer/prompts/store.py -> parents[3] is repo root
    if len(here.parents) > 3 and (here.parents[3] / "dataset").is_dir():
        return here.parents[3].resolve()
    return here.parents[2].resolve()


def prompts_dir() -> Path:
    env = os.environ.get("AGENTIC_PROMPTS_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    default_dir = repo_root() / "dataset"
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir.resolve()


def schema_file_path() -> Path:
    pdir = prompts_dir()
    schema_path = pdir / "kv_description.json"
    if schema_path.is_file():
        return schema_path.resolve()
    alt = repo_root() / "dataset" / "kv_description.json"
    if alt.is_file():
        return alt.resolve()
    return schema_path.resolve()


def load_schema_keys() -> List[Dict[str, Any]]:
    path = schema_file_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            res: List[Dict[str, Any]] = []
            for item in data:
                if isinstance(item, dict) and item.get("key"):
                    res.append({
                        "key": str(item["key"]).strip(),
                        "description": str(item.get("description") or "").strip(),
                    })
            return res
    except Exception:
        pass
    return []


def _resolve_prompt_path(prompt_id: str) -> Path:
    cfg = PROMPT_CONFIGS.get(prompt_id)
    if not cfg:
        raise ValueError(f"Unknown prompt id: {prompt_id}. Must be one of: {list(PROMPT_CONFIGS.keys())}")
    return (prompts_dir() / cfg["filename"]).resolve()


def _backup_pattern(filename: str) -> re.Pattern[str]:
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    return re.compile(rf"^{re.escape(stem)}\.bak\.(\d{{8}}T[\d_]+Z){re.escape(suffix)}$")


def list_prompts() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    pdir = prompts_dir()

    for pid, cfg in PROMPT_CONFIGS.items():
        path = pdir / cfg["filename"]
        exists = path.is_file()
        line_count = 0
        size_bytes = 0
        updated_at = None

        if exists:
            stat = path.stat()
            size_bytes = stat.st_size
            updated_at = datetime.fromtimestamp(stat.st_mtime, tz=KST).isoformat()
            try:
                line_count = len(path.read_text(encoding="utf-8").splitlines())
            except Exception:
                pass

        backups = list_backups(pid)

        item = {
            "id": pid,
            "title": cfg["title"],
            "filename": cfg["filename"],
            "path": str(path),
            "role": cfg["role"],
            "category": cfg["category"],
            "tags": cfg["tags"],
            "pipeline_file": cfg["pipeline_file"],
            "description": cfg["description"],
            "exists": exists,
            "size_bytes": size_bytes,
            "line_count": line_count,
            "updated_at": updated_at,
            "backup_count": len(backups),
        }
        results.append(item)

    return results


def get_prompt(prompt_id: str) -> Dict[str, Any]:
    cfg = PROMPT_CONFIGS.get(prompt_id)
    if not cfg:
        raise ValueError(f"Unknown prompt id: {prompt_id}")

    path = _resolve_prompt_path(prompt_id)
    content = ""
    exists = path.is_file()
    size_bytes = 0
    line_count = 0
    updated_at = None

    if exists:
        stat = path.stat()
        size_bytes = stat.st_size
        updated_at = datetime.fromtimestamp(stat.st_mtime, tz=KST).isoformat()
        try:
            content = path.read_text(encoding="utf-8")
            line_count = len(content.splitlines())
        except Exception as exc:
            content = f"Error reading file: {exc}"

    backups = list_backups(prompt_id)
    schema_keys = load_schema_keys() if prompt_id in ("master", "search", "kv_schema") else []

    return {
        "id": prompt_id,
        "title": cfg["title"],
        "filename": cfg["filename"],
        "path": str(path),
        "role": cfg["role"],
        "category": cfg["category"],
        "tags": cfg["tags"],
        "pipeline_file": cfg["pipeline_file"],
        "description": cfg["description"],
        "exists": exists,
        "size_bytes": size_bytes,
        "line_count": line_count,
        "updated_at": updated_at,
        "content": content,
        "contract": cfg["contract"],
        "schema_keys": schema_keys,
        "backups": backups,
    }


def list_backups(prompt_id: str) -> List[Dict[str, Any]]:
    cfg = PROMPT_CONFIGS.get(prompt_id)
    if not cfg:
        return []

    pdir = prompts_dir()
    pattern = _backup_pattern(cfg["filename"])
    backups: List[Dict[str, Any]] = []

    try:
        for entry in pdir.iterdir():
            if not entry.is_file():
                continue
            match = pattern.match(entry.name)
            if match:
                ts_str = match.group(1)
                stat = entry.stat()
                backups.append({
                    "filename": entry.name,
                    "path": str(entry.resolve()),
                    "timestamp": ts_str,
                    "size_bytes": stat.st_size,
                    "updated_at": datetime.fromtimestamp(stat.st_mtime, tz=KST).isoformat(),
                })
    except Exception:
        pass

    backups.sort(key=lambda b: b["timestamp"], reverse=True)
    return backups


def get_backup(prompt_id: str, timestamp_or_filename: str) -> Dict[str, Any]:
    cfg = PROMPT_CONFIGS.get(prompt_id)
    if not cfg:
        raise ValueError(f"Unknown prompt id: {prompt_id}")

    pdir = prompts_dir()
    candidate: Optional[Path] = None

    if timestamp_or_filename.endswith(".txt") or timestamp_or_filename.endswith(".json"):
        candidate = pdir / timestamp_or_filename
    else:
        stem = Path(cfg["filename"]).stem
        suffix = Path(cfg["filename"]).suffix
        candidate = pdir / f"{stem}.bak.{timestamp_or_filename}{suffix}"

    if not candidate.is_file():
        raise FileNotFoundError(f"Backup file not found: {timestamp_or_filename}")

    stat = candidate.stat()
    content = candidate.read_text(encoding="utf-8")
    return {
        "filename": candidate.name,
        "path": str(candidate.resolve()),
        "size_bytes": stat.st_size,
        "updated_at": datetime.fromtimestamp(stat.st_mtime, tz=KST).isoformat(),
        "content": content,
    }


def save_prompt(prompt_id: str, content: str) -> Dict[str, Any]:
    if not isinstance(content, str):
        raise ValueError("Prompt content must be a string")

    cfg = PROMPT_CONFIGS.get(prompt_id)
    if not cfg:
        raise ValueError(f"Unknown prompt id: {prompt_id}")

    if prompt_id == "kv_schema":
        try:
            parsed = json.loads(content)
            if not isinstance(parsed, list):
                raise ValueError("KV schema must be a JSON array of objects with 'key' and 'description'")
            for idx, item in enumerate(parsed):
                if not isinstance(item, dict) or "key" not in item:
                    raise ValueError(f"Item at index {idx} must be an object with at least a 'key' field")
            content = json.dumps(parsed, ensure_ascii=False, indent=2) + "\n"
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON syntax for KV schema: {exc}")

    path = _resolve_prompt_path(prompt_id)
    pdir = path.parent
    pdir.mkdir(parents=True, exist_ok=True)

    backup_info: Optional[Dict[str, Any]] = None
    if path.is_file():
        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        stem = Path(cfg["filename"]).stem
        suffix = Path(cfg["filename"]).suffix
        backup_path = pdir / f"{stem}.bak.{stamp}{suffix}"
        if backup_path.is_file():
            stamp = now.strftime("%Y%m%dT%H%M%S_%fZ")
            backup_path = pdir / f"{stem}.bak.{stamp}{suffix}"
        shutil.copy2(path, backup_path)
        stat = backup_path.stat()
        backup_info = {
            "filename": backup_path.name,
            "timestamp": stamp,
            "path": str(backup_path.resolve()),
            "size_bytes": stat.st_size,
        }

    # Atomic write to avoid partial corruption
    fd, tmp_path_str = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(pdir))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)

    os.replace(tmp_path_str, str(path))

    stat = path.stat()
    return {
        "id": prompt_id,
        "filename": cfg["filename"],
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "line_count": len(content.splitlines()),
        "updated_at": datetime.fromtimestamp(stat.st_mtime, tz=KST).isoformat(),
        "backup": backup_info,
    }


def restore_prompt(prompt_id: str, timestamp_or_filename: str) -> Dict[str, Any]:
    backup = get_backup(prompt_id, timestamp_or_filename)
    return save_prompt(prompt_id, backup["content"])


def save_schema_keys(keys: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(keys, list):
        raise ValueError("keys must be a list of dicts with 'key' and 'description'")
    content = json.dumps(keys, ensure_ascii=False, indent=2) + "\n"
    return save_prompt("kv_schema", content)


def compute_diff(
    original: str,
    modified: str,
    fromfile: str = "current",
    tofile: str = "modified",
) -> str:
    orig_lines = original.splitlines(keepends=True)
    mod_lines = modified.splitlines(keepends=True)
    diff = difflib.unified_diff(
        orig_lines,
        mod_lines,
        fromfile=fromfile,
        tofile=tofile,
        n=3,
    )
    return "".join(diff)

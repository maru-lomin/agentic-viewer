#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PRED="${PRED:-../inference-pipeline/outputs/result.json}"
ANSWER_SHEET="${ANSWER_SHEET:-../dataset/answer_sheet.json}"
OUTPUT="${OUTPUT:-./outputs/eval_report.json}"

EXTRA_ARGS=()
if [[ -n "${DOCUMENT:-}" ]]; then
  EXTRA_ARGS+=(--document "${DOCUMENT}")
fi

uv run python -m agentic_viewer.eval.evaluate_kv \
  --pred "${PRED}" \
  --answer-sheet "${ANSWER_SHEET}" \
  --output "${OUTPUT}" \
  "${EXTRA_ARGS[@]}" \
  "$@"

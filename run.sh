#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Default: sibling inference-pipeline runs dir when present.
if [[ -z "${AGENTIC_RUNS_DIR:-}" ]]; then
  SIBLING="$ROOT/../inference-pipeline/outputs/runs"
  if [[ -d "$SIBLING" || -d "$ROOT/../inference-pipeline" ]]; then
    mkdir -p "$SIBLING"
    export AGENTIC_RUNS_DIR="$(cd "$SIBLING" && pwd)"
  else
    mkdir -p "$ROOT/runs"
    export AGENTIC_RUNS_DIR="$ROOT/runs"
  fi
fi

echo "runs_root=$AGENTIC_RUNS_DIR"
echo "inference_api=${INFERENCE_API_URL:-http://127.0.0.1:8010}  (for agentic-evaluation)"
exec uv run python -m agentic_viewer.app

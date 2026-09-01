#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Default: shared repo outputs/runs (see ../outputs/README.md).
if [[ -z "${AGENTIC_RUNS_DIR:-}" ]]; then
  SHARED="$ROOT/../outputs/runs"
  LEGACY="$ROOT/../inference-pipeline/outputs/runs"
  if [[ -d "$SHARED" || -d "$ROOT/../outputs" ]]; then
    mkdir -p "$SHARED"
    export AGENTIC_RUNS_DIR="$(cd "$SHARED" && pwd)"
  elif [[ -d "$LEGACY" || -d "$ROOT/../inference-pipeline" ]]; then
    mkdir -p "$LEGACY"
    export AGENTIC_RUNS_DIR="$(cd "$LEGACY" && pwd)"
  else
    mkdir -p "$ROOT/runs"
    export AGENTIC_RUNS_DIR="$ROOT/runs"
  fi
fi

echo "runs_root=$AGENTIC_RUNS_DIR"
echo "inference_api=${INFERENCE_API_URL:-http://127.0.0.1:8010}  (for agentic-evaluation)"
exec uv run python -m agentic_viewer.app

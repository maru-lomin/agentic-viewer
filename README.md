# Agentic run-trace viewer

FastAPI UI for browsing `outputs/runs/<request_id>/` artifacts produced by the
Korean Re agentic inference pipeline (parse / chunk / agent traces).

## Quick start

```bash
# From this repo
export AGENTIC_RUNS_DIR=/path/to/inference-pipeline/outputs/runs
./run.sh
# → http://127.0.0.1:8099
```

If `AGENTIC_RUNS_DIR` is unset and this repo sits next to `inference-pipeline/`,
it defaults to `../inference-pipeline/outputs/runs`.

## Tabs

- **Chat** — message-level transcript (`conversation.jsonl`)
- **Agent steps (visualize)** — per LLM turn: request → assistant → tool results
- Timeline / Pages / Result — pipeline stage dumps

## Env

| Variable | Default | Meaning |
|----------|---------|---------|
| `AGENTIC_RUNS_DIR` | sibling `inference-pipeline/outputs/runs` | runs root |
| `TRACE_VIEWER_HOST` | `0.0.0.0` | bind host |
| `TRACE_VIEWER_PORT` | `8099` | bind port |

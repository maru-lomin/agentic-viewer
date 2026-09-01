# Agentic run-trace viewer

FastAPI UI for browsing `outputs/runs/<request_id>/` artifacts produced by the
Korean Re agentic inference pipeline (parse / chunk / agent traces).

## Quick start

```bash
# From this repo
export AGENTIC_RUNS_DIR=/path/to/outputs/runs
./run.sh
# → http://127.0.0.1:8099
```

If `AGENTIC_RUNS_DIR` is unset, defaults to shared `../outputs/runs`
(legacy fallback: `../inference-pipeline/outputs/runs`). See `../outputs/README.md`.

## Tabs

- **Agent hierarchy** — Master → tools → SearchAgent sessions
- **Timing** — wall / model time breakdown
- **Pages** — parsed page markdown
- **Eval** — value Exact Match, search page P/R/F1, evidence token F1 vs `dataset/answer_sheet.json` (cached as `05_eval.json` when the runs dir is writable)

## CLI evaluation

```bash
./evaluate.sh
# PRED=../inference-pipeline/outputs/result.json ./evaluate.sh
# AGENTIC_ANSWER_SHEET=/path/to/answer_sheet.json ./evaluate.sh
```

## Env

| Variable | Default | Meaning |
|----------|---------|---------|
| `AGENTIC_RUNS_DIR` | `../outputs/runs` | runs root |
| `AGENTIC_ANSWER_SHEET` | `../dataset/answer_sheet.json` | gold labels for Eval |
| `TRACE_VIEWER_HOST` | `0.0.0.0` | bind host |
| `TRACE_VIEWER_PORT` | `8099` | bind port |

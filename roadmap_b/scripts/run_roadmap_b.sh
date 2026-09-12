#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="$(realpath "${BASELINE_ROOT:-../roadmap_a_source}")"
PY="${PYTHON:-$BASE/.venv/bin/python}"
OUT="${ROADMAP_B_OUTPUT:-artifacts/roadmap_b}"
ARGS=(--baseline-root "$BASE" --output-root "$OUT")
if [[ -n "${PROCESSED_ROOT:-}" ]]; then ARGS+=(--audio-root "$PROCESSED_ROOT"); fi
case "${1:-}" in
  replay) "$PY" -m src.b_data replay "${ARGS[@]}" ;;
  extract) "$PY" -m src.b_data extract "${ARGS[@]}" ;;
  extract-text) "$PY" -m src.b_data extract-text "${ARGS[@]}" ;;
  train) "$PY" -m src.b_train "${ARGS[@]}" ;;
  evaluate) "$PY" -m src.b_evaluate "${ARGS[@]}" ;;
  tests) "$PY" scripts/validate_b.py --baseline-root "$BASE" --output-root "$OUT" ;;
  validate)
    "$PY" scripts/validate_b.py --baseline-root "$BASE" --output-root "$OUT"
    "$PY" -m src.b_inference validate "${ARGS[@]}"
    ;;
  report) "$PY" -m src.b_report "${ARGS[@]}"; "$PY" scripts/complete_b_report.py --output-root "$OUT" ;;
  all)
    "$PY" -m src.b_data replay "${ARGS[@]}"
    "$PY" -m src.b_data extract "${ARGS[@]}"
    "$PY" -m src.b_data extract-text "${ARGS[@]}"
    "$PY" -m src.b_train "${ARGS[@]}"
    "$PY" -m src.b_evaluate "${ARGS[@]}"
    "$PY" scripts/validate_b.py --baseline-root "$BASE" --output-root "$OUT"
    "$PY" -m src.b_inference validate "${ARGS[@]}"
    "$PY" -m src.b_report "${ARGS[@]}"
    "$PY" scripts/complete_b_report.py --output-root "$OUT"
    ;;
  *) echo "Usage: $0 {replay|extract|extract-text|train|evaluate|tests|validate|report|all}" >&2; exit 2 ;;
esac

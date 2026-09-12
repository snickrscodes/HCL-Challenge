# Finalization sequence (no more training)

Current stage: provisional delivery smoke complete (60 accepted takes); independent L01 judgments and final presentation validation remain pending; FINAL_FREEZE.json and official-test STARTED.json must remain absent. A clean diagnostic failure is publishable and must not trigger tuning.

## Collection and local diagnostic

Follow `recording_kit/README.md`. Keep accepted human WAVs local. With the exact frozen A snapshot available as `$BASELINE_ROOT`:

```bash
python -m src.delivery status
python -m scripts.controlled_delivery_smoke listener
```

For this completed collection, the listener action verifies original private saved outputs and reinterprets them without rerunning encoders. It produces text/audio/A/B distributions, gate, actual influence, residual norms, listener summaries, failures and hardware/timing metadata in `artifacts/submission/delivery_results`. It refuses incomplete collection. No classifier or temperature is fitted. Keep model outputs and intended-delivery results hidden from L01 until blinded judgments are saved. The earlier provisional smoke is explicitly not listener-validated evidence.

## Presentation

Figures use saved development predictions only. Use a separate environment with `requirements-presentation.txt` for plotting:

```bash
python scripts/final_figures.py --help
python scripts/final_report.py --help
python -m src.submission_demo --help
```

The current `demo_draft` has five development cases. Build the final `demo` in a new destination after the fixed S01/p01 human recordings are complete, regardless of how their outputs look. Verify all six cases, report failures, record the trace demonstration and save `presentation_validation.json` with actual checks. Do not fabricate a passed validation record.

## Server validation and freeze

Run on the authorized RTX4090, with A preserved separately. Copy only new submission source/presentation/diagnostic results as needed; human audio stays local unless separately authorized. All commands below use A's existing environment. Check `--help` for explicit root overrides.

```bash
python -m src.final_evaluate policy --baseline-root "$BASELINE_ROOT"
python -m src.final_evaluate preflight --baseline-root "$BASELINE_ROOT"
python -m src.final_evaluate freeze --baseline-root "$BASELINE_ROOT"
```

Preflight tests loaded models, raw/cache equivalence, missing/long-audio fallback, parameter counts, fresh process and warm runtime without test inference. It preserves older validation attempts. Freeze requires completed human evaluation, current-source checks, six-case demo and presentation validation. It hashes source, checkpoints, temperatures, policy, manifests/splits and evidence. No FINAL_FREEZE.json exists at the present preparation stage.

## One official test pass — only after freeze

```bash
python -m src.final_evaluate test --baseline-root "$BASELINE_ROOT"
```

This is an explicitly separate command, never part of recording or reporting. It atomically creates `official_test/STARTED.json` before inference and refuses a second invocation, including automatic retry after failure. Investigate an interrupted run without silently restarting. The planned models are canonical causal text, fixed A, B adaptive1337, constant1337 and matched-text-only1337, plus eligible audio diagnostics. No batched historical text policy is reintroduced. The two >60s clips use text fallback; expected total support2610/audio-eligible2608 must be reconciled against actual availability.

All temperatures, alpha, weights, precision, threshold, labels and seed are frozen before test. No fitting/retraining follows test. The report states actual metrics, class harms and bootstrap uncertainty, then applies the predeclared production recommendation rule. Preserve A-original historical evidence distinctly from canonical final inference.

## Packaging after results

Finish README/report/test tables, record the demo, then create full and small evidence archives with SHA256 and extraction verification. Exclude virtual environments, package caches and raw MELD media. Preserve selected checkpoints, small heads, calibration, locks, results, source and tests. Respect each human take's separate public-release permission. Preparation artifacts are not a completed final submission.

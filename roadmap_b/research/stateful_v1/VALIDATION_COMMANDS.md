# Reproducing validation

Use the exact locked environment in `roadmap_b/requirements.lock.txt`, the documented external33-file T+A asset bundle, and Python3.12 with local RTX5080 Laptop for GPU acceptance. No research worktree belongs on PYTHONPATH. All commands run from the tested checkout.

```bash
cd roadmap_b
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests -q
RUN_PRETRAINED=1 B_BASELINE_ROOT=/path/to/retained-roadmap-a HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests -q -m pretrained
cd ..
python prepare_ta_assets.py --asset-root /path/to/ta-assets --verify-only
python roadmap_b/scripts/validate_stateful.py --asset-root /path/to/ta-assets --baseline-root /path/to/retained-roadmap-a --evidence-root /path/to/retained-roadmap-b/artifacts --output-root /new/validation-directory --benchmark
python roadmap_b/scripts/validate_stateful_resources.py --output /new/resource-validation.json
python ta.py --asset-root /path/to/ta-assets --text 'Hello.'
python stateful_ta.py --asset-root /path/to/ta-assets --text 'Hello.'
python stateful_ta.py --asset-root /path/to/ta-assets --events roadmap_b/examples/stateful_events.jsonl
```

The perception validator needs retained development assets only for validation, never production inference: raw15 manifest and float WAVs under `ta_finalization_v1/validation_data`, frozen B features/text_features and adaptive/1337 predictions for dev_model/dev_calib, plus original observed dev manifests. It explicitly restricts the GPU and development inventory. It performs zero training/calibration, records failures, rejects an existing output directory and preserves complete fixed timing samples.

Raw15 comparisons are bit-exact in one local environment. Saved historical CPU-head numeric differences are diagnostic; every frozen category/availability must match, and adapter equivalence is exact on identical regenerated inputs. Deterministic replay recomputes successful final estimates; recorded cancelled/failed executions are dispositions, not synthetic model evidence.

Also run existing `ta.py --replay` and new `stateful_ta.py --replay` on the retained four-turn raw fixture, waveform and missing-audio CLI checks, and asset verification from a foreign working directory. Public synthetic event fixture covers two externally supplied participant IDs. Debug logs/private media remain outside Git.

# Source provenance

Base: verified roadmap_a_full_continuation_2026-09-11.tar.gz.
Archive SHA256: bd8ec38403035ef378efa5a01035efe0ebb1a9f1cbf8fd08ec668013d774a2d0.
The baseline_lock.json is copied byte-for-byte from the completed review.

All original src modules and A tests are retained unchanged from that archive.
New B files:
- src/b_fusion.py: bounded correction and four controlled variants.
- src/b_data.py: immutable references, real numerical replay, operational policy, canonical caches.
- src/b_train.py: one fixed controlled matrix and teacher diagnostics.
- src/b_evaluate.py: calibration, paired evaluation, interventions, bootstrap.
- src/b_inference.py: shared A preprocessing/models/session with B correction and raw probes.
- src/b_report.py: measured report and advancement criteria.
- configs/roadmap_b.yaml, scripts/run_roadmap_b.sh, scripts/validate_b.py and B tests.

The raw B path reuses A's waveform preparation, WavLM wrapper, tokenization/current-span masks,
model loaders and response/session policy. It does not modify A's original inference method.
B wraps acoustic extraction to enforce the empirically selected singleton policy when required.

All added implementation is AI-assisted and must be judged by tests and real experiment evidence.
No new external model, framework, generated speech, reinforcement learning or response generator is used.

## Approved canonical text repair

Real RTX4090 replay confirmed BF16 text batch dependence. The user authorized separate singleton FP32 B text caches using unchanged T1 weights. Run `bash scripts/run_roadmap_b.sh extract-text` after audio extraction and before training. A-original is immutable; A-replay-canonical uses both canonical modalities with original alpha/temperatures. No A source was modified. The four numerical-replay keys supplement the predeclared raw validation probes before B training.

## Evaluation-only repair after head freeze

The first evaluation stopped while serializing an undefined correlation for the constant-gate control. Float32 standard-deviation roundoff incorrectly treated an exactly constant vector as varying. Correlation now converts inputs to float64 before checking variance, returning null for undefined correlation. The original freeze and failed evaluation are retained; evaluation_repair.json records old/new source hashes and the rerun. No checkpoint, cache, temperature selection rule, or model selection changed.

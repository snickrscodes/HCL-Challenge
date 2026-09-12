# Implementation and reproduction

Read the [project overview](../README.md) for current claims/results. This directory contains the current A and B implementation; the surrounding workspace's older source trees are not part of the public snapshot.

**PERCEPTION ARCHITECTURE SEARCH COMPLETE.** Adaptive seed 1337 is the fixed reference. Both speaker sessions (60 takes) and provisional intended-delivery inference are complete. Blinded L01 interpretation, final freeze, official test and final packaging are still pending. No completed research phase should be rerun into its populated artifact directory.

## Layout

- `src/data.py`, `context.py`, `audio.py`: audited media conversion, causal text packing, common audio preprocessing.
- `src/models.py`, `train_text.py`, `extract_audio.py`, `train_audio.py`, `train_fusion.py`, `calibrate.py`, `evaluate.py`, `pipeline.py`: Roadmap A.
- `src/b_data.py`, `b_fusion.py`, `b_train.py`, `b_evaluate.py`, `b_report.py`: canonical replay/cache policy and four controlled B heads.
- `src/inference.py`, `b_inference.py`, `session.py`, `responses.py`, `replay.py`: raw-input state/response interaction.
- `src/recording_kit.py`, `delivery.py`, `recording_kit/`, `scripts/controlled_delivery_smoke.py`: private human collection and frozen-model interpretation.
- `src/final_evaluate.py`, `submission_demo.py`, `scripts/final_figures.py`, `final_report.py`: downstream guarded evaluation and presentation tooling.
- `tests/`: A/B regression, residual mathematics, availability, numerical policy, recording and finalization checks. Synthetic fixtures are software tests, not scientific evidence.
- `evidence/`: explicit public exports of completed development results. Not a replacement for operational `artifacts/`.

## Environment

Linux, Python 3.12.14, PyTorch 2.8.0 with CUDA 12.8, Transformers 4.56.2. The full pinned environment is in `requirements.lock.txt`. The server used an RTX 4090; no system CUDA reinstallation was needed. Bootstrap installs the pinned environment and downloads dependencies; it does not download MELD or task checkpoints.

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
pytest -q
ruff check .
ruff format --check .
```

Ordinary tests skip the explicitly opt-in pretrained tests. `scripts/validate_b.py` documents/runs the additional loaded-checkpoint integration checks and requires the preserved A/B assets. Do not interpret unit-test success alone as reproducing measured real-model results. Presentation dependencies are separate in `requirements-presentation.txt`.

## Media acquisition and Roadmap A from scratch

Obtain media from the [official MELD project](https://github.com/declare-lab/MELD) / [declare-lab MELD dataset distribution](https://huggingface.co/datasets/declare-lab/MELD), subject to the owners' terms. Media are not redistributed in Git. Extract the train/dev/test split archives into a supplied root containing `train_splits/`, `dev_splits_complete/`, and `test_splits_complete/`. On filesystems rejecting archive ownership, use `tar --no-same-owner --no-same-permissions` when extracting split archives.

Use the current official pinned annotation revision, not an older nested train CSV: the latter differed in 2,697 rows, mostly Unicode punctuation corruption. The preprocessing code acquires/verifies the pinned official annotation files. Exact canonical `dia{Dialogue_ID}_utt{Utterance_ID}` stems avoid AppleDouble and alternate-video junk. FFmpeg produces mono 16-kHz float WAV without loudness normalization, denoising or silence trimming; missing/corrupt clips remain explicit failures.

```bash
python -m src.data --raw-root /path/to/MELD.Raw --output-root data/processed
bash scripts/run_roadmap_a.sh
```

This is a new training reproduction, **not** the immutable historical A snapshot. It trains T0/T1, extracts frozen WavLM features and trains the small audio/concat heads, selects raw-logit alpha and fits calibration on a separate grouped partition. The regular pipeline does not run official-test inference. Do not add its final-test flag during development.

Published counts/split IDs and annotation hashes are in [data audit](evidence/data_audit/) and [baseline_lock.json](baseline_lock.json). Historical A used 86 dev-model dialogues / 28 calibration dialogues, seed 1337 and calibration fraction .25. Never regenerate the actual continuation split after observing outcomes. Fresh training can yield different weights even with identical settings; do not rewrite the historical lock to call it the original result.

## Exact selected checkpoint inference

The public repository intentionally has no learned blobs, media, pooled features or private human outputs. The exact artifact bundle must be supplied separately; no public task-checkpoint hosting endpoint currently exists. Restore:

1. Original `roadmap_a_source/` separately, including the 71 source/checkpoint/cache/split files named by `baseline_lock.json` and required processed data. Do not extract B over A.
2. Completed B `artifacts/roadmap_b/` under this directory, including original `model_freeze.json`, canonical/text policy, adaptive seed-1337 checkpoint and calibration; preserve controls/evidence as supplied.
3. For raw MELD validation, original processed WAVs. A supplied standalone WAV is sufficient as audio for a single prediction, but the current integrity loader still expects the locked A continuation files.

These are the **original operational artifacts**, not path-redacted files under `evidence/`. Their hashes have not been changed by public export. This prototype prioritizes exact continuation verification; it does not yet implement a separate minimal-weight distribution loader.

```bash
export BASELINE_ROOT=/path/to/roadmap_a_source
export PYTHON="$PWD/.venv/bin/python"
python -m src.b_inference predict --baseline-root "$BASELINE_ROOT" \
  --audio /path/to/example.wav --text "Yeah, sure." --history examples/history.json
```

Omit `--audio` for calibrated text fallback. History is a JSON list of `text`/`speaker` turns. The session receives chunks, waits for end-of-turn, encodes text singleton FP32 and audio singleton BF16, applies the selected residual and emits state/response. No feature cache substitutes for waveform encoding in this path.

## Reproducing the completed B experiment

This is an audit/reproduction sequence, not a request to repeat the finished search. It requires the exact original A bundle and separate new B output directory. `BASELINE_ROOT` points to A; `PROCESSED_ROOT`, if set, points to the directory containing `audio/`, not `audio/` itself. `ROADMAP_B_OUTPUT` selects a new destination. Commands refuse overwrites.

```bash
export ROADMAP_B_OUTPUT=/path/to/new_b_reproduction
bash scripts/run_roadmap_b.sh replay
# Preserve/review numerical replay before proceeding.
bash scripts/run_roadmap_b.sh extract
# extract-text additionally requires the recorded FP32 text replay evidence.
bash scripts/run_roadmap_b.sh extract-text
bash scripts/run_roadmap_b.sh train
bash scripts/run_roadmap_b.sh evaluate
bash scripts/run_roadmap_b.sh validate
bash scripts/run_roadmap_b.sh report
```

The four-clip WavLM replay preceded training. Separate text FP32 diagnostic evidence and approval were recorded during the original investigation; `extract-text` expects those diagnostic files in `numerical_replay/`. The phase script does not recreate that original investigation automatically. Review [text repair evidence](evidence/roadmap_b/numerical_replay/text_fp32_diagnostics.json) and the checks in `b_data.py`; do not fabricate an approval or bypass them. For studying the measured run, use saved evidence rather than rerunning training.

All four heads use frozen representations and raw original logits, ordinary CE, AdamW LR .001, weight decay .01, batch 128, 20-epoch ceiling, patience 3, and `dev_model` macro F1. Heads run at 1337/1338/1339. Final scalar temperatures use only `dev_calib`. Mean/std, seed-1337 paired results, confidence slices, dialogue bootstrap, shuffle/mean/same-speaker interventions, real silence and fixed 10-dB noise are already measured. No loss sweep or optional imbalance follow-up was run.

## Human diagnostic

```bash
python -m src.recording_kit --port 8765
python -m src.delivery status
```

Follow [the collection guide](recording_kit/README.md). Keep raw audio, consent/acceptance metadata, private blinded-token mappings and ratings local. Participation consent does not grant public audio release. Ten fixed phrases × three takes × two speakers were already accepted; do not record new takes based on model behavior.

For a new fully accepted collection, the explicit provisional action is:

```bash
python -m scripts.controlled_delivery_smoke run --baseline-root "$BASELINE_ROOT"
```

**Do not rerun it on the completed collection.** When L01 is finished, reuse original saved private outputs:

```bash
python -m scripts.controlled_delivery_smoke listener
```

Only the interpretation layer runs. Intended labels are temporary assumptions, never human ground truth. The legacy small `examples/delivery_manifest.json` / `b_inference delivery` interface remains as historical tooling; the fixed 60-take browser protocol supersedes it for this project. Do not use the older `src.delivery run` to re-infer the completed collection.

## Evidence and finalization

[Evidence map](evidence/README.md) lists original-source/public-export hashes and what each directory establishes. Source files used in completed model experiments were not edited for this Git preparation; documentation/export tooling is newer and requires the later current-source final preflight. Existing model/source locks remain historical identities, not a declaration that this entire Git snapshot has passed final freeze.

Read [FINALIZATION.md](FINALIZATION.md) before any downstream evaluation. **Official test remains unopened.** Complete blinded listener interpretation and presentation, rerun current-source preflight, create `FINAL_FREEZE.json`, then run the separately authorized one-pass test. No training or model selection follows test. A remains the permanent fallback.

[External components](EXTERNAL_COMPONENTS.md) and [source changes](SOURCE_CHANGES.md) document ownership and numerical repairs. Private study material is deliberately absent.

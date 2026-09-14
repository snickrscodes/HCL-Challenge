# HCL Challenge: local T+A release candidate

**Scientific scope: `EVIDENCE_ONLY_BASELINE_WITH_RESEARCH_RESULTS`.** The selected system is frozen B/adaptive/1337 plus frozen D2/1337. Audio can change B's seven-category MELD state through its learned bounded correction, and that state drives the authored response. D2 supplies a separate six-category vocal estimate and normalized 128D vector; it does not change the default response.

One fixed role-specific response hypothesis was evaluated and rejected for insufficient effect and coverage. It remains explicitly selectable as `--policy role_specific_h1` for research review. This release does not solve delivery-sensitive categorical integration or establish human response usefulness. Independent evaluation is pending; no official MELD test or former-confirmation inference was run in this continuation.

Start with [START_HERE_TA.md](START_HERE_TA.md) for equations, ownership and self-checks, [TA_SYSTEM_CONTRACT.md](TA_SYSTEM_CONTRACT.md) for field responsibilities, and [TA_SYSTEM_REPORT.md](TA_SYSTEM_REPORT.md) for the measured decision and validation. [The release manifest](roadmap_b/evidence/ta_system_closure/RELEASE.json) selects the default. Earlier A/B research remains in [roadmap_b/README.md](roadmap_b/README.md).

## Environment

The tested environment is Linux/WSL, Python 3.12.14, PyTorch 2.8.0+cu128, Transformers 4.56.2 and an RTX 5080 Laptop GPU. Using an installed `uv`, create an environment from the repository root:

```bash
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match roadmap_b/requirements.lock.txt
```

The exact 55-package installation was tested in a new environment using the existing local package cache and `--offline`. Fresh network downloads were not tested. Versions are pinned; the additional index supplies the CUDA PyTorch build. `--device cpu` explicitly selects a separate FP32 execution scope; current raw validation and timing cover the RTX 5080 only. The recorded RTX 4090 BF16 cache discrepancy remains unresolved.

## External assets

The repository is **not self-contained**: selected weights remain external. The tested minimal bundle contains 33 files, 880,734,827 bytes, with exact identities in [ASSET_MANIFEST.json](roadmap_b/evidence/ta_system_closure/ASSET_MANIFEST.json). Pretrained revisions and third-party terms are in [EXTERNAL_COMPONENTS.md](roadmap_b/EXTERNAL_COMPONENTS.md). No new weight download URL or redistribution permission is implied.

Prepare once from retained local A, B and C1 roots:

```bash
.venv/bin/python prepare_ta_assets.py --asset-root /path/to/ta_assets --baseline-root /path/to/retained/roadmap_a_source --b-root /path/to/retained/artifacts/roadmap_b --c1-root /path/to/retained/artifacts/roadmap_c1_v2
.venv/bin/python prepare_ta_assets.py --asset-root /path/to/ta_assets --verify-only
```

The preparer copies only the hash-verified allowlist, never downloads or overwrites changed assets. The resulting bundle needs no corpus, training cache, D0/D3 checkpoint, separate normalization file or original source tree. Complete runtime source, policy, selection and root entry points are checked before model loading. Keep the model bundle backed up separately from Git; a source archive cannot replace it.

## Run from the repository root

```bash
.venv/bin/python ta.py --asset-root /path/to/ta_assets --audio /path/to/utterance.wav --text 'The exact observed transcript.' --chunk-samples 1600
.venv/bin/python ta.py --asset-root /path/to/ta_assets --text 'I am thinking about what to say.'
.venv/bin/python ta.py --asset-root /path/to/ta_assets --replay roadmap_b/examples/ta_replay.json --chunk-samples 1600
```

The replay example contains observed human turns without audio and demonstrates fallback. To replay actual PCM, use a JSON object with `turns`, each containing `text`, optional `speaker`, and optional `audio` path. Audio paths resolve relative to the replay JSON. A four-turn chunked raw development replay was also tested; its two absent recordings explicitly fall back to text. The transcript is supplied, not transcribed by ASR. PCM is buffered incrementally; WavLM runs at utterance end.

Output contains `turns[].meld_state`, `acoustic_evidence`, `interaction` and authoritative `response`, plus provenance, timing and the 218,895,839-parameter ledger. Nested `meld_state.response` retains B's reference response. Use `--output new-result.json` to save without overwriting. Missing/corrupt audio falls back explicitly; decoded audio over 60 seconds is valid but ineligible; silence is valid. Only observed human turns enter the last-three-turn context. Paths are configurable and the entry point works from other directories when invoked by absolute path.

## Verification and aggregate regeneration

```bash
cd roadmap_b
../.venv/bin/python -m pytest -q
../.venv/bin/python scripts/ta_system_evaluate.py --evidence-root /path/to/retained/roadmap_b/artifacts --output-root /path/to/new/aggregate-output
```

Aggregate regeneration reads fixed saved training/development/working evidence and verifies hashes; it runs no model inference or fitting. The large evidence dependency is separate from inference assets. It refuses forbidden partitions and never reopens former confirmation. Original historical failures remain in the report. See the report for the separate pretrained and raw-validation commands and hardware scope.

No push, remote PR, release publication, upload, cloud-resource change, model retraining or extension was performed. The local review branch is intended for user review and a user-initiated push.

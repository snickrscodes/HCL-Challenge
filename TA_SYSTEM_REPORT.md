# T+A system closure report

## Decision and scope

The chosen contract separates contextual MELD interpretation, vocal listener evidence and authored response. The delivered default is **EVIDENCE_ONLY_BASELINE_WITH_RESEARCH_RESULTS**, not a successful integrated or role-specific interaction release. Frozen B/adaptive/1337 remains authoritative for the seven-category state and default response. Frozen D2/1337 emits genuine `crema6_audio_votes_v1` evidence and a normalized128D vector. B's learned acoustic correction can change the state and response; D2 does not change them by default.

The prospective H1 policy remains available only through explicit `--policy role_specific_h1`, clearly labeled rejected/research-only. It selects three authored offers from D2 in contextual-neutral or low-confidence turns. No model-generated estimate becomes observed history. The complete field contract is in `TA_SYSTEM_CONTRACT.md`.

Launch was 2026-09-14 03:30 America/New_York. Research cutoff was 07:30 and handoff deadline 09:00 on that same date. Research ended early after one hypothesis, zero fits, zero calibrations and zero threshold retries. The remaining hypothesis and twelve-fit allowance were not filled. Coverage failure alone did not justify lowering a cutoff against repeatedly used working data. Finalization proceeded without extending either boundary.

## Authority and baseline

Git identified the original repository root; it was on `main` at `312d93cac817fb08188352f19c3827418ff1c440`, with the expected `git@github.com:snickrscodes/HCL-Challenge.git` remote and substantial untracked research source. Operational source was therefore identified by current-file hashes, not HEAD alone. An ignored private baseline inventory records full staged/unstaged/untracked status, index hash and source hashes. The separate branch `ta/finalize-20260914-0330` starts at that HEAD. Only necessary current-source dependencies and task-scoped additions enter its commit.

The old TA release manifest is an immutable preconfirmation selection snapshot. Its completion attestation establishes that former RAVDESS confirmation was consumed. Later `joint_supervision_v1/canonical_closure_v2` evidence also completes the former J raw/benchmark block. Neither update promotes J or a later rejected acoustic model. All 71 retained A baseline entries, six B release entries and thirteen historical C1 manifest entries matched their recorded hashes. Actual model dependencies were available locally, so no cloud connection or transfer was necessary.

## Fixed H1 comparison

`roadmap_b/evidence/ta_system_closure/HYPOTHESIS_01.md` and `POLICY_CASES.json` were written before H1 outcomes. The candidate preserved B and allowed a vocal route only for D2 angry/fear/disgust with maximum probability>=.60 and top-two margin>=.15, when B was neutral or below.50 confidence. Farewell/thanks and confident non-neutral contextual states kept the historical response. Wording, thresholds and fixed controls were not tuned afterward.

All 720 previously inspected working recordings were joined by identity and checked against the immutable output lock. The primary target used all 412 recordings meeting the existing>=.80 supported-six vote-mass requirement; 398 had unique pluralities. Calm, surprise and none were neither relabeled nor assigned invented probabilities. The fixed 720-entry shuffle moved the whole D2 evidence record together while B and transcript stayed fixed. Missing-evidence and a CREMA-training-only prior were additional controls. No working labels fitted a head, temperature or prior.

The primary actor-equal score awarded `2*q[route_label]-1` to a vocal route and0 to baseline response. It measures alignment of routing with matching listener targets, not human usefulness. Minimum useful gain was.05 against both baseline and shuffle, with>=10% eligible coverage,>=8 actors and>=.65 mean routed agreement.

| Metric | Candidate | Control/requirement | Status |
|---|---:|---:|---|
| Signed actor-equal routed agreement | .0409550744 | baseline0; gain>=.05 | FAIL |
| Gain against coupled shuffle | .0405938419 | >=.05; shuffle score.0003612326 | FAIL |
| Eligible routes |20/412 (4.854%) | >=10% | FAIL |
| Actors with eligible routes |9 | >=8 | PASS |
| Mean routed listener agreement |.9175 | >=.65 | PASS |
| Stable-proxy route changes |5/67 | <=D2 top-one12/67 | PASS |
| B state preservation on MELD development |839/839 exact | no probability/category/temperature changes | PASS |

Candidate routes changed 21 of720 responses:18 angry and3 fear. Within the supported-six subset there were 18 angry and 2 fear routes; no disgust route fired. Happy, sad, neutral and disgust examples remain in coverage and class accounting. The shuffle changed the same 21 total responses, but only 11 were eligible and their mean listener agreement was.511616. Missing-evidence and training-prior controls emitted no new routes. Thus actual aligned evidence mattered, but the declared useful effect and coverage were not attained: **REJECT_CHANGE**.

There were 11 route changes across all 360 performed repeats. Among 67 listener-consistent proxies the worst actor rate was 1/1, despite the pooled5/67 result. All actor counts and transitions remain visible. Only three actors meet the old eight-actor matched-selectivity support rule, so selective nuisance robustness is **INCONCLUSIVE**. The discrete route metric does not repair continuous acoustic-distribution stability. No blanket stability claim follows from its pass.

The evaluator's 16 fixture tests passed. A second literal arithmetic implementation, importing neither the policy nor evaluator, independently reproduced scores, coverage, agreement and repeat counts. Aggregate results and exact dependencies are in `H1_RESULTS.json`, `H1_DECISION.json`, and `H1_INDEPENDENT_ARITHMETIC.json`.

## Source and historical scientific evidence

The original 839-row B development predictions recompute to macroF1 .5004668339, weightedF1 .6195674456, calibrated NLL1.0830059210 and15-bin ECE.0584720362. The H1 policy left those probabilities, categories and calibration exactly intact while changing one authored response. This is cached complete-system policy preservation, independently complemented by raw same-device equivalence; it is not a new official benchmark. Two historical B cache streams differ by up to4.77e-7 in logits and were not described as bit-identical.

Frozen D2's working conditional-six NLL recomputes to1.3510569641, Brier.3439323712, JSD.2190041096 and actor-equal macroF1 .4353478706. Unique-plurality true positives/support are neutral42/46, happy1/36, sad12/68, angry79/97, fear51/84 and disgust35/67. These are separate listener-target metrics, never averaged with MELD losses. D2's improved distributional fit does not establish uniform categorical competence.

Historical negative findings remain negative: the former bridge obtained only0.715% working seven-class NLL reduction against5% required and no macro gain. Joint J exceeded the preferred ECE allowance and underperformed D2 on the six-target working task. Stage A and all latest conditional-gradient candidates failed their original repeat-tail conditions. Removing the MELD conditional gradient improved working NLL in all three seeds but increased standalone acoustic MELD NLL by .328209/.292423/.290963. All candidate repeat means passed and all maximum-actor tails failed; the unchanged excess limits were .01 mean and .03 maximum actor-p95. Those standalone regressions are not misrepresented as measurements of an unrun integrated system. No rejected model was loaded as default.

## Engineering validation and limits

The portable loader changes asset resolution and D2-only initialization, reusing the original preprocessing, B inference, shared summary tap and D2 prediction logic. The minimal 33-file bundle is 880,734,827 bytes. D2's embedded mean/scale exactly match the historical normalization file; D3/D0 and training caches are unnecessary. All 218,895,839 learned parameters, including frozen encoders and conservative learned scalars, are below six billion; normalization buffers occupy 79,924bytes separately.

A new environment installed all 55 locked packages from the local cache. The Linux/WSL RTX5080 run used PyTorch2.8.0+cu128 and Transformers4.56.2: singleton FP32 text, BF16 WavLM and FP32 summaries/heads. All 15 fixed development waveforms matched legacy B state exactly, D2 posterior/vector differences were zero, and same-environment cached D2-head replay was exact. Every eligible turn used one encoder pass. Missing/corrupt/long audio skipped encoding with distinct reasons; decoded long audio remained valid; silence was encoded. Context changed B while D2 evidence remained exact. A four-turn raw/chunked replay used only observed human history, and injected failure cleared stale acoustic state.

The single matched benchmark used B, legacyB+D2 and the selected evidence-only wrapper in that fixed order, 30 warmups plus 200 turns each. The timing boundary includes predecoded waveform/transcript/history to returned state/response, synchronized on GPU; disk decoding and model loading are excluded. For<=12s turns, B p95 was 31.453ms and selected-system p95 was 33.428ms, a difference of 1.975ms. Absolute100ms and incremental10ms criteria passed. All-duration selected p95 was 66.379ms. **RTF p95 .320553 exceeded.1: FAIL.** The 27 ultra-short timed observations remained included. All duration strata and empty strata are reported. Within-block median drift ratios were 1.094/1.278/.888, so this single sequential block is not a precise universal speed comparison and was not rerun for favorable timing.

`RAW_VALIDATION_SUMMARY.json` records the actual evidence-only policy and launch selection identity separately from engineering status and performance flags. The historical RTX4090-versus5080 BF16 failure remains; no cross-GPU equivalence is asserted. The new wrapper was not revalidated on a remote4090, and cloud resources were untouched.

The final applicable ordinary suite passed **118 tests**, with **3 pretrained tests skipped** in that invocation. A separate opt-in pretrained run passed **all 3**. The final root help, asset verification, raw waveform, fallback and four-turn replay commands passed from an unrelated directory using the newly installed environment. Fixtures, cached-policy metrics, raw checks and pretrained checks remain separate evidence; these counts are not added into a scientific success score. Clean-commit validation is recorded after commit in the external final attestation, avoiding a circular self-commit identity.

## Running and regenerating

The root README contains installation, preparation, raw waveform, fallback and replay commands. Inference uses an explicit asset root, pinned source and external files; it does not import source from the dirty original worktree. The clean-commit check uses the same external asset mechanism. No pretrained/project weight download URL was invented. Redistribution rights remain separate from a runnable source commit.

To repeat the separate pretrained checks with retained local baseline assets, run from `roadmap_b`:

```bash
RUN_PRETRAINED=1 B_BASELINE_ROOT=/path/to/retained/roadmap_a_source HF_HUB_OFFLINE=1 ../.venv/bin/python -m pytest -q -m pretrained
```

To reproduce the raw comparison and a single fresh benchmark on an explicitly identified local environment, use an unused output directory:

```bash
../.venv/bin/python scripts/validate_ta_system.py --asset-root /path/to/ta_assets --evidence-root /path/to/retained/roadmap_b/artifacts --baseline-root /path/to/retained/roadmap_a_source --output-root /path/to/new/raw-validation --benchmark
```

This is a reproduction command, not authorization to repeatedly seek favorable timing. Aggregate H1 regeneration is the root README's saved-evidence command; its large hash-locked inputs remain external. No confirmation or official-test inference is required by either command.

## Remaining claims and exclusions

Independent evaluation and the volunteer pilot remain pending. Former RAVDESS confirmation is consumed and was not reopened. No official MELD test, repeated CREMA external inference, IEMOCAP acquisition, agreements, volunteer substitutes, synthetic human judgments, new rental, cloud upload, remote inference API, extension, backbone training, ASR, VAD, diarization or learned affect recurrence occurred. Human response usefulness remains **NOT_MEASURED**. The unresolved decision is how to obtain sufficient independently supported delivery-sensitive behavior under the stated contract, not whether to relabel this rejected policy as successful.

The new commit contains audited task-related source, tests, configuration, aggregate evidence and documentation. It excludes environments, weights, media, private listener/consent records, per-record embeddings and machine-local logs. No blanket third-party license is assigned. Original worktree/index/unrelated changes are preserved. Git publication remains solely the user's action.

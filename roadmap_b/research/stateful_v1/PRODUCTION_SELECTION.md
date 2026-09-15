# Production selection

Selected: **stateful core, final frozen B/D2, deterministic evidence-only response**. Partial text model evidence: DEFERRED. Prefix acoustic evidence: DEFERRED. Optional scientific hypotheses executed:0/2. These deferrals are not failed experiments. RunPod was not used.

## Evidence

The first local RTX5080 matched validation passed all15 raw probes with exact B categorical state/logits/probabilities, exact D2 evidence/probabilities/128D representation, exact human context and deterministic response. Four raw temporal turns preserve history lengths0/1/2/3 and audio availability true/false/false/true.

All1,109 frozen development categories and availability were preserved across114 dialogues. Saved-feature CPU head replay reports historical probability drift at most8.43e-8 and logit drift4.77e-7; these are last-bit differences in historical cached CPU-head regeneration, not adapter error or newly permitted tolerance. Old/new adapter outputs on the identical regenerated values remain bit-exact. Saved-feature replay is not raw WavLM inference and does not fabricate D2 evidence. Raw15 supplies the independent full-model check.

## Engineering and performance

First fixed200-turn-per-interface local comparison,30 warmups: ordinary ingestion p95 0.0645ms; finalization overhead excluding model p95 0.5174ms. Total old p50/p95 15.30/57.09ms; stateful17.09/66.49ms. These sequential blocks include hardware variation and do not establish a speedup. Model execution is unchanged; state overhead is measured directly. Short-clip p95RTF remains unfavorable (0.456 in the stateful block), retaining the baseline limitation. Later clean-checkout checks report their own full measurements; no best-of selection.

The10,000-turn fixed stress passed all retained-collection/PCM bounds, later live-memory growth294bytes, and100-session weak-reference cleanup. Ordinary event p95 0.5884ms and model-free finalization p95 0.1209ms meet the5ms targets.

## Review corrections

Initial software tests found test-factory keyword and concurrency-contract mismatches. Review identified that observation-only replay cannot encode completion order. The selected implementation records separate execution boundaries with explicit configuration; cancelled/failed inference and later recovery replay deterministically. A combined cancellation-plus-predictor-error regression was added and fixed before production construction. See SPEC_AMENDMENTS.json for contract clarifications.

## No mathematical changes

Existing model sources, frozen asset identities, ta.py and response policy remain byte-identical to the accepted base. Core adds0 learned parameters. There is no Qwen/Outlines/new model dependency, vision, diarization, ASR integration, RL training, new data or official test inference.

## Release process

The production branch must be constructed separately from accepted base3a403f4418a4399f0a40741ca9f54bd3b5038006. The selected implementation is copied under an explicit allowlist and validated independently. One audited local commit is followed by a new immutable detached-checkout validation. Final source/asset identities and counts are in the external FINAL_ATTESTATION.json and CLEAN_CHECKOUT_VALIDATION.json, avoiding circular self-commit hashes. Never push automatically.

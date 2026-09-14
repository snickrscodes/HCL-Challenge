# Start here: owning the T+A system

The release keeps contextual interpretation and vocal evidence as separate estimates. Frozen B/adaptive/1337 owns the seven-category MELD state. Frozen D2/1337 owns a six-category estimate of audio-only listener judgments. A deterministic policy owns the proposed response. The point of this continuation is to test a concrete interaction between those roles while preserving the contextual predictor.

The final scope is **EVIDENCE_ONLY_BASELINE_WITH_RESEARCH_RESULTS**. H1 was rejected for insufficient effect and coverage; `evidence_only` remains the default and `role_specific_h1` is an explicit research option. All15 raw probes matched the legacy implementation locally on RTX5080. The single benchmark passed absolute/incremental latency but failed short-clip RTF. See the root [T+A report](TA_SYSTEM_REPORT.md), [H1 decision](roadmap_b/evidence/ta_system_closure/H1_DECISION.json), and [raw validation summary](roadmap_b/evidence/ta_system_closure/RAW_VALIDATION_SUMMARY.json).

## What changed after A, B and C1

Roadmap A selected a causal-context RoBERTa text classifier and an original WavLM audio classifier. Its simple logit combination established a baseline. Roadmap B learned a small bounded correction to the selected text logits, with a gate that depends on both modalities. It retained the original text and audio components and introduced explicit audio eligibility and calibration behavior.

C1 supplied a different target: how listeners categorized acted speech from audio alone. D2 reads richer WavLM summaries and produces a genuine six-class posterior plus a normalized 128-dimensional representation. It was previously exposed as evidence without controlling B's response. A better listener distribution therefore did not automatically improve the final interaction.

The new engineering delta is a portable, strictly verified loader and a root interface. It loads only the selected B and D2 dependencies, preserves the existing computation, and returns immutable observations with distinct state, evidence and action fields. H1 adds one prospectively specified response policy. There is no new trained model, ASR, vision, VAD, learned state feedback, or response generator.

## Follow one turn through the models

The application receives a supplied transcript and optional floating-point PCM. Chunks are buffered until the utterance ends. Shared preprocessing validates finite samples, downmixes channels and resamples to mono 16 kHz. This is incremental input buffering; WavLM itself runs on the completed eligible utterance.

T1 receives token and mask tensors of shape `[1,L]`, with `L <= 128`. It can see at most three preceding observed human turns. Relative speaker markers distinguish the current speaker from other speakers. The oldest context turns are dropped when necessary; current content is truncated only if it exceeds its own capacity. The current-content pooling mask excludes markers and prior-turn tokens, although attention can use the retained context.

If the final RoBERTa hidden states are `H_t`, T1 computes the mean of current-content hidden states:

```text
h_t = masked_mean_current(H_t)       # [1,768]
z_t = W_text h_t + b_text            # [1,7]
```

One frozen WavLM execution produces thirteen hidden-state tensors `[1,T,768]`: the pre-Transformer output and twelve Transformer-layer outputs. The shared tap preserves the original last-layer masked mean `h_a [1,768]`. The original audio classifier maps that mean through its 256-unit hidden layer to `z_a [1,7]`.

B's adaptive correction is precisely the implementation in `b_fusion.py`. For each seven-dimensional raw-logit vector, form its softmax maximum and entropy divided by `log(7)`. Concatenate text and audio representations and these four summaries:

```text
gate_input = [h_t, h_a, max(p_t), H(p_t)/log(7),
                       max(p_a), H(p_a)/log(7)]  # 1540
g = sigmoid(Linear_1(GELU(Linear_32(gate_input))))
r = W_residual h_a + b_residual                  # 7
u = tanh(r)
v = u - mean(u)
delta = 2*v / max(1, max(abs(v)))
z_B = z_t + available*g*delta                    # 7
```

Dropout is disabled at inference. Each correction coordinate has absolute value at most two, and the correction sums to zero. The gate is a learned mixing control, not a calibrated probability that audio is reliable. The original audio logits influence gate summaries; B does not simply average their seven probabilities with text probabilities.

Eligible audio uses `softmax(z_B / 1.7140600388082663)`. Without eligible audio, B uses the explicit text fallback `softmax(z_t / 1.6053059396751617)`. The resulting MELD argmax, confidence, distribution and authored reference response are B's outputs.

D2 reads the same WavLM hidden states. For each layer and valid-frame mask, compute FP32 population mean and standard deviation. Concatenating them gives `S [1,13,1536]`. There is no sample-standard-deviation correction and no second encoder call.

```text
X_l = (S_l - training_mean_l) / training_scale_l
w = softmax(learned_layer_logits)                # 13
x = sum_l w_l*X_l                                # [1,1536]
v = GELU(LayerNorm(W_project x + b_project))      # [1,128]
d = v / max(norm(v), 1e-8)                        # [1,128]
z_D2 = W_classifier v + b_classifier              # [1,6]
p_D2 = softmax(z_D2 / 1.0316526942307542)
```

The classifier consumes the unnormalized `v`, while the retained representation is normalized `d`. Do not replace one with the other. The normalization buffers are `mean [13,1536]` and `scale [13,1]`. Their exact equality to the historical normalization file was checked; the deployable D2 checkpoint already embeds them.

## Who owns each field

`meld_state` owns the MELD7 categorical interpretation: neutral, joy, sadness, anger, surprise, fear and disgust. `acoustic_evidence` owns only `crema6_audio_votes_v1`: neutral, happy, sad, angry, fear and disgust. D2 has no surprise output. Calm and none are not silently folded into neutral. Do not append a zero surprise probability or compare losses from different label spaces as if they measured the same task.

`interaction` explains which response rule fired and retains `reference_response`. The top-level `response` is the proposed user-facing action. H1 can change that action while leaving every B categorical probability intact. This is a narrower contract than improving delivery-sensitive seven-class classification.

Observed context contains transcript and speaker only. Generated estimates and robot actions have separate records. A response becomes an emitted action only after acknowledgment; acknowledgment never inserts it into T1 history. Snapshot JSON is stored immutably, and callers receive freshly decoded dictionaries. Failed turns do not become completed human history. The system wrapper clears aborted-turn buffers and inherited runtime error handling clears transient acoustic summaries.

Missing or corrupt audio produces calibrated text fallback and unavailable evidence. Successfully decoded audio over sixty seconds is valid but ineligible; the encoder is skipped without cropping. Silence of sufficient length remains valid input and is not forced neutral. Arrays shorter than the 400-sample receptive field are invalid. Transcript contents are data: they do not authorize shell, filesystem, network or credential actions.

## H1: the exact proposed behavioral change

The declared bottleneck was concrete: B returned neutral for all 720 already-used RAVDESS working recordings, while D2 contained matching listener information that never reached the response. H1 preserves B and considers a vocal offer only when B is neutral or its confidence is below 0.50. D2 must be available, its top probability at least 0.60, and its top-minus-second margin at least 0.15. Only three tops have routes:

- Angry: “Would you like to vent, or focus on what to do next?”
- Fear: “Would it help to slow down and talk through your concern?”
- Disgust: “Would you like to say more about this, or change the subject?”

Farewell and thanks retain lexical precedence. Confident non-neutral contextual disagreement, unsupported vocal categories and low vocal confidence retain the original B response. These thresholds are fixed design choices, not calibrated psychological certainty. The offers do not assert the speaker's internal emotion.

The baseline is evidence-only B+D2. Controls remove D2 availability or apply one predeclared coupled shuffle to its logits, posterior and vector, while B and text stay fixed. No labels, actor IDs or recording filenames enter the policy. No threshold or wording search is permitted after results.

The primary score is actor-equal signed routed listener agreement. On an eligible row, a new route receives `2*q[route_label]-1`; a baseline action receives zero. This penalizes unsupported routing and does not reward abstention as a successful response. Required gain is at least 0.05 against both baseline and shuffle. Safeguards require at least 10% route coverage, routes for at least eight actors, and mean routed agreement of at least 0.65.

All 720 working rows remain accounted for. Matching six-target scores use the historical supported-vote-mass threshold of at least 0.80, expected to retain 412 rows. B probabilities and decisions must remain exactly unchanged on all 839 MELD development-model rows. The fixed case inventory and real local availability, history, source-equivalence and one-pass checks must pass.

All 67 listener-consistent repeat proxies remain in the stability inventory. H1 route-ID changes may not exceed D2's historical top-one change count of 12/67. Every actor, maximum actor rate, route-to-baseline transition and all 360 performed repeats must be reported separately. Sparse selective-nuisance support remains inconclusive; a new discrete metric cannot erase earlier continuous-distribution tail failures.

H1 scored .040955 against baseline0 and gained .040594 over the fixed shuffle, both below .05. It routed20/412 eligible recordings (4.85%, below10%) across nine actors, with .9175 mean routed listener agreement. Aggregate stable-proxy changes were5/67 against D2 top-one12/67, but the worst actor had1/1; selective nuisance support remains inconclusive. All839 B development probabilities/categories were preserved, and fixed-case and integrity checks passed. The gain and coverage failures determine REJECT_CHANGE. No threshold was lowered, no second hypothesis was run, and no new head was fitted.

## Frozen parameters and portable assets

The required learned-parameter ledger is:

| Component | Parameters |
|---|---:|
| Selected T1 encoder, marker embeddings and head | 124,062,727 |
| Shared WavLM | 94,381,936 |
| Original audio classifier | 198,663 |
| B adaptive residual and gate | 54,728 |
| Retained original/B scalar state | 5 |
| D2 neural head | 197,779 |
| D2 temperature | 1 |
| **Total** | **218,895,839** |

All neural parameters are frozen for deployment; this continuation trains none. Normalization adds 79,924 buffer bytes, reported separately. The loader checks the inherited unique-parameter ledger, including frozen components, against the six-billion limit.

`ASSET_MANIFEST.json` names 33 exact files totaling 880,734,827 bytes. It retains the selected T1, WavLM, original audio head, B head, D2 head, calibrations and historical provenance. It excludes alternate text models, D0/D3 heads, separate normalization, training caches and corpus media. Source and asset hashes are checked before model construction; missing, changed, unexpected or symlinked dependencies fail closed.

The historical research loader constructs D3 before switching to D2 and uses working-directory-based research configuration. `PortableTADual` instead builds an explicit asset-root configuration and loads D2 directly from its checkpoint. It inherits the same shared tap and prediction logic. This is inference-only dependency verification; it does not pretend to rerun historical full experiment freezes without their research files.

The recorded runtime is Python 3.12, PyTorch 2.8.0+cu128 and Transformers 4.56.2. CUDA uses singleton FP32 text, BF16 WavLM and FP32 summaries/head. Explicit CPU execution is FP32 and has a separate numerical scope. Known RTX 4090/5080 discrepancies prohibit a cross-device equality claim. Preserve RoBERTa and WavLM revision attribution in the manifest and external-component disclosures; the commit supplies no pretrained weights.

## Run from the repository root

Prepare from retained local frozen roots, or verify an existing prepared directory:

```bash
python prepare_ta_assets.py --asset-root /path/to/ta-assets \
  --baseline-root /path/to/frozen-A \
  --b-root /path/to/artifacts/roadmap_b \
  --c1-root /path/to/artifacts/roadmap_c1_v2
python prepare_ta_assets.py --asset-root /path/to/ta-assets --verify-only
python ta.py --asset-root /path/to/ta-assets \
  --text "I am still thinking about it." --audio /path/to/utterance.wav
python ta.py --asset-root /path/to/ta-assets \
  --replay roadmap_b/examples/ta_replay.json
```

Use the provided pinned environment. The preparer copies only checked local files and never downloads replacements. Paths are independent of current working directory. Replay audio paths are relative to the replay JSON; its ordered turns must be observed human inputs. The example contains no private recordings. `--chunk-samples` controls buffering, `--device cpu` explicitly changes numerical scope, and `--output` writes a new file without overwriting one. Omitting audio exercises text fallback. The policy override is explicit; inspect its returned status before interpreting research output.

## Keep the historical failures visible

D2 improved working conditional-six NLL over D0, approximately 1.3511 versus 1.7147, but that does not establish every class or final decision improved. Its working happy recall was 1/36 and sad recall 12/68. All classes remain in accounting even where H1 offers no route.

Earlier categorical bridges, joint-supervision variants and factorized acoustic candidates did not earn replacement. The latest conditional-gradient ablation improved working NLL in all three seeds, but raised standalone acoustic MELD development NLL by about 0.328, 0.292 and 0.291. Every candidate passed its repeat-mean safeguard and failed its maximum-actor-tail safeguard. Different clipping frequencies also prevent identifying annotation semantics as the unique cause.

Those standalone regressions were not measurements of this integrated response policy. Conversely, changing the system contract does not retroactively accept those models. Both working and former confirmation results have been inspected; neither is pristine independent confirmation now. Official MELD test inference and human response usefulness remain unmeasured here. Same-text acoustic route changes demonstrate evidence-dependent execution only when the declared controls support them; they do not prove empathy, fresh-domain transfer or selective robustness.

## Read these functions in order

Start with `ta.py:main`, then `ta_assets.verify_asset_root` and `ta_runtime.PortableTADual.__init__`. Read `context.encode_turn`, `b_inference.RoadmapB.predict_turn` and `b_fusion.ResidualFusion.forward` for contextual state. Follow `c1_transfer_runtime.RefinedSummaryTap.forward`, `c1_model.EvidenceHead.forward` and `c1_runtime_v2.C1Dual.predict_turn` for shared acoustic evidence. Finish with `ta_policy.select_response`, `ta_interface.SystemPredictor.predict_turn` and `SystemSession.end_turn`. The contract, hypothesis card, asset manifest and final scorecard explain why these functions are used.

## Self-check questions

1. Which field owns the MELD category?
2. Does D2 add a second WavLM execution?
3. What has 1,540 dimensions in B?
4. Is the gate an audio-trust probability?
5. Does the D2 classifier consume normalized `d`?
6. Why is the separate D2 normalization file unnecessary?
7. What happens to decoded 61-second audio?
8. Can emitted responses become observed history?
9. What does a successful H1 routing test establish?
10. May former confirmation actors become fresh confirmation again?

## Answers

1. `meld_state`; D2 and the response policy do not replace it.
2. No. The shared tap preserves B's mean and derives D2 summaries.
3. Two 768-dimensional representations plus four confidence/entropy summaries.
4. No. It is a learned bounded-correction control.
5. No. It consumes unnormalized `v`; `d` is separately retained.
6. Exact checked mean/scale buffers are embedded in the selected checkpoint.
7. It stays valid but acoustically unavailable and uses calibrated text fallback.
8. No. Actions, estimates and observed human transcripts remain separate.
9. Supported deterministic behavior on disclosed evaluation cases, subject to safeguards; human usefulness remains unmeasured.
10. No. Their inspected results remain consumed historical evidence.

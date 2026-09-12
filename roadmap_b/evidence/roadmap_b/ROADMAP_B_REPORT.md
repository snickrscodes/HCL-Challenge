# Roadmap B controlled experiment

Question: can adaptive text-primary acoustic correction improve fixed fusion for a reason that requires matched audio?

Roadmap A is preserved. Official test inference is forbidden in this workflow.

## Baseline integrity

Baseline lock SHA256: fcfddbe1d9413b6ba9bb77d60eb5948e07ee476d1a735744474346b07fe0652f.
Original archive SHA256: bd8ec38403035ef378efa5a01035efe0ebb1a9f1cbf8fd08ec668013d774a2d0.
Exact split IDs and checkpoint hashes are in the supplied baseline lock; they were not regenerated.
A-original and A-replay-canonical are separate references. Original alpha/temperatures remain fixed.

## Numerical policy

Canonical encoding: **singleton**, precision bfloat16.
Real equal-batch discrepancy found: True.
Whole-utterance audio limit: 60 seconds. Longer decoded clips keep audio_valid=true but are acoustically ineligible; calibrated text fallback is used.
No cropping, chunked pooling, label changes, or original-cache replacement.
A separate user-approved numerical repair uses singleton FP32 text with the exact unchanged T1 weights. A-original remains the historical BF16-cache result; A-replay-canonical uses FP32 text and singleton BF16 audio with the original alpha and temperatures.
Real text diagnostics found a maximum singleton/batch FP32 probability difference of 0.0000063. Compared with the original BF16 cache, 198/839 examples differed by more than 0.005 and six argmaxes changed. No performance scores selected this policy.

| Probe | BF16 paired/single max embedding difference | Max fused probability difference |
|---|---:|---:|
| dev/99/3 | 0.0620483 | 0.0257038 |
| dev/108/5 | 0.0220257 | 0.00656054 |
| dev/62/4 | 0.000890304 | 0.00053984 |
| dev/72/0 | 0.000866592 | 0.000218774 |
| dev/85/7 | 0.0110965 | 0.0014241 |
| dev/108/7 | 0.00143909 | 0.000122309 |
| dev/65/0 | 0.00195421 | 0.000252111 |
| dev/108/2 | 0.00106265 | 0.00020966 |
| dev/77/3 | 0.00117299 | 0.000358119 |
| dev/10/9 | 0.00239611 | 0.000472353 |
| dev/74/19 | 0 | 0 |

All BF16/FP32/cached pairwise comparisons, logits, probabilities, waveform hashes and runtime metadata are in numerical_replay/.

## Architecture

Frozen T1/WavLM/original heads supply h_t, h_a, z_t and z_a. Raw softmax confidence and entropy feed a 1540→32→1 sigmoid gate.
u=tanh(W_r h_a+b_r); c=u−mean(u); delta=2c/max(1,max_abs(c)); z_B=z_t+availability*g*delta.
The residual and final gate weights initialize to zero; final gate bias is −2. Initial logits equal text exactly, with nonzero residual gradients.
Controls: centered bias, constant-gate acoustic residual, adaptive residual, and the same architecture with text replacing every acoustic input.
All variants use identical eligible training identities and CE/AdamW. Seeds 1337/1338/1339 are fixed; 1337 is the reference, not a best-seed choice.

## Development results

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| A_original_text_context | 0.6138 | 0.6045 | 0.4863 |
| A_original_concat | 0.6162 | 0.6058 | 0.4873 |
| A_original_late_fusion | 0.6377 | 0.6178 | 0.4894 |
| A_replay_canonical | 0.6377 | 0.6178 | 0.4894 |
| text | 0.6174 | 0.6074 | 0.4896 |
| bias, three head seeds | 0.6202 ± 0.0050 | 0.6067 ± 0.0035 | 0.4846 ± 0.0012 |
| constant, three head seeds | 0.6238 ± 0.0007 | 0.6089 ± 0.0007 | 0.4846 ± 0.0005 |
| adaptive, three head seeds | 0.6361 ± 0.0036 | 0.6194 ± 0.0019 | 0.4999 ± 0.0009 |
| text_only, three head seeds | 0.6190 ± 0.0042 | 0.6054 ± 0.0042 | 0.4845 ± 0.0012 |

Standard deviations describe head-seed variability, not encoder-seed robustness.

| Variant | Seed | Best epoch | Macro F1 | Weighted F1 | Fit seconds |
|---|---:|---:|---:|---:|---:|
| bias | 1337 | 4 | 0.4844 | 0.6040 | 0.47 |
| bias | 1338 | 11 | 0.4858 | 0.6106 | 1.01 |
| bias | 1339 | 4 | 0.4835 | 0.6055 | 0.57 |
| constant | 1337 | 8 | 0.4851 | 0.6096 | 1.09 |
| constant | 1338 | 7 | 0.4842 | 0.6084 | 1.25 |
| constant | 1339 | 6 | 0.4846 | 0.6086 | 1.22 |
| adaptive | 1337 | 4 | 0.5005 | 0.6196 | 1.70 |
| adaptive | 1338 | 4 | 0.5004 | 0.6212 | 1.65 |
| adaptive | 1339 | 5 | 0.4989 | 0.6175 | 1.82 |
| text_only | 1337 | 2 | 0.4843 | 0.6009 | 1.33 |
| text_only | 1338 | 2 | 0.4858 | 0.6064 | 1.14 |
| text_only | 1339 | 1 | 0.4834 | 0.6090 | 0.93 |

## Reference-seed class effects against A-replay-canonical

| Group | Changed | Corrected | Harmed | Net correct |
|---|---:|---:|---:|---:|
| all | 82 | 28 | 30 | -2 |
| non_neutral | 54 | 20 | 16 | 4 |
| neutral | 28 | 8 | 14 | -6 |
| joy | 9 | 6 | 0 | 6 |
| sadness | 11 | 2 | 6 | -4 |
| anger | 13 | 4 | 4 | 0 |
| surprise | 11 | 4 | 6 | -2 |
| fear | 5 | 0 | 0 | 0 |
| disgust | 5 | 4 | 0 | 4 |

Reference-seed fear true positives: 7; disgust: 7; combined: 14.
Full per-class metrics, confusion matrices, recovery/harm rates, fixed slice supports and seed-paired results are saved for every run.

## Gate and posterior influence

Audio influence is TV(softmax(z_B/T_B),softmax(z_t/T_B)), using the same temperature.
Gate values are not causal attribution, a fraction of emotion, or compute-saving routing.

| Diagnostic, seed 1337 | Mean | Median | p10 | p90 | p95 |
|---|---:|---:|---:|---:|---:|
| gate | 0.5799 | 0.6569 | 0.1023 | 0.9334 | 0.9512 |
| delta_l2 | 1.8895 | 1.8898 | 1.6574 | 2.1250 | 2.2053 |
| delta_max_abs | 1.2909 | 1.2896 | 1.1630 | 1.4366 | 1.4837 |
| audio_influence | 0.0604 | 0.0568 | 0.0121 | 0.1112 | 0.1239 |

Correlations with confidence and correction/harm outcomes are descriptive; complete ranges and saturation rates are saved.

## Audio interventions, reference seed

| Condition | Macro F1 |
|---|---:|
| Matched | 0.5005 |
| training_mean_audio | 0.4945 |
| same_speaker | 0.4986 |
| gate_zero | 0.4896 |
| training_mean_gate | 0.4853 |
| Ten shuffles, mean [range] | 0.4920 [0.4880, 0.4947] |

Shuffling also disrupts lexical/background correspondence; same-speaker swaps do not isolate prosody.
Real silence/noise probes, when available, use actual WavLM on a predetermined subset, not zero features.

## Calibration and uncertainty

| Variant/seed | Temperature | Calibration NLL before → after | ECE before → after |
|---|---:|---:|---:|
| bias/1337 | 1.6084 | 1.2697 → 1.1341 | 0.1834 → 0.0781 |
| bias/1338 | 1.6175 | 1.2680 → 1.1298 | 0.1863 → 0.0773 |
| bias/1339 | 1.6106 | 1.2696 → 1.1333 | 0.1770 → 0.0770 |
| constant/1337 | 1.6133 | 1.2640 → 1.1273 | 0.1811 → 0.0642 |
| constant/1338 | 1.6138 | 1.2641 → 1.1273 | 0.1839 → 0.0649 |
| constant/1339 | 1.6150 | 1.2662 → 1.1288 | 0.1810 → 0.0809 |
| adaptive/1337 | 1.7141 | 1.3066 → 1.1314 | 0.1913 → 0.0698 |
| adaptive/1338 | 1.7290 | 1.3073 → 1.1260 | 0.2028 → 0.0573 |
| adaptive/1339 | 1.7081 | 1.3025 → 1.1303 | 0.1919 → 0.0569 |
| text_only/1337 | 1.7612 | 1.3334 → 1.1397 | 0.1979 → 0.0809 |
| text_only/1338 | 1.8351 | 1.3702 → 1.1470 | 0.2142 → 0.0651 |
| text_only/1339 | 1.7327 | 1.3157 → 1.1357 | 0.2027 → 0.0611 |

NLL fitting does not guarantee lower ECE. dev_calib selects no architecture or seed.
Paired dialogue-cluster bootstrap uses 1000 replicates. Intervals are conditional post-selection descriptions, not significance claims.


Complete bootstrap estimates and intervals: `evaluation/adaptive/1337/metrics.json`.

Frozen teacher training errors: canonical FP32 1340; original BF16 1341. These are in-sample errors, not generalization estimates.

## Runtime and resources

GPU: NVIDIA GeForce RTX 4090; model load 0.375 seconds.
Loaded required parameters: 218,698,059.
Median real-time factor: 0.00655; peak allocated GPU bytes: 1,363,681,280.
Peak sampled process RAM bytes: 1,864,208,384.

| Stage | p50 ms | p95 ms |
|---|---:|---:|
| text | 4.965 | 5.601 |
| audio | 12.591 | 51.840 |
| fusion_response | 0.611 | 0.803 |
| total | 18.216 | 58.213 |

| Same-workload stage | A p50/p95 ms | B p50/p95 ms |
|---|---:|---:|
| text | 5.001 / 5.821 | 4.965 / 5.601 |
| audio | 13.483 / 50.513 | 12.591 / 51.840 |
| fusion_response | 0.202 / 0.282 | 0.611 / 0.803 |
| total | 18.651 / 56.224 | 18.216 / 58.213 |

Timing starts after explicit end-of-turn and excludes recording, disk read, endpointing and ASR. WavLM is not streaming.

## Same-words/different-delivery diagnostic

Pending consented recordings. The recording manifest is provided; no synthetic recording is presented as human evidence.

## Limitations

MELD is imbalanced TV dialogue with overlapping speakers, lexical information, laughter/music and imperfect clip boundaries. The encoders may exploit these features instead of prosody.
Teacher training outputs are in-sample. Repeated development selection and three cheap head seeds do not establish population or encoder-seed robustness.
No ASR, VAD, LLM, vision, RL, new encoder training, or optional imbalance search was added.

## Additional measured evidence

The declared engineering advancement rule passes. This is a macro-F1 and class-tradeoff result, not a claim of fewer overall errors or demonstrated prosodic causality.

A-original and A-replay-canonical differ on 2 decisions even though their aggregate accuracy/F1 values coincide. They remain distinct references.

### Preservation of A corrections (retrospective)

Against canonical text, fixed A makes 43 useful corrections and 26 harms. Reference B preserves 17 of those corrections and avoids 18 of those harms. It also makes different decisions elsewhere. These outcome-defined subsets are retrospective diagnostics, not predeclared slices.

Against A directly, reference B corrects 28 errors and introduces 30 errors: net −2. Its non-neutral net is +4 and neutral net is −6. This does not establish the simple story that B preserves most useful A corrections while only removing harms.

| Adaptive seed | Non-neutral net vs A | Fear TP | Disgust TP | Combined TP |
|---|---:|---:|---:|---:|
| 1337 | 4 | 7 | 7 | 14 |
| 1338 | -1 | 7 | 7 | 14 |
| 1339 | 2 | 7 | 7 | 14 |

The non-neutral criterion was operationalized before training for the seed mean and reference seed; seed 1338 individually loses one non-neutral correct decision. Fixed A has 7 fear and 3 disgust true positives. B does not improve fear detections.

### Reference-seed class metrics

| Class | Support | A F1 | B F1 | B precision | B recall |
|---|---:|---:|---:|---:|---:|
| neutral | 353 | 0.7726 | 0.7750 | 0.7163 | 0.8442 |
| joy | 121 | 0.5487 | 0.5862 | 0.6126 | 0.5620 |
| sadness | 83 | 0.3876 | 0.3256 | 0.4565 | 0.2530 |
| anger | 121 | 0.5022 | 0.4891 | 0.5185 | 0.4628 |
| surprise | 114 | 0.6316 | 0.6468 | 0.6281 | 0.6667 |
| fear | 30 | 0.3333 | 0.2917 | 0.3889 | 0.2333 |
| disgust | 17 | 0.2500 | 0.3889 | 0.3684 | 0.4118 |

Confusion matrices use the immutable label order: neutral, joy, sadness, anger, surprise, fear, disgust.

A-replay-canonical:
```json
[[304, 14, 8, 8, 17, 2, 0], [40, 62, 1, 7, 11, 0, 0], [33, 10, 25, 11, 1, 2, 1], [30, 9, 6, 56, 17, 0, 3], [15, 7, 2, 11, 78, 1, 0], [6, 3, 3, 5, 6, 7, 0], [6, 0, 1, 4, 3, 0, 3]]
```
B seed 1337:
```json
[[298, 16, 10, 10, 13, 4, 2], [31, 68, 1, 9, 12, 0, 0], [36, 8, 21, 11, 1, 2, 4], [30, 9, 6, 56, 13, 1, 6], [12, 8, 3, 12, 76, 3, 0], [5, 2, 4, 8, 4, 7, 0], [4, 0, 1, 2, 2, 1, 7]]
```

### Fixed slices

| Slice | Support | Text macro F1 | A macro F1 | B macro F1 |
|---|---:|---:|---:|---:|
| all | 839 | 0.4896 | 0.4894 | 0.5005 |
| audio_available | 838 | 0.4895 | 0.4893 | 0.5004 |
| text_confidence_lt_0.6 | 210 | 0.3698 | 0.3767 | 0.4002 |
| lt_0.4 | 50 | 0.3347 | 0.2231 | 0.3630 |
| 0.4_to_0.6 | 160 | 0.3773 | 0.4214 | 0.4050 |
| 0.6_to_0.8 | 168 | 0.4506 | 0.4572 | 0.4517 |
| gte_0.8 | 461 | 0.4993 | 0.4993 | 0.4993 |
| unimodal_disagreement | 411 | 0.4522 | 0.4756 | 0.4759 |

### Paired dialogue uncertainty

| Difference, B 1337 − A | Observed | 95% bootstrap interval |
|---|---:|---:|
| accuracy | -0.00238 | [-0.01609, +0.01202] |
| weighted_f1 | +0.00182 | [-0.01188, +0.01664] |
| macro_f1 | +0.01104 | [-0.01919, +0.04746] |

All three intervals include zero. They are paired dialogue-cluster, conditional post-selection descriptions (1,000 replicates), not significance evidence. Head-seed standard deviations measure a different source of variability.

### Gate relationships and acoustic interventions

Reference gate correlations: influence 0.495; text confidence 0.197; audio confidence 0.175. The gate is not simply a low-text-confidence switch.

| Adaptive seed | Matched macro F1 | Shuffle mean | Training-mean audio | Same-speaker swap | Fixed training-mean gate |
|---|---:|---:|---:|---:|---:|
| 1337 | 0.5005 | 0.4920 | 0.4945 | 0.4986 | 0.4853 |
| 1338 | 0.5004 | 0.4906 | 0.4936 | 0.4947 | 0.4830 |
| 1339 | 0.4989 | 0.4896 | 0.4928 | 0.4942 | 0.4917 |

Same-speaker swaps cover 829/839 rows. Training-mean replacements and training-mean gates use training data only. Confidence summaries use raw logits, without calibration inputs.

### Raw silence/noise probes

| Condition | Probe support | Correct | Prediction changes vs matched | Mean posterior TV vs matched |
|---|---:|---:|---:|---:|
| matched | 15 | 8 | 0 | 0.00000 |
| silence | 15 | 8 | 2 | 0.04210 |
| noise | 15 | 8 | 0 | 0.01378 |

These 15 predetermined duration/numerical probes are a small diagnostic set. Silence is actual zero PCM encoded by WavLM; noise is the fixed 10 dB condition. Equal correct counts do not establish robustness or expression sensitivity.

### Raw input to both outputs

The predetermined dev/99/3 probe produces the following measured output. It illustrates restrained correction when the audio classifier disagrees with text; it is not a newly selected success benchmark.
```json
{
  "emotion": "surprise",
  "confidence": 0.6594182526194377,
  "distribution": {
    "neutral": 0.019824707259132324,
    "joy": 0.18854538937412513,
    "sadness": 0.03056194709556351,
    "anger": 0.04486641853647808,
    "surprise": 0.6594182526194377,
    "fear": 0.028134504070521784,
    "disgust": 0.02864878104474141
  },
  "audio_valid": true,
  "audio_available": true,
  "audio_unavailable_reason": null,
  "text_prediction": "surprise",
  "audio_prediction": "neutral",
  "gate": 0.05208805575966835,
  "audio_influence": 0.008601562941674539,
  "response": "That sounds unexpected. What happened next?",
  "delta_l2": 1.945749044418335,
  "delta_max_abs": 1.2642518281936646,
  "current_truncated": false,
  "latency_ms": {
    "text": 4.914575984003022,
    "audio": 11.918677017092705,
    "fusion_response": 0.6099739985074848,
    "total": 17.443226999603212
  }
}
```

### Reproducibility and runtime details

All 15 raw/cache cases passed, including all four original discrepancy clips. Maximum fused-probability difference was 2.21e-08; text probability differences were zero. No tolerance was loosened.
Runtime uses 45 warm turns (15 clips × 3 repeats), explicit end-of-turn, and real WavLM. Loads were measured in a warm process/filesystem context, not cold machine startup. Same-workload A uses canonical precision; historical A timing is preserved separately.

| Canonical cache | Split | Rows | Extraction seconds |
|---|---|---:|---:|
| WavLM singleton BF16 | train | 9989 | 197.469 |
| WavLM singleton BF16 | dev_model | 839 | 15.702 |
| WavLM singleton BF16 | dev_calib | 270 | 4.794 |
| T1 singleton FP32 | train | 9989 | 46.666 |
| T1 singleton FP32 | dev_model | 839 | 3.887 |
| T1 singleton FP32 | dev_calib | 270 | 1.221 |

Direct recomputation from all 12 prediction files reproduces accuracy, macro/weighted F1 and confusion matrices. The validation suite reports 47 passing tests, including inherited A regressions and pretrained checks.

An evaluation-only reporting repair handles undefined constant-gate correlations as null. The initial failed output, original source, model freeze and exact old/new source hashes are retained in evaluation_repair.json and evaluation_failed_constant_correlation_v1/. No head was retrained or reselected.

Consented same-words recordings remain pending; the recording manifest is supplied. No optional imbalance experiment was run. The next continuation should keep seed 1337 fixed and finish the controlled delivery diagnostic and presentation, rather than search more architectures. The official test still requires a separate planned final pass.

### Fresh-process temporal demonstration

Input:
```json
{
  "key": "dev/54/8",
  "text": "Probably kill myself!",
  "speaker": "Joey",
  "history": [
    {
      "text": "See, there's always one guy.  \"If I had a wish, I'd wish for three more wishes.\"",
      "speaker": "Rachel"
    },
    {
      "text": "Hey Joey. Hi. Hey, buddy.",
      "speaker": "All"
    },
    {
      "text": "Hey, Joey, what would you do if you were omnipotent?",
      "speaker": "Monica"
    }
  ],
  "waveform_sha256": "92ead1b63971c791d37d347ff90ff9b59f95e0652eb053713c6febeab0a2a722",
  "num_samples": 36864,
  "sample_rate": 16000,
  "audio_source": "<COMPUTE_WORKSPACE>/roadmap_a_source/data/processed/audio/dev/dia54_utt8.wav",
  "scope": "Predetermined dev probe, fresh-process raw inference; no feature cache"
}
```
Output after buffered audio chunks and explicit end-of-turn:
```json
{
  "emotion": "anger",
  "confidence": 0.43589787407681485,
  "distribution": {
    "neutral": 0.04366641961489998,
    "joy": 0.11472067473559495,
    "sadness": 0.15403768454425998,
    "anger": 0.43589787407681485,
    "surprise": 0.03939770591155448,
    "fear": 0.09694581715016888,
    "disgust": 0.11533382396670679
  },
  "audio_valid": true,
  "audio_available": true,
  "audio_unavailable_reason": null,
  "text_prediction": "anger",
  "audio_prediction": "neutral",
  "gate": 0.6418665647506714,
  "audio_influence": 0.06302699736376935,
  "response": "I'm not sure how that felt for you. Would you tell me a little more?",
  "delta_l2": 1.6309046745300293,
  "delta_max_abs": 1.2724394798278809,
  "current_truncated": false,
  "latency_ms": {
    "text": 573.970447992906,
    "audio": 1065.9098860051017,
    "fusion_response": 198.70955299120396,
    "total": 1838.5898869892117
  }
}
```

## Decision

- PASS: mean_macro_gain
- PASS: positive_gain_in_at_least_two_seeds
- PASS: weighted_f1_not_materially_worse
- PASS: beats_constant_residual
- PASS: beats_matched_text_only
- PASS: non_neutral_not_worse_mean_and_reference_seed
- PASS: fear_disgust_tp_not_worse_mean_and_reference_seed
- PASS: matched_beats_shuffled_and_training_mean
- PASS: gate_not_effectively_constant
- PASS: raw_inference_fallback_parameter_ledger
- PASS: tests_passed

Engineering continuation criteria on selected development models, not significance tests

ADVANCE ROADMAP B

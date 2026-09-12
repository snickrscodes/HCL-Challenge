# Controlled delivery — provisional intended-delivery smoke

> **Provisional analysis using speaker-intended delivery labels. Independent listener validation (L01) is pending; these results are not human-validated evidence.**

## Engineering readout — provisional assumptions only

The frozen audio classifier responds to these recordings, but neither fixed A nor adaptive B changes its argmax within any of the 60 within-speaker delivery pairs. Text has exactly zero posterior movement. Therefore this smoke test does **not** demonstrate categorical delivery correction by either fused system. Posterior movement is the informative quantity here.

Using the pre-output mapping, 36 pairs have distinct mapped intended categories, 12 are intended-neutral stable pairs, and 12 involve an unmapped delivery. A aligns with the temporary category contrast on 28/36 pairs and moves oppositely on 8; B aligns on 23/36 and moves oppositely on 13. Both intended categories gain probability for their own respective delivery on 22 pairs for A and 15 for B. These are descriptive counts of correlated contrasts under an oracle assumption, not accuracy or independent trials.

Mean calibrated pairwise TV across all 60 pairs is .026253 for A versus .007295 for B. Audio-only mean TV is .172151. At the common diagnostic temperature T=1, A TV is .023458 and B TV .002098: the sensitivity gap is not solely due to the different saved calibration temperatures. This does not isolate which feature, residual or logit-scale mechanism causes the gap.

B is less sensitive, and this is not uniformly a benefit: its smaller movement accompanies fewer intended-direction alignments. Conversely, the 12 intended stable-control pairs have no categorical flips for either A or B, with mean/max TV .013337/.039055 for A and .006506/.014078 for B. Stability alone does not establish that the common category is appropriate; independent judgments are pending.

B gate median is .705521 (range .036373–.951342), while its median within-recording audio influence is .031032 (range .003520–.065804). A high gate is not large delivery sensitivity. Within-recording influence compares B to text under the same B temperature; between-recording TV compares two deliveries. These are different comparisons and should not be conflated.

The conservative provisional conclusion is: paired recordings produce measurable acoustic/posterior variation, but B does not show an advantage over A on the temporary intended-direction diagnostic, and neither fused model flips category within a delivery set. This does not revoke development selection or establish a human-validated failure. L01 may perceive categories or contrasts differently. No tuning, re-recording, seed change or freeze decision follows.

The inference ran locally on the NVIDIA GeForce RTX 5080 Laptop GPU with Python 3.12.14, PyTorch 2.8.0+cu128 and Transformers 4.56.2. It reused the exact selected weights and canonical singleton FP32 text/BF16 audio path. These are not fresh RTX 4090 measurements. Accepted recordings stayed local; no human audio was sent to RunPod.

Full recording-by-recording intended probabilities, predictions and diagnostics are indexed in [RECORDING_INDEX.md](RECORDING_INDEX.md); every full distribution and waveform hash remains in [outputs.json](outputs.json). The full paired sensitivity/sign comparison is in [engineering_readout.json](engineering_readout.json).

## Scope and interpretation

All 60 accepted recordings are included: two speakers, ten fixed phrases, three deliveries. No recording was chosen, changed or rejected based on a model output. Text/context are fixed; comparisons are the three within-speaker pairs per phrase (60 correlated pairs, 20 sets). No benchmark accuracy is reported. More acoustic sensitivity is not automatically better.

The mapping was saved before inference in intended_mapping.json. Willing/friendly, warm appreciation and relaxed acceptance remain unmapped. Other categories are temporary coarse assumptions. Neutral controls are intended stable controls, not independently validated ones.

For distinct mapped targets c1/c2, D=(p2[c2]-p2[c1])-(p1[c2]-p1[c1]). D>1e-12 is intention-aligned; D<−1e-12 is opposite. This numerical tolerance is not an effect-size threshold. Both per-target changes and magnitudes are saved; D can be positive even if only one target moves appropriately. Each pair shares recordings with other pairs; counts are descriptive.

## Posterior movement

| Model | Mapped distinct pairs | Aligned / opposite / zero | Both targets increase | Mean D | Mean TV, all pairs |
|---|---:|---|---:|---:|---:|
| text | 36 | 0 / 0 / 36 | 0 | 0.000000 | 0.000000 |
| audio | 36 | 29 / 7 / 0 | 22 | 0.114824 | 0.172151 |
| A | 36 | 28 / 8 / 0 | 22 | 0.019911 | 0.026253 |
| B | 36 | 23 / 13 / 0 | 15 | 0.002822 | 0.007295 |

A is the unchanged fixed-alpha model under canonical numerical inputs; B is adaptive seed 1337. Historical A-original batch-dependent outputs are not reintroduced. Each model uses its saved temperature; TV comparisons therefore describe deployed posterior sensitivity, not architecture alone. Unit-temperature sensitivities are also saved in uncalibrated_pairs.json as a scale diagnostic.

## Intended stable controls

| Model | Pairs | Argmax changes | Mean TV | Max TV |
|---|---:|---:|---:|---:|
| text | 12 | 0 | 0.000000 | 0.000000 |
| audio | 12 | 0 | 0.140675 | 0.299940 |
| A | 12 | 0 | 0.013337 | 0.039055 |
| B | 12 | 0 | 0.006506 | 0.014078 |

## Every same-text set

| Speaker / phrase | Exact text | Intended stable | Text invariant | Text TV | Audio TV | A TV | B TV |
|---|---|---|---|---:|---:|---:|---:|
| S01 / p01 | That's great. | False | True | 0.000000 | 0.240609 | 0.047014 | 0.003343 |
| S01 / p02 | Sure. | False | True | 0.000000 | 0.124498 | 0.025566 | 0.018014 |
| S01 / p03 | I'm fine. | False | True | 0.000000 | 0.216321 | 0.033046 | 0.012811 |
| S01 / p04 | Really? | False | True | 0.000000 | 0.210141 | 0.018797 | 0.000254 |
| S01 / p05 | Okay. | False | True | 0.000000 | 0.131418 | 0.015039 | 0.010088 |
| S01 / p06 | I can't believe that. | False | True | 0.000000 | 0.190604 | 0.061259 | 0.001237 |
| S01 / p07 | Thanks. | False | True | 0.000000 | 0.271828 | 0.022050 | 0.005033 |
| S01 / p08 | Whatever. | False | True | 0.000000 | 0.185063 | 0.020851 | 0.006549 |
| S01 / p09 | All right. | True | True | 0.000000 | 0.106053 | 0.008357 | 0.003304 |
| S01 / p10 | I understand. | True | True | 0.000000 | 0.093201 | 0.010547 | 0.009527 |
| S02 / p01 | That's great. | False | True | 0.000000 | 0.359219 | 0.064984 | 0.005134 |
| S02 / p02 | Sure. | False | True | 0.000000 | 0.139666 | 0.022247 | 0.018942 |
| S02 / p03 | I'm fine. | False | True | 0.000000 | 0.112052 | 0.021758 | 0.010809 |
| S02 / p04 | Really? | False | True | 0.000000 | 0.127390 | 0.014618 | 0.000257 |
| S02 / p05 | Okay. | False | True | 0.000000 | 0.165236 | 0.018618 | 0.007746 |
| S02 / p06 | I can't believe that. | False | True | 0.000000 | 0.203531 | 0.063892 | 0.001792 |
| S02 / p07 | Thanks. | False | True | 0.000000 | 0.079591 | 0.007841 | 0.006966 |
| S02 / p08 | Whatever. | False | True | 0.000000 | 0.123159 | 0.014142 | 0.010897 |
| S02 / p09 | All right. | True | True | 0.000000 | 0.148885 | 0.007660 | 0.004096 |
| S02 / p10 | I understand. | True | True | 0.000000 | 0.214562 | 0.026783 | 0.009097 |

## All directional failures and unchanged contrasts

These are failures relative to intended-category assumptions, not listener-confirmed mistakes. Unmapped pairs are excluded from this sign assessment but retained in pairs.json.

- S01_p05_d1 / S01_p05_d3, targets ['joy', 'anger']: B D=-0.001116 (opposite), A D=-0.007198; B TV=0.008773.
- S01_p06_d1 / S01_p06_d2, targets ['surprise', 'anger']: B D=-0.001163 (opposite), A D=-0.021864; B TV=0.001028.
- S01_p06_d2 / S01_p06_d3, targets ['anger', 'sadness']: B D=-0.000180 (opposite), A D=-0.036536; B TV=0.001375.
- S02_p01_d2 / S02_p01_d3, targets ['anger', 'sadness']: B D=-0.000265 (opposite), A D=+0.025807; B TV=0.004515.
- S02_p02_d2 / S02_p02_d3, targets ['anger', 'fear']: B D=-0.003698 (opposite), A D=+0.001813; B TV=0.008060.
- S02_p03_d1 / S02_p03_d3, targets ['neutral', 'anger']: B D=-0.002843 (opposite), A D=+0.021341; B TV=0.006953.
- S02_p04_d1 / S02_p04_d2, targets ['surprise', 'anger']: B D=-0.000038 (opposite), A D=+0.017751; B TV=0.000074.
- S02_p04_d1 / S02_p04_d3, targets ['surprise', 'fear']: B D=-0.000344 (opposite), A D=+0.022473; B TV=0.000336.
- S02_p05_d1 / S02_p05_d2, targets ['joy', 'sadness']: B D=-0.001371 (opposite), A D=+0.011997; B TV=0.011047.
- S02_p05_d1 / S02_p05_d3, targets ['joy', 'anger']: B D=-0.005366 (opposite), A D=+0.008012; B TV=0.007423.
- S02_p05_d2 / S02_p05_d3, targets ['sadness', 'anger']: B D=-0.003008 (opposite), A D=+0.002312; B TV=0.004770.
- S02_p06_d1 / S02_p06_d2, targets ['surprise', 'anger']: B D=-0.001788 (opposite), A D=+0.039157; B TV=0.001644.
- S02_p06_d1 / S02_p06_d3, targets ['surprise', 'sadness']: B D=-0.000292 (opposite), A D=+0.087240; B TV=0.001999.

## Every pair, including stable and unmapped cases

| Pair | Targets | Text TV | Audio TV | A TV | B TV | A D | B D | B direction |
|---|---|---:|---:|---:|---:|---|---|---|
| S01_p01_d1 / S01_p01_d2 | ['joy', 'anger'] | 0.000000 | 0.300588 | 0.064923 | 0.003469 | +0.079146 | +0.003662 | aligned |
| S01_p01_d1 / S01_p01_d3 | ['joy', 'sadness'] | 0.000000 | 0.178745 | 0.035433 | 0.003491 | +0.044945 | +0.004630 | aligned |
| S01_p01_d2 / S01_p01_d3 | ['anger', 'sadness'] | 0.000000 | 0.242494 | 0.040687 | 0.003070 | +0.023213 | +0.001286 | aligned |
| S01_p02_d1 / S01_p02_d2 | [None, 'anger'] | 0.000000 | 0.143442 | 0.025247 | 0.015267 | n/a | n/a | unmapped_or_same_category |
| S01_p02_d1 / S01_p02_d3 | [None, 'fear'] | 0.000000 | 0.071883 | 0.017230 | 0.025886 | n/a | n/a | unmapped_or_same_category |
| S01_p02_d2 / S01_p02_d3 | ['anger', 'fear'] | 0.000000 | 0.158170 | 0.034221 | 0.012890 | +0.014832 | +0.002906 | aligned |
| S01_p03_d1 / S01_p03_d2 | ['neutral', 'sadness'] | 0.000000 | 0.166185 | 0.037217 | 0.017716 | +0.052842 | +0.028196 | aligned |
| S01_p03_d1 / S01_p03_d3 | ['neutral', 'anger'] | 0.000000 | 0.274146 | 0.046119 | 0.007052 | +0.057174 | +0.006057 | aligned |
| S01_p03_d2 / S01_p03_d3 | ['sadness', 'anger'] | 0.000000 | 0.208631 | 0.015801 | 0.013665 | +0.011609 | +0.008322 | aligned |
| S01_p04_d1 / S01_p04_d2 | ['surprise', 'anger'] | 0.000000 | 0.189148 | 0.011057 | 0.000283 | -0.004794 | +0.000307 | aligned |
| S01_p04_d1 / S01_p04_d3 | ['surprise', 'fear'] | 0.000000 | 0.155564 | 0.022163 | 0.000194 | -0.024633 | +0.000041 | aligned |
| S01_p04_d2 / S01_p04_d3 | ['anger', 'fear'] | 0.000000 | 0.285712 | 0.023170 | 0.000286 | +0.005273 | +0.000006 | aligned |
| S01_p05_d1 / S01_p05_d2 | ['joy', 'sadness'] | 0.000000 | 0.144543 | 0.015147 | 0.006494 | +0.002740 | +0.000451 | aligned |
| S01_p05_d1 / S01_p05_d3 | ['joy', 'anger'] | 0.000000 | 0.135247 | 0.014036 | 0.008773 | -0.007198 | -0.001116 | opposite |
| S01_p05_d2 / S01_p05_d3 | ['sadness', 'anger'] | 0.000000 | 0.114465 | 0.015935 | 0.014996 | +0.007514 | +0.005980 | aligned |
| S01_p06_d1 / S01_p06_d2 | ['surprise', 'anger'] | 0.000000 | 0.226334 | 0.088249 | 0.001028 | -0.021864 | -0.001163 | opposite |
| S01_p06_d1 / S01_p06_d3 | ['surprise', 'sadness'] | 0.000000 | 0.191708 | 0.050966 | 0.001308 | +0.042249 | +0.001539 | aligned |
| S01_p06_d2 / S01_p06_d3 | ['anger', 'sadness'] | 0.000000 | 0.153771 | 0.044561 | 0.001375 | -0.036536 | -0.000180 | opposite |
| S01_p07_d1 / S01_p07_d2 | [None, 'sadness'] | 0.000000 | 0.250374 | 0.027791 | 0.006865 | n/a | n/a | unmapped_or_same_category |
| S01_p07_d1 / S01_p07_d3 | [None, 'anger'] | 0.000000 | 0.293363 | 0.019677 | 0.005457 | n/a | n/a | unmapped_or_same_category |
| S01_p07_d2 / S01_p07_d3 | ['sadness', 'anger'] | 0.000000 | 0.271747 | 0.018681 | 0.002776 | +0.016727 | +0.004496 | aligned |
| S01_p08_d1 / S01_p08_d2 | [None, 'anger'] | 0.000000 | 0.174334 | 0.025961 | 0.004415 | n/a | n/a | unmapped_or_same_category |
| S01_p08_d1 / S01_p08_d3 | [None, 'sadness'] | 0.000000 | 0.134681 | 0.019886 | 0.008912 | n/a | n/a | unmapped_or_same_category |
| S01_p08_d2 / S01_p08_d3 | ['anger', 'sadness'] | 0.000000 | 0.246173 | 0.016707 | 0.006320 | +0.015121 | +0.006762 | aligned |
| S01_p09_d1 / S01_p09_d2 | ['neutral', 'neutral'] | 0.000000 | 0.125163 | 0.011039 | 0.003110 | n/a | n/a | unmapped_or_same_category |
| S01_p09_d1 / S01_p09_d3 | ['neutral', 'neutral'] | 0.000000 | 0.077883 | 0.008139 | 0.004912 | n/a | n/a | unmapped_or_same_category |
| S01_p09_d2 / S01_p09_d3 | ['neutral', 'neutral'] | 0.000000 | 0.115113 | 0.005892 | 0.001890 | n/a | n/a | unmapped_or_same_category |
| S01_p10_d1 / S01_p10_d2 | ['neutral', 'neutral'] | 0.000000 | 0.108035 | 0.014155 | 0.010674 | n/a | n/a | unmapped_or_same_category |
| S01_p10_d1 / S01_p10_d3 | ['neutral', 'neutral'] | 0.000000 | 0.046560 | 0.004483 | 0.014078 | n/a | n/a | unmapped_or_same_category |
| S01_p10_d2 / S01_p10_d3 | ['neutral', 'neutral'] | 0.000000 | 0.125009 | 0.013003 | 0.003830 | n/a | n/a | unmapped_or_same_category |
| S02_p01_d1 / S02_p01_d2 | ['joy', 'anger'] | 0.000000 | 0.422367 | 0.081653 | 0.007629 | +0.085089 | +0.008358 | aligned |
| S02_p01_d1 / S02_p01_d3 | ['joy', 'sadness'] | 0.000000 | 0.351702 | 0.079721 | 0.003258 | +0.097133 | +0.004130 | aligned |
| S02_p01_d2 / S02_p01_d3 | ['anger', 'sadness'] | 0.000000 | 0.303590 | 0.033578 | 0.004515 | +0.025807 | -0.000265 | opposite |
| S02_p02_d1 / S02_p02_d2 | [None, 'anger'] | 0.000000 | 0.185190 | 0.031090 | 0.023419 | n/a | n/a | unmapped_or_same_category |
| S02_p02_d1 / S02_p02_d3 | [None, 'fear'] | 0.000000 | 0.125276 | 0.020415 | 0.025346 | n/a | n/a | unmapped_or_same_category |
| S02_p02_d2 / S02_p02_d3 | ['anger', 'fear'] | 0.000000 | 0.108531 | 0.015236 | 0.008060 | +0.001813 | -0.003698 | opposite |
| S02_p03_d1 / S02_p03_d2 | ['neutral', 'sadness'] | 0.000000 | 0.087167 | 0.017819 | 0.010935 | -0.021610 | +0.016034 | aligned |
| S02_p03_d1 / S02_p03_d3 | ['neutral', 'anger'] | 0.000000 | 0.097423 | 0.016954 | 0.006953 | +0.021341 | -0.002843 | opposite |
| S02_p03_d2 / S02_p03_d3 | ['sadness', 'anger'] | 0.000000 | 0.151565 | 0.030501 | 0.014538 | -0.008576 | +0.011103 | aligned |
| S02_p04_d1 / S02_p04_d2 | ['surprise', 'anger'] | 0.000000 | 0.124513 | 0.015205 | 0.000074 | +0.017751 | -0.000038 | opposite |
| S02_p04_d1 / S02_p04_d3 | ['surprise', 'fear'] | 0.000000 | 0.169743 | 0.020638 | 0.000336 | +0.022473 | -0.000344 | opposite |
| S02_p04_d2 / S02_p04_d3 | ['anger', 'fear'] | 0.000000 | 0.087915 | 0.008010 | 0.000360 | +0.006290 | +0.000071 | aligned |
| S02_p05_d1 / S02_p05_d2 | ['joy', 'sadness'] | 0.000000 | 0.147433 | 0.013461 | 0.011047 | +0.011997 | -0.001371 | opposite |
| S02_p05_d1 / S02_p05_d3 | ['joy', 'anger'] | 0.000000 | 0.192715 | 0.020218 | 0.007423 | +0.008012 | -0.005366 | opposite |
| S02_p05_d2 / S02_p05_d3 | ['sadness', 'anger'] | 0.000000 | 0.155559 | 0.022175 | 0.004770 | +0.002312 | -0.003008 | opposite |
| S02_p06_d1 / S02_p06_d2 | ['surprise', 'anger'] | 0.000000 | 0.213544 | 0.070600 | 0.001644 | +0.039157 | -0.001788 | opposite |
| S02_p06_d1 / S02_p06_d3 | ['surprise', 'sadness'] | 0.000000 | 0.231020 | 0.080217 | 0.001999 | +0.087240 | -0.000292 | opposite |
| S02_p06_d2 / S02_p06_d3 | ['anger', 'sadness'] | 0.000000 | 0.166029 | 0.040858 | 0.001735 | +0.048276 | +0.001714 | aligned |
| S02_p07_d1 / S02_p07_d2 | [None, 'sadness'] | 0.000000 | 0.076376 | 0.009430 | 0.009479 | n/a | n/a | unmapped_or_same_category |
| S02_p07_d1 / S02_p07_d3 | [None, 'anger'] | 0.000000 | 0.076288 | 0.003725 | 0.003664 | n/a | n/a | unmapped_or_same_category |
| S02_p07_d2 / S02_p07_d3 | ['sadness', 'anger'] | 0.000000 | 0.086108 | 0.010368 | 0.007754 | +0.007039 | +0.006665 | aligned |
| S02_p08_d1 / S02_p08_d2 | [None, 'anger'] | 0.000000 | 0.165017 | 0.020652 | 0.012045 | n/a | n/a | unmapped_or_same_category |
| S02_p08_d1 / S02_p08_d3 | [None, 'sadness'] | 0.000000 | 0.064130 | 0.011397 | 0.005302 | n/a | n/a | unmapped_or_same_category |
| S02_p08_d2 / S02_p08_d3 | ['anger', 'sadness'] | 0.000000 | 0.140330 | 0.010377 | 0.015342 | -0.013116 | +0.000344 | aligned |
| S02_p09_d1 / S02_p09_d2 | ['neutral', 'neutral'] | 0.000000 | 0.070651 | 0.003663 | 0.002249 | n/a | n/a | unmapped_or_same_category |
| S02_p09_d1 / S02_p09_d3 | ['neutral', 'neutral'] | 0.000000 | 0.154357 | 0.007987 | 0.005141 | n/a | n/a | unmapped_or_same_category |
| S02_p09_d2 / S02_p09_d3 | ['neutral', 'neutral'] | 0.000000 | 0.221645 | 0.011331 | 0.004896 | n/a | n/a | unmapped_or_same_category |
| S02_p10_d1 / S02_p10_d2 | ['neutral', 'neutral'] | 0.000000 | 0.299940 | 0.039055 | 0.012434 | n/a | n/a | unmapped_or_same_category |
| S02_p10_d1 / S02_p10_d3 | ['neutral', 'neutral'] | 0.000000 | 0.147026 | 0.017341 | 0.003086 | n/a | n/a | unmapped_or_same_category |
| S02_p10_d2 / S02_p10_d3 | ['neutral', 'neutral'] | 0.000000 | 0.196721 | 0.023954 | 0.011769 | n/a | n/a | unmapped_or_same_category |

## Artifacts and limits

- outputs.json / recordings.jsonl: every metadata item and waveform hash; text/audio/A/B full distributions, logits, argmaxes, intended-category probabilities, gate, TV influence, residual norms and timing.
- pairs.json / sets.json / summary.json: all paired distances, both target-probability changes, direction signs and stability, with no favorable-case filtering.
- recording_lock.json / integrity.json: accepted input hashes, frozen model/policy hashes and local hardware.
- No training, tuning, test inference, re-recording, selection change or final freeze occurs. The public listener UI is unchanged; keep this provisional report away from L01 until ratings are complete.
- Intended labels may disagree with perceived emotion. Room/microphone, timing, breath and recording variation remain confounds. No prosodic causality or human-validated performance claim follows.
- When L01 finishes, the listener action verifies the same input/output/model hashes and calls the existing blinded-judgment interpretation on saved outputs. It does not reload encoders or run inference.

## Verification and listener continuation

All 60 waveform/metadata identities are unchanged. Every saved pairwise TV and directional contrast was independently recomputed. All 65 existing preflight source/config hashes still match; only the standalone diagnostic tooling and its focused tests were added. The ordinary suite reports **58 passed, 3 skipped**; lint and formatting passed. The three skipped cases are pretrained tests, not missing human validation. Sixty actual raw-recording predictions were completed in this pass.

L01 remains incomplete; no final human report, FINAL_FREEZE or official-test outputs were created. See [NEXT_STEPS.md](NEXT_STEPS.md) for the interpretation-only command and [validation.json](validation.json) for the checks. Do not expose this provisional interpretation to L01 before completion.

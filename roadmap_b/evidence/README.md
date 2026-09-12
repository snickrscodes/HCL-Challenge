# Public evidence map

These are **development** artifacts and a provisional human-delivery smoke test. Official MELD test has not been evaluated. `FINAL_INFERENCE_POLICY.json` is the fixed operational policy; it is not `FINAL_FREEZE.json` and does not authorize final test.

The original local A/B artifacts were retained unchanged and remain outside Git. `PUBLIC_EXPORT_MANIFEST.json` lists each exported file, original logical source, original SHA256, public SHA256, size and transformation. Most exports are byte-identical; only local path labels are replaced where needed. Human integrity is an explicit projection containing model hashes, baseline integrity and hardware, without consent/recording metadata. No metric was recomputed, rounded or edited during export. Original embedded source/artifact hashes still refer to the originals, not redacted public bytes.

## Where to inspect a claim

| Directory/file | Evidence |
|---|---|
| `roadmap_a/ROADMAP_A_REPORT.md`, `results_bundle.json` | Historical A progression, metrics, environment and limits |
| `data_audit/` | Counts, failure/outlier rows, duration summaries and exact dev/calibration dialogue IDs; no raw annotation/transcript dump |
| `../baseline_lock.json` | Original A archive and 71 file identities, fixed splits and annotation hashes |
| `roadmap_b/numerical_replay/` | Real singleton/equal-batch BF16/FP32 comparison and approved text FP32 repair |
| `roadmap_b/canonical_policy.json`, `text_policy.json` | B's canonical audio/text contract |
| `roadmap_b/model_freeze.json` | Completed B model/cache identities; not the later final-test freeze |
| `roadmap_b/training_summary.json` | All four variants × three seeds, best epochs, duration and parameter counts |
| `roadmap_b/evaluation/` | Saved numeric development logits/probabilities, per-class metrics, paired changes, calibration, predetermined slices, bootstrap and interventions |
| `roadmap_b/advancement.json` | Predeclared engineering continuation checks, not statistical significance |
| `roadmap_b/raw_validation/` | Raw/cache equivalence, availability, parameter ledger, silence/noise and measured warm runtime |
| `roadmap_b/evaluation_repair.json` | Operational evaluation repair provenance, without concealing the failed first calculation |
| `roadmap_b/ROADMAP_B_REPORT.md` | Completed controlled experiment and class tradeoffs; historical human-pending wording reflects its creation date |
| `figures/` | Six development plots with underlying CSV data and captions |
| `policy/FINAL_INFERENCE_POLICY.json` | Singleton FP32 text, singleton BF16 audio, 60-second availability policy |
| `controlled_delivery_provisional/` | All derived pairs/sets, intended mapping, summary, numerical readout, original model identities and validation; L01 pending |

Historical reports are preserved as evidence, with only declared path redactions in separate copies. Read the repository README for the **current** status: the 60 recordings and provisional smoke are now complete. Do not treat a report's older human-pending statement as the current collection count. Figure/report generator source lives in `../scripts/`; published CSVs support inspection without model downloads.

## Reproduce this export

From `roadmap_b/`, with original artifacts restored separately:

```bash
python scripts/export_public_evidence.py --baseline-root /path/to/roadmap_a_source \
  --output-root /path/to/new_public_export
```

The exporter uses an explicit file allowlist and refuses an existing destination. It has no model loading, training or test-inference step. Root Git ignores keep operational artifacts private; new scientific outputs are reviewed and exported deliberately rather than automatically published.

Full MELD media, complete annotations/transcripts, all learned weights and feature caches, human WAVs, raw `outputs.json`/`recordings.jsonl`, consent data, timestamps, blinded listener maps/responses, private study notes, environments and continuation archives are omitted. The derived human figures use only pseudonymous speaker IDs and intended-delivery comparisons. Public derivations cannot run the later L01 interpretation by themselves: that requires the original private collection/output files.

Some historical reports show brief example utterances as discussion, not a redistributed transcript dataset. Dataset/media rights remain with their owners. No raw voice release permission is inferred from study participation.

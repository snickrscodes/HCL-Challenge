# Provisional smoke: continuation instructions

All accepted recordings were inferred once using the frozen selected models, with one explicit warm-up. The outputs are provisional engineering evidence; no independent listener labels were used. L01 is still incomplete. Keep this report and model outputs hidden from L01 until the blinded ratings are submitted.

From the Roadmap B repository, with the existing Python environment active:

```bash
python -m src.delivery status
```

Only after the independent listener pass is complete:

```bash
python -m scripts.controlled_delivery_smoke listener
```

The listener action verifies recording, output and frozen model/policy identities, strips provisional category annotations, then calls the existing blinded-listener summarizer on saved distributions. It does not instantiate an encoder, recompute embeddings, retrain, calibrate or select models. It writes the separate final diagnostic to `artifacts/submission/delivery_results/` and refuses to overwrite it. Do not run the older `src.delivery run` for this collection: that path would unnecessarily repeat inference.

The provisional directory remains intact. Final listener categories may change which pairs are evaluable, aligned, stable or unclear. Report those differences honestly. Completion of this smoke does not satisfy the final human diagnostic or authorize final freeze/test.

The new standalone diagnostic utility/test files must be included in the later current-source preflight. Existing frozen perception modules and policies remain unchanged. Script snapshots are retained under `source/`; `delivery_smoke_readout.py` derives the readout from saved JSON only. Full raw recordings remain in the local collection directory, not copied into this result bundle or uploaded to RunPod.

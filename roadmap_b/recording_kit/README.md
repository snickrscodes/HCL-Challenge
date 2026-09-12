# Human delivery diagnostic: collection guide

This is a fixed post-selection diagnostic, outside training, calibration and model selection. No human recordings have been collected as part of software tests. Synthetic browser-device fixtures are disposable software tests, never human evidence.

## Start the local kit

From the `roadmap_b/` implementation directory, using the existing project environment:

```bash
python -m src.recording_kit --port 8765
```

Open http://localhost:8765 in Chrome or Edge on this computer. Use the same computer for both speakers and the listener. The server binds to loopback; another device cannot connect. Keep the terminal running. If browser microphone permission is denied, enable it for localhost and retry. In Windows/WSL, open the URL in the Windows browser using the actual microphone.

## Speakers

1. Speaker S01 reads the consent statement and checks their own consent box. Public release of voice recordings is a separate optional checkbox.
2. Record the displayed exact words with the requested natural delivery. Keep microphone, distance and room as comparable as practical. Do not add words or spoken commentary.
3. Stop, listen back and confirm the words and recording are correct. Retry a misread or technical failure before accepting. Accepted takes are immutable; do not delete/re-record based on a model output.
4. Complete 30 takes: ten fixed phrases, three deliveries each. Select S02 and repeat with a second consenting speaker. Each speaker provides their own consent.
5. Eight phrases have contrasting deliveries. Two are stable neutral controls: slight volume/rate variation should not automatically imply a different emotion.

The kit saves mono float32 WAV at the browser's native sample rate. Browser processing settings, capture backend, timestamps, sample count, duration, transcript, delivery and SHA256 are recorded. Browser and server hashes must agree. AudioWorklet is preferred; browsers with unavailable/stalled worklets use a logged ScriptProcessor PCM compatibility path (a deprecated browser API, with no compression). Accepted waveforms must be finite and nonzero. Listen back to catch clipping, dropped audio or background interruptions. Runtime later uses the shared 16-kHz preprocessing path.

## Independent listener

After all 60 takes are accepted, give http://localhost:8765/listen to someone who did not record them or see model outputs/delivery instructions. Do not show them this phrase manifest or speaker instructions.

Use L01. Rate 60 individual clips in a fixed randomized order, choosing the closest seven-class interpretation or unclear/mixed. Then rate 60 same-words pairs as meaningfully different, not different, or unsure. Optional notes are welcome. Intended delivery and model outputs are hidden. Saved judgments cannot be changed after results are shown. A second independent listener may use L02, if practical.

The pages resume accepted work after a refresh. Complete the listener pass before revealing model outputs. Intended delivery is an instruction, not objective ground truth.

## Files and privacy

The default collection root is `artifacts/submission/controlled_delivery/`:

- `protocol.json`: immutable phrase/delivery/speaker plan.
- `records/*.wav` and `records/*.json`: accepted audio and metadata.
- `listener_plan_private.json`: local blind-token mapping; do not show to listeners.
- `ratings/L01/*.json`: independent judgments.

These files remain local. Do not upload raw human recordings to RunPod or a public package by default. Public voice release requires the separate recorded permission for every included take. Private diagnostic waveforms can be omitted from the reviewer archive while retaining hashes and aggregate results.

Check completion without loading models:

```bash
python -m src.delivery status
```

Tell the project operator when S01, S02 and L01 are complete. The operator then runs the frozen A/B comparison locally. No recording is used to tune models, select examples for a favorable result, or change final inference policy. Official MELD test remains closed until this diagnostic and final validation are complete.

## Current collection and provisional analysis

S01/S02 have completed all 60 accepted takes. The frozen-model provisional intended-delivery smoke is complete; L01 remains pending. Do not repeat inference or change accepted takes based on results. Keep provisional output hidden from the blinded listener. After L01 finishes, run `python -m scripts.controlled_delivery_smoke listener` to reinterpret unchanged saved outputs against listener judgments. Public aggregate/pairwise evidence is separate from the local raw outputs, consent/timestamp metadata and ratings. This provisional smoke does not satisfy the final listener-validated diagnostic.

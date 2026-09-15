# Stateful perception v1

The stateful API extends the frozen utterance-final T+A system with causal observations, session-local participants and immutable snapshots. Existing `ta.py`, `SystemSession` and `predict_turn(...)` remain available unchanged.

## What the state means

- **Observations:** supplied human text/PCM, external speaker associations and actually emitted robot text.
- **Perception:** model estimates at a specific observation boundary. B controls the established MELD categorical state. D2 remains separately namespaced acoustic/listener evidence and representation.
- **Execution:** requests, completion, cancellation, failure and emission acknowledgements. Selecting a response does not establish emission.

The system never turns a predicted emotion into observed text or identity. Only the application assertion `record_robot_emitted` adds robot dialogue. No dynamic generation is loaded.

## Run

Use the same locked Python environment and hash-verified external assets described in the root README. Run from the repository root:

```bash
python stateful_ta.py --asset-root /path/to/ta-assets --text 'Hello.'
python stateful_ta.py --asset-root /path/to/ta-assets --text 'Hello.' --audio /path/to/utterance.wav
python stateful_ta.py --asset-root /path/to/ta-assets --replay /path/to/human-turns.json
python stateful_ta.py --asset-root /path/to/ta-assets --events /path/to/events.jsonl
```

The CLI supports up to64 completed turns per invocation. The session API supports longer interactions with bounded retention. CUDA uses the frozen FP32-text/BF16-WavLM scope; explicit `--device cpu` is a separate FP32 scope. Offline model loading is enforced.

## Python API

With `roadmap_b` on the Python import path:

```python
from src.interaction import StatefulSession
from src.ta_runtime import PortableTADual

model = PortableTADual(asset_root)
session = StatefulSession(model, release_manifest, session_id="interaction-1")
session.start()
participant = session.register_participant("external-speaker-1")
turn = session.start_turn(participant)
session.observe_text_fragment(turn, "Hello")
session.observe_audio_chunk(turn, mono_float32_16khz_pcm)
partial = session.get_latest_perception(turn)
session.finalize_text(turn, "Hello.")
final = session.end_turn(turn)
response = session.get_response(turn)
# After the application actually emits this exact response:
session.record_robot_emitted(turn, response)
session.close()
model.close()
```

The caller owns the model. Serialize all access when sharing a model between sessions: its WavLM summary scratch state is mutable. A session accepts one active/in-flight turn. Cancellation invalidates pending publication; a new turn cannot begin until the old model call exits. In-flight model execution is not interrupted, so its bounded request PCM remains alive until that call returns. Unsupported overlap raises explicitly.

## Lifecycle and causal context

Fragments append to current observed text. Final text may correct them without changing past snapshots. Text may finalize before audio finishes; audio may finish before final text arrives. End requires finalized text. Cancellation and session close release buffers and exclude the turn from future T1 context. Inference failure remains an observed finalized utterance, with a failed execution disposition, but follows the original success-only T1 context rule.

T1 receives exactly the preceding three successfully perceived finalized **human** turns across participants, with unchanged external speaker keys and token packing. Partial text, robot responses, inferred affect and cancelled/failed turns do not enter T1. An optional initial history is explicitly recorded as externally supplied human observations at session start.

## Snapshots and audio

Snapshots store canonical immutable JSON; `to_dict()` returns a fresh defensive copy. IDs, as-of boundary and semantic creation time are replay-derived. Measured latency stays in `session.last_timing`, outside snapshot hashes. Partial snapshots are **input checkpoints with no model evidence**, not early emotion predictions. Final snapshots contain frozen B/D2 evidence and the proposed deterministic action.

PCM ingestion is incremental; **WavLM remains utterance-final**, with one shared execution per eligible final turn. The chunk API requires finite mono floating PCM at16kHz, up to65,536 samples per event. Existing raw one-shot APIs retain their broader preprocessing support. There is no per-chunk normalization or resampling. Missing/corrupt/over-limit audio follows the original fallback. Valid silence remains audio.

After more than60 seconds, the state releases PCM and retains count/digest plus duration-limit status. Finalization uses a documented transient duration sentinel solely to select the unchanged fallback before WavLM. It is never represented as observed silence.

## Bounded retention and replay

Defaults:32 participants;64 turns/dialogue entries;128 snapshots;256 observation/execution/dedup entries;3 T1 context turns;16,384 text characters;60seconds PCM. Participant capacity raises instead of silently losing identity. All raw PCM is released on finalization/cancellation/failure/close. Closing clears session-owned content; caller-retained immutable snapshots remain valid.

Events use `session_id:event_seq` IDs. Strict contiguous sequence and nondecreasing finite receipt timestamps reject stale/out-of-order input atomically. Exact duplicates inside the last256 accepted-event window acknowledge without effects; older duplicates reject as stale and never reapply.

`src.interaction.replay.write_events/read_events/replay` supplies explicit JSONL debug replay, including immutable PCM bytes and identity hashes. Files have a64MiB bound and are never recorded automatically. Production retained observation metadata contains only audio counts/hashes, not duplicate PCM. A retained suffix is not a complete replay log.

## Future extensions

SPEAKER_ASSOCIATION = EXTERNALLY_SUPPLIED. Future speaker embeddings, anonymous voice/face tracks, active-speaker evidence and visual-expression estimates need interval timestamps, receipt boundaries, provenance, availability and uncertain participant association. They must create later evidence without rewriting historical ownership. Missing evidence stays absent; no fake vectors exist.

Early text/audio model estimates, learned temporal fusion and WAIT/COMMIT policies are deferred. The research specification and future RL plan live in `roadmap_b/research/stateful_v1`; production does not import them. Core adds **0 learned parameters**; frozen loaded total remains **218,895,839**.

### Completion-aware replay

For cancellation during inference, failure/recovery, or custom retention, pass a caller-owned `ReplayRecorder` from `src.interaction.replay` as `recorder=` when constructing the session. `recorder.write(path)` writes separate observation and execution completion records plus the exact limits/release identity. `replay(read_events(path), model, release)` recomputes successful final perception at recorded completion boundaries and reproduces recorded failed/discarded execution without inventing model estimates. Observation-only event files support sequential successful replay; they cannot encode concurrent completion order. Recording is opt-in, capped at64MiB; overflow disables complete export explicitly. The recorder retains its debug artifact independently after session close.

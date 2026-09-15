# Stateful Perception v1 controlling specification

Status: FROZEN BEFORE IMPLEMENTATION. Machine-readable companion: STATEFUL_SPEC.json. Accepted base: 3a403f4418a4399f0a40741ca9f54bd3b5038006. Amendments must record reason and precede rerun; no threshold relaxation.

## Production selection

A small session owner, immutable input/perception checkpoints, final frozen B/D2 and deterministic evidence-only response. Zero learned parameters. No partial model inference or acoustic prefix experiment selected. Partial input checkpoints explicitly contain no model estimate. Existing APIs and ta.py remain unchanged; stateful_ta.py supplies the stateful root CLI. No production import from research.

## Types and ownership

- ObservationEvent: immutable envelope with session ID, sequence-bound event ID, nondecreasing monotonic receipt timestamp, type, optional participant/turn, source, availability, provenance and immutable payload. PCM uses immutable little-endian float32 bytes, separate from metadata JSON. Model predictions are not event types.
- ParticipantState: frozen session-local anonymous ID, external speaker key and last observed event. Turn associations refer to these IDs. Registration-order IDs are deterministic; same supplied key returns the same local participant. Capacity exhaustion raises, rather than silently forgetting identities. No real-world recognition. SPEAKER_ASSOCIATION = EXTERNALLY_SUPPLIED.
- TurnState: session-owned mutable lifecycle state; externally exposed through defensive JSON summaries. One active/pending turn, receiving/finalized/cancelled status, partial/final text, participant binding, PCM count/digest and bounded chunks. No caller may mutate owned buffers.
- PerceptionSnapshot: frozen canonical JSON, identity, session/turn/participant, as-of sequence/time, partial/final stage, observed-input summary, original human context references, availability and optional final B/D2 evidence. All nested values detached by JSON serialization. Returned dictionaries are fresh copies.
- InteractionState: exported defensive view with separate observations, perception and execution; bounded participants, turns, dialogue, snapshots, emission and cancellation metadata. No raw PCM in exported state or retained observation log. Model outputs are never stored in observed dialogue.

## Events and transitions

SESSION_STARTED must be first. Participants register before use. HUMAN_TURN_STARTED rejects another active or pending turn. Turn IDs are deterministic from the start event sequence, preventing reuse after retention eviction. TEXT_FRAGMENT_OBSERVED appends; TEXT_FINALIZED supplies the complete final transcript and may correct earlier fragments without rewriting checkpoints. Text-final status rejects later fragments, but audio may still arrive. If audio ends first, callers simply stop sending it until text finalization. HUMAN_TURN_ENDED requires finalized text and seals input; it synchronously requests final inference. SESSION_ENDED cancels any incomplete/pending turn and clears all session-owned content except sequence/closed-state and bounded duplicate acknowledgements; callers may retain immutable snapshots themselves.

TURN_CANCELLED rejects already successfully finalized turns; it cancels receiving or pending inference. A request token contains session/turn/generation/event boundary. Cancellation or close invalidates it. Predictor execution occurs outside the state lock; completion checks the token before changing snapshots/context. Stale completion returns no new snapshot and cannot replace later state. No inference retries or parallel model execution are added. Applications sharing a model must serialize access to its mutable summary tap.

AUDIO_UNAVAILABLE records an externally asserted decode failure before any PCM. Invalid ordinary PCM chunks reject atomically, matching Session. A malformed whole input may be represented as decode failure through the compatibility helper. Valid silence stays available. No event after close except exact retained duplicate. Unknown types, foreign sessions, sequence gaps, stale sequence, decreasing/nonfinite timestamp, illegal association and oversized payload reject before mutation.

Event IDs are session_id:event_seq. Exact duplicates within the bounded digest cache acknowledge without any side effect or model rerun. A conflicting duplicate raises. Older expired events raise as stale; they never reapply. This makes bounded idempotency explicit instead of claiming indefinite arbitrary-ID memory. Rejection diagnostics are outside authoritative state.

## Observation, inference and action boundaries

Text/audio observations generate partial input checkpoints with unavailable model evidence. Final snapshots call the unmodified PortableTADual.predict_turn once, then select_response(policy=evidence_only). B remains MELD operational state and D2 remains CREMA6 evidence plus representation. Original model/checkpoint/numerical provenance is preserved, and session metadata is additional. Volatile latency is stored separately as telemetry, never hashed into semantic snapshots. created_at is the semantic event timestamp; actual wall execution timing is telemetry.

Retrieving a response records no emitted dialogue. ROBOT_RESPONSE_EMITTED requires a matching retained final turn response and is an application assertion that emission actually happened. It is valid at most once per response, including after unrelated later observations; it never becomes human T1 context. There is no draft-generation API. Cancellation prevents an unpublished response from acquiring a final snapshot.

## Exact compatibility

Use the original human speaker key when calling the predictor; anonymous participant IDs never replace SAME/OTHER marker semantics. Maintain the last three successful finalized human turns in a separate deque. Inference failure retains the actual end observation and a failed execution disposition, but does not add that turn to T1 context, matching SystemSession. Raw model B logits are inspected by validation hooks, not introduced as a changed predictor return contract.

PCM arrives as finite mono 16kHz float PCM, copied to float32 as in existing SystemSession. Retain at most 960,000 samples (60 seconds). Once exceeded, release chunks and retain only count/hash and a duration-limit flag. At final inference a transient zero array of 960,001 samples invokes the unchanged duration-limit branch: it is explicitly an execution sentinel, not observed silent evidence, and never reaches WavLM. Missing uses None; corrupt uses the established invalid-audio sentinel. Whole-utterance preparation and the one shared WavLM execution are unchanged. New chunk API supports canonical 16kHz; existing one-shot APIs continue supporting their prior input formats.

## Retention and replay

Defaults are in JSON: 32 participants (reject at capacity), 64 turns/dialogue entries, 128 snapshots, 256 observation/execution/dedup entries, 16,384 text characters, 65,536 samples per chunk. IDs and external keys are limited to 128 characters. Each participant stores only a last-event reference; retained turns provide bounded observed-turn references. Latest-snapshot indexes are pruned with turn/snapshot eviction. Finalized/cancelled/failed raw PCM is released. No hidden adapter snapshot list grows unbounded.

Optional explicit JSONL event files encode PCM as base64 with hashes and a schema version. A bounded reader/writer enforces 64 MiB total. No automatic disk memory. Ordinary retained observations hold PCM digest/count only. Replaying identical events invokes the same final predictor, snapshots and execution dispositions under the same environment; volatile runtime telemetry is excluded. Complete debug replay requires the original explicit event file; a truncated in-memory retention window is not claimed to reconstruct an entire session.

## Future extension contract

Future speaker_embedding, visual_expression, active_speaker, spatial_audio and ASR_partial producers need schema-versioned evidence with source, capture interval, receipt sequence, availability, uncertainty and track/participant association. Inferred associations belong to perception. No placeholder arrays are implemented. Historical turn bindings and snapshots remain immutable; later associations create later evidence. Learned temporal state and RL policy state are separate private consumers and may not feed guesses into frozen T1.

## Gates

H0: all15 raw probes, full accessible1,109 saved-output/feature inventory with comparison level stated, four-turn raw replay, missing/corrupt/over-limit/silence, one shared WavLM call. Raw same-environment equality is bit-exact, including captured B logits, probabilities, D2 vector and response/provenance (excluding volatile timing). No cross-GPU claim.

State tests include atomic rejection, duplicates/expiration, cancellation/failure/stale results, participant/session isolation, deep immutability, human-only bounded context, future isolation, shared-prefix counterfactual, deterministic JSONL replay and close cleanup. Fixed10,000-turn synthetic stress must plateau at retention bounds. Event handling and finalization outside predictor p95 each <=5ms. Benchmark30 warmups and200 cyclic raw turns per interface on the same local5080, retain all samples and short-clip RTF limitation. No best-of reruns.

Only after mandatory gates pass: construct a separate production worktree from accepted base, validate with external assets and checkout-local imports, audit one commit, then validate a detached checkout of that immutable commit. Never push.

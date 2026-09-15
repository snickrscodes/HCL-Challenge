# Current architecture map

Inspected accepted source: reviewed T+A baseline
`3a403f4418a4399f0a40741ca9f54bd3b5038006`.
All source paths below are relative to this checkout.

## Operational call path

```text
ta.py
  -> PortableTADual(asset_root, explicit device)
  -> SystemPredictor(model, release, policy="evidence_only")
  -> SystemSession
     start -> begin_turn -> push_audio* -> set_transcript -> end_turn
       -> SystemPredictor.predict_turn
         -> inherited RefinedC1Dual / C1Dual.predict_turn
           -> validate/prepare waveform
           -> RoadmapB.predict_turn
             -> encode_turn -> singleton FP32 T1/RoBERTa
             -> singleton shared BF16 WavLM for eligible audio
               -> last-layer mean -> frozen B acoustic path
               -> all-layer summaries -> separate D2 head
             -> B categorical state and authored response
           -> separate D2 evidence
         -> select_response(evidence_only) -> immutable TurnSnapshot
       -> append successful human turn to bounded T1 context
```


## State ownership

| Owner | State | Limit or gap |
|---|---|---|
| src/session.py::Session | Mutable start/active flags, speaker, transcript, PCM chunks and history | One active turn; no participant/session IDs, events, cancellation or deduplication |
| src/ta_interface.py::SystemSession | Decode-failure flag and guaranteed end-turn cleanup | Production wrapper; generic Session has weaker failure cleanup |
| src/ta_interface.py::SystemPredictor | Model, policy, manifest and snapshots | Snapshots unbounded |
| src/ta_release.py::SnapshotPredictor | Snapshot and acknowledged-action lists | Unbounded lists; linear acknowledgement lookup |
| src/ta_release.py::TurnSnapshot | Frozen canonical JSON string | Deeply immutable; to_dict returns a fresh copy |
| src/c1_transfer_runtime.py::RefinedSummaryTap | Latest WavLM summaries, call count and timing | Mutable per-model scratch state; no concurrency protection |
| src/c1_runtime_v2.py::BackboneClock | Forward hooks and latest timing/counters | Removed by PortableTADual.close |

Snapshot IDs derive from snapshot-list length. Bounding the old list alone would cause ID reuse.
Existing snapshot timing fields vary between repeated executions.

## Exact causal context and failure semantics

The root runtime uses SystemSession, not generic Session.

SystemSession.end_turn (src/ta_interface.py, lines 68-87) requires a transcript,
concatenates PCM, calls the predictor, and appends Turn(transcript, speaker) only
after a successful predictor return. It then retains history[-3:]. A finally block
clears active state, PCM, transcript and decode-failure status on success or failure.

Thus failed inference never enters frozen T1 context. The existing
test_failure_aborts_buffers_without_promoting_history_and_next_turn_is_fresh
asserts failure cleanup, empty history, and the next successful ID turn-000001.
The new observation history may retain speech actually received despite failure;
its separate T1-eligible history must preserve this existing success-only rule.

Context contains three preceding successful human turns across all speakers,
not three turns per participant. context.py::encode_turn uses loaded T1 metadata
(reviewed values: 128 maximum tokens, three history turns). It marks historical
speakers SELF or OTHER relative to the current original speaker key, reserves the
current utterance, drops oldest complete history entries when needed, and pools
only current content. Empty tokenized current text raises.

Robot replies, drafts, B/D2 predictions and partial turns are not model context.

## PCM and availability

Session.push_audio accepts finite mono floating-point PCM, copies nonempty chunks
to FP32, and accumulates them without a duration bound. end_turn concatenates once.

SystemSession.mark_decode_failed is allowed only before PCM and blocks later
chunks. Finalization passes the established NaN execution sentinel to the existing
preprocessing boundary; this is not observed acoustic evidence.

audio.py::prepare_waveform validates positive integer sample rate, floating PCM,
finite nonempty samples and channel layout; converts to FP32; downmixes and
resamples the complete waveform to 16 kHz; and requires at least 400 samples.
It does not reject silence or normalize amplitude.

C1Dual catches preprocessing ValueError and reports D2 invalid_audio while B follows
its established calibrated text fallback. B rejects acoustic inference beyond
60 seconds after preparation. Missing, corrupt and over-limit audio do not run
WavLM. Valid silence does. D2 unavailable distributions and representations are None.

## Inference and provenance

PortableTADual verifies external assets before construction, loads T1, B adaptive
seed 1337 and D2 seed 1337, and loads no D0/D3. RefinedSummaryTap executes one shared
WavLM pass, preserves B's last-layer mean operation and supplies all 13 layer
summaries to D2. D2 exports a six-class CREMA6 posterior and normalized 128D vector.

RefinedC1Dual clears summary scratch state following exceptions; each prediction
clears it before inference. close removes clock hooks and clears summaries.

Loaded learned parameters: 218,895,839, including six learned scalars and one
shared WavLM. CUDA scope is singleton FP32 text, singleton BF16 WavLM and FP32
summaries/head. Explicit CPU FP32 has a separate numerical scope.

## Responses and replay

responses.py::respond is deterministic from transcript, B emotion and confidence.
ta_policy.py defaults to evidence_only. The release explicitly marks
role_specific_h1 as rejected research opt-in; its old CLI flag is not an accepted
new production behavior.

ta.py --replay accepts human-only transcript/waveform turns; assistant records are
rejected. Emission acknowledgement occurs after printing structured output.
roadmap_b/examples/ta_replay.json is the supplied fixture. src/replay.py is the
older Roadmap A path, not the current T+A entry point.

Tests already cover context packing/causality, immutable final snapshots, emission
acknowledgement, failure cleanup, missing/corrupt distinction, asset verification
and summary cleanup. Existing replay is completed-turn replay, not event replay.

## Races and extension boundaries

Concurrent model calls could overwrite shared summary scratch state/timing.
Concurrent session calls could interleave histories and snapshot numbering.
The current system assumes sequential ownership and has no shared-predictor
session isolation.

New orchestration belongs outside the unchanged final predictor signature.
Current external speaker keys support only same/other context association.
There is no automatic participant tracking, vision, partial inference or native
streaming WavLM.

## Useful reviewed response lifecycle concepts

Read-only reference:
artifacts/local_llm_extension_v1/review/response_generation/core.py and execution.py.

Useful concepts: session/turn/generation identity checks; immutable request
snapshots; distinct observed dialogue, perception, execution and private draft
stores; delivery-only robot history; serialized cancellation/delivery; rejection
of stale completion; explicit terminal dispositions; bounded collections; and
truthful handling of uncertain delivery.

Do not import generation models, prompt validators, workers, or alternative
safety/template responses. These are outside this migration's accepted behavior.

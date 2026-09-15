# Design space

Read with STATEFUL_RESEARCH.md for external source links. Entries below are project-specific engineering assessments.

| Family | Problem and assumptions | Causality | State / training | Latency | Fit / decision now |
|---|---|---|---|---|---|
| Event-driven state owner | Order observations and validate lifecycle | Strict receipt-order reduction | Bounded application structures; no training | Small per-event work | Selected, single process |
| Immutable snapshots | Preserve estimates at event boundaries | Safe if input prefix frozen | Copied immutable evidence; no training | Serialization/copy cost | Selected with bounded retention |
| Durable event sourcing | Recover arbitrary history after restart | Ordered replay can be causal | Unbounded log unless explicit lifecycle; no training | Replay cost grows | Only explicit bounded debug replay, no database |
| Sliding temporal window | Bound prior evidence | Causal if past-only | O(window); no training for context | Recomputes selected window | Preserve existing T1 policy exactly |
| RNN hidden state | Learn compressed temporal dependency | Forward RNN causal; bidirectional variant not | Fixed hidden state plus new trained weights | Cheap incremental step | Deferred: changes mathematics, opaque replay dependencies |
| Selective state space | Long-sequence recurrent memory | Forward scan causal | Fixed recurrent state, trained architecture | Efficient step, new encoder required | Deferred, no checkpoint-compatible wrapper |
| Transformer memory | Attend longer context with cached states | Requires masks and permitted context | Bounded cache plus trained compatible model | Cache reduces repeated work | Deferred; frozen RoBERTa/WavLM are not converted by buffering |
| Bayesian / filtering updates | Smooth noisy beliefs via dynamics/likelihood model | Filtering causal; smoothing can use future | Posterior and transition/calibration assumptions | Cheap small-state update | Deferred; dependent prefixes risk double-counting |
| Prefix recomputation | Estimate from partial input | Causal prefix input, no internal cache guarantee | Reuse frozen weights; optional evaluation | Multiple complete encoder calls | H2 deferred; cost and correctness gates required |
| Chunk-aware speech encoder | Bounded-context streaming audio | Explicit right context defines effective latency | Different pretrained/trained encoder | Cached blocks and lookahead | Future architecture research, not v1 replacement |
| Temporal multimodal fusion | Associate asynchronous speech/visual evidence | Only received intervals; many published systems use lookahead | Tracks, timing, availability and learned fusion | Alignment/encoder cost | Interfaces only; no fake visual evidence |
| Optimal stopping | Decide when to act | State restricted to current receipt prefix | Policy state; future RL training | Policy overhead plus encoder cadence | Specification only; final responses remain unchanged |

## Selected boundary

Observation history, model estimates and system execution have independent serialized keys. Model outputs never become observed text or speaker identity. Emission is an external observation, not a consequence of retrieving an authored response. Sessions persist across turns but introduce no cross-session identity storage.

## Risks to settle in the source-dependent specification

Bound audio, text, participants, completed turns, snapshots, execution logs and duplicate recognition; reject overlap explicitly. Whole-utterance preprocessing must remain byte/numerically equivalent across chunk sizes. Idempotency must specify a finite acknowledgement window and stale-watermark behavior. Semantic timestamps and snapshot IDs must be replay-derived; operational latency belongs outside canonical snapshots.

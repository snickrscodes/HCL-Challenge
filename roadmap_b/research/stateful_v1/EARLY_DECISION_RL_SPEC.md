# Early-decision RL specification

Status: future research only, written after state/causality/replay contract stabilization. Stateful v1 trains no policy, adds zero learned parameters, and emits no partial model estimates. No model-based early trajectories were generated in this loop.

## Sequential decision process

One episode belongs to a session, participant and human turn. At each received evidence boundary: WAIT to receive later evidence, or terminal COMMIT(label). Initial formulation disallows REVISE. Subsequent perception can still be recorded without rewriting a committed action. At valid final perception, force commitment; cancellation, session close and inference failure terminate with explicit dispositions and no invented emotion label.

## Causal state

Candidate features: current separately namespaced B/D2 or future partial evidence, modality availability, entropy/margin/confidence, preceding posterior changes and switches, elapsed seconds, observed token/sample counts, causal participant activity and private policy memory. Participant identifier strings should not become memorization features. Exclude eventual duration, final transcript length, future snapshots/labels and affect treated as observations. Offline prefix fraction is an evaluation coordinate, not online state unless its horizon was externally declared.

## Reward and commitment

Prospective candidate reward: R = 1[correct] - 3[wrong] - lambda_d * min(elapsed/H, 1), with 0 <= lambda_d <=0.25 and H a fixed application time budget declared before inference. WAIT may accumulate the same delay cost. Sweep costs on development trajectories before a later independent evaluation. Verify explicitly that immediate uninformed guessing does not outperform always-final; the arithmetic alone cannot guarantee every application distribution. No switching penalty is needed with terminal commitment. Revision-enabled policy requires a separate action/reward contract.

## Offline trajectories

Later authorized development trajectories must carry source/input/model/environment identities, event receipt boundary, elapsed time, evidence namespaces, availability, posterior, entropy, margin, current prediction, final reference, permitted development label and execution disposition. Immutable snapshots provide the evidence boundary; execution records distinguish deliberation from action.

Whole-utterance labels target the eventual annotation, not verified instantaneous affect at every prefix. MELD7 labels are not CREMA6 listener truth. Cancelled trajectories are censored rather than fabricated classification errors. Current observed-only partial checkpoints are insufficient for RL policy training. No official MELD test, consumed confirmation or private recordings become training data by default.

## Baselines and evaluation

Match evidence and compute across always-final, fixed elapsed-time commitment, confidence threshold, entropy threshold and margin threshold. Evaluate accuracy, macro/per-class metrics, mean/p95 decision time, accuracy-latency frontier, wrong early commit rate, calibration, stability, compute multiplier and failure/cancellation coverage. Freeze a future held-out evaluation and uncertainty estimates before training. RL must improve meaningful simple baselines, not merely a weak stopping rule.

## Failure modes

Future-information leakage; unknown-duration normalization; partial-input miscalibration; correlated prefixes counted repeatedly; participant memorization; immediate-guessing reward incentives; confidence mistaken for truth; hidden encoder compute multiplication; replay omitting execution boundaries.

[Stop&Hop](https://arxiv.org/abs/2208.09795) motivates irregular observation timing and learned halting; [FIRMBOUND](https://arxiv.org/html/2501.18059v1) motivates finite-horizon risk-based stopping. These are external research foundations, not evidence of benefit for this robot.

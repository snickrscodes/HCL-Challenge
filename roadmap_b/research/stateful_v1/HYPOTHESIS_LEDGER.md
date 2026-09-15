# Hypothesis ledger

Frozen before implementation and before outcomes. JSON companion contains each exact candidate, control, allowed data, metrics, thresholds, failure criteria, compute and production consequence.

## H0 — SELECTED

- **kind:** mandatory_engineering
- **rank:** 0
- **question:** Does stateful orchestration preserve final T+A and deterministic causality?
- **motivation:** Architecture migration must not change frozen inference
- **candidate:** Frozen event/session adapter
- **control:** Accepted SystemPredictor/SystemSession on same model and same inputs
- **data:** raw15, raw4 replay, saved1109 development inputs/outputs, synthetic fixtures
- **metrics:** Exact B logits/category/probabilities; D2 probabilities/vector; context, response, availability/provenance; replay hashes; state correctness; p95 overhead and retained counts
- **effect threshold:** zero numeric/semantic differences; all tests pass; event/finalization p95<=5ms; bounded stress
- **failure criteria:** Any mismatch, leakage, unbounded retention, mandatory gate failure
- **compute:** Local5080 raw15 plus fallback/replay and fixed30 warmups+200 turns per interface; CPU saved heads and stress; no fits
- **production consequence:** Release core only on PASS

## H1 — DEFERRED

- **kind:** optional_scientific
- **rank:** 1
- **question:** Do frozen T1 text prefixes provide correct useful early advisory evidence?
- **motivation:** Early transcript evidence may inform later stopping research
- **candidate:** 25/50/75/100 percent whitespace prefixes, unchanged previous3 human turns; singleton text-side path; no audio; final utterance labels only as imperfect diagnostic targets
- **control:** 100 percent identical text-side model/context
- **data:** Existing dev_model839 records, adaptive development; freeze IDs before any future run
- **metrics:** macroF1, final agreement, entropy/margin, earliest stable label, wrong p>=0.9 fraction, p95 and total compute
- **effect threshold:** At50 percent: agreement>=0.85; macroF1 within0.05 of final; incorrect high-confidence<=0.05; p95<=25ms and total extra compute<=2x final text cost
- **failure criteria:** Any threshold fails, no reproducible causal context, future data leakage
- **compute:** At most4 singleton text calls per example; zero fits
- **production consequence:** Advisory partial text only on prospective PASS; never change final response
- **decision:** No optional run selected: no natural fragment timing/instantaneous labels; diagnostic agreement alone cannot justify added hot-path cost. Core delivers observation state without this computation.

## H2 — DEFERRED

- **kind:** optional_scientific
- **rank:** 2
- **question:** Does coarse WavLM prefix recomputation yield useful acoustic estimates at acceptable cost?
- **motivation:** Possible future early audio evidence
- **candidate:** 25/50/75/100 percent causal waveform prefixes with frozen D2, exclude prefixes shorter than400 samples
- **control:** Same-device full waveform D2
- **data:** Existing allowed development raw media only; no private tuning or new corpus
- **metrics:** D2 posterior divergence, stability, entropy, final agreement, permitted listener-label metric if present, p50/p95 and compute multiplier
- **effect threshold:** At50 percent: final agreement>=0.85; mean JS divergence<=0.05; total compute<=2x single final; p95 per prefix<=50ms; independent label evidence required for usefulness claim
- **failure criteria:** Missing label evidence, cost/quality gate failure or future leakage
- **compute:** Up to4 full prefix executions, zero fits
- **production consequence:** Advisory prefix API only on PASS; no default response change
- **decision:** Existing frozen D2 not prefix-calibrated; MELD7 labels are not CREMA6 listener truth; repeated encoder cost insufficiently justified.

## H3 — DEFERRED

- **kind:** future_scientific
- **rank:** 3
- **question:** Can stopping policy improve accuracy-delay frontier beyond simple rules?
- **motivation:** Future early-decisionRL
- **candidate:** WAIT/terminalCOMMIT policy on causal trajectories
- **control:** always final, fixed elapsed time, confidence, entropy, margin
- **data:** Future separately authorized causal dev trajectories, evaluation holdout fixed later
- **metrics:** accuracy-delay frontier, wrong early commit, calibration, per-class behavior, compute
- **effect threshold:** Future preregistration: Pareto improvement on meaningful simple baselines with uncertainty bounds; no training now
- **failure criteria:** Future leakage, immediate guessing incentive, no baseline improvement
- **compute:** Deferred training; zero executions this loop
- **production consequence:** None in v1

## H4 — DEFERRED

- **kind:** future_scientific
- **rank:** 4
- **question:** Can anonymous audio/visual association improve participant continuity?
- **motivation:** Externally supplied v1 association has no tracking
- **candidate:** Interval-stamped speaker/face-track association
- **control:** external participant labels
- **data:** No new corpus or recordings this loop
- **metrics:** association errors, latency, overlap, identity switches
- **effect threshold:** Future prospective contract required
- **failure criteria:** retroactive mutation or identity leakage
- **compute:** Not authorized for implementation
- **production consequence:** Interfaces only

## H5 — DEFERRED

- **kind:** future_scientific
- **rank:** 5
- **question:** Would recurrent or Bayesian temporal fusion improve evidence?
- **motivation:** Possible temporal continuity
- **candidate:** separately namespaced filter/learned recurrence
- **control:** independent frozen final estimates
- **data:** Future authorized trajectories only
- **metrics:** calibration, correctness, switching, delay
- **effect threshold:** Future prospective contract required
- **failure criteria:** correlated-evidence double counting or baseline changes
- **compute:** No training this loop
- **production consequence:** None

Optional experiments selected: **0 of at most2**. Deferral is a prospective scope decision, not a failed experimental outcome. H0 and engineering stress/performance are mandatory invariants, not scientific experiments. No thresholds may be selected after results.

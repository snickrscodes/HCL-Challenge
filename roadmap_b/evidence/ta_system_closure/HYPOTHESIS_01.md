# H1: fixed role-specific authored response routing

Declared before any H1 candidate result. One hypothesis; zero fits or calibrations. Requirement: a complete T+A system may retain contextual MELD utility while making a concrete authored offer depend on supported vocal evidence.

Observed bottleneck: frozen B predicts neutral on all 720 already-used RAVDESS working recordings. Frozen D2 carries more matching six-class listener information, but its evidence did not reach the historical response. Its happy and sad working recall are weak (1/36 and 12/68); all classes remain in score/coverage accounting.

Candidate: B remains exactly authoritative for MELD state. Only when B is neutral or confidence <0.50, D2 is available, its genuine six-class top probability >=0.60 and top-minus-second margin >=0.15, and top is angry, fear, or disgust, emit one fixed authored offer below. Farewell and thanks lexical precedence are retained. All other cases use the original B response, including confident contextual disagreement and unsupported D2 categories. Thresholds are fixed design choices, not calibrated trust probabilities. Evidence-only B+D2 is the matched baseline. No universal head or posterior mixing.

Routes (exact wording, no phrase search):
- angry: "Would you like to vent, or focus on what to do next?"
- fear: "Would it help to slow down and talk through your concern?"
- disgust: "Would you like to say more about this, or change the subject?"
These are offers, not assertions of internal emotion. Human usefulness is not measured.

Inputs are text, the frozen B state, and frozen D2's six-class posterior. No recording identity, labels, or performed-emotion metadata enters routing. One fixed test inventory covers eligible routes, boundaries, confidence, disagreement, missing/corrupt/long audio, silence, farewell/thanks, labels, and inert adversarial transcript text. All 720 existing working records are processed; matched six-target metrics use the historical >=0.80 supported-vote-mass rule, expected412 rows, with calm/surprise/none retained as excluded mass, never silently mapped. No confirmation or official test comparison.

Training/selection/calibration: none. Existing frozen temperatures only. One fixed coupled shuffle from acoustic_capability_v1/working/SHUFFLE.json (720source_ids) moves D2 logits/posterior/vector together while B and transcript stay fixed; no-new-evidence control makes D2 unavailable. No cutoff/phrase retries. Actual source/manifest/checkpoint hashes are recorded.

Primary metric: actor-equal signed routed listener agreement. On each eligible row, a new vocal route earns 2*q[route_label]-1, using the matching conditional-six listener target; no new vocal route earns0. Thus incorrectly supported routes can hurt, and abstention is not scored as a successful response. Minimum useful effect: >=0.05 above evidence-only baseline and >=0.05 above the fixed shuffle. Safeguards: >=10% eligible route coverage and >=8 actors with a route; routed mean listener agreement >=0.65; B categorical probabilities/decisions/calibration exactly unchanged on all839 MELD dev_model rows; fixed cases pass; local raw source-equivalence/availability/history/one-pass checks pass.

Response stability: use all67 original listener-consistent repeat proxies; candidate route-ID change rate must not exceed frozen D2's historical top-one change rate12/67. Report every actor's rate, maximum actor rate, route-to-baseline transitions, and all360 performed repeats separately. Retain original full pair inventory/support and the existing three-of-eight selective-nuisance support limitation as INCONCLUSIVE, never a pass. Existing continuous-TV and rejected-model mean/tail safeguards remain visible and unchanged; this discrete routing metric does not repair them.

Falsification: insufficient primary gain, coverage, agreement, excessive stable-repeat route changes, or any hard runtime/provenance/source-preservation failure rejects default routing. Sparse selective-nuisance support limits claims independently. Outcome may be ACCEPT_DEVELOPMENT_CANDIDATE/ROLE_SPECIFIC_TA_RELEASE only if the declared primary and measured safeguards pass; otherwise REJECT_CHANGE or NARROW_RELEASE_CLAIM and retain evidence-only default. An opt-in research policy may remain if it is useful to review and clearly labeled. A second hypothesis is used only for a specific newly established bottleneck whose resolution could change this decision, never to chase thresholds.

Budget: one CPU pass over saved working predictions, one fixed comparison, no learned fits; at most30 minutes of research implementation/evaluation and bounded source validation. Local raw tests/benchmark belong to finalization. No further candidate inference on CREMA external, confirmation, or MELD official test.

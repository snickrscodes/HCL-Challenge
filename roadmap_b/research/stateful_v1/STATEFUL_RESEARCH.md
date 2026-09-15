# Stateful perception research

Research date: 2026-09-14. Bounded primary-source pass, completed before implementation inspection. External findings below are distinguished from project design inferences. No model training, new dataset, test-set inference, or remote compute is involved.

## External findings and sources

1. [Microsoft Event Sourcing pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/event-sourcing) (official documentation, accessed 2026-09-14): immutable ordered events support reconstruction and derived views; persistent event-store adoption introduces schema, concurrency and retention complexity.
2. [WavLM](https://arxiv.org/abs/2110.13900) (2021, revised 2022): masked speech prediction plus denoising learns multipurpose speech representations. This does not establish causal cached inference.
3. [Official WavLM implementation](https://github.com/microsoft/unilm/blob/master/wavlm/WavLM.py) (accessed 2026-09-14): encoder attention has an optional streaming mask defaulting to none, and positional convolution. Buffering chunks outside this model does not provide a streaming cache.
4. [Transformer-XL](https://arxiv.org/abs/1901.02860) (2019): segment-level recurrent attention memory extends context. Reusing states requires matching trained architecture and attention semantics.
5. [Mamba](https://arxiv.org/abs/2312.00752) (2023): selective state-space computation supports sequence modeling with compact recurrent inference state. It is a learned replacement, not an orchestration wrapper.
6. [Emformer](https://arxiv.org/abs/2010.10759) (2020/2021): streaming ASR uses memory and cached left-context keys/values, with explicitly managed block context. Reported ASR latency is not an emotion-recognition guarantee.
7. [Chunk SSL](https://arxiv.org/abs/2509.15579) (2025 preprint): chunk-aware pretraining uses same and preceding chunks; evaluates speech recognition/translation. It illustrates why streaming restrictions belong in training. It does not validate the frozen B/D2 checkpoints on prefixes.
8. [TRACE: Emotion Understanding in Streaming Video with Trajectory-Aware Reliability](https://arxiv.org/html/2608.26786v1) (2026-08-27 preprint): separates evolving prefix beliefs, trajectory reliability and commitment; examines confidence, entropy, stability and switching. Uses Qwen and learned components, plus contextual reinterpretation. Its architectural motivation is relevant; its models, full-context reasoning and reported gains are not adopted.
9. [Overlap-aware low-latency online diarization](https://arxiv.org/abs/2109.06483) (2021): rolling local segmentation, speaker embeddings and incremental clustering support anonymous speaker association with a latency/quality tradeoff. Diarization associations are estimates, not real-world identity.
10. [TalkNet](https://arxiv.org/abs/2107.06592) (2021): temporal audio/visual encoders and cross-attention estimate whether visible faces are speaking. Temporal context requires an explicit future-context audit before online use.
11. [Stop&Hop](https://arxiv.org/abs/2208.09795) (2022): a recurrent representation and learned halting policy address irregular observation times; demonstrates a sequential stopping formulation in other domains, not evidence of benefit for this robot.
12. [Early classification with reinforcement learning](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2018/papers/1570433964.pdf) (2018): formulates decisions on partial time-series observations with an earliness/accuracy tradeoff.

## Project-specific conclusions (inferences, not literature claims)

Select a small, synchronous event/session owner with bounded observed state, separately namespaced inference snapshots and execution dispositions. Deep immutable snapshots provide causal audit boundaries without a new learned recurrence or durable personal memory. The mathematical final predictor remains untouched.

A sliding window is appropriate only for the frozen human-turn context and bounded retention. It is not evidence that affect follows a Markov process. Bayesian filtering could smooth estimates, but correlated prefixes are not independent observations; multiplying their probabilities would overcount evidence. No transition model or calibration is justified here.

Production PCM ingestion may be incremental, while WavLM executes once on complete permitted audio. Prefix recomputation is causal with respect to received input but not native streaming: earlier hidden features can change when the prefix grows. Four prefix lengths require four executions; both information value and cost require validation.

Partial transcript states are immediately useful as observations. Model predictions on transcript prefixes may be diagnostically interesting, but suffix negation and changed tokenization make confidence and final agreement insufficient evidence of correctness. No partial affect computation is required to establish the core architecture.

Future participant association must retain separate track IDs, observed intervals, association confidence and producer provenance. Historical turn ownership must not silently change. Face tracks, active-speaker probabilities and expression estimates belong to separate namespaces; absent vision stays absent. Timestamp both capture intervals and receipt boundaries. A future causal consumer may use only evidence received by its boundary, including declared lookahead latency.

Future stopping policies should consume causal evidence and elapsed time, choose WAIT or COMMIT(label), and be tested against fixed-time and uncertainty-threshold baselines. Final duration/token count may be used as offline analysis coordinates, never as online state before it is known. Belief updates do not authorize external commitment or inferred-state feedback into T1.

## Scope decision

Zero new learned parameters. H0 equivalence is mandatory. Default scientific selection is no optional experiment: preserve partial observation hooks, defer learned partial text and acoustic prefixes until useful data/control and benefit gates justify a separate run. RunPod is unnecessary for the core. Research is complete; subsequent source inspection determines exact adapters, retention limits and validation inventory.

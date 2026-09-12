# Roadmap A report

Status: **development only; test not evaluated**.

## Environment

```json
{
  "git_commit": null,
  "git_dirty": null,
  "python": "3.12.14",
  "pytorch": "2.8.0+cu128",
  "transformers": "4.56.2",
  "cuda_runtime": "12.8",
  "gpu": "NVIDIA GeForce RTX 4090",
  "gpu_memory_bytes": 25252724736,
  "cpu": "AMD EPYC 75F3 32-Core Processor",
  "cpu_count": 128,
  "ram_bytes": 540844646400,
  "process_rss_bytes": 2323185664,
  "seed": 1337,
  "peak_gpu_memory_bytes": 1373408768
}
```

## Data audit

Rows: {'dev': 1109, 'test': 2610, 'train': 9989}. Valid media: 13706; invalid: 2.
Outliers: 104. Duration quantiles (seconds): {'min': 0.064, 'p50': 2.4746875, 'p90': 6.3146875, 'p95': 7.8933125, 'p99': 11.771734375000062, 'max': 304.96}.
FFmpeg float WAV, mono 16000 Hz, no normalization/trim/denoising; failures retained

Failures and anomalies are retained in data audit JSONL. Audio-only scores use valid audio only; concat and fusion fall back to calibrated text for unavailable audio.
Dev dialogues: 86 model / 28 calibration. Explicit IDs are saved in audit/dev_dialogues.json.

## Models

Text: FacebookAI/roberta-base @ e2da8e2f811d1448a5b465c236feacd80ffbac7b.
Audio: microsoft/wavlm-base-plus @ 4c66d4806a428f2e922ccfa1a962776e232d487b.
T0: current only. T1: at most three preceding same-dialogue turns, relative speaker markers, current-only mean pooling, maximum 128 tokens. Old context removed first; target truncations logged.
Frozen WavLM: mono 16 kHz, final-layer masked mean. Unequal-length waveforms are encoded separately to prevent group-normalization padding dependence. Audio head 768→256→7; concat head 1536→256→7 (tiny smoke fixtures use smaller encoders).

```json
{
  "components": {
    "selected_text": {
      "total": 124062727,
      "trainable": 124062727
    },
    "frozen_wavlm": {
      "total": 94381936,
      "trainable": 0
    },
    "audio_head": {
      "total": 198663,
      "trainable": 198663
    }
  },
  "learned_scalars": {
    "alpha": 1,
    "temperatures": 3
  },
  "total_required_parameters": 218643330,
  "within_6b": true,
  "note": "Includes frozen parameters and all diagnostic temperatures; unused alternate text/concat are separate experiments."
}
```

- text_current: parameters {'total': 124062727, 'trainable': 124062727}; elapsed 108.86 s; peak GPU allocated 2867105280 bytes.
- text_context: parameters {'total': 124062727, 'trainable': 124062727}; elapsed 113.68 s; peak GPU allocated 3105074688 bytes.
- audio: parameters {'total': 198663, 'trainable': 198663}; elapsed 53.80 s; peak GPU allocated 35917312 bytes.
- concat: parameters {'total': 395271, 'trainable': 395271}; elapsed 31.19 s; peak GPU allocated 17039360 bytes.
- wavlm_extraction: parameters {'total': 94381936, 'trainable': 0}; elapsed 145.52 s; peak GPU allocated 1702252032 bytes.

## Results: dev_model

| Model | Accuracy | Weighted F1 | Macro F1 | n |
|---|---:|---:|---:|---:|
| majority | 0.4207 | 0.2492 | 0.0846 | 839 |
| text_current | 0.5912 | 0.5739 | 0.4421 | 839 |
| text_context | 0.6138 | 0.6045 | 0.4863 | 839 |
| audio | 0.4785 | 0.4214 | 0.2611 | 838 |
| concat | 0.6162 | 0.6058 | 0.4873 | 839 |
| late_fusion | 0.6377 | 0.6178 | 0.4894 | 839 |

### majority: per-class results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| neutral | 0.4207 | 1.0000 | 0.5923 | 353 |
| joy | 0.0000 | 0.0000 | 0.0000 | 121 |
| sadness | 0.0000 | 0.0000 | 0.0000 | 83 |
| anger | 0.0000 | 0.0000 | 0.0000 | 121 |
| surprise | 0.0000 | 0.0000 | 0.0000 | 114 |
| fear | 0.0000 | 0.0000 | 0.0000 | 30 |
| disgust | 0.0000 | 0.0000 | 0.0000 | 17 |

Confusion matrix: rows=true, columns=predicted; canonical label order.

```json
[[353, 0, 0, 0, 0, 0, 0], [121, 0, 0, 0, 0, 0, 0], [83, 0, 0, 0, 0, 0, 0], [121, 0, 0, 0, 0, 0, 0], [114, 0, 0, 0, 0, 0, 0], [30, 0, 0, 0, 0, 0, 0], [17, 0, 0, 0, 0, 0, 0]]
```


### text_current: per-class results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| neutral | 0.7068 | 0.7989 | 0.7500 | 353 |
| joy | 0.4820 | 0.5537 | 0.5154 | 121 |
| sadness | 0.4364 | 0.2892 | 0.3478 | 83 |
| anger | 0.4933 | 0.3058 | 0.3776 | 121 |
| surprise | 0.5245 | 0.6579 | 0.5837 | 114 |
| fear | 0.4091 | 0.3000 | 0.3462 | 30 |
| disgust | 0.3333 | 0.1176 | 0.1739 | 17 |

Confusion matrix: rows=true, columns=predicted; canonical label order.

```json
[[282, 22, 13, 7, 25, 4, 0], [32, 67, 1, 3, 17, 1, 0], [29, 11, 24, 9, 4, 4, 2], [37, 21, 7, 37, 15, 3, 1], [10, 14, 4, 9, 75, 1, 1], [3, 3, 4, 7, 4, 9, 0], [6, 1, 2, 3, 3, 0, 2]]
```


### text_context: per-class results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| neutral | 0.7323 | 0.7904 | 0.7602 | 353 |
| joy | 0.5913 | 0.5620 | 0.5763 | 121 |
| sadness | 0.4464 | 0.3012 | 0.3597 | 83 |
| anger | 0.5213 | 0.4050 | 0.4558 | 121 |
| surprise | 0.5338 | 0.6930 | 0.6031 | 114 |
| fear | 0.3750 | 0.3000 | 0.3333 | 30 |
| disgust | 0.2857 | 0.3529 | 0.3158 | 17 |

Confusion matrix: rows=true, columns=predicted; canonical label order.

```json
[[279, 18, 13, 9, 27, 5, 2], [30, 68, 1, 7, 15, 0, 0], [29, 8, 25, 11, 2, 4, 4], [27, 11, 8, 49, 16, 2, 8], [8, 8, 3, 12, 79, 3, 1], [4, 2, 4, 5, 6, 9, 0], [4, 0, 2, 1, 3, 1, 6]]
```


### audio: per-class results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| neutral | 0.5077 | 0.8466 | 0.6347 | 352 |
| joy | 0.2632 | 0.1653 | 0.2030 | 121 |
| sadness | 0.4091 | 0.1084 | 0.1714 | 83 |
| anger | 0.5125 | 0.3388 | 0.4080 | 121 |
| surprise | 0.4571 | 0.2807 | 0.3478 | 114 |
| fear | 0.5000 | 0.0333 | 0.0625 | 30 |
| disgust | 0.0000 | 0.0000 | 0.0000 | 17 |

Confusion matrix: rows=true, columns=predicted; canonical label order.

```json
[[298, 20, 6, 11, 17, 0, 0], [84, 20, 3, 7, 7, 0, 0], [58, 6, 9, 6, 4, 0, 0], [56, 15, 2, 41, 7, 0, 0], [59, 13, 2, 6, 32, 1, 1], [21, 2, 0, 6, 0, 1, 0], [11, 0, 0, 3, 3, 0, 0]]
```


### concat: per-class results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| neutral | 0.7308 | 0.8074 | 0.7672 | 353 |
| joy | 0.5583 | 0.5537 | 0.5560 | 121 |
| sadness | 0.4211 | 0.1928 | 0.2645 | 83 |
| anger | 0.4887 | 0.5372 | 0.5118 | 121 |
| surprise | 0.6600 | 0.5789 | 0.6168 | 114 |
| fear | 0.3125 | 0.3333 | 0.3226 | 30 |
| disgust | 0.3077 | 0.4706 | 0.3721 | 17 |

Confusion matrix: rows=true, columns=predicted; canonical label order.

```json
[[285, 21, 9, 14, 12, 9, 3], [31, 67, 1, 13, 9, 0, 0], [33, 8, 16, 14, 1, 6, 5], [23, 9, 5, 65, 10, 2, 7], [10, 12, 3, 16, 66, 4, 3], [4, 3, 3, 9, 1, 10, 0], [4, 0, 1, 2, 1, 1, 8]]
```


### late_fusion: per-class results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| neutral | 0.7005 | 0.8612 | 0.7726 | 353 |
| joy | 0.5905 | 0.5124 | 0.5487 | 121 |
| sadness | 0.5435 | 0.3012 | 0.3876 | 83 |
| anger | 0.5490 | 0.4628 | 0.5022 | 121 |
| surprise | 0.5865 | 0.6842 | 0.6316 | 114 |
| fear | 0.5833 | 0.2333 | 0.3333 | 30 |
| disgust | 0.4286 | 0.1765 | 0.2500 | 17 |

Confusion matrix: rows=true, columns=predicted; canonical label order.

```json
[[304, 14, 8, 8, 17, 2, 0], [40, 62, 1, 7, 11, 0, 0], [33, 10, 25, 11, 1, 2, 1], [30, 9, 6, 56, 17, 0, 3], [15, 7, 2, 11, 78, 1, 0], [6, 3, 3, 5, 6, 7, 0], [6, 0, 1, 4, 3, 0, 3]]
```


## Results: dev_calib

| Model | Accuracy | Weighted F1 | Macro F1 | n |
|---|---:|---:|---:|---:|
| majority | 0.4333 | 0.2620 | 0.0864 | 270 |
| text_current | 0.5963 | 0.5861 | 0.4743 | 270 |
| text_context | 0.5963 | 0.5810 | 0.4395 | 270 |
| audio | 0.5037 | 0.4398 | 0.2640 | 270 |
| concat | 0.5519 | 0.5445 | 0.4057 | 270 |
| late_fusion | 0.6185 | 0.5979 | 0.4647 | 270 |

### majority: per-class results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| neutral | 0.4333 | 1.0000 | 0.6047 | 117 |
| joy | 0.0000 | 0.0000 | 0.0000 | 42 |
| sadness | 0.0000 | 0.0000 | 0.0000 | 28 |
| anger | 0.0000 | 0.0000 | 0.0000 | 32 |
| surprise | 0.0000 | 0.0000 | 0.0000 | 36 |
| fear | 0.0000 | 0.0000 | 0.0000 | 10 |
| disgust | 0.0000 | 0.0000 | 0.0000 | 5 |

Confusion matrix: rows=true, columns=predicted; canonical label order.

```json
[[117, 0, 0, 0, 0, 0, 0], [42, 0, 0, 0, 0, 0, 0], [28, 0, 0, 0, 0, 0, 0], [32, 0, 0, 0, 0, 0, 0], [36, 0, 0, 0, 0, 0, 0], [10, 0, 0, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0, 0]]
```


### text_current: per-class results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| neutral | 0.6885 | 0.7179 | 0.7029 | 117 |
| joy | 0.4655 | 0.6429 | 0.5400 | 42 |
| sadness | 0.6250 | 0.5357 | 0.5769 | 28 |
| anger | 0.5625 | 0.2812 | 0.3750 | 32 |
| surprise | 0.5476 | 0.6389 | 0.5897 | 36 |
| fear | 0.3333 | 0.2000 | 0.2500 | 10 |
| disgust | 0.5000 | 0.2000 | 0.2857 | 5 |

Confusion matrix: rows=true, columns=predicted; canonical label order.

```json
[[84, 16, 1, 3, 10, 3, 0], [11, 27, 1, 1, 2, 0, 0], [9, 1, 15, 0, 3, 0, 0], [10, 7, 4, 9, 2, 0, 0], [5, 5, 1, 0, 23, 1, 1], [2, 1, 1, 2, 2, 2, 0], [1, 1, 1, 1, 0, 0, 1]]
```


### text_context: per-class results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| neutral | 0.7193 | 0.7009 | 0.7100 | 117 |
| joy | 0.4677 | 0.6905 | 0.5577 | 42 |
| sadness | 0.5714 | 0.5714 | 0.5714 | 28 |
| anger | 0.4706 | 0.2500 | 0.3265 | 32 |
| surprise | 0.5682 | 0.6944 | 0.6250 | 36 |
| fear | 0.0000 | 0.0000 | 0.0000 | 10 |
| disgust | 0.5000 | 0.2000 | 0.2857 | 5 |

Confusion matrix: rows=true, columns=predicted; canonical label order.

```json
[[82, 19, 2, 3, 10, 1, 0], [9, 29, 0, 2, 1, 1, 0], [8, 1, 16, 0, 3, 0, 0], [9, 7, 5, 8, 2, 1, 0], [3, 5, 1, 1, 25, 0, 1], [2, 1, 2, 2, 3, 0, 0], [1, 0, 2, 1, 0, 0, 1]]
```


### audio: per-class results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| neutral | 0.5474 | 0.8889 | 0.6775 | 117 |
| joy | 0.3000 | 0.1429 | 0.1935 | 42 |
| sadness | 0.5000 | 0.1786 | 0.2632 | 28 |
| anger | 0.4643 | 0.4062 | 0.4333 | 32 |
| surprise | 0.3810 | 0.2222 | 0.2807 | 36 |
| fear | 0.0000 | 0.0000 | 0.0000 | 10 |
| disgust | 0.0000 | 0.0000 | 0.0000 | 5 |

Confusion matrix: rows=true, columns=predicted; canonical label order.

```json
[[104, 1, 2, 5, 5, 0, 0], [30, 6, 2, 3, 1, 0, 0], [13, 5, 5, 1, 3, 1, 0], [13, 5, 0, 13, 1, 0, 0], [20, 3, 1, 4, 8, 0, 0], [6, 0, 0, 2, 2, 0, 0], [4, 0, 0, 0, 1, 0, 0]]
```


### concat: per-class results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| neutral | 0.6852 | 0.6325 | 0.6578 | 117 |
| joy | 0.4242 | 0.6667 | 0.5185 | 42 |
| sadness | 0.5000 | 0.3571 | 0.4167 | 28 |
| anger | 0.3548 | 0.3438 | 0.3492 | 32 |
| surprise | 0.6579 | 0.6944 | 0.6757 | 36 |
| fear | 0.0000 | 0.0000 | 0.0000 | 10 |
| disgust | 0.2500 | 0.2000 | 0.2222 | 5 |

Confusion matrix: rows=true, columns=predicted; canonical label order.

```json
[[74, 22, 5, 8, 7, 1, 0], [9, 28, 0, 3, 1, 1, 0], [11, 2, 10, 0, 3, 0, 2], [9, 7, 3, 11, 1, 1, 0], [2, 4, 0, 4, 25, 0, 1], [2, 3, 1, 3, 1, 0, 0], [1, 0, 1, 2, 0, 0, 1]]
```


### late_fusion: per-class results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| neutral | 0.6718 | 0.7521 | 0.7097 | 117 |
| joy | 0.5192 | 0.6429 | 0.5745 | 42 |
| sadness | 0.6250 | 0.5357 | 0.5769 | 28 |
| anger | 0.5882 | 0.3125 | 0.4082 | 32 |
| surprise | 0.5909 | 0.7222 | 0.6500 | 36 |
| fear | 0.0000 | 0.0000 | 0.0000 | 10 |
| disgust | 1.0000 | 0.2000 | 0.3333 | 5 |

Confusion matrix: rows=true, columns=predicted; canonical label order.

```json
[[88, 15, 3, 2, 8, 1, 0], [13, 27, 0, 1, 1, 0, 0], [10, 0, 15, 0, 3, 0, 0], [11, 5, 4, 10, 2, 0, 0], [5, 4, 0, 1, 26, 0, 0], [3, 1, 1, 1, 4, 0, 0], [1, 0, 1, 2, 0, 0, 1]]
```

## Fusion

Matched text baseline: text_context. Raw-logit alpha: 0.40.
Temperatures: text 1.6053, audio 1.0172, fusion 1.0856.
Alpha uses dev_model; all temperatures use dev_calib after alpha is frozen.

- dev_model: macro-F1 difference +0.0031; changed 98, corrected 46, harmed 26. This does not prove prosodic causality.
- dev_calib: macro-F1 difference +0.0252; changed 26, corrected 9, harmed 3. This does not prove prosodic causality.

## Runtime

Utterance-final causal inference, timed from explicit end_turn; PCM buffered incrementally.

Observed warm latency (milliseconds):

```json
{
  "text": {
    "p50": 7.472384997527115,
    "p95": 11.147275535040574
  },
  "audio": {
    "p50": 19.56499449443072,
    "p95": 78.90181271213807
  },
  "fusion_response": {
    "p50": 0.12190600682515651,
    "p95": 0.16305550234392288
  },
  "total": {
    "p50": 28.606930500245653,
    "p95": 86.53718565328747
  }
}
```

Load time: 1.699 s. Median real-time factor: 0.0179.
Extraction throughput:

```json
{
  "train": {
    "elapsed_s": 119.46134382797754,
    "valid_utterances": 9988,
    "utterances_per_s": 83.608635898007,
    "audio_seconds_per_s": 262.70869299102975
  },
  "dev_model": {
    "elapsed_s": 17.146714263013564,
    "valid_utterances": 838,
    "utterances_per_s": 48.87233712219802,
    "audio_seconds_per_s": 157.34698255395227
  },
  "dev_calib": {
    "elapsed_s": 7.378654853004264,
    "valid_utterances": 270,
    "utterances_per_s": 36.592035456173676,
    "audio_seconds_per_s": 103.2572352520037
  }
}
```

There is no predeclared latency SLA, VAD delay, generator TTFT, or streaming acoustic claim. The authored response is emitted with the final state. Samples and durations are in runtime.json.

## Failures and limitations

- Strong class imbalance; macro F1 and all minority-class scores remain necessary.
- Friends speakers and episodes overlap across official splits; this is not a speaker-independent benchmark.
- TV music, laughter, overlapping voices, and clip boundaries can provide non-emotional shortcuts.
- Clip/alignment failures remain explicit; no failed or silent clip is relabeled neutral.
- WavLM can exploit speaker, lexical content, or background. Improvements are complementarity evidence, not causal proof.
- Ground-truth transcripts confer an advantage over a live speech-recognition system.
- Buffering is incremental; acoustic computation is utterance-final.
- Responses are authored examples, without MELD response supervision or a completed human-quality study.
- Single seed and small calibration partition limit statistical certainty; dev_model is a selection set.

## Decision

Roadmap A development run is complete; final fallback validation awaits frozen test evaluation. Audio complementarity warrants considering Roadmap B after validation.

The continuation rule uses development evidence only: alpha > 0, improved macro F1, and more corrected than harmed examples. Test scores never trigger retuning. Parameter/runtime and data checks are necessary but do not establish deployment quality.

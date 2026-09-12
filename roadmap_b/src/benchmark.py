"""Measure warm raw-turn inference separately from model loading and buffering."""

import time
from pathlib import Path

import psutil
import numpy as np
import torch

from .audio import read_waveform
from .context import histories
from .inference import RoadmapA
from .utils import audio_file, environment, key, split_rows, write_json


def benchmark(cfg):
    load_start = time.perf_counter()
    predictor = RoadmapA(cfg)
    load_s = time.perf_counter() - load_start
    rows = split_rows(cfg, "dev_model")
    history = histories(rows, cfg["text"]["history_turns"])
    valid = sorted([r for r in rows if r["audio_valid"]], key=lambda r: r["duration_s"])
    count = min(len(valid), cfg["runtime"]["benchmark_turns"])
    indices = np.unique(np.linspace(0, len(valid) - 1, count).astype(int))
    first = valid[indices[0]]
    predictor.predict_turn(
        first["text"],
        read_waveform(audio_file(cfg, first)),
        16000,
        history[key(first)],
        speaker=first["speaker"],
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    measurements = []
    rss_samples = [psutil.Process().memory_info().rss]
    for index in indices:
        row = valid[index]
        # WAV disk read is outside inference timing; waveform preparation is inside.
        prediction = predictor.predict_turn(
            row["text"], read_waveform(audio_file(cfg, row)), 16000, history[key(row)], speaker=row["speaker"]
        )
        rss_samples.append(psutil.Process().memory_info().rss)
        measurements.append(
            {
                "key": key(row),
                "duration_s": row["duration_s"],
                "latency_ms": prediction.latency_ms,
                "real_time_factor": prediction.latency_ms["total"] / 1000 / row["duration_s"],
                "prediction": prediction.to_dict(),
            }
        )
    result = {
        "model_load_s": load_s,
        "warmup_turns": 1,
        "samples": measurements,
        "definition": "Utterance-final causal inference, timed from explicit end_turn; PCM buffered incrementally.",
        "excluded": ["recording time", "WAV disk read", "automatic endpoint detection", "ASR"],
        "environment": environment(cfg),
        "parameter_ledger": predictor.parameter_ledger(),
        "summary_ms": {
            name: {
                "p50": float(np.median([r["latency_ms"][name] for r in measurements])),
                "p95": float(np.quantile([r["latency_ms"][name] for r in measurements], 0.95)),
            }
            for name in ("text", "audio", "fusion_response", "total")
        },
        "median_real_time_factor": float(np.median([r["real_time_factor"] for r in measurements])),
        "caution": "Small duration-stratified sample; quantiles are initial observations, not an SLA.",
    }
    if not result["parameter_ledger"]["within_6b"]:
        raise ValueError("Runtime exceeds six billion total parameters")
    result["peak_process_rss_bytes"] = max(rss_samples)
    write_json(Path(cfg["output_root"]) / "runtime.json", result)
    write_json(Path(cfg["output_root"]) / "parameter_ledger.json", result["parameter_ledger"])
    return result


def synthetic_benchmark(cfg):
    """Compute-only diagnostic, explicitly separate from MELD evaluation."""
    predictor = RoadmapA(cfg)
    rng = np.random.default_rng(cfg["seed"])
    samples = []
    rss = []
    for duration in cfg["runtime"]["synthetic_durations_s"]:
        wave = rng.normal(0, 0.05, round(duration * 16000)).astype(np.float32)
        predictor.predict_turn("Yeah, sure.", wave, 16000, [])  # warm each input shape
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for repeat in range(cfg["runtime"]["synthetic_repeats"]):
            prediction = predictor.predict_turn("Yeah, sure.", wave, 16000, [])
            samples.append(
                {
                    "duration_s": duration,
                    "repeat": repeat,
                    "latency_ms": prediction.latency_ms,
                    "real_time_factor": prediction.latency_ms["total"] / 1000 / duration,
                    "gpu_peak_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                }
            )
            rss.append(psutil.Process().memory_info().rss)
    result = {
        "scope": "Synthetic noise: compute/length diagnostic only, not emotion quality or MELD performance",
        "environment": environment(cfg),
        "samples": samples,
        "peak_sampled_rss_bytes": max(rss),
        "summary": {
            str(duration): {
                "p50_ms": float(
                    np.median([r["latency_ms"]["total"] for r in samples if r["duration_s"] == duration])
                ),
                "p95_ms": float(
                    np.quantile(
                        [r["latency_ms"]["total"] for r in samples if r["duration_s"] == duration], 0.95
                    )
                ),
            }
            for duration in cfg["runtime"]["synthetic_durations_s"]
        },
    }
    write_json(Path(cfg["output_root"]) / "synthetic_runtime.json", result)
    return result


def main():
    from .utils import common_parser, load_config

    parser = common_parser(__doc__)
    parser.add_argument(
        "--synthetic", action="store_true", help="Measure configured noise lengths, without MELD metrics"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    synthetic_benchmark(cfg) if args.synthetic else benchmark(cfg)


if __name__ == "__main__":
    main()

"""Freeze alpha first, then fit scalar temperatures on dev_calib only."""

from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from .evaluate import probabilities, metrics
from .train_fusion import cache_logits
from .utils import (
    key,
    common_parser,
    finish_run,
    load_config,
    read_json,
    split_rows,
    start_run,
    write_json,
    write_rows,
)


def late_logits(text_logits, audio_logits, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    return (1 - alpha) * np.asarray(text_logits) + alpha * np.asarray(audio_logits)


def fit_temperature(logits, labels):
    logits, labels = np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=int)
    if len(labels) == 0:
        raise ValueError("Cannot calibrate empty subset")

    def loss(log_temperature):
        probs = probabilities(logits, float(np.exp(log_temperature)))
        return -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-15, 1)).mean()

    fit = minimize_scalar(loss, bounds=(-5, 5), method="bounded", options={"xatol": 1e-7})
    if not fit.success:
        raise RuntimeError("Temperature optimizer failed")
    value = float(np.exp(fit.x))
    return value if loss(fit.x) < loss(0) else 1.0


def calibrate(cfg):
    path, started = start_run(cfg, "calibration")
    selection = read_json(Path(cfg["output_root"]) / "selection.json")
    if selection["alpha"] is None:
        raise ValueError("Freeze alpha before calibration")
    zt, za, valid = cache_logits(cfg, "dev_calib")
    y = np.array([r["label"] for r in split_rows(cfg, "dev_calib")])
    zf = late_logits(zt, za, selection["alpha"])
    temperatures = {
        "text": fit_temperature(zt, y),
        "audio": fit_temperature(za[valid], y[valid]),
        "fusion": fit_temperature(zf[valid], y[valid]),
        "fitted_on": "dev_calib",
        "alpha_frozen": selection["alpha"],
        "audio_and_fusion_fit_subset": "valid audio",
        "log_temperature_bounds": [-5, 5],
    }
    write_json(Path(cfg["output_root"]) / "calibration.json", temperatures)
    diagnostics = {}
    for name, z, targets in (
        ("text", zt, y),
        ("audio", za[valid], y[valid]),
        ("fusion", zf[valid], y[valid]),
    ):
        diagnostics[name] = {
            "before": metrics(targets, probabilities(z)),
            "after": metrics(targets, probabilities(z, temperatures[name])),
            "temperature": temperatures[name],
        }
    write_json(path / "metrics.json", diagnostics)
    write_rows(path / "train_log.jsonl", [])
    pt, pa = probabilities(zt, temperatures["text"]), probabilities(za, temperatures["audio"])
    pf = probabilities(zf, temperatures["fusion"])
    pf[~valid] = pt[~valid]
    write_rows(
        path / "predictions.jsonl",
        [
            {
                "key": key(row),
                "label": row["label"],
                "text_probabilities": pt[i].tolist(),
                "audio_probabilities": pa[i].tolist() if valid[i] else None,
                "fusion_probabilities": pf[i].tolist(),
            }
            for i, row in enumerate(split_rows(cfg, "dev_calib"))
        ],
    )
    write_json(path / "checkpoint" / "temperatures.json", temperatures)
    finish_run(cfg, path, started)
    return temperatures


def main():
    args = common_parser(__doc__).parse_args()
    calibrate(load_config(args.config))


if __name__ == "__main__":
    main()

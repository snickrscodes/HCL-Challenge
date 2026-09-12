"""Frozen-feature concatenation and deterministic raw-logit late-fusion selection."""

from pathlib import Path

import numpy as np
import torch

from .evaluate import metrics, probabilities, save_evaluation
from .models import load_head
from .train_audio import aligned_cache, fit_head
from .train_text import extract_text
from .utils import (
    common_parser,
    finish_run,
    load_config,
    load_tensor,
    read_json,
    split_rows,
    start_run,
    write_json,
    write_rows,
)


def select_text(cfg):
    root = Path(cfg["output_root"])
    # Exact macro-F1 ties prefer the simpler current-only model.
    current = read_json(root / "text_current" / "metrics.json")["macro_f1"]
    context = read_json(root / "text_context" / "metrics.json")["macro_f1"]
    return "text_context" if context > current else "text_current"


def cache_logits(cfg, split):
    selection = read_json(Path(cfg["output_root"]) / "selection.json")
    rows = split_rows(cfg, split)
    text = aligned_cache(load_tensor(Path(cfg["feature_root"]) / selection["text"] / f"{split}.pt"), rows)
    audio = aligned_cache(load_tensor(Path(cfg["feature_root"]) / "wavlm_base_plus" / f"{split}.pt"), rows)
    head = load_head(Path(cfg["checkpoint_root"]) / "audio").eval()
    with torch.inference_mode():
        za = head(audio["features"]).numpy()
    return text["logits"].numpy(), za, audio["audio_valid"].numpy()


def select_alpha(zt, za, labels, valid, step=0.01):
    from .calibrate import late_logits

    if step <= 0 or step > 1 or not np.isclose(round(1 / step) * step, 1):
        raise ValueError("alpha_step must divide one")
    scores, best = [], None
    for alpha in np.linspace(0, 1, round(1 / step) + 1):
        logits = late_logits(zt, za, float(alpha))
        logits[~valid] = zt[~valid]
        values = metrics(labels, probabilities(logits))
        scores.append(
            {"alpha": float(alpha), "macro_f1": values["macro_f1"], "weighted_f1": values["weighted_f1"]}
        )
        # Iteration order breaks exact ties in favor of less audio.
        if best is None or values["macro_f1"] > best["macro_f1"]:
            best = scores[-1]
    return best["alpha"], scores


def train(cfg):
    from .calibrate import late_logits

    path, started = start_run(cfg, "late_fusion")
    selected = select_text(cfg)
    write_json(Path(cfg["output_root"]) / "selection.json", {"text": selected, "alpha": None})
    data, rows = {}, {}
    for split in ("train", "dev_model", "dev_calib"):
        rows[split] = split_rows(cfg, split)
        text = extract_text(cfg, selected, split)
        audio = aligned_cache(
            load_tensor(Path(cfg["feature_root"]) / "wavlm_base_plus" / f"{split}.pt"), rows[split]
        )
        valid = audio["audio_valid"]
        data[split] = (
            torch.cat([text["features"], audio["features"]], 1)[valid],
            [row for row, keep in zip(rows[split], valid) if keep],
        )
    fit_head(
        cfg,
        "concat",
        data["train"][0],
        data["dev_model"][0],
        data["train"][1],
        data["dev_model"][1],
        cfg["fusion"]["concat_hidden"],
    )
    zt, za, valid = cache_logits(cfg, "dev_model")
    alpha, grid = select_alpha(
        zt, za, [r["label"] for r in rows["dev_model"]], valid, cfg["fusion"]["alpha_step"]
    )
    write_json(
        Path(cfg["output_root"]) / "selection.json",
        {
            "text": selected,
            "alpha": alpha,
            "selected_on": "dev_model",
            "metric": "macro_f1",
            "tie_break": "current text; smallest alpha",
            "logit_policy": "raw, independently trained",
        },
    )
    write_json(path / "checkpoint" / "selection.json", read_json(Path(cfg["output_root"]) / "selection.json"))
    logits = late_logits(zt, za, alpha)
    logits[~valid] = zt[~valid]
    save_evaluation(path, rows["dev_model"], probabilities(logits))
    write_rows(path / "train_log.jsonl", grid)
    finish_run(cfg, path, started)
    return alpha


def main():
    args = common_parser(__doc__).parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()

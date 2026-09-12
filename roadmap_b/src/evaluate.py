"""One fixed-label metric implementation, including saved-prediction recomputation."""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .constants import LABELS
from .utils import common_parser, key, load_config, read_json, read_rows, split_rows, write_json, write_rows


def probabilities(logits, temperature=1.0):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    logits = torch.as_tensor(logits, dtype=torch.float64)
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite logits")
    return torch.softmax(logits / temperature, dim=-1).numpy()


def metrics(labels, probs):
    labels, probs = np.asarray(labels, dtype=int), np.asarray(probs, dtype=float)
    if len(labels) == 0:
        raise ValueError("Cannot evaluate an empty subset")
    if probs.shape != (len(labels), len(LABELS)) or not np.isfinite(probs).all():
        raise ValueError("Probability shape/finite check failed")
    if (probs < 0).any() or not np.allclose(probs.sum(1), 1, atol=1e-6):
        raise ValueError("Invalid probability distribution")
    pred = probs.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, pred, labels=np.arange(len(LABELS)), zero_division=0
    )
    confidence = probs.max(1)
    correct = pred == labels
    bins, ece = [], 0.0
    for lower, upper in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        count = int(mask.sum())
        gap = abs(float(correct[mask].mean() - confidence[mask].mean())) if count else 0.0
        ece += count / len(labels) * gap
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "accuracy": float(correct[mask].mean()) if count else None,
                "confidence": float(confidence[mask].mean()) if count else None,
            }
        )
    return {
        "n": len(labels),
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(labels, pred, labels=np.arange(len(LABELS))).tolist(),
        "nll": float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-15, 1)).mean()),
        "ece_15": float(ece),
        "reliability_bins": bins,
        "labels": list(LABELS),
    }


def save_evaluation(path, rows, probs, **extra):
    path = Path(path)
    values = metrics([r["label"] for r in rows], probs)
    values.update(extra)
    predictions = [
        {
            "key": key(row),
            "label": row["label"],
            "emotion": LABELS[int(p.argmax())],
            "prediction": int(p.argmax()),
            "probabilities": p.tolist(),
            "audio_available": row["audio_valid"],
        }
        for row, p in zip(rows, probs)
    ]
    write_rows(path / "predictions.jsonl", predictions)
    write_json(path / "metrics.json", values)
    recomputed = recompute(path / "predictions.jsonl")
    for name in ("accuracy", "macro_f1", "weighted_f1"):
        if values[name] != recomputed[name]:
            raise AssertionError("Saved prediction metric mismatch")
    return values


def recompute(path):
    rows = read_rows(path)
    return metrics([r["label"] for r in rows], np.array([r["probabilities"] for r in rows]))


def majority(cfg, splits=("dev_model", "dev_calib")):
    train = split_rows(cfg, "train")
    label = int(np.bincount([r["label"] for r in train], minlength=len(LABELS)).argmax())
    result = {}
    for split in splits:
        rows = split_rows(cfg, split)
        probs = np.eye(len(LABELS))[np.repeat(label, len(rows))]
        result[split] = save_evaluation(
            Path(cfg["output_root"]) / "evaluation" / split / "majority",
            rows,
            probs,
            note="Training-set majority; NLL of a hard reference is not meaningful.",
        )
    return result


def evaluate_split(cfg, split):
    from .train_fusion import cache_logits, aligned_cache
    from .calibrate import late_logits
    from .models import load_head
    from .utils import load_tensor

    artifact = Path(cfg["output_root"])
    selection = read_json(artifact / "selection.json")
    calibration = read_json(artifact / "calibration.json")
    rows = split_rows(cfg, split)
    for name in ("text_current", "text_context"):
        from .train_text import extract_text

        extract_text(cfg, name, split)
    zt, za, valid = cache_logits(cfg, split)
    outputs = {}
    for name in ("text_current", "text_context"):
        cache = load_tensor(Path(cfg["feature_root"]) / name / f"{split}.pt")
        aligned_cache(cache, rows)
        temperature = calibration["text"] if name == selection["text"] else 1.0
        outputs[name] = save_evaluation(
            artifact / "evaluation" / split / name,
            rows,
            probabilities(cache["logits"], temperature),
            temperature=temperature,
        )
    pa = probabilities(za, calibration["audio"])
    valid_rows = [row for row, keep in zip(rows, valid) if keep]
    outputs["audio"] = save_evaluation(
        artifact / "evaluation" / split / "audio",
        valid_rows,
        pa[valid],
        coverage=len(valid_rows) / len(rows),
        temperature=calibration["audio"],
    )
    text_cache = load_tensor(Path(cfg["feature_root"]) / selection["text"] / f"{split}.pt")
    audio_cache = load_tensor(Path(cfg["feature_root"]) / "wavlm_base_plus" / f"{split}.pt")
    head = load_head(Path(cfg["checkpoint_root"]) / "concat").eval()
    with torch.inference_mode():
        concat = head(torch.cat([text_cache["features"], audio_cache["features"]], dim=1))
    pc = probabilities(concat)
    pc[~valid] = probabilities(zt[~valid], calibration["text"])
    outputs["concat"] = save_evaluation(
        artifact / "evaluation" / split / "concat",
        rows,
        pc,
        missing_audio_policy="calibrated selected text fallback",
    )
    pf = probabilities(late_logits(zt, za, selection["alpha"]), calibration["fusion"])
    pf[~valid] = probabilities(zt[~valid], calibration["text"])
    text_pred, fused_pred = zt.argmax(1), pf.argmax(1)
    labels = np.array([row["label"] for row in rows])
    outputs["late_fusion"] = save_evaluation(
        artifact / "evaluation" / split / "late_fusion",
        rows,
        pf,
        temperature=calibration["fusion"],
        matched_text=selection["text"],
        alpha=selection["alpha"],
        predictions_changed=int((text_pred != fused_pred).sum()),
        text_wrong_fusion_correct=int(((text_pred != labels) & (fused_pred == labels)).sum()),
        text_correct_fusion_wrong=int(((text_pred == labels) & (fused_pred != labels)).sum()),
    )
    majority(cfg, (split,))
    return outputs


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--majority", action="store_true")
    parser.add_argument("--split", choices=("dev_model", "dev_calib"), default="dev_model")
    args = parser.parse_args()
    cfg = load_config(args.config)
    majority(cfg) if args.majority else evaluate_split(cfg, args.split)


if __name__ == "__main__":
    main()

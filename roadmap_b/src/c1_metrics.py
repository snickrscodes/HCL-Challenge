"""Six-label soft-vote metrics/calibration, separate from fixed seven-label A/B metrics."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, xlogy
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .c1_data import contract
from .evaluate import probabilities


def checked(logits, targets):
    z, y = np.asarray(logits, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if z.ndim != 2 or z.shape != y.shape or z.shape[1] != 6 or len(z) == 0:
        raise ValueError("Expected nonempty six-class logits and listener distributions")
    if (
        not np.isfinite(z).all()
        or not np.isfinite(y).all()
        or (y < 0).any()
        or not np.allclose(y.sum(1), 1, atol=1e-6)
    ):
        raise ValueError("Invalid logits/targets")
    return z, y


def ece(confidence, reference):
    bins = []
    result = 0.0
    for i in range(15):
        lo, hi = i / 15, (i + 1) / 15
        mask = (confidence > lo) & (confidence <= hi)
        count = int(mask.sum())
        c = float(confidence[mask].mean()) if count else None
        r = float(reference[mask].mean()) if count else None
        if count:
            result += count / len(confidence) * abs(c - r)
        bins.append(
            {"lower_exclusive": lo, "upper_inclusive": hi, "n": count, "confidence": c, "reference": r}
        )
    return result, bins


def metrics(logits, targets, temperature=1.0):
    z, y = checked(logits, targets)
    p = probabilities(z, temperature)
    logp = log_softmax(z / temperature, axis=-1)
    pred = p.argmax(1)
    confidence = p.max(1)
    middle = (p + y) / 2
    jsd = 0.5 * (
        np.sum(xlogy(p, p) - xlogy(p, middle), axis=1) + np.sum(xlogy(y, y) - xlogy(y, middle), axis=1)
    )
    vote_ece, vote_bins = ece(confidence, y[np.arange(len(y)), pred])
    unique = (y == y.max(1, keepdims=True)).sum(1) == 1
    hard = None
    if unique.any():
        truth = y[unique].argmax(1)
        predicted = pred[unique]
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, predicted, labels=range(6), zero_division=0
        )
        hard_ece, hard_bins = ece(confidence[unique], (truth == predicted).astype(float))
        hard = {
            "n": int(unique.sum()),
            "accuracy": float((truth == predicted).mean()),
            "macro_f1": float(f1.mean()),
            "weighted_f1": float(np.average(f1, weights=support)),
            "per_class": {
                name: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i, name in enumerate(contract()["labels"])
            },
            "confusion_matrix": confusion_matrix(truth, predicted, labels=range(6)).tolist(),
            "ece": hard_ece,
            "bins": hard_bins,
        }
    return {
        "namespace": contract()["namespace"],
        "labels": contract()["labels"],
        "n": len(y),
        "ties": int((~unique).sum()),
        "soft_nll": float(-(y * logp).sum(1).mean()),
        "brier": float(np.square(p - y).sum(1).mean()),
        "jsd": float(jsd.mean()),
        "vote_consistency_ece": vote_ece,
        "vote_bins": vote_bins,
        "unique_plurality": hard,
        "temperature": float(temperature),
        "log_base": "e",
    }


def fit_temperature(logits, targets, partition):
    if partition != "calib":
        raise ValueError("Only external calibration may fit a delivery temperature")
    z, y = checked(logits, targets)

    def loss(t):
        return float(-(y * log_softmax(z / np.exp(t), axis=-1)).sum(1).mean())

    fit = minimize_scalar(loss, bounds=(-4, 4), method="bounded", options={"xatol": 1e-7})
    if not fit.success:
        raise RuntimeError("Soft-target temperature fitting failed")
    candidates = [float(fit.x), -4.0, 0.0, 4.0]
    best = min(candidates, key=loss)
    return {
        "temperature": float(np.exp(best)),
        "log_temperature": best,
        "boundary_hit": abs(best) >= 4 - 1e-6,
        "nll_before": loss(0),
        "nll_after": loss(best),
        "partition": "calib",
        "bounds": [-4, 4],
    }

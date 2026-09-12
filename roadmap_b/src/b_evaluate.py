"""Frozen development evaluation, calibration, controls, interventions and uncertainty."""

from pathlib import Path

import numpy as np
import torch

from .b_data import eligibility, frozen_policy, load_inputs, output, parser, settings, verify_baseline
from .b_fusion import VARIANTS
from .b_train import load_model, predict
from .calibrate import fit_temperature
from .constants import LABELS
from .evaluate import metrics, probabilities
from .models import load_head
from .utils import digest, key, read_json, read_rows, write_json, write_rows


def distributions(logits, inputs, temperature, text_temperature):
    p = probabilities(logits, temperature)
    missing = ~inputs["available"].numpy()
    p[missing] = probabilities(inputs["zt"][missing], text_temperature)
    return p


def transition_stats(labels, reference, candidate, mask=None):
    y, a, b = map(np.asarray, (labels, reference, candidate))
    m = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    y, a, b = y[m], a[m], b[m]
    changed = a != b
    corrected = (a != y) & (b == y)
    harmed = (a == y) & (b != y)
    wrong_different = (a != y) & (b != y) & changed

    def divide(num, den):
        return float(num / den) if den else None

    return {
        "support": len(y),
        "changed": int(changed.sum()),
        "corrected": int(corrected.sum()),
        "harmed": int(harmed.sum()),
        "wrong_to_different_wrong": int(wrong_different.sum()),
        "net_correct": int(corrected.sum() - harmed.sum()),
        "correction_precision_among_changed": divide(corrected.sum(), changed.sum()),
        "recovery_rate_among_reference_errors": divide(corrected.sum(), (a != y).sum()),
        "harm_rate_among_reference_correct": divide(harmed.sum(), (a == y).sum()),
    }


def grouped_changes(labels, reference, candidate):
    y = np.asarray(labels)
    groups = {"all": np.ones(len(y), bool), "non_neutral": y != 0}
    groups.update({label: y == i for i, label in enumerate(LABELS)})
    return {name: transition_stats(y, reference, candidate, mask) for name, mask in groups.items()}


def true_positives(labels, pred):
    y, p = np.asarray(labels), np.asarray(pred)
    values = {name: int(((y == i) & (p == i)).sum()) for i, name in enumerate(LABELS)}
    values["fear_plus_disgust"] = values["fear"] + values["disgust"]
    values["non_neutral"] = sum(v for n, v in values.items() if n in LABELS and n != "neutral")
    return values


def fixed_slices(inputs):
    text_prob = probabilities(inputs["zt"]).max(-1)
    available = inputs["available"].numpy()
    return {
        "all": np.ones(len(text_prob), bool),
        "audio_available": available,
        "text_confidence_lt_0.6": text_prob < 0.6,
        "lt_0.4": text_prob < 0.4,
        "0.4_to_0.6": (text_prob >= 0.4) & (text_prob < 0.6),
        "0.6_to_0.8": (text_prob >= 0.6) & (text_prob < 0.8),
        "gte_0.8": text_prob >= 0.8,
        "unimodal_disagreement": available
        & (inputs["zt"].argmax(-1).numpy() != inputs["za"].argmax(-1).numpy()),
    }


def describe(values):
    v = np.asarray(values, float)
    if not len(v):
        return {"support": 0}
    return {
        "support": len(v),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p10": float(np.quantile(v, 0.1)),
        "p90": float(np.quantile(v, 0.9)),
        "p95": float(np.quantile(v, 0.95)),
        "min": float(v.min()),
        "max": float(v.max()),
        "std": float(v.std()),
        "near_zero_rate_lt_0.05": float((v < 0.05).mean()),
        "near_one_rate_gt_0.95": float((v > 0.95).mean()),
    }


def correlation(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and x.std() > 1e-12 and y.std() > 1e-12 else None


def bootstrap(rows, labels, reference, candidate, repetitions, seed):
    """Paired resampling of entire dialogues, using vectorized confusion matrices."""
    ids = sorted({r["dialogue_id"] for r in rows})
    groups = np.array([ids.index(r["dialogue_id"]) for r in rows])
    y = np.asarray(labels)
    matrices = []
    for pred in (reference, candidate):
        c = np.zeros((len(ids), len(LABELS), len(LABELS)), dtype=np.int64)
        np.add.at(c, (groups, y, np.asarray(pred)), 1)
        matrices.append(c)
    weights = np.random.default_rng(seed).multinomial(
        len(ids), np.full(len(ids), 1 / len(ids)), size=repetitions
    )

    def scores(c):
        cm = np.einsum("rg,gij->rij", weights, c)
        tp = np.diagonal(cm, axis1=1, axis2=2)
        support, predicted = cm.sum(2), cm.sum(1)
        f1 = np.divide(
            2 * tp, support + predicted, out=np.zeros_like(tp, float), where=(support + predicted) > 0
        )
        n = support.sum(1)
        return {"accuracy": tp.sum(1) / n, "weighted_f1": (f1 * support).sum(1) / n, "macro_f1": f1.mean(1)}

    a, b = map(scores, matrices)
    return {
        "replicates": repetitions,
        "unit": "dialogue",
        "seed": seed,
        "caution": "Conditional descriptive intervals after model selection; not significance or encoder-seed uncertainty",
        "differences": {
            n: {"low": float(np.quantile(b[n] - a[n], 0.025)), "high": float(np.quantile(b[n] - a[n], 0.975))}
            for n in a
        },
    }


def derangement(indices, seed):
    indices = np.asarray(indices)
    if len(indices) < 2:
        raise ValueError("Derangement needs two available examples")
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        perm = rng.permutation(indices)
        if np.all(perm != indices):
            return perm
    raise RuntimeError("Could not sample derangement")


def swapped(inputs, mapping):
    value = dict(inputs)
    value["ha"] = inputs["ha"][mapping]
    value["za"] = inputs["za"][mapping]
    return value


def intervention_inputs(cfg, rows, train, dev):
    n = len(rows)
    valid = dev["available"].nonzero().flatten().numpy()
    conditions = {}
    for seed in cfg["evaluation"]["shuffle_seeds"]:
        mapping = np.arange(n)
        mapping[valid] = derangement(valid, seed)
        conditions[f"shuffle_{seed}"] = (
            swapped(dev, mapping),
            {"support": len(valid), "mapping": mapping.tolist()},
        )
    mean_h = train["ha"][train["available"]].mean(0, keepdim=True)
    head = load_head(Path(cfg["baseline_root"]) / "checkpoints/audio").eval()
    with torch.inference_mode():
        mean_z = head(mean_h)
    mean = dict(dev)
    mean["ha"], mean["za"] = mean_h.expand(n, -1), mean_z.expand(n, -1)
    conditions["training_mean_audio"] = (
        mean,
        {"source": "eligible train mean embedding, then original audio head"},
    )
    mapping = np.arange(n)
    speakers = {}
    for i in valid:
        speakers.setdefault(rows[i]["speaker"], []).append(i)
    support = []
    for group in speakers.values():
        if len(group) > 1:
            mapping[group] = np.roll(group, 1)
            support.extend(group)
    conditions["same_speaker"] = (
        swapped(dev, mapping),
        {
            "support": len(support),
            "indices": sorted(map(int, support)),
            "mapping": mapping.tolist(),
            "rule": "cyclic shift in locked row order within speaker",
        },
    )
    return conditions


def verify_freeze(cfg):
    root = output(cfg)
    frozen_policy(cfg)
    freeze = read_json(root / "model_freeze.json")
    for relative, expected in freeze["checkpoints_sha256"].items():
        if digest(root / relative) != expected:
            raise ValueError("Frozen B checkpoint changed")
    for filename, expected in freeze["implementation_sha256"].items():
        actual = digest(Path("src") / filename)
        if actual != expected:
            repair_file = root / "evaluation_repair.json"
            repair = read_json(repair_file) if repair_file.exists() else {}
            accepted = repair.get("source_hash_amendments", {}).get(filename, {})
            if accepted.get("original_sha256") != expected or accepted.get("repaired_sha256") != actual:
                raise ValueError("B implementation changed after model freeze without recorded repair")
    if digest(root / "canonical_policy.json") != freeze["policy_sha256"]:
        raise ValueError("Audio policy changed after selection")
    if digest(root / "text_policy.json") != freeze["text_policy_sha256"]:
        raise ValueError("Text policy changed after selection")
    for relative, expected in freeze["cache_sha256"].items():
        if digest(root / relative) != expected:
            raise ValueError("Frozen B cache changed")
    if digest(root / "resolved_config.yaml") != freeze["resolved_config_sha256"]:
        raise ValueError("Resolved experimental configuration changed")
    return freeze


def evaluate(cfg):
    verify_baseline(cfg)
    verify_freeze(cfg)
    root = output(cfg)
    dest = root / "evaluation"
    dest.mkdir(parents=True, exist_ok=False)
    rows, dev = load_inputs(cfg, "dev_model")
    _, train = load_inputs(cfg, "train")
    calibration_rows, calib = load_inputs(cfg, "dev_calib")
    base = Path(cfg["baseline_root"])
    temperatures = read_json(base / "artifacts/calibration.json")
    alpha = read_json(base / "artifacts/selection.json")["alpha"]
    y = dev["labels"].numpy()
    text_probs = probabilities(dev["zt"], temperatures["text"])
    canonical_z = (1 - alpha) * dev["zt"] + alpha * dev["za"]
    canonical_p = distributions(canonical_z, dev, temperatures["fusion"], temperatures["text"])
    original_refs = {}
    for name in ("text_context", "concat", "late_fusion"):
        records = read_rows(base / f"artifacts/evaluation/dev_model/{name}/predictions.jsonl")
        assert [r["key"] for r in records] == [key(r) for r in rows]
        original_refs["A_original_" + name] = np.array([r["probabilities"] for r in records])
    references = {**original_refs, "A_replay_canonical": canonical_p, "text": text_probs}
    reference_results = {}
    for name, p in references.items():
        reference_results[name] = metrics(y, p)
        write_rows(
            dest / f"{name}_predictions.jsonl",
            [
                {"key": key(row), "label": row["label"], "probabilities": prob.tolist()}
                for row, prob in zip(rows, p)
            ],
        )
    eligible = dev["available"].numpy()
    audio_p = probabilities(dev["za"], temperatures["audio"])
    audio_metrics = metrics(y[eligible], audio_p[eligible])
    audio_metrics["coverage"] = float(eligible.mean())
    audio_metrics["excluded_rows_retained_in_overall_models"] = int((~eligible).sum())
    write_json(dest / "A_replay_audio_eligible_only.json", audio_metrics)
    write_rows(
        dest / "A_replay_audio_eligible_predictions.jsonl",
        [
            {"key": key(rows[i]), "label": rows[i]["label"], "probabilities": audio_p[i].tolist()}
            for i in np.flatnonzero(eligible)
        ],
    )
    write_json(dest / "references.json", reference_results)
    slices = fixed_slices(dev)
    write_json(
        dest / "slice_membership.json",
        {name: [key(rows[i]) for i in np.flatnonzero(mask)] for name, mask in slices.items()},
    )
    conditions = intervention_inputs(cfg, rows, train, dev)
    results = {}
    a_interventions = {}
    for cname, (modified, metadata) in conditions.items():
        p = distributions(
            (1 - alpha) * modified["zt"] + alpha * modified["za"],
            modified,
            temperatures["fusion"],
            temperatures["text"],
        )
        a_interventions[cname] = {"metrics": metrics(y, p), "metadata": metadata}
    write_json(dest / "A_replay_interventions.json", a_interventions)
    for variant in VARIANTS:
        for seed in cfg["seeds"]:
            name = f"{variant}/{seed}"
            model, _ = load_model(root / "runs" / name / "checkpoint.pt")
            run_dir = dest / name
            run_dir.mkdir(parents=True)
            raw = predict(model, dev)
            calibration = predict(model, calib)
            valid = calib["available"]
            temperature = fit_temperature(calibration["logits"][valid], calib["labels"][valid])
            p = distributions(raw["logits"], dev, temperature, temperatures["text"])
            unscaled = distributions(raw["logits"], dev, 1.0, temperatures["text"])
            outcome = p.argmax(-1)
            influence = 0.5 * np.abs(
                probabilities(raw["logits"], temperature) - probabilities(dev["zt"], temperature)
            ).sum(-1)
            deltas = raw["delta"].numpy()
            diagnostics = {
                "gate": raw["gate"].numpy(),
                "delta_l2": np.linalg.norm(deltas, axis=-1),
                "delta_max_abs": np.abs(deltas).max(-1),
                "audio_influence": influence,
                "text_max_probability": raw["text_max_probability"].numpy(),
                "audio_max_probability": raw["audio_max_probability"].numpy(),
                "text_entropy": raw["text_entropy"].numpy(),
                "audio_entropy": raw["audio_entropy"].numpy(),
            }
            records = []
            for i, row in enumerate(rows):
                records.append(
                    {
                        "key": key(row),
                        "label": row["label"],
                        "prediction": int(outcome[i]),
                        "raw_logits": raw["logits"][i].tolist(),
                        "text_logits": dev["zt"][i].tolist(),
                        "audio_logits": dev["za"][i].tolist(),
                        "probabilities": p[i].tolist(),
                        "audio_valid": row["audio_valid"],
                        "audio_unavailable_reason": eligibility(row)[1],
                        "audio_available": bool(dev["available"][i]),
                        "prediction_changed": bool(outcome[i] != dev["zt"][i].argmax()),
                        **{n: float(v[i]) for n, v in diagnostics.items()},
                    }
                )
            write_rows(run_dir / "predictions.jsonl", records)
            paired = {rname: grouped_changes(y, ref.argmax(-1), outcome) for rname, ref in references.items()}
            gate = diagnostics["gate"]
            a_pred = canonical_p.argmax(-1)
            corrected, harmed = (a_pred != y) & (outcome == y), (a_pred == y) & (outcome != y)
            diag = {n: describe(v[dev["available"]]) for n, v in diagnostics.items()}
            diag["relationships"] = {
                "gate_vs_influence": correlation(gate, influence),
                "gate_vs_text_confidence": correlation(gate, diagnostics["text_max_probability"]),
                "gate_vs_audio_confidence": correlation(gate, diagnostics["audio_max_probability"]),
                "corrected_gate": describe(gate[corrected]),
                "harmed_gate": describe(gate[harmed]),
                "corrected_influence": describe(influence[corrected]),
                "harmed_influence": describe(influence[harmed]),
            }
            calibration_metrics = {
                "temperature": temperature,
                "fitted_on": "dev_calib eligible rows only, after checkpoint freeze",
                "before": metrics(calib["labels"][valid], probabilities(calibration["logits"][valid])),
                "after": metrics(
                    calib["labels"][valid], probabilities(calibration["logits"][valid], temperature)
                ),
            }
            write_json(run_dir / "calibration.json", calibration_metrics)
            calibration_p = distributions(calibration["logits"], calib, temperature, temperatures["text"])
            write_rows(
                run_dir / "dev_calib_predictions.jsonl",
                [
                    {
                        "key": key(row),
                        "label": row["label"],
                        "raw_logits": z.tolist(),
                        "probabilities": prob.tolist(),
                        "audio_available": bool(available),
                    }
                    for row, z, prob, available in zip(
                        calibration_rows, calibration["logits"], calibration_p, calib["available"]
                    )
                ],
            )

            training_mean_gate = read_json(root / "runs" / name / "training.json")["training_mean_gate"]
            interventions = {}
            for cname, (modified, metadata) in conditions.items():
                z = predict(model, modified)["logits"]
                ip = distributions(z, modified, temperature, temperatures["text"])
                values = {"metrics": metrics(y, ip), "metadata": metadata}
                if cname == "same_speaker" and metadata["support"]:
                    idx = metadata["indices"]
                    values["swappable_subset"] = metrics(y[idx], ip[idx])
                    values["matched_same_subset"] = metrics(y[idx], p[idx])
                interventions[cname] = values
            zero = predict(model, dev, 0)["logits"]
            if not torch.equal(zero, dev["zt"]):
                raise AssertionError("Forced-zero gate failed exact text regression")
            for cname, forced in (("gate_zero", 0.0), ("training_mean_gate", training_mean_gate)):
                z = predict(model, dev, forced)["logits"]
                interventions[cname] = {
                    "metrics": metrics(y, distributions(z, dev, temperature, temperatures["text"])),
                    "forced_gate": forced,
                }
            shuffled = [
                v["metrics"]["macro_f1"] for n, v in interventions.items() if n.startswith("shuffle_")
            ]
            interventions["shuffle_summary"] = {
                "macro_f1_mean": float(np.mean(shuffled)),
                "macro_f1_min": float(np.min(shuffled)),
                "macro_f1_max": float(np.max(shuffled)),
                "outcome_range_not_confidence_interval": True,
            }
            values = {
                "metrics": metrics(y, p),
                "uncalibrated_metrics": metrics(y, unscaled),
                "temperature": temperature,
                "true_positives": true_positives(y, outcome),
                "paired_changes": paired,
                "diagnostics": diag,
                "slices": {n: metrics(y[m], p[m]) if m.any() else {"n": 0} for n, m in slices.items()},
                "bootstrap_vs_A_replay": bootstrap(
                    rows,
                    y,
                    a_pred,
                    outcome,
                    cfg["evaluation"]["bootstrap_replicates"],
                    cfg["evaluation"]["bootstrap_seed"],
                ),
                "bootstrap_vs_A_original": bootstrap(
                    rows,
                    y,
                    original_refs["A_original_late_fusion"].argmax(-1),
                    outcome,
                    cfg["evaluation"]["bootstrap_replicates"],
                    cfg["evaluation"]["bootstrap_seed"],
                ),
                "interventions": interventions,
            }
            write_json(run_dir / "metrics.json", values)
            results[name] = values
    summary = {}
    for variant in VARIANTS:
        summary[variant] = {}
        for metric in ("accuracy", "weighted_f1", "macro_f1"):
            v = [results[f"{variant}/{s}"]["metrics"][metric] for s in cfg["seeds"]]
            summary[variant][metric] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
    write_json(
        dest / "summary.json",
        {
            "variants": summary,
            "reference": "A_replay_canonical",
            "A_replay_true_positives": true_positives(y, canonical_p.argmax(-1)),
            "encoder_seed_variability": "not measured; frozen A encoders are one realization",
        },
    )
    verify_baseline(cfg)


def main():
    args = parser(__doc__).parse_args()
    evaluate(settings(args))


if __name__ == "__main__":
    main()

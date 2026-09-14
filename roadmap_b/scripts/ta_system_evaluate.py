"""Frozen H1 policy evaluation. Reads only locked training/dev/working evidence.

No model forward, fit, calibration, threshold search, or holdout comparison occurs.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.ta_policy import LABELS6, select_response  # noqa: E402

LABELS7 = ("neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust")
EXPECTED = {
    "ta_closure_v1/EVALUATION_PROTOCOL.json": "8af56aa4d8184b33afd045b0ef61cd1005818f4c5afa3f704c6f8920ab0e2c32",
    "ta_closure_v1/working/OUTPUT_LOCK.json": "b1b596c79eda124f7c47008404a5c39adb7ab909126e6a5bfba1a2bb4f7d3e0a",
    "acoustic_capability_v1/working/SHUFFLE.json": "10ec8912914479ea9f97c0a8743e1aef3484aae823b99522b87f93df4a126b5f",
    "joint_supervision_v1/working_local5080/PAIR_INVENTORY.json": "6446912545f77233a20dcb411e64ec5f7ca581996f7423bf841aaa15bdccfd79",
    "roadmap_b/evaluation/adaptive/1337/predictions.jsonl": "9bc75968236877cfccd057ad0c16ba9a40054aa1f083ca255faeb6bc75cdab0f",
    "roadmap_b/evaluation/adaptive/1337/metrics.json": "b6520b0592b869237e1616bfa28f5188cd5a6c4c40e31a9af3b83330718cccee",
    "joint_supervision_v1/features/meld_dev_model.pt": "8ff39055ef45c54e206a72bcc28a07993bbe6462c26ac9afa7f3838764c059fb",
    "acoustic_capability_v1/features/meld_dev_model.pt": "73ae6c65369bdae7be1b15b4d26f6ed4999b6363c84bc2888319ad251d786ace",
    "joint_supervision_v1/features/crema_train.pt": "965d997fc4fb014dc3d53210198387a67b26cf0a9e164c2215061d6350b6aaaa",
    "ta_finalization_v1/TA_RELEASE_CANDIDATE.json": "0cc0ae6f0e3fdc024242cb8b60e98d66945626d60d6438a11c3edeb3b5a9ec68",
    "conditional_gradient_ablation_v1/ACCEPTANCE_MATRIX.json": "b7cc4f70552f18afd402bab7d21abc760a7a378bc02975cd1636c03edea32af9",
}
SOURCE_LOCK = {
    "src/ta_policy.py": "9814915d9fd0bdd45256227ed251c42d31cb8aa260b63f36dd494f0c6e724d9a",
    "evidence/ta_system_closure/HYPOTHESIS_01.md": "f222ea56474f66b4a31ebfa5da732421a00e2659bca17a8eb14d3ff2fb9280f6",
    "evidence/ta_system_closure/POLICY_CASES.json": "0061c5aeef8c79a970899c090437211cb268f2fa150659d6c3b46df40ad67612",
    "src/responses.py": "9c6a0cf2b373fe7875db3f941b0f1024a7fc337571c2766a564a7793871c61a1",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_path(root, relative):
    parts = Path(relative).parts
    require(
        not any("confirmation" in p.lower() or p.lower() in {"test", "external_test"} for p in parts),
        "Forbidden evaluation partition",
    )
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Evidence path escapes root")
    return path


def probabilities(values, size):
    p = np.asarray(values, dtype=np.float64)
    require(
        p.shape == (size,) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(),
        "Malformed probability vector",
    )
    require(np.isclose(p.sum(), 1, atol=1e-6, rtol=0), "Probabilities do not sum to one")
    return p


def conditional_target(counts):
    counts = np.asarray(counts)
    require(
        counts.shape == (9,) and (counts >= 0).all() and np.equal(counts, counts.astype(int)).all(),
        "Invalid nine-label counts",
    )
    supported = counts[[0, 2, 3, 4, 5, 6]]
    if counts.sum() <= 0 or 5 * supported.sum() < 4 * counts.sum():
        return None
    return supported.astype(np.float64) / supported.sum()


def actor_mean(values, actors):
    groups = defaultdict(list)
    for value, actor in zip(values, actors, strict=True):
        groups[actor].append(float(value))
    return float(np.mean([np.mean(v) for v in groups.values()])) if groups else None


def verify_shuffle(mapping, ids):
    require(
        set(mapping) == set(ids) and len(mapping) == len(ids) and set(mapping.values()) == set(ids),
        "Shuffle must be a complete bijection",
    )


def evidence(model, available=True):
    return {
        "available": bool(available),
        "namespace": "crema6_audio_votes_v1",
        "distribution": dict(zip(LABELS6, model["probabilities"], strict=True)) if available else None,
    }


def route_rows(rows, *, mapping=None, prior=None, policy="role_specific_h1", unavailable=False):
    by_id = {r["id"]: r for r in rows}
    if mapping is not None:
        verify_shuffle(mapping, by_id)
    actions = {}
    for row in rows:
        # The whole model record moves together; never combine different recordings.
        model = deepcopy(by_id[mapping[row["id"]]]["models"]["D2"] if mapping else row["models"]["D2"])
        if prior is not None:
            model = {"probabilities": prior}
        state = row["B_state"]
        before = deepcopy(state)
        actions[row["id"]] = select_response(
            row["transcript"], state, evidence(model, not unavailable), policy=policy
        )
        require(state == before, "Policy mutated B state")
    return actions


def route_summary(rows, actions):
    selected = [(r, conditional_target(r["counts9"])) for r in rows]
    selected = [(r, q) for r, q in selected if q is not None]
    scores, actors, agreements = [], [], []
    per_actor = defaultdict(lambda: {"support": 0, "routes": 0, "signed_sum": 0.0, "agreement_sum": 0.0})
    per_class = {
        label: {"unique_plurality_support": 0, "routed": 0, "routes": Counter()} for label in LABELS6
    }
    route_counts = Counter()
    for row, q in selected:
        action = actions[row["id"]]
        label = action["evidence_label"]
        routed = label is not None
        agreement = float(q[LABELS6.index(label)]) if routed else 0.0
        score = 2 * agreement - 1 if routed else 0.0
        scores.append(score)
        actors.append(row["actor"])
        stats = per_actor[row["actor"]]
        stats["support"] += 1
        stats["routes"] += int(routed)
        stats["signed_sum"] += score
        stats["agreement_sum"] += agreement
        route_counts[action["route_id"]] += 1
        if routed:
            agreements.append(agreement)
        winners = np.flatnonzero(q == q.max())
        if len(winners) == 1:
            stats = per_class[LABELS6[int(winners[0])]]
            stats["unique_plurality_support"] += 1
            stats["routed"] += int(routed)
            stats["routes"][action["route_id"]] += 1
    for stats in per_actor.values():
        stats["signed_mean"] = stats.pop("signed_sum") / stats["support"]
        stats["coverage"] = stats["routes"] / stats["support"]
        stats["routed_mean_agreement"] = (
            stats.pop("agreement_sum") / stats["routes"] if stats["routes"] else None
        )
    return {
        "eligible_support": len(selected),
        "all_recordings": len(rows),
        "actor_equal_signed_routed_agreement": actor_mean(scores, actors),
        "pooled_signed_routed_agreement": float(np.mean(scores)),
        "eligible_route_count": len(agreements),
        "eligible_route_coverage": len(agreements) / len(selected),
        "actors_with_eligible_route": sum(s["routes"] > 0 for s in per_actor.values()),
        "routed_mean_listener_agreement": float(np.mean(agreements)) if agreements else None,
        "eligible_route_counts": dict(route_counts),
        "per_actor": dict(sorted(per_actor.items())),
        "unique_plurality_class_slices": per_class,
        "all_recording_route_counts": dict(Counter(a["route_id"] for a in actions.values())),
        "all_recording_reason_counts": dict(Counter(a["reason"] for a in actions.values())),
        "all_recording_response_changes": sum(
            a["response"] != a["reference_response"] for a in actions.values()
        ),
        "unsupported_mass_policy": "All720 retained for routing. Score explicitly conditions412 compatible rows. Calm, surprise, none never mapped.",
    }


def repeat_summary(rows, pairs, actions):
    by_id = {r["id"]: r for r in rows}
    result = {}
    for name, selected in [
        ("stable_proxy", [p for p in pairs if p["listener_consistent_repeat_proxy"]]),
        ("all_performed_repeats", [p for p in pairs if p["kind"] == "repeat"]),
    ]:
        actors = defaultdict(lambda: {"support": 0, "route_changes": 0, "D2_top1_changes": 0, "TV": []})
        transitions = Counter()
        for pair in selected:
            left, right = pair["left"], pair["right"]
            p = np.asarray(by_id[left]["models"]["D2"]["probabilities"])
            q = np.asarray(by_id[right]["models"]["D2"]["probabilities"])
            lr, rr = actions[left]["route_id"], actions[right]["route_id"]
            stats = actors[pair["actor"]]
            stats["support"] += 1
            stats["route_changes"] += int(lr != rr)
            stats["D2_top1_changes"] += int(p.argmax() != q.argmax())
            stats["TV"].append(float(np.abs(p - q).sum() / 2))
            transitions[f"{lr}->{rr}"] += 1
        for stats in actors.values():
            values = stats.pop("TV")
            stats["route_change_rate"] = stats["route_changes"] / stats["support"]
            stats["D2_TV_mean"] = float(np.mean(values))
            stats["D2_TV_p95_linear"] = float(np.quantile(values, 0.95, method="linear"))
        result[name] = {
            "support": len(selected),
            "route_changes": sum(v["route_changes"] for v in actors.values()),
            "D2_top1_changes": sum(v["D2_top1_changes"] for v in actors.values()),
            "maximum_actor_route_change_rate": max(v["route_change_rate"] for v in actors.values()),
            "actor_equal_D2_TV_mean": float(np.mean([v["D2_TV_mean"] for v in actors.values()])),
            "per_actor": dict(sorted(actors.items())),
            "transitions": dict(transitions),
            "route_to_baseline_transitions": sum(
                n for k, n in transitions.items() if k.endswith("->b_policy") and not k.startswith("b_policy")
            ),
            "baseline_to_route_transitions": sum(
                n
                for k, n in transitions.items()
                if k.startswith("b_policy->") and not k.endswith("->b_policy")
            ),
        }
    return result


def classification_metrics(probability_rows, labels, classes):
    p = np.asarray(probability_rows, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    prediction = p.argmax(1)
    confusion = np.zeros((classes, classes), dtype=int)
    np.add.at(confusion, (y, prediction), 1)
    support = confusion.sum(1)
    denominator = confusion.sum(0) + support
    f1 = np.divide(2 * confusion.diagonal(), denominator, out=np.zeros(classes), where=denominator != 0)
    confidence = p.max(1)
    bins = np.minimum((confidence * 15).astype(int), 14)
    ece = sum(
        float((bins == b).mean())
        * abs(float((prediction[bins == b] == y[bins == b]).mean()) - float(confidence[bins == b].mean()))
        for b in range(15)
        if (bins == b).any()
    )
    return {
        "n": len(y),
        "accuracy": float((prediction == y).mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.sum(f1 * support) / len(y)),
        "nll": float(-np.log(p[np.arange(len(y)), y]).mean()),
        "ece_15": ece,
        "confusion_matrix": confusion.tolist(),
    }


def frozen_d2_metrics(rows):
    selected = [(r, conditional_target(r["counts9"])) for r in rows]
    selected = [(r, q) for r, q in selected if q is not None]
    nll, brier, jsd, actors = [], [], [], []
    hard = defaultdict(lambda: [[], []])
    counts = {label: {"support": 0, "true_positives": 0} for label in LABELS6}
    for row, q in selected:
        p = probabilities(row["models"]["D2"]["probabilities"], 6)
        nll.append(float(-np.sum(q * np.log(p))))
        brier.append(float(np.square(p - q).sum()))
        mid = (p + q) / 2
        positive = q > 0
        jsd.append(
            float(
                0.5 * np.sum(p * np.log(p / mid))
                + 0.5 * np.sum(q[positive] * np.log(q[positive] / mid[positive]))
            )
        )
        actors.append(row["actor"])
        winners = np.flatnonzero(q == q.max())
        if len(winners) == 1:
            label = int(winners[0])
            hard[row["actor"]][0].append(p)
            hard[row["actor"]][1].append(label)
            counts[LABELS6[label]]["support"] += 1
            counts[LABELS6[label]]["true_positives"] += int(p.argmax() == label)
    metrics = [classification_metrics(p, y, 6) for p, y in hard.values()]
    for s in counts.values():
        s["recall"] = s["true_positives"] / s["support"] if s["support"] else None
    return {
        "soft_support": len(selected),
        "unique_plurality_support": sum(v["support"] for v in counts.values()),
        "actor_equal_nll6": actor_mean(nll, actors),
        "actor_equal_brier6": actor_mean(brier, actors),
        "actor_equal_jsd6": actor_mean(jsd, actors),
        "actor_equal_macro_f1": float(np.mean([m["macro_f1"] for m in metrics])),
        "pooled_class_accounting": counts,
        "interpretation": "Frozen listener evidence on used working data; unchanged by response routing.",
    }


def source_preservation(root, manifest):
    rows = [
        json.loads(line)
        for line in (root / "roadmap_b/evaluation/adaptive/1337/predictions.jsonl").read_text().splitlines()
    ]
    joint = torch.load(
        root / "joint_supervision_v1/features/meld_dev_model.pt", map_location="cpu", weights_only=False
    )
    acoustic = torch.load(
        root / "acoustic_capability_v1/features/meld_dev_model.pt", map_location="cpu", weights_only=False
    )
    require(
        len(rows) == 839
        and [r["key"] for r in rows] == joint["identity"]["keys"] == acoustic["identity"]["keys"],
        "MELD ID join failed",
    )
    require(
        joint["identity"]["wavehashes"] == acoustic["identity"]["wavehashes"], "MELD waveform join failed"
    )
    for field in ["available", "labels", "d2"]:
        require(torch.equal(joint["inputs"][field], acoustic["inputs"][field]), "MELD tensor join failed")
    z = acoustic["inputs"]["d2_logits"].double().numpy() / manifest["temperature_values"]["D2"]
    p6 = np.exp(z - z.max(1, keepdims=True))
    p6 /= p6.sum(1, keepdims=True)
    changed = 0
    original = []
    after = []
    for index, (record, metadata) in enumerate(zip(rows, joint["rows"], strict=True)):
        p = probabilities(record["probabilities"], 7)
        require(record["label"] == int(acoustic["inputs"]["labels"][index]), "MELD label join failed")
        state = {
            "emotion": LABELS7[int(p.argmax())],
            "confidence": float(p.max()),
            "distribution": dict(zip(LABELS7, p.tolist(), strict=True)),
        }
        before = deepcopy(state)
        action = select_response(
            metadata["text"],
            state,
            evidence({"probabilities": p6[index]}, bool(acoustic["inputs"]["available"][index])),
            policy="role_specific_h1",
        )
        require(state == before, "Categorical state changed")
        original.append(p.tolist())
        after.append(list(state["distribution"].values()))
        changed += action["response"] != action["reference_response"]
    metrics = classification_metrics(original, [r["label"] for r in rows], 7)
    historical = read_json(root / "roadmap_b/evaluation/adaptive/1337/metrics.json")["metrics"]
    errors = {k: abs(metrics[k] - historical[k]) for k in ["macro_f1", "weighted_f1", "nll", "ece_15"]}
    require(max(errors.values()) < 1e-12, "Independent B metrics disagree")
    return {
        "records": 839,
        "audio_available": int(acoustic["inputs"]["available"].sum()),
        "exact_preservation": original == after,
        "categorical_probability_max_abs_difference": 0.0,
        "categorical_prediction_changes": 0,
        "B_temperature_before_and_after": manifest["temperature_values"]["B"],
        "independently_recomputed_B_metrics": metrics,
        "metric_recompute_absolute_errors": errors,
        "response_changes": int(changed),
        "separate_saved_B_stream_logit_max_difference": float(
            (torch.tensor([r["raw_logits"] for r in rows]) - joint["inputs"]["zb"]).abs().max()
        ),
        "scope": "Cached response-policy validation, not fresh waveform equivalence. D2 is not scored as contextual MELD truth.",
    }


def acceptance(result):
    c = result["working"]["candidate"]
    primary = c["actor_equal_signed_routed_agreement"]
    baseline = result["working"]["evidence_only"]["actor_equal_signed_routed_agreement"]
    shuffled = result["working"]["fixed_coupled_shuffle"]["actor_equal_signed_routed_agreement"]
    agreement = c["routed_mean_listener_agreement"]
    checks = {
        "primary_gain_over_baseline_at_least_0.05": primary - baseline >= 0.05,
        "primary_gain_over_shuffle_at_least_0.05": primary - shuffled >= 0.05,
        "eligible_coverage_at_least_0.10": c["eligible_route_coverage"] >= 0.1,
        "actors_with_routes_at_least_8": c["actors_with_eligible_route"] >= 8,
        "routed_listener_agreement_at_least_0.65": agreement is not None and agreement >= 0.65,
        "stable_route_changes_at_most_12_of_67": result["stability"]["stable_proxy"]["route_changes"] <= 12,
        "MELD839_categorical_probabilities_calibration_preserved": result["source_preservation"][
            "exact_preservation"
        ],
        "fixed_cases": result["fixed_cases"]["passed"],
        "input_integrity_and_joins": result["integrity"]["passed"],
    }
    passed = all(checks.values())
    return {
        "decision": "ACCEPT_DEVELOPMENT_CANDIDATE_PENDING_RUNTIME" if passed else "REJECT_CHANGE",
        "recommended_default": "pending_local_runtime_validation" if passed else "evidence_only",
        "measured_requirements": {k: "PASS" if v else "FAIL" for k, v in checks.items()},
        "all_measured_requirements_pass": passed,
        "primary_gain_over_baseline": primary - baseline,
        "primary_gain_over_shuffle": primary - shuffled,
        "local_raw_runtime_and_availability": "NOT_MEASURED",
        "independent_transfer": "NOT_MEASURED",
        "human_response_usefulness": "NOT_MEASURED",
        "selective_nuisance_claim": "INCONCLUSIVE",
        "claim": "Fixed adaptive working-data response-routing experiment; no improved categorical accuracy or human-response-quality claim.",
        "budget": {"new_hypotheses": 1, "new_fits": 0, "new_calibrations": 0, "threshold_retries": 0},
    }


def evaluate(root):
    root = Path(root).resolve()
    hashes = {}
    for relative, expected in EXPECTED.items():
        actual = sha(checked_path(root, relative))
        require(actual == expected, f"Frozen input hash mismatch: {relative}")
        hashes[relative] = actual
    for relative, expected in SOURCE_LOCK.items():
        require(sha(PROJECT / relative) == expected, f"Prospective source lock mismatch: {relative}")
    manifest = read_json(root / "ta_finalization_v1/TA_RELEASE_CANDIDATE.json")
    for relative, expected in manifest["B_required_files"].items():
        require(sha(checked_path(root / "roadmap_b", relative)) == expected, "Frozen B asset changed")
    for relative in ["normalization/D2.pt", "runs/D2/1337/checkpoint.pt", "runs/D2/1337/calibration.json"]:
        require(
            sha(checked_path(root / "roadmap_c1_v2", relative)) == manifest["c1_required_files"][relative],
            "Frozen D2 asset changed",
        )
    protocol = read_json(root / "ta_closure_v1/EVALUATION_PROTOCOL.json")
    require(
        tuple(protocol["C1_labels"]) == LABELS6 and tuple(protocol["B_labels"]) == LABELS7,
        "Namespaces changed",
    )
    working = root / "ta_closure_v1/working"
    lock = read_json(working / "OUTPUT_LOCK.json")
    files = sorted((working / "predictions").glob("*.json"))
    require(
        len(files) == 720 and set(lock["files_sha256"]) == {"predictions/" + f.name for f in files},
        "Working720 inventory changed",
    )
    rows = []
    for file in files:
        require(
            sha(file) == lock["files_sha256"]["predictions/" + file.name], "Working prediction hash mismatch"
        )
        row = read_json(file)
        require(
            row["partition"] == "working"
            and row["actor"] in protocol["working_actors"]
            and row["history"] == [],
            "Prohibited partition/history",
        )
        require(
            row["hardware"] == "NVIDIA GeForce RTX 5080 Laptop GPU" and row["id"] == file.stem,
            "Working numerical identity mismatch",
        )
        for name, size in [("B", 7), ("text", 7), ("D2", 6)]:
            probabilities(row["models"][name]["probabilities"], size)
        require(
            row["models"]["D2"]["temperature"] == manifest["temperature_values"]["D2"],
            "D2 calibration changed",
        )
        rows.append(row)
    ids = {r["id"] for r in rows}
    require(len(ids) == 720, "Duplicate working IDs")
    pairs = read_json(root / "joint_supervision_v1/working_local5080/PAIR_INVENTORY.json")["pairs"]
    require(all(p["left"] in ids and p["right"] in ids for p in pairs), "Pair joins leave inventory")
    require(
        len(pairs) == 2712 and sum(p["listener_consistent_repeat_proxy"] for p in pairs) == 67,
        "Historical pairs changed",
    )
    shuffle = read_json(root / "acoustic_capability_v1/working/SHUFFLE.json")
    verify_shuffle(shuffle["source_ids"], ids)
    require(shuffle["fields_replaced_together"] == ["d2", "d2_logits", "x_pre"], "Coupled shuffle changed")
    train = torch.load(
        root / "joint_supervision_v1/features/crema_train.pt", map_location="cpu", weights_only=False
    )
    prior = train["inputs"]["targets"][train["inputs"]["available"]].double().mean(0).numpy()
    prior = (prior / prior.sum()).tolist()
    controls = {
        "candidate": route_rows(rows),
        "evidence_only": route_rows(rows, policy="evidence_only"),
        "no_new_evidence": route_rows(rows, unavailable=True),
        "fixed_coupled_shuffle": route_rows(rows, mapping=shuffle["source_ids"]),
        "CREMA_training_listener_prior": route_rows(rows, prior=prior),
    }
    summaries = {name: route_summary(rows, actions) for name, actions in controls.items()}
    require(summaries["candidate"]["eligible_support"] == 412, "Conditional mask changed")
    stability = repeat_summary(rows, pairs, controls["candidate"])
    require(
        stability["stable_proxy"]["D2_top1_changes"] == 12
        and stability["all_performed_repeats"]["support"] == 360,
        "Repeat reference mismatch",
    )
    cases = read_json(PROJECT / "evidence/ta_system_closure/POLICY_CASES.json")
    case_results = [
        {
            "id": c["id"],
            "passed": select_response(c["text"], c["state"], c["evidence"], policy="role_specific_h1")[
                "route_id"
            ]
            == c["expected_route"],
        }
        for c in cases
    ]
    historical = read_json(root / "conditional_gradient_ablation_v1/ACCEPTANCE_MATRIX.json")
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "H1 fixed role-specific authored routing",
        "numerical_scope": "Saved internally consistent RTX5080 working outputs; RTX4090 MELD development outputs; CPU float64 metric arithmetic. No model forwards.",
        "integrity": {
            "passed": True,
            "all720_prediction_hashes_match": True,
            "input_hashes": hashes,
            "source_hashes": {
                rel: sha(PROJECT / rel)
                for rel in [*SOURCE_LOCK, "src/ta_policy.py", "scripts/ta_system_evaluate.py"]
            },
            "manifest_assets_verified": "All B manifest assets and selected D2 normalization/checkpoint/calibration",
            "shuffle_bijection720": True,
            "no_confirmation_or_official_test_opened": True,
        },
        "working": summaries,
        "stability": stability,
        "source_preservation": source_preservation(root, manifest),
        "frozen_D2": frozen_d2_metrics(rows),
        "training_prior_control": {
            "source": "CREMA training listener targets only",
            "rows": 3697,
            "probabilities": prior,
        },
        "fixed_cases": {
            "count": len(case_results),
            "passed": all(c["passed"] for c in case_results),
            "cases": case_results,
        },
        "historical_failures_retained": {
            "latest_ablation_status": historical["status"],
            "candidate_ready": historical["candidate_ready"],
            "per_seed": historical["seed_results"],
            "pair_selectivity_support": historical["pair_selectivity_support"],
            "original_repeat_thresholds": {
                "actor_mean_TV_excess_max": 0.01,
                "maximum_within_actor_p95_TV_excess_max": 0.03,
            },
            "original_repeat_result": "All3 candidate means passed; all3 maximum-actor tails failed. H1 does not alter D2 or repair those rejected models.",
            "Stage_A": "Rejected: reference macroF1gain .003665 below .02; all3 D2NLL and repeat tails failed; source safeguards passed.",
            "joint_J": "Preferred ECE increase .021882241 exceeds .02; workingNLL6 1.826067 above D2 1.351057;19/67 stable flips;3/8 actor support.",
            "runtime": "Historical canonical B+D2 p95RTF failed on64ms probes. Changed-runtime benchmark NOT_MEASURED here.",
        },
        "scope_limits": [
            "Used working actors are adaptive development evidence",
            "Independent evaluation pending",
            "Human response usefulness unmeasured",
            "Source preservation is cached policy validation, not fresh waveform equivalence",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        required=True,
        help="Existing roadmap_b/artifacts root; only locked allowed inputs are read",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination with neither H1_RESULTS.json nor H1_DECISION.json already present",
    )
    args = parser.parse_args()
    outputs = [args.output_root / name for name in ["H1_RESULTS.json", "H1_DECISION.json"]]
    require(
        not any(p.exists() for p in outputs),
        "Refusing to overwrite completed H1 evidence; choose unused output destination",
    )
    results = evaluate(args.evidence_root)
    decision = acceptance(results)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for path, value in zip(outputs, [results, decision], strict=True):
        with path.open("x") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

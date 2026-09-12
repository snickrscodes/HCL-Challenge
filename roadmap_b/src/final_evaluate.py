"""Final policy lock and one planned official test pass; no fitting or model selection."""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .audio import read_waveform
from .b_data import encoder_config, parser as b_parser
from .b_data import rows_for, settings, verify_baseline, waveform_path
from .b_evaluate import bootstrap, fixed_slices, grouped_changes, true_positives, verify_freeze
from .b_inference import RoadmapB, raw_validation
from .b_train import load_model
from .constants import LABELS
from .context import histories
from .evaluate import metrics, probabilities
from .utils import digest, environment, key, read_json, read_rows, write_json, write_rows

FINAL_MODELS = (
    "text_causal_canonical",
    "A_fixed_canonical",
    "B_adaptive_1337",
    "constant_1337",
    "text_only_1337",
)


def final_policy(cfg):
    root = Path(cfg["output_root"])
    return {
        "version": 1,
        "labels": list(LABELS),
        "text": {"encoding": "singleton", "dtype": "float32", "model": "exact unchanged selected causal T1"},
        "audio": {"encoding": "singleton", "dtype": "bfloat16", "model": "exact unchanged WavLM"},
        "max_audio_seconds": 60,
        "duration_limit": {
            "audio_valid": True,
            "audio_available": False,
            "reason": "duration_limit",
            "action": "skip WavLM; calibrated text fallback; retain overall row; no cropping/chunking",
        },
        "missing_corrupt_audio": "existing calibrated text fallback",
        "selected_B": {
            "variant": "adaptive",
            "seed": 1337,
            "temperature": read_json(root / "evaluation/adaptive/1337/calibration.json")["temperature"],
        },
        "A_alpha": 0.40,
        "A_temperatures": read_json(Path(cfg["baseline_root"]) / "artifacts/calibration.json"),
        "control_seeds": {"constant": 1337, "text_only": 1337},
        "models": list(FINAL_MODELS),
        "historical_A": "A-original development evidence remains immutable; final inference uses A_fixed_canonical, never reintroduces historical batched BF16 text.",
        "calibration": "All saved dev_calib temperatures preserved; no fitting on test.",
        "confidence_slices": ["<.4", ".4-.6", ".6-.8", ">=.8"],
        "bootstrap": {"replicates": 1000, "seed": 20260912, "unit": "dialogue"},
        "production_recommendation_rule": {
            "B_default_requires": {
                "macro_F1_gain_at_least": 0.005,
                "weighted_F1_loss_at_most": 0.005,
                "non_neutral_net_at_least": 0,
                "fear_disgust_TP_loss_at_most": 0,
            },
            "otherwise": "A fixed canonical default, B documented experimental mode; no retraining",
        },
        "parameter_rule": "Total learned parameters including frozen models and calibration scalars; quantization is not a parameter discount.",
        "timing": "Warm loaded session; explicitly separate fresh-process/model load; exclude recording, endpointing, ASR and disk reads from per-turn inference timing.",
    }


def write_policy(cfg, dest):
    dest = Path(dest).resolve()
    for protected in (Path(cfg["baseline_root"]).resolve(), Path(cfg["output_root"]).resolve()):
        if dest == protected or dest.is_relative_to(protected):
            raise ValueError("Final submission artifacts must not overwrite A or B development evidence.")
    dest.mkdir(parents=True, exist_ok=True)
    file = dest / "FINAL_INFERENCE_POLICY.json"
    value = final_policy(cfg)
    if file.exists():
        if read_json(file) != value:
            raise ValueError("Final inference policy already exists and differs.")
    else:
        write_json(file, value)
    write_json(dest / "FINAL_INFERENCE_POLICY.sha256.json", {"sha256": digest(file)})
    return value


def source_hashes():
    paths = [
        Path("requirements.lock.txt"),
        Path("requirements.txt"),
        Path("pyproject.toml"),
        *Path("src").glob("*.py"),
        *Path("tests").glob("*.py"),
        *Path("scripts").glob("*.py"),
        *Path("scripts").glob("*.sh"),
        *Path("recording_kit").glob("*"),
        *Path("configs").glob("*"),
    ]
    return {str(p): digest(p) for p in sorted(paths) if p.is_file()}


def preflight(cfg, dest):
    dest = Path(dest).resolve()
    write_policy(cfg, dest)
    verify_baseline(cfg)
    verify_freeze(cfg)
    root = Path(cfg["output_root"])
    view = dest / "preflight"
    if view.exists():
        if not view.resolve().is_relative_to(dest):
            raise ValueError("Preflight view must remain inside the submission namespace.")
        previous = dest / "preflight_attempts" / str(time.time_ns())
        previous.parent.mkdir(parents=True, exist_ok=True)
        view.rename(previous)
    view.mkdir(parents=True, exist_ok=False)
    # A read-only artifact view lets existing raw validation write new measurements, not B history.
    for name in (
        "canonical_policy.json",
        "text_policy.json",
        "numerical_replay",
        "model_freeze.json",
        "evaluation_repair.json",
        "resolved_config.yaml",
        "runs",
        "evaluation",
        "features",
        "text_features",
        "raw_probe_manifest.json",
    ):
        source = root / name
        if source.exists():
            (view / name).symlink_to(source.resolve(), target_is_directory=source.is_dir())
    local = {**cfg, "output_root": str(view)}
    command = [
        sys.executable,
        "scripts/validate_b.py",
        "--baseline-root",
        cfg["baseline_root"],
        "--output-root",
        str(view),
    ]
    subprocess.run(command, check=True)
    raw_validation(local)
    subprocess.run(
        [
            sys.executable,
            "scripts/raw_demo.py",
            "--baseline-root",
            cfg["baseline_root"],
            "--output-root",
            str(view),
        ],
        check=True,
        capture_output=True,
    )
    checks = {
        "passed": True,
        "source_sha256": source_hashes(),
        "B_seed": 1337,
        "baseline_integrity": verify_baseline(cfg),
        "policy_sha256": digest(dest / "FINAL_INFERENCE_POLICY.json"),
        "test_inferred": False,
        "raw_cache_equivalence": read_json(view / "raw_validation/cache_consistency.json")["passed"],
        "parameter_ledger": read_json(view / "raw_validation/parameter_ledger.json"),
        "validation_sha256": digest(view / "validation.json"),
    }
    write_json(view / "PREFLIGHT.json", checks)


def freeze(cfg, dest):
    dest = Path(dest).resolve()
    policy = write_policy(cfg, dest)
    file = dest / "FINAL_FREEZE.json"
    if file.exists():
        raise ValueError("Final freeze already exists; never overwrite.")
    ready = read_json(dest / "delivery_results/summary.json")["collection"]
    if not ready["complete"] or len(ready["speakers"]) < 2 or not ready["complete_listeners"]:
        raise ValueError("Human recording/listener diagnostic must be completed, regardless of its outcomes.")
    pre = read_json(dest / "preflight/PREFLIGHT.json")
    if not pre["passed"] or pre["source_sha256"] != source_hashes():
        raise ValueError("Run current-source pre-final validation first.")
    if (
        not (dest / "figures/figure_manifest.json").is_file()
        or not (dest / "FINAL_REPORT_DRAFT.md").is_file()
    ):
        raise ValueError("Presentation figures and report dry run must exist before final freeze.")
    demo = read_json(dest / "demo/manifest.json")
    required_cases = {
        "audio_helps",
        "audio_changes_little",
        "failure",
        "same_words",
        "missing_audio",
        "causal_history",
    }
    if (
        not required_cases.issubset(demo["cases"])
        or not read_json(dest / "presentation_validation.json")["passed"]
    ):
        raise ValueError("Validate the complete predetermined demonstration before final freeze.")
    integrity = verify_baseline(cfg)
    verify_freeze(cfg)
    base = Path(cfg["baseline_root"])
    broot = Path(cfg["output_root"])
    checkpoints = {}
    for name in ("adaptive", "constant", "text_only"):
        for rel in (f"runs/{name}/1337/checkpoint.pt", f"evaluation/{name}/1337/calibration.json"):
            checkpoints[rel] = digest(broot / rel)
    artifacts = {
        str(p.relative_to(dest)): digest(p)
        for folder in ("delivery_results", "figures", "demo")
        for p in sorted((dest / folder).rglob("*"))
        if p.is_file()
    }
    value = {
        "version": 1,
        "models": list(FINAL_MODELS),
        "policy": policy,
        "policy_sha256": digest(dest / "FINAL_INFERENCE_POLICY.json"),
        "baseline_integrity": integrity,
        "A_files_sha256": read_json(cfg["baseline_lock"])["baseline_files_sha256"],
        "B_selected_files_sha256": checkpoints,
        "B_model_freeze_sha256": digest(broot / "model_freeze.json"),
        "B_evaluation_repair_sha256": digest(broot / "evaluation_repair.json"),
        "source_sha256": source_hashes(),
        "development_evidence_sha256": artifacts,
        "manifest_sha256": digest(base / "data/processed/manifest.jsonl"),
        "split_ids_sha256": {
            str(p.relative_to(base)): digest(p)
            for p in sorted((base / "data/processed").rglob("*dialogue*"))
            if p.is_file()
        },
        "preflight_sha256": digest(dest / "preflight/PREFLIGHT.json"),
        "environment": environment(encoder_config(cfg)),
        "test_prediction_state": "NOT STARTED",
        "recommendation_is_not_model_tuning": True,
    }
    write_json(file, value)
    write_json(dest / "FINAL_FREEZE.sha256.json", {"sha256": digest(file)})
    return value


def verify_final(cfg, dest):
    dest = Path(dest)
    value = read_json(dest / "FINAL_FREEZE.json")
    if read_json(dest / "FINAL_FREEZE.sha256.json")["sha256"] != digest(dest / "FINAL_FREEZE.json"):
        raise ValueError("Final freeze changed.")
    if value["source_sha256"] != source_hashes():
        raise ValueError("Source changed after final freeze.")
    if value["policy_sha256"] != digest(dest / "FINAL_INFERENCE_POLICY.json"):
        raise ValueError("Policy changed.")
    verify_baseline(cfg)
    verify_freeze(cfg)
    for rel, h in value["B_selected_files_sha256"].items():
        if digest(Path(cfg["output_root"]) / rel) != h:
            raise ValueError("Selected B/control artifact changed.")
    for rel, h in value["development_evidence_sha256"].items():
        if digest(dest / rel) != h:
            raise ValueError("Frozen diagnostic/presentation evidence changed.")
    return value


def claim_test_run(dest, freeze_sha):
    folder = Path(dest) / "official_test"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "STARTED.json"
    with path.open("x") as stream:
        json.dump(
            {
                "freeze_sha256": freeze_sha,
                "started_unix": time.time(),
                "policy": "Exactly one planned inference pass. No automatic retry or seed/model reselection.",
            },
            stream,
            indent=2,
        )
    return folder


def run_test_once(cfg, dest):
    dest = Path(dest).resolve()
    frozen = verify_final(cfg, dest)
    base = Path(cfg["baseline_root"])
    broot = Path(cfg["output_root"])
    # Metadata validation is allowed; actual test inference starts only after the atomic run claim.
    rows = read_rows(base / "data/processed/splits/test.jsonl")
    if len(rows) != 2610 or len({key(r) for r in rows}) != 2610:
        raise ValueError("Official test identity/count mismatch.")
    hist = histories(rows, 3)
    model = RoadmapB(cfg, variant="adaptive", seed=1337)
    controls = {n: load_model(broot / f"runs/{n}/1337/checkpoint.pt")[0] for n in ("constant", "text_only")}
    temps = {n: read_json(broot / f"evaluation/{n}/1337/calibration.json")["temperature"] for n in controls}
    dev = rows_for(cfg, "dev_model")
    warm = next(r for r in dev if key(r) == "dev/48/5")
    model.predict_turn(
        warm["text"],
        read_waveform(waveform_path(cfg, warm)),
        16000,
        histories(dev, 3)[key(warm)],
        speaker=warm["speaker"],
    )
    folder = claim_test_run(dest, digest(dest / "FINAL_FREEZE.json"))
    all_records = []
    with (folder / "all_predictions.jsonl").open("x") as stream:
        for row in rows:
            wave = read_waveform(waveform_path(cfg, row)) if row["audio_valid"] else None
            state, raw = model.predict_turn(
                row["text"],
                wave,
                16000 if wave is not None else None,
                hist[key(row)],
                speaker=row["speaker"],
                return_internal=True,
            )
            pt = probabilities(raw["zt"][0], model.calibration["text"])
            pa = probabilities(raw["za"][0], model.calibration["audio"]) if state["audio_available"] else None
            alpha = model.selection["alpha"]
            a = (
                probabilities((1 - alpha) * raw["zt"][0] + alpha * raw["za"][0], model.calibration["fusion"])
                if state["audio_available"]
                else pt
            )
            pred = {
                "text_causal_canonical": pt.tolist(),
                "A_fixed_canonical": a.tolist(),
                "B_adaptive_1337": list(state["distribution"].values()),
            }
            with torch.inference_mode():
                for name, head in controls.items():
                    z = head(raw["ht"], raw["ha"], raw["zt"], raw["za"], raw["available"])["logits"][0]
                    pred[name + "_1337"] = (
                        probabilities(z, temps[name]) if state["audio_available"] else pt
                    ).tolist()
            value = {
                "key": key(row),
                "dialogue_id": row["dialogue_id"],
                "label": row["label"],
                "audio_valid": row["audio_valid"],
                "audio_available": state["audio_available"],
                "audio_unavailable_reason": state["audio_unavailable_reason"],
                "duration_s": row["duration_s"],
                "waveform_sha256": digest(waveform_path(cfg, row)) if wave is not None else None,
                "text_logits": raw["zt"][0].tolist(),
                "audio_logits": raw["za"][0].tolist(),
                "B_logits": raw["logits"][0].tolist(),
                "audio_probabilities": None if pa is None else pa.tolist(),
                "probabilities": pred,
                "B_state": state,
            }
            stream.write(json.dumps(value, allow_nan=False) + "\n")
            stream.flush()
            all_records.append(value)
    # All models were evaluated from each shared canonical raw representation once. Scoring is pure.
    score_test(cfg, dest, rows, all_records, frozen)
    write_json(
        folder / "COMPLETED.json",
        {
            "freeze_sha256": digest(dest / "FINAL_FREEZE.json"),
            "rows": len(rows),
            "predictions_sha256": digest(folder / "all_predictions.jsonl"),
            "finished_unix": time.time(),
        },
    )
    verify_final(cfg, dest)


def score_test(cfg, dest, rows, values, frozen):
    dest = Path(dest)
    folder = dest / "official_test"
    y = np.array([r["label"] for r in values])
    eligible = np.array([r["audio_available"] for r in values])
    p = {n: np.array([r["probabilities"][n] for r in values]) for n in FINAL_MODELS}
    available = torch.tensor(eligible)
    inputs = {
        "zt": torch.tensor([r["text_logits"] for r in values]),
        "za": torch.tensor([r["audio_logits"] for r in values]),
        "available": available,
    }
    slices = fixed_slices(inputs)
    reference = p["A_fixed_canonical"].argmax(1)
    result = {}
    for name, prob in p.items():
        result[name] = {
            "metrics": metrics(y, prob),
            "true_positives": true_positives(y, prob.argmax(1)),
            "changes_vs_A": grouped_changes(y, reference, prob.argmax(1)),
            "changes_vs_text": grouped_changes(y, p["text_causal_canonical"].argmax(1), prob.argmax(1)),
            "slices": {k: metrics(y[v], prob[v]) if v.any() else {"n": 0} for k, v in slices.items()},
        }
        write_json(folder / (name + "_metrics.json"), result[name])
        write_rows(
            folder / (name + "_predictions.jsonl"),
            [
                {
                    "key": r["key"],
                    "label": r["label"],
                    "probabilities": q.tolist(),
                    "audio_available": r["audio_available"],
                }
                for r, q in zip(values, prob)
            ],
        )
    audio = metrics(y[eligible], np.array([r["audio_probabilities"] for r in values if r["audio_available"]]))
    write_json(folder / "audio_eligible_metrics.json", audio)
    interval = bootstrap(
        rows,
        y,
        reference,
        p["B_adaptive_1337"].argmax(1),
        frozen["policy"]["bootstrap"]["replicates"],
        frozen["policy"]["bootstrap"]["seed"],
    )
    write_json(folder / "B_vs_A_bootstrap.json", interval)
    a, b = result["A_fixed_canonical"], result["B_adaptive_1337"]
    rule = frozen["policy"]["production_recommendation_rule"]["B_default_requires"]
    tests = {
        "macro_gain": b["metrics"]["macro_f1"] >= a["metrics"]["macro_f1"] + rule["macro_F1_gain_at_least"],
        "weighted_preserved": b["metrics"]["weighted_f1"]
        >= a["metrics"]["weighted_f1"] - rule["weighted_F1_loss_at_most"],
        "non_neutral_preserved": b["changes_vs_A"]["non_neutral"]["net_correct"] >= 0,
        "fear_disgust_preserved": b["true_positives"]["fear_plus_disgust"]
        >= a["true_positives"]["fear_plus_disgust"],
    }
    write_json(
        folder / "summary.json",
        {
            "models": result,
            "overall_support": len(y),
            "audio_eligible_support": int(eligible.sum()),
            "unavailable_rows": [
                {"key": r["key"], "reason": r["audio_unavailable_reason"]}
                for r in values
                if not r["audio_available"]
            ],
            "production_recommendation": "B default; A fallback"
            if all(tests.values())
            else "A canonical default; B experimental adaptive mode",
            "predeclared_recommendation_checks": tests,
            "no_retraining_or_reselection": True,
        },
    )


def main():
    p = b_parser(__doc__)
    p.add_argument("action", choices=["policy", "preflight", "freeze", "test"])
    p.add_argument("--submission-root", default="artifacts/submission")
    args = p.parse_args()
    cfg = settings(args)
    {"policy": write_policy, "preflight": preflight, "freeze": freeze, "test": run_test_once}[args.action](
        cfg, args.submission_root
    )


if __name__ == "__main__":
    main()

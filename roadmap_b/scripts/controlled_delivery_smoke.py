"""Provisional intended-delivery analysis; final listener interpretation reuses saved outputs."""

import itertools
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from src.b_data import parser, settings, verify_baseline
from src.b_evaluate import verify_freeze
from src.constants import LABELS
from src.context import Turn
from src.delivery import readiness, summarize
from src.evaluate import probabilities
from src.recording_kit import planned, records
from src.utils import digest, environment, setup, write_json

BANNER = (
    "Provisional analysis using speaker-intended delivery labels. Independent listener validation "
    "(L01) is pending; these results are not human-validated evidence."
)
# Fixed from the phrase manifest before any model output is inspected.
CATEGORY = {
    "warmly pleased": "joy",
    "irritated or annoyed": "anger",
    "disappointed and subdued": "sadness",
    "willing and friendly": None,
    "reluctant and irritated": "anger",
    "hesitant or apprehensive": "fear",
    "calm and matter-of-fact": "neutral",
    "hurt or sad": "sadness",
    "defensive and irritated": "anger",
    "pleasantly surprised": "surprise",
    "skeptical and annoyed": "anger",
    "worried or apprehensive": "fear",
    "pleased acceptance": "joy",
    "sad resignation": "sadness",
    "irritated acceptance": "anger",
    "angry disbelief": "anger",
    "sad disbelief": "sadness",
    "warm appreciation": None,
    "hurt or disappointed": "sadness",
    "irritated or resentful": "anger",
    "relaxed acceptance": None,
    "irritated dismissal": "anger",
    "neutral, ordinary volume": "neutral",
    "neutral, slightly softer": "neutral",
    "neutral, slightly slower": "neutral",
}
NAMES = ("text", "audio", "A", "B")


def accepted(root):
    root = Path(root).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    expected = planned(protocol)
    values = records(root)
    if len(values) != len(expected) or {r["recording_id"] for r in values} != set(expected):
        raise ValueError("All planned accepted recordings are required; no output-based subset.")
    hashes = {"protocol.json": digest(root / "protocol.json")}
    for row in values:
        for field, value in expected[row["recording_id"]].items():
            if row[field] != value:
                raise ValueError(f"Accepted metadata differs from protocol: {field}")
        if not all(row.get(k) is True for k in ("speaker_consented", "human_recording", "words_confirmed")):
            raise ValueError("Accepted consenting human recordings are required.")
        audio = (root / row["audio"]).resolve()
        if not audio.is_relative_to(root) or digest(audio) != row["waveform_sha256"]:
            raise ValueError("Accepted waveform changed or lies outside collection.")
        for path in (audio, root / "records" / (row["recording_id"] + ".json")):
            hashes[str(path.relative_to(root))] = digest(path)
    return values, hashes


def policy_for(values):
    mappings = {}
    for row in values:
        delivery = row["intended_delivery"]
        if delivery not in CATEGORY:
            raise ValueError(f"Unplanned delivery: {delivery}")
        mappings[delivery] = {
            "category": CATEGORY[delivery],
            "rationale": (
                "No unique emotion target: pragmatic/social stance alone does not establish a MELD category."
                if CATEGORY[delivery] is None
                else "Temporary coarse interpretation of the explicit intended affect; not a listener label."
            ),
        }
    return {
        "banner": BANNER,
        "labels": list(LABELS),
        "mapping": mappings,
        "scope": "All accepted recordings; within-speaker/phrase pairs only; no training or selection.",
        "pair_order": "lexicographic delivery ID; all three unordered pairs in each set",
        "direction": "D=(p2[c2]-p2[c1])-(p1[c2]-p1[c1]); positive is intention-consistent only",
        "sign_epsilon": 1e-12,
        "effect_size": "Always retain magnitude; no practical-significance claim from a positive sign.",
        "stable_controls": "Manifest p09/p10; expected neutral category, not listener-confirmed stability.",
        "unmapped": "Retain all distributions and TV; directional category analysis null.",
        "text_invariance_atol": 2e-5,
        "final_controlled_delivery_requirement_satisfied": False,
    }


def pair_metrics(p1, p2, c1, c2):
    p1, p2 = np.asarray(p1), np.asarray(p2)
    result = {
        "posterior_tv": float(np.abs(p2 - p1).sum() / 2),
        "argmaxes": [LABELS[int(p1.argmax())], LABELS[int(p2.argmax())]],
        "argmax_changed": bool(p1.argmax() != p2.argmax()),
        "intended_category_changes": None,
        "directional_contrast_change": None,
        "direction": "unmapped_or_same_category",
        "both_targets_increase_for_their_own_delivery": None,
    }
    if c1 in LABELS and c2 in LABELS:
        i, j = LABELS.index(c1), LABELS.index(c2)
        result["intended_category_changes"] = {
            "first_category": c1,
            "second_category": c2,
            "p_first_category_first": float(p1[i]),
            "p_first_category_second": float(p2[i]),
            "p_second_category_first": float(p1[j]),
            "p_second_category_second": float(p2[j]),
            "first_category_second_minus_first": float(p2[i] - p1[i]),
            "second_category_second_minus_first": float(p2[j] - p1[j]),
        }
        if c1 != c2:
            delta = float((p2[j] - p2[i]) - (p1[j] - p1[i]))
            result["directional_contrast_change"] = delta
            result["direction"] = "aligned" if delta > 1e-12 else "opposite" if delta < -1e-12 else "zero"
            result["both_targets_increase_for_their_own_delivery"] = bool(p1[i] > p2[i] and p2[j] > p1[j])
    return result


def provisional_summary(outputs):
    groups = {}
    for rid, row in outputs.items():
        meta = row["metadata"]
        groups.setdefault((meta["speaker"], meta["phrase_id"]), []).append(rid)
    sets, pairs = [], []
    for (speaker, phrase), ids in sorted(groups.items()):
        ids.sort()
        signatures = {
            json.dumps([outputs[i]["metadata"]["text"], outputs[i]["metadata"]["history"]]) for i in ids
        }
        if len(signatures) != 1:
            raise ValueError("Text/context not fixed within delivery set.")
        current = []
        for first, second in itertools.combinations(ids, 2):
            a, b = outputs[first], outputs[second]
            c1, c2 = a["intended_category"], b["intended_category"]
            row = {
                "first": first,
                "second": second,
                "speaker": speaker,
                "phrase_id": phrase,
                "intended_categories": [c1, c2],
                "stable_control": a["metadata"]["stable_control"],
                "direction_evaluable": c1 is not None and c2 is not None and c1 != c2,
                "models": {n: pair_metrics(a[n], b[n], c1, c2) for n in NAMES},
            }
            row["B_minus_A_TV"] = row["models"]["B"]["posterior_tv"] - row["models"]["A"]["posterior_tv"]
            current.append(row)
        maximum = max(np.abs(np.array(outputs[i]["text"]) - outputs[ids[0]]["text"]).max() for i in ids)
        sets.append(
            {
                "speaker": speaker,
                "phrase_id": phrase,
                "text": outputs[ids[0]]["metadata"]["text"],
                "recordings": ids,
                "stable_control": outputs[ids[0]]["metadata"]["stable_control"],
                "text_max_abs_difference": float(maximum),
                "text_invariant": bool(maximum <= 2e-5),
                "mean_pairwise_tv": {
                    n: float(np.mean([p["models"][n]["posterior_tv"] for p in current])) for n in NAMES
                },
                "pairs": [p["first"] + "/" + p["second"] for p in current],
            }
        )
        pairs.extend(current)
    counts = {}
    for name in NAMES:
        directional = [p["models"][name] for p in pairs if p["direction_evaluable"]]
        stable = [p["models"][name] for p in pairs if p["stable_control"]]
        counts[name] = {
            "directional_support": len(directional),
            **{
                sign: sum(p["direction"] == sign for p in directional)
                for sign in ("aligned", "opposite", "zero")
            },
            "both_targets_increase": sum(
                p["both_targets_increase_for_their_own_delivery"] for p in directional
            ),
            "mean_directional_change": float(
                np.mean([p["directional_contrast_change"] for p in directional])
            ),
            "mean_pairwise_tv": float(np.mean([p["models"][name]["posterior_tv"] for p in pairs])),
            "stable_support": len(stable),
            "stable_argmax_changes": sum(p["argmax_changed"] for p in stable),
            "stable_mean_tv": float(np.mean([p["posterior_tv"] for p in stable])),
            "stable_max_tv": max(p["posterior_tv"] for p in stable),
        }
    return {
        "banner": BANNER,
        "human_validated": False,
        "final_requirement_satisfied": False,
        "recordings": len(outputs),
        "sets": sets,
        "pairs": pairs,
        "counts": counts,
        "all_text_sets_invariant": all(s["text_invariant"] for s in sets),
    }


def render(summary, outputs, dest):
    lines = [
        "# Controlled delivery — provisional intended-delivery smoke",
        "",
        f"> **{BANNER}**",
        "",
        "## Scope and interpretation",
        "",
        "All 60 accepted recordings are included: two speakers, ten fixed phrases, three deliveries. "
        "No recording was chosen, changed or rejected based on a model output. Text/context are fixed; "
        "comparisons are the three within-speaker pairs per phrase (60 correlated pairs, 20 sets). "
        "No benchmark accuracy is reported. More acoustic sensitivity is not automatically better.",
        "",
        "The mapping was saved before inference in intended_mapping.json. Willing/friendly, warm appreciation "
        "and relaxed acceptance remain unmapped. Other categories are temporary coarse assumptions. "
        "Neutral controls are intended stable controls, not independently validated ones.",
        "",
        "For distinct mapped targets c1/c2, D=(p2[c2]-p2[c1])-(p1[c2]-p1[c1]). "
        "D>1e-12 is intention-aligned; D<−1e-12 is opposite. This numerical tolerance is not an effect-size "
        "threshold. Both per-target changes and magnitudes are saved; D can be positive even if only one "
        "target moves appropriately. Each pair shares recordings with other pairs; counts are descriptive.",
        "",
        "## Posterior movement",
        "",
        "| Model | Mapped distinct pairs | Aligned / opposite / zero | Both targets increase | Mean D | Mean TV, all pairs |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for n, c in summary["counts"].items():
        lines.append(
            f"| {n} | {c['directional_support']} | {c['aligned']} / {c['opposite']} / {c['zero']} | "
            f"{c['both_targets_increase']} | {c['mean_directional_change']:.6f} | {c['mean_pairwise_tv']:.6f} |"
        )
    lines += [
        "",
        "A is the unchanged fixed-alpha model under canonical numerical inputs; B is adaptive seed 1337. "
        "Historical A-original batch-dependent outputs are not reintroduced. Each model uses its saved "
        "temperature; TV comparisons therefore describe deployed posterior sensitivity, not architecture "
        "alone. Unit-temperature sensitivities are also saved in uncalibrated_pairs.json as a scale diagnostic.",
        "",
        "## Intended stable controls",
        "",
        "| Model | Pairs | Argmax changes | Mean TV | Max TV |",
        "|---|---:|---:|---:|---:|",
    ]
    for n, c in summary["counts"].items():
        lines.append(
            f"| {n} | {c['stable_support']} | {c['stable_argmax_changes']} | {c['stable_mean_tv']:.6f} | {c['stable_max_tv']:.6f} |"
        )
    lines += [
        "",
        "## Every same-text set",
        "",
        "| Speaker / phrase | Exact text | Intended stable | Text invariant | Text TV | Audio TV | A TV | B TV |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for s in summary["sets"]:
        v = s["mean_pairwise_tv"]
        lines.append(
            f"| {s['speaker']} / {s['phrase_id']} | {s['text']} | {s['stable_control']} | "
            f"{s['text_invariant']} | {v['text']:.6f} | {v['audio']:.6f} | {v['A']:.6f} | {v['B']:.6f} |"
        )
    lines += [
        "",
        "## All directional failures and unchanged contrasts",
        "",
        "These are failures relative to intended-category assumptions, not listener-confirmed mistakes. "
        "Unmapped pairs are excluded from this sign assessment but retained in pairs.json.",
        "",
    ]
    for p in summary["pairs"]:
        if p["direction_evaluable"] and p["models"]["B"]["direction"] != "aligned":
            a, b = p["models"]["A"], p["models"]["B"]
            lines.append(
                f"- {p['first']} / {p['second']}, targets {p['intended_categories']}: "
                f"B D={b['directional_contrast_change']:+.6f} ({b['direction']}), "
                f"A D={a['directional_contrast_change']:+.6f}; B TV={b['posterior_tv']:.6f}."
            )
    lines += [
        "",
        "## Every pair, including stable and unmapped cases",
        "",
        "| Pair | Targets | Text TV | Audio TV | A TV | B TV | A D | B D | B direction |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for p in summary["pairs"]:
        v = p["models"]

        def d(n):
            return (
                "n/a"
                if v[n]["directional_contrast_change"] is None
                else f"{v[n]['directional_contrast_change']:+.6f}"
            )

        lines.append(
            f"| {p['first']} / {p['second']} | {p['intended_categories']} | {v['text']['posterior_tv']:.6f} | "
            f"{v['audio']['posterior_tv']:.6f} | {v['A']['posterior_tv']:.6f} | {v['B']['posterior_tv']:.6f} | "
            f"{d('A')} | {d('B')} | {v['B']['direction']} |"
        )
    lines += [
        "",
        "## Artifacts and limits",
        "",
        "- outputs.json / recordings.jsonl: every metadata item and waveform hash; text/audio/A/B full "
        "distributions, logits, argmaxes, intended-category probabilities, gate, TV influence, residual norms and timing.",
        "- pairs.json / sets.json / summary.json: all paired distances, both target-probability changes, "
        "direction signs and stability, with no favorable-case filtering.",
        "- recording_lock.json / integrity.json: accepted input hashes, frozen model/policy hashes and local hardware.",
        "- No training, tuning, test inference, re-recording, selection change or final freeze occurs. "
        "The public listener UI is unchanged; keep this provisional report away from L01 until ratings are complete.",
        "- Intended labels may disagree with perceived emotion. Room/microphone, timing, breath and recording "
        "variation remain confounds. No prosodic causality or human-validated performance claim follows.",
        "- When L01 finishes, the listener action verifies the same input/output/model hashes and calls the "
        "existing blinded-judgment interpretation on saved outputs. It does not reload encoders or run inference.",
        "",
    ]
    (dest / "CONTROLLED_DELIVERY_PROVISIONAL_SMOKE.md").write_text("\n".join(lines))


def model_lock(cfg):
    root = Path(cfg["output_root"])
    paths = [
        Path(cfg["baseline_lock"]),
        root / "model_freeze.json",
        root / "canonical_policy.json",
        root / "text_policy.json",
        root / "runs/adaptive/1337/checkpoint.pt",
        root / "evaluation/adaptive/1337/calibration.json",
        root.parent / "submission/FINAL_INFERENCE_POLICY.json",
    ]
    return {str(p.resolve()): digest(p) for p in paths}


def run(cfg, root, dest):
    # Importing the inference class is deliberately confined to this action.
    from src.b_inference import RoadmapB

    root, dest = Path(root).resolve(), Path(dest).resolve()
    values, hashes = accepted(root)
    mapped = policy_for(values)
    a_lock = verify_baseline(cfg)
    verify_freeze(cfg)
    frozen = model_lock(cfg)
    dest.mkdir(parents=True, exist_ok=False)
    write_json(dest / "recording_lock.json", hashes)
    write_json(dest / "intended_mapping.json", mapped)
    write_json(
        dest / "STARTED.json",
        {
            "scope": "Provisional intended labels only; L01 pending; no final requirement satisfied",
            "model_lock": frozen,
            "script_sha256": digest(__file__),
            "selected_seed": 1337,
        },
    )
    setup(1337)
    start = time.perf_counter()
    model = RoadmapB(cfg, variant="adaptive", seed=1337)
    load_s = time.perf_counter() - start
    first = values[0]
    wave, rate = sf.read(root / first["audio"], dtype="float32")
    model.predict_turn(
        first["text"], wave, rate, [Turn(**t) for t in first["history"]], speaker=first["speaker"]
    )
    outputs, seen = {}, {}
    for row in values:
        wave, rate = sf.read(root / row["audio"], dtype="float32")
        if (
            wave.ndim != 1
            or len(wave) != row["num_samples"]
            or rate != row["sample_rate"]
            or not np.isfinite(wave).all()
        ):
            raise ValueError("Decoded accepted WAV differs from frozen metadata.")
        state, internal = model.predict_turn(
            row["text"],
            wave,
            rate,
            [Turn(**t) for t in row["history"]],
            speaker=row["speaker"],
            return_internal=True,
        )
        if not state["audio_available"]:
            raise ValueError("A planned short accepted recording is unexpectedly audio-ineligible.")
        zt, za = internal["zt"][0], internal["za"][0]
        za_fixed = (1 - model.selection["alpha"]) * zt + model.selection["alpha"] * za
        distributions = {
            "text": probabilities(zt, model.calibration["text"]).tolist(),
            "audio": probabilities(za, model.calibration["audio"]).tolist(),
            "A": probabilities(za_fixed, model.calibration["fusion"]).tolist(),
            "B": [state["distribution"][label] for label in LABELS],
        }
        signature = json.dumps([row["text"], row["history"]], sort_keys=True)
        if signature in seen and not np.allclose(distributions["text"], seen[signature], atol=2e-5, rtol=0):
            raise ValueError("Fixed transcript/context text inference is not invariant.")
        seen[signature] = distributions["text"]
        category = mapped["mapping"][row["intended_delivery"]]["category"]
        logits = {
            "text": zt.tolist(),
            "audio": za.tolist(),
            "A": za_fixed.tolist(),
            "B": internal["logits"][0].tolist(),
        }
        value = {
            "metadata": row,
            **distributions,
            "labels": list(LABELS),
            "state": state,
            "text_logits": logits["text"],
            "audio_logits": logits["audio"],
            "A_logits": logits["A"],
            "B_logits": logits["B"],
            "uncalibrated_distributions": {n: probabilities(z).tolist() for n, z in logits.items()},
            "argmaxes": {n: LABELS[int(np.argmax(p))] for n, p in distributions.items()},
            "intended_category": category,
            "intended_category_probabilities": {
                n: p[LABELS.index(category)] if category is not None else None
                for n, p in distributions.items()
            },
            "waveform_sha256": row["waveform_sha256"],
            "human_validated": False,
        }
        outputs[row["recording_id"]] = value
        with (dest / "recordings.jsonl").open("a") as stream:
            stream.write(json.dumps({"recording_id": row["recording_id"], **value}, allow_nan=False) + "\n")
    if accepted(root)[1] != hashes or model_lock(cfg) != frozen:
        raise ValueError("Input/model evidence changed during inference.")
    verify_baseline(cfg)
    verify_freeze(cfg)
    summary = provisional_summary(outputs)
    write_json(dest / "outputs.json", outputs)
    write_json(dest / "pairs.json", summary["pairs"])
    write_json(dest / "sets.json", summary["sets"])
    write_json(dest / "summary.json", summary)
    raw_outputs = {rid: {**v, **v["uncalibrated_distributions"]} for rid, v in outputs.items()}
    write_json(dest / "uncalibrated_pairs.json", provisional_summary(raw_outputs)["pairs"])
    write_json(
        dest / "integrity.json",
        {
            "environment": environment(model.cfg),
            "A_integrity": a_lock,
            "model_files_sha256": frozen,
            "recording_sha256": hashes,
            "load_seconds": load_s,
            "parameter_ledger": model.parameter_ledger(),
            "scope": "Local real raw recordings; canonical singleton FP32 text/BF16 audio; no training or test inference",
            "timing": "One warm-up then 60 raw predictions. Model clocks exclude WAV disk reads and recording duration. "
            "This is the local GPU, not a new RTX 4090 benchmark.",
            "code_addition": "Only this standalone diagnostic utility and focused tests; frozen src modules unchanged.",
            "final_preflight_note": "New tooling source must be included in later current-source preflight; no freeze now.",
        },
    )
    render(summary, outputs, dest)
    write_json(
        dest / "COMPLETED.json",
        {
            "outputs_sha256": digest(dest / "outputs.json"),
            "recording_lock_sha256": digest(dest / "recording_lock.json"),
            "integrity_sha256": digest(dest / "integrity.json"),
            "mapping_sha256": digest(dest / "intended_mapping.json"),
            "summary_sha256": digest(dest / "summary.json"),
            "only_provisional": True,
            "final_requirement_satisfied": False,
        },
    )
    print(
        json.dumps({"output": str(dest), "recordings": len(outputs), "counts": summary["counts"]}, indent=2)
    )


def listener(root, smoke, dest):
    """Interpret saved predictions only; never construct or invoke a model."""
    root, smoke, dest = Path(root).resolve(), Path(smoke).resolve(), Path(dest).resolve()
    if not readiness(root)["complete"]:
        raise ValueError(
            "Independent blinded listener pass is incomplete; provisional analysis cannot satisfy it."
        )
    completed = json.loads((smoke / "COMPLETED.json").read_text())
    for field, file in (
        ("outputs_sha256", "outputs.json"),
        ("recording_lock_sha256", "recording_lock.json"),
        ("integrity_sha256", "integrity.json"),
        ("mapping_sha256", "intended_mapping.json"),
    ):
        if digest(smoke / file) != completed[field]:
            raise ValueError("Provisional saved evidence changed.")
    if accepted(root)[1] != json.loads((smoke / "recording_lock.json").read_text()):
        raise ValueError("Accepted recording evidence changed; cannot reuse predictions.")
    provenance = json.loads((smoke / "integrity.json").read_text())
    for path, expected in provenance["model_files_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Frozen model/policy identity changed.")
    outputs = json.loads((smoke / "outputs.json").read_text())
    # Discard provisional category annotations before final listener interpretation.
    allowed = {
        "metadata",
        "text",
        "audio",
        "A",
        "B",
        "state",
        "text_logits",
        "audio_logits",
        "A_logits",
        "B_logits",
    }
    clean = {rid: {k: v for k, v in row.items() if k in allowed} for rid, row in outputs.items()}
    summary = summarize(root, clean)
    dest.mkdir(parents=True, exist_ok=False)
    write_json(dest / "outputs.json", clean)
    write_json(dest / "summary.json", summary)
    write_json(
        dest / "integrity.json",
        {
            "prediction_source": str(smoke),
            "prediction_source_sha256": completed["outputs_sha256"],
            "model_inference_rerun": False,
            "original_inference_provenance": provenance,
            "recording_and_rating_sha256": {
                str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()
            },
            "scope": "Independent blinded-listener interpretation of unchanged frozen predictions; no tuning/test",
        },
    )
    lines = [
        "# Same words / different delivery — independent listener diagnostic",
        "",
        summary["interpretation"],
        "",
        summary["listener_procedure"],
        "",
        "Saved frozen predictions were reused; only interpretation against completed blinded judgments ran. "
        "Intended-category assumptions were removed from final analysis. Human judgments remain subjective, "
        "not objective internal-state ground truth.",
        "",
        f"Collection: {summary['collection']}",
        "",
        "| Model | Listener-distinct pairs | Directionally aligned | Listener-stable pairs | Stable label changes |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, c in summary["counts"].items():
        lines.append(
            f"| {name} | {c['listener_distinct_support']} | {c['posterior_moves_toward_listener_contrast']} | "
            f"{c['listener_stable_support']} | {c['undesired_label_changes_on_stable_pairs']} |"
        )
    lines += [
        "",
        "## Every listener pair, including failures",
        "",
        "| Pair | Listener categories | Difference | A D / TV | B D / TV |",
        "|---|---|---|---|---|",
    ]
    for c in summary["cases"]:
        a, b = c["models"]["A"], c["models"]["B"]
        lines.append(
            f"| {c['first']} / {c['second']} | {c['listener_categories']} | {c['listener_difference']} | "
            f"{a['directional_contrast_change']} / {a['posterior_tv']:.6f} | "
            f"{b['directional_contrast_change']} / {b['posterior_tv']:.6f} |"
        )
    lines += [
        "",
        "Full text/A/B comparisons, directional eligibility, stability and agreement are in summary.json. "
        "Unclear or same-category judgments are not silently scored as directional failures. "
        "No model selection, calibration or freeze action was taken.",
        "",
    ]
    (dest / "DELIVERY_DIAGNOSTIC.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(dest), "inference_rerun": False, "counts": summary["counts"]}, indent=2))


def main():
    p = parser(__doc__)
    p.add_argument("action", choices=["run", "listener"])
    p.add_argument("--recordings-root", default="artifacts/submission/controlled_delivery")
    p.add_argument("--smoke-output", default="artifacts/submission/controlled_delivery_provisional_smoke")
    p.add_argument("--listener-output", default="artifacts/submission/delivery_results")
    args = p.parse_args()
    if args.action == "run":
        run(settings(args), args.recordings_root, args.smoke_output)
    else:
        listener(args.recordings_root, args.smoke_output, args.listener_output)


if __name__ == "__main__":
    main()

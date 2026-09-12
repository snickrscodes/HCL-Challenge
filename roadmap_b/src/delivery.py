"""Post-selection human delivery diagnostic; no training, selection or test-set access."""

import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from .b_data import parser as b_parser
from .b_data import settings, verify_baseline
from .b_evaluate import verify_freeze
from .b_inference import RoadmapB
from .constants import LABELS
from .context import Turn
from .evaluate import probabilities
from .recording_kit import listener_plan, planned, records
from .utils import digest, environment, write_json


def majority(values):
    counted = Counter(values)
    if not counted:
        return None
    top = counted.most_common()
    return top[0][0] if top[0][1] > sum(counted.values()) / 2 else None


def readiness(root):
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text())
    values = records(root)
    expected = planned(protocol)
    result = {
        "expected_recordings": len(expected),
        "recordings": len(values),
        "speakers": sorted({r["speaker"] for r in values}),
        "phrases": len({r["phrase_id"] for r in values}),
        "complete": False,
        "complete_listeners": [],
    }
    if {r["recording_id"] for r in values} != set(expected):
        result["reason"] = "Consented recordings are incomplete."
        return result
    for r in values:
        if r["speaker_consented"] is not True or r.get("human_recording") is not True:
            raise ValueError("Real consenting human recordings are required.")
        if digest(root / r["audio"]) != r["waveform_sha256"]:
            raise ValueError("Accepted waveform changed.")
    public, _ = listener_plan(root, protocol)
    tokens = {r["token"] for r in public["clips"]} | {r["pair_token"] for r in public["pairs"]}
    for folder in sorted((root / "ratings").glob("L*")):
        ratings = [json.loads(p.read_text()) for p in folder.glob("*.json")]
        if {r["token"] for r in ratings} == tokens and all(r["independent_listener"] for r in ratings):
            result["complete_listeners"].append(folder.name)
    result["complete"] = len(result["complete_listeners"]) >= protocol["required_independent_listeners"]
    result["reason"] = None if result["complete"] else "Independent blinded listener pass is incomplete."
    return result


def summarize(root, outputs):
    root = Path(root)
    status = readiness(root)
    if not status["complete"]:
        raise ValueError(status["reason"])
    public, private = listener_plan(root, json.loads((root / "protocol.json").read_text()))
    ratings = [
        json.loads(p.read_text())
        for listener in status["complete_listeners"]
        for p in (root / "ratings" / listener).glob("*.json")
    ]
    clip_votes = {
        c["token"]: [r["category"] for r in ratings if r["kind"] == "clip" and r["token"] == c["token"]]
        for c in public["clips"]
    }
    pair_votes = {
        c["pair_token"]: [
            r["meaningfully_different"]
            for r in ratings
            if r["kind"] == "pair" and r["token"] == c["pair_token"]
        ]
        for c in public["pairs"]
    }
    agreement = []
    for votes in clip_votes.values():
        valid = [v for v in votes if v in LABELS]
        agreement.extend(a == b for a, b in itertools.combinations(valid, 2))
    cases = []
    for pair in public["pairs"]:
        first = private["token_to_recording"][pair["first"]]
        second = private["token_to_recording"][pair["second"]]
        c1, c2 = majority(clip_votes[pair["first"]]), majority(clip_votes[pair["second"]])
        difference = majority(pair_votes[pair["pair_token"]])
        eligible = difference == "yes" and c1 in LABELS and c2 in LABELS and c1 != c2
        stable = difference == "no" and c1 in LABELS and c1 == c2
        case = {
            "pair_token": pair["pair_token"],
            "first": first,
            "second": second,
            "listener_categories": [c1, c2],
            "listener_difference": difference,
            "direction_evaluable": eligible,
            "listener_stable": stable,
            "models": {},
        }
        for name in ("text", "A", "B"):
            p1 = np.array(outputs[first][name])
            p2 = np.array(outputs[second][name])
            delta = None
            if eligible:
                i, j = LABELS.index(c1), LABELS.index(c2)
                delta = float((p2[j] - p2[i]) - (p1[j] - p1[i]))
            case["models"][name] = {
                "directional_contrast_change": delta,
                "moves_toward_listener_contrast": delta > 0 if delta is not None else None,
                "posterior_tv": float(0.5 * np.abs(p1 - p2).sum()),
                "argmax_changed": bool(p1.argmax() != p2.argmax()),
                "argmaxes": [LABELS[int(p1.argmax())], LABELS[int(p2.argmax())]],
            }
        cases.append(case)
    counts = {}
    for name in ("text", "A", "B"):
        distinct = [c for c in cases if c["direction_evaluable"]]
        stable = [c for c in cases if c["listener_stable"]]
        counts[name] = {
            "listener_distinct_support": len(distinct),
            "posterior_moves_toward_listener_contrast": sum(
                c["models"][name]["moves_toward_listener_contrast"] for c in distinct
            ),
            "listener_stable_support": len(stable),
            "undesired_label_changes_on_stable_pairs": sum(
                c["models"][name]["argmax_changed"] for c in stable
            ),
        }
    return {
        "collection": status,
        "listener_procedure": "Blinded independent clip categories followed by same-speaker/same-phrase contrast judgments; no intended delivery or model outputs displayed.",
        "interlistener_category_agreement": float(np.mean(agreement)) if agreement else None,
        "interlistener_comparison_support": len(agreement),
        "interpretation": "Intention is not ground truth. Correlated delivery pairs are descriptive diagnostics, not a benchmark or independent statistical trials.",
        "counts": counts,
        "cases": cases,
    }


def run(cfg, root, dest):
    root, dest = Path(root).resolve(), Path(dest).resolve()
    ready = readiness(root)
    if not ready["complete"]:
        raise ValueError(ready["reason"])
    verify_baseline(cfg)
    verify_freeze(cfg)
    dest.mkdir(parents=True, exist_ok=False)
    model = RoadmapB(cfg, variant="adaptive", seed=1337)
    # Warm-up is explicit and uses an existing real recording, never a fabricated speech example.
    values = records(root)
    first = values[0]
    wave, rate = sf.read(root / first["audio"], dtype="float32")
    model.predict_turn(
        first["text"], wave, rate, [Turn(**v) for v in first["history"]], speaker=first["speaker"]
    )
    outputs = {}
    seen_text = {}
    for row in values:
        wave, rate = sf.read(root / row["audio"], dtype="float32")
        state, internal = model.predict_turn(
            row["text"],
            wave,
            rate,
            [Turn(**v) for v in row["history"]],
            speaker=row["speaker"],
            return_internal=True,
        )
        zt, za = internal["zt"][0], internal["za"][0]
        pt = probabilities(zt, model.calibration["text"])
        pa = probabilities(za, model.calibration["audio"])
        alpha = model.selection["alpha"]
        pf = (
            probabilities((1 - alpha) * zt + alpha * za, model.calibration["fusion"])
            if state["audio_available"]
            else pt
        )
        signature = json.dumps([row["text"], row["history"]], sort_keys=True)
        if signature in seen_text and not np.allclose(pt, seen_text[signature], atol=2e-5, rtol=0):
            raise ValueError(
                "Fixed text changed across recordings; investigate before interpreting delivery."
            )
        seen_text[signature] = pt
        outputs[row["recording_id"]] = {
            "metadata": row,
            "text": pt.tolist(),
            "audio": pa.tolist(),
            "A": pf.tolist(),
            "B": list(state["distribution"].values()),
            "state": state,
            "text_logits": zt.tolist(),
            "audio_logits": za.tolist(),
            "B_logits": internal["logits"][0].tolist(),
        }
    summary = summarize(root, outputs)
    evidence = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}
    write_json(dest / "outputs.json", outputs)
    write_json(dest / "summary.json", summary)
    write_json(
        dest / "integrity.json",
        {
            "recording_and_rating_sha256": evidence,
            "environment": environment(model.cfg),
            "scope": "Post-selection, unchanged models, no training/calibration; test untouched",
        },
    )
    lines = [
        "# Same words / different delivery",
        "",
        summary["interpretation"],
        "",
        summary["listener_procedure"],
        "",
        f"{ready['recordings']} recordings; {ready['phrases']} phrases; speakers {', '.join(ready['speakers'])}; complete listeners {', '.join(ready['complete_listeners'])}.",
        "",
        "| Model | Listener-distinct contrasts | Directionally aligned posterior movements | Listener-stable pairs | Label changes on stable pairs |",
        "|---|---:|---:|---:|---:|",
    ]
    for n, c in summary["counts"].items():
        lines.append(
            f"| {n} | {c['listener_distinct_support']} | {c['posterior_moves_toward_listener_contrast']} | {c['listener_stable_support']} | {c['undesired_label_changes_on_stable_pairs']} |"
        )
    lines += ["", "## Failures and inconclusive cases", ""]
    for c in summary["cases"]:
        if (c["direction_evaluable"] and not c["models"]["B"]["moves_toward_listener_contrast"]) or (
            c["listener_stable"] and c["models"]["B"]["argmax_changed"]
        ):
            lines.append(
                f"- {c['first']} / {c['second']}: listeners {c['listener_categories']}, difference={c['listener_difference']}; B {c['models']['B']}."
            )
    lines += [
        "",
        f"Contrasts not direction-evaluable: {sum(not c['direction_evaluable'] for c in summary['cases'])}. Stable or unclear judgments are not silently counted as directional failures or successes.",
        "",
        "All metadata, distributions, argmaxes, gate, actual influence, residual norms and timings are retained in outputs.json. No tuning uses these results.",
    ]
    (dest / "DELIVERY_DIAGNOSTIC.md").write_text("\n".join(lines) + "\n")
    verify_baseline(cfg)


def main():
    p = b_parser(__doc__)
    p.add_argument("action", choices=["status", "run"])
    p.add_argument("--recordings-root", default="artifacts/submission/controlled_delivery")
    p.add_argument("--diagnostic-output", default="artifacts/submission/delivery_results")
    args = p.parse_args()
    if args.action == "status":
        print(json.dumps(readiness(Path(args.recordings_root)), indent=2))
    else:
        run(settings(args), args.recordings_root, args.diagnostic_output)


if __name__ == "__main__":
    main()

"""Reproducible development evidence figures from saved predictions only; no inference."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.constants import LABELS

p = argparse.ArgumentParser(__doc__)
p.add_argument("--baseline-root", required=True)
p.add_argument("--b-root", default="artifacts/roadmap_b")
p.add_argument("--output-root", default="artifacts/submission/figures")
a = p.parse_args()
base = Path(a.baseline_root)
broot = Path(a.b_root)
out = Path(a.output_root)
out.mkdir(parents=True, exist_ok=True)
inputs = {}


def read(path):
    path = Path(path)
    inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def rows(path):
    path = Path(path)
    inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return [json.loads(s) for s in path.read_text().splitlines()]


def table(name, values):
    with (out / (name + ".csv")).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader()
        w.writerows(values)


def finish(fig, name, caption):
    fig.savefig(out / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out / (name + ".svg"), bbox_inches="tight")
    (out / (name + ".txt")).write_text(caption + "\n")
    plt.close(fig)


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titleweight": "bold",
        "svg.hashsalt": "meld-final",
    }
)
colors = ["#666f79", "#d39333", "#2367a6"]
refs = read(broot / "evaluation/references.json")
summary = read(broot / "evaluation/summary.json")
runs = {
    f"{v}/{s}": read(broot / f"evaluation/{v}/{s}/metrics.json")
    for v in ("bias", "constant", "adaptive", "text_only")
    for s in (1337, 1338, 1339)
}
progress = []
for label, name in [
    ("Current text", "text_current"),
    ("Causal text", "text_context"),
    ("Audio", "audio"),
    ("Concat", "concat"),
    ("Fixed A", "late_fusion"),
]:
    m = read(base / f"artifacts/evaluation/dev_model/{name}/metrics.json")
    progress.append({"model": label, **{k: m[k] for k in ("macro_f1", "weighted_f1", "accuracy")}})
progress.append(
    {
        "model": "Adaptive B (1337)",
        **{k: runs["adaptive/1337"]["metrics"][k] for k in ("macro_f1", "weighted_f1", "accuracy")},
    }
)
table("A_progression", progress)
fig, ax = plt.subplots(figsize=(10, 5))
y = np.arange(len(progress))
for i, m in enumerate(("macro_f1", "weighted_f1", "accuracy")):
    ax.barh(
        y + (i - 1) * 0.24, [r[m] for r in progress], height=0.23, label=m.replace("_", " "), color=colors[i]
    )
ax.set(
    yticks=y,
    yticklabels=[r["model"] for r in progress],
    xlim=(0, 1),
    xlabel="Score",
    title="Development progression",
)
ax.invert_yaxis()
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.15)
finish(
    fig,
    "A_progression",
    "dev_model only. A rows are historical development results; B uses canonical singleton FP32 text/BF16 audio. Audio support is the eligible subset; other rows retain missing audio. The numerical-policy references remain distinct.",
)
control = []
names = {
    "bias": "Bias only",
    "constant": "Constant residual",
    "adaptive": "Adaptive residual",
    "text_only": "Matched text only",
}
fig, ax = plt.subplots(figsize=(9, 4.7))
for i, (v, label) in enumerate(names.items()):
    vals = np.array([runs[f"{v}/{s}"]["metrics"]["macro_f1"] for s in (1337, 1338, 1339)])
    ax.scatter(
        vals, [i - 0.08, i, i + 0.08], s=25, color=colors[2] if v == "adaptive" else colors[0], alpha=0.7
    )
    ax.errorbar(
        vals.mean(),
        i,
        xerr=vals.std(ddof=1),
        fmt="D",
        color=colors[2] if v == "adaptive" else colors[0],
        capsize=4,
    )
    control.append(
        {
            "variant": v,
            "macro_f1_mean": vals.mean(),
            "macro_f1_std": vals.std(ddof=1),
            **{f"seed_{s}": runs[f"{v}/{s}"]["metrics"]["macro_f1"] for s in (1337, 1338, 1339)},
        }
    )
ax.axvline(refs["A_replay_canonical"]["macro_f1"], color=colors[1], ls="--", label="Fixed A canonical")
ax.set(
    yticks=np.arange(4),
    yticklabels=list(names.values()),
    xlabel="Macro F1 (expanded scale)",
    title="Extra capacity alone does not explain the development gain",
)
ax.invert_yaxis()
ax.legend()
ax.grid(axis="x", alpha=0.15)
table("B_controls", control)
finish(
    fig,
    "B_controls",
    "Dots are three frozen-encoder head seeds; diamonds ±1 sample standard deviation. This is not encoder-seed robustness or a confidence interval.",
)
conditions = {
    "matched": "Matched",
    "shuffle": "Shuffled (10 means)",
    "training_mean_audio": "Training-mean audio",
    "same_speaker": "Same-speaker swap",
    "training_mean_gate": "Training-mean gate",
    "gate_zero": "Gate zero",
}
interventions = []
fig, ax = plt.subplots(figsize=(9, 4.7))
for i, (name, label) in enumerate(conditions.items()):
    vals = []
    for seed in (1337, 1338, 1339):
        r = runs[f"adaptive/{seed}"]
        iv = r["interventions"]
        vals.append(
            r["metrics"]["macro_f1"]
            if name == "matched"
            else iv["shuffle_summary"]["macro_f1_mean"]
            if name == "shuffle"
            else iv[name]["metrics"]["macro_f1"]
        )
    ax.errorbar(
        np.mean(vals),
        i,
        xerr=np.std(vals, ddof=1),
        fmt="o",
        capsize=4,
        color=colors[2] if name == "matched" else colors[0],
    )
    interventions.append(
        {"condition": name, "macro_f1_mean": np.mean(vals), "head_seed_std": np.std(vals, ddof=1)}
    )
ax.set(
    yticks=np.arange(len(conditions)),
    yticklabels=list(conditions.values()),
    xlabel="Macro F1 (expanded scale)",
    title="Matched audio versus fixed interventions",
)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.15)
table("C_interventions", interventions)
finish(
    fig,
    "C_interventions",
    "Three head-seed means ±1 sample SD. Each shuffled value first averages the 10 predeclared derangements. Matched h_a and z_a always move together. These interventions do not isolate prosody.",
)
classes = []
fig, ax = plt.subplots(figsize=(10, 5))
y = np.arange(7)
for name in LABELS:
    aa = refs["A_replay_canonical"]["per_class"][name]
    bb = runs["adaptive/1337"]["metrics"]["per_class"][name]
    classes.append(
        {
            "emotion": name,
            "support": bb["support"],
            "A_f1": aa["f1"],
            "B_f1": bb["f1"],
            "B_minus_A": bb["f1"] - aa["f1"],
        }
    )
ax.barh(y - 0.18, [r["A_f1"] for r in classes], 0.34, color=colors[0], label="A canonical")
ax.barh(y + 0.18, [r["B_f1"] for r in classes], 0.34, color=colors[2], label="B seed 1337")
for i, r in enumerate(classes):
    ax.text(max(r["A_f1"], r["B_f1"]) + 0.015, i, f"Δ {r['B_minus_A']:+.3f}", va="center", fontsize=10)
ax.set(
    yticks=y,
    yticklabels=LABELS,
    xlim=(0, 1),
    xlabel="Per-class F1",
    title="Class tradeoffs: all seven emotions",
)
ax.invert_yaxis()
ax.legend()
ax.grid(axis="x", alpha=0.15)
table("D_classes", classes)
finish(
    fig,
    "D_classes",
    "A canonical vs predeclared B seed 1337, dev_model. Regressions and improvements use the same scale and annotation. Fear true positives do not improve; sadness detections decline.",
)
records = rows(broot / "evaluation/adaptive/1337/predictions.jsonl")
keys = {r["key"]: i for i, r in enumerate(records)}
y = np.array([r["label"] for r in records])
pred = {
    "Text": np.array([np.argmax(r["text_logits"]) for r in records]),
    "A": np.array(
        [
            np.argmax(r["probabilities"])
            for r in rows(broot / "evaluation/A_replay_canonical_predictions.jsonl")
        ]
    ),
    "B": np.array([r["prediction"] for r in records]),
}
membership = read(broot / "evaluation/slice_membership.json")


def macro(y, p):
    cm = np.zeros((7, 7))
    np.add.at(cm, (y, p), 1)
    d = cm.sum(0) + cm.sum(1)
    return float(np.divide(2 * cm.diagonal(), d, out=np.zeros(7), where=d > 0).mean())


bins = ("lt_0.4", "0.4_to_0.6", "0.6_to_0.8", "gte_0.8")
slices = []
for b in bins:
    ix = np.array([keys[k] for k in membership[b]])
    slices.append({"bin": b, "support": len(ix), **{n: macro(y[ix], pr[ix]) for n, pr in pred.items()}})
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, name in enumerate(pred):
    ax.plot(range(4), [r[name] for r in slices], marker="o", color=colors[i], label=name)
ax.set(
    xticks=range(4),
    xticklabels=[f"{b}\n(n={r['support']})" for b, r in zip(("<.4", ".4–.6", ".6–.8", "≥.8"), slices)],
    xlabel="Frozen raw text confidence",
    ylabel="Macro F1",
    title="Predeclared text-confidence slices",
)
ax.legend()
ax.grid(alpha=0.15)
table("E_confidence_slices", slices)
finish(
    fig,
    "E_confidence_slices",
    "Slice membership comes from fixed pre-B raw text probabilities, not labels or B outcomes. Seven-class macro F1 is retained even when a small slice lacks a class; supports are shown.",
)
gate = np.array([r["gate"] for r in records])
influence = np.array([r["audio_influence"] for r in records])
available = np.array([r["audio_available"] for r in records])
ap = pred["A"]
bp = pred["B"]
fig, ax = plt.subplots(figsize=(8, 5))
groups = [
    ("Other decisions", ~(((ap != y) & (bp == y)) | ((ap == y) & (bp != y))), "#a2a9af"),
    ("Corrected A error", (ap != y) & (bp == y), "#2367a6"),
    ("Harmed A decision", (ap == y) & (bp != y), "#b74b43"),
]
for label, mask, color in groups:
    m = mask & available
    ax.scatter(gate[m], influence[m], s=19, alpha=0.6, label=label, color=color)
ax.set(
    xlim=(0, 1),
    ylim=(0, None),
    xlabel="Gate value (not attribution)",
    ylabel="Actual posterior TV influence",
    title="Gate and posterior influence are different quantities",
)
ax.legend(fontsize=9)
ax.grid(alpha=0.15)
finish(
    fig,
    "F_gate_influence",
    "Reference seed 1337, eligible dev rows. Influence is TV(softmax(z_B/T_B), softmax(z_t/T_B)) with the same T_B. Outcome colors are retrospective diagnostics, not causal attribution or a selection rule.",
)
table(
    "F_gate_influence",
    [
        {
            "key": r["key"],
            "gate": r["gate"],
            "posterior_TV": r["audio_influence"],
            "audio_available": r["audio_available"],
        }
        for r in records
    ],
)
manifest = {
    "scope": "Development only; no test inference or test-driven figure selection",
    "inputs_sha256": inputs,
    "files_sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.iterdir())
        if p.is_file() and p.name != "figure_manifest.json"
    },
}
(out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Six development figures plus CSV/SVG/PNG outputs generated from saved evidence.")

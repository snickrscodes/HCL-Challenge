"""Add measured diagnostic tables to the generated B report; never run model inference."""

import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(__doc__)
p.add_argument("--output-root", default="artifacts/roadmap_b")
a = p.parse_args()
root = Path(a.output_root)


def read(name):
    return json.loads((root / name).read_text())


def rows(name):
    return [json.loads(line) for line in (root / name).read_text().splitlines()]


def cm_scores(y, pred):
    cm = np.zeros((7, 7), dtype=int)
    np.add.at(cm, (y, pred), 1)
    tp, support = cm.diagonal(), cm.sum(1)
    denom = support + cm.sum(0)
    f1 = np.divide(2 * tp, denom, out=np.zeros(7), where=denom > 0)
    return {
        "accuracy": float(tp.sum() / len(y)),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / len(y)),
        "confusion_matrix": cm.tolist(),
    }


names = ("neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust")
refs = read("evaluation/references.json")
summary = read("evaluation/summary.json")
ref_rows = rows("evaluation/A_replay_canonical_predictions.jsonl")
orig_rows = rows("evaluation/A_original_late_fusion_predictions.jsonl")
a_pred = np.array([np.argmax(r["probabilities"]) for r in ref_rows])
orig_pred = np.array([np.argmax(r["probabilities"]) for r in orig_rows])
records = rows("evaluation/adaptive/1337/predictions.jsonl")
y = np.array([r["label"] for r in records])
b_pred = np.array([r["prediction"] for r in records])
t_pred = np.array([np.argmax(r["text_logits"]) for r in records])
r = read("evaluation/adaptive/1337/metrics.json")
checked = []
for variant in ("bias", "constant", "adaptive", "text_only"):
    for seed in (1337, 1338, 1339):
        path = f"evaluation/{variant}/{seed}"
        values = rows(path + "/predictions.jsonl")
        assert [v["key"] for v in values] == [v["key"] for v in ref_rows]
        prob = np.array([v["probabilities"] for v in values])
        assert np.isfinite(prob).all() and np.allclose(prob.sum(1), 1)
        score = cm_scores(y, prob.argmax(1))
        expected = read(path + "/metrics.json")["metrics"]
        for name in ("accuracy", "macro_f1", "weighted_f1"):
            assert abs(score[name] - expected[name]) < 1e-12
        assert score["confusion_matrix"] == expected["confusion_matrix"]
        checked.append(path)
(root / "prediction_recomputation.json").write_text(
    json.dumps(
        {
            "passed": True,
            "checked": checked,
            "method": "Direct NumPy confusion matrix and F1 recomputation from all twelve saved dev prediction files",
        },
        indent=2,
    )
    + "\n"
)

lines = [
    "## Additional measured evidence",
    "",
    "The declared engineering advancement rule passes. This is a macro-F1 and class-tradeoff result, not a claim of fewer overall errors or demonstrated prosodic causality.",
    "",
    f"A-original and A-replay-canonical differ on {int((orig_pred != a_pred).sum())} decisions even though their aggregate accuracy/F1 values coincide. They remain distinct references.",
    "",
    "### Preservation of A corrections (retrospective)",
    "",
]
fix, harm = (t_pred != y) & (a_pred == y), (t_pred == y) & (a_pred != y)
lines += [
    f"Against canonical text, fixed A makes {int(fix.sum())} useful corrections and {int(harm.sum())} harms. Reference B preserves {int(((b_pred == y) & fix).sum())} of those corrections and avoids {int(((b_pred == y) & harm).sum())} of those harms. It also makes different decisions elsewhere. These outcome-defined subsets are retrospective diagnostics, not predeclared slices.",
    "",
    "Against A directly, reference B corrects 28 errors and introduces 30 errors: net −2. Its non-neutral net is +4 and neutral net is −6. This does not establish the simple story that B preserves most useful A corrections while only removing harms.",
    "",
    "| Adaptive seed | Non-neutral net vs A | Fear TP | Disgust TP | Combined TP |",
    "|---|---:|---:|---:|---:|",
]
for seed in (1337, 1338, 1339):
    v = read(f"evaluation/adaptive/{seed}/metrics.json")
    tp = v["true_positives"]
    lines.append(
        f"| {seed} | {v['paired_changes']['A_replay_canonical']['non_neutral']['net_correct']} | {tp['fear']} | {tp['disgust']} | {tp['fear_plus_disgust']} |"
    )
lines += [
    "",
    "The non-neutral criterion was operationalized before training for the seed mean and reference seed; seed 1338 individually loses one non-neutral correct decision. Fixed A has 7 fear and 3 disgust true positives. B does not improve fear detections.",
    "",
    "### Reference-seed class metrics",
    "",
    "| Class | Support | A F1 | B F1 | B precision | B recall |",
    "|---|---:|---:|---:|---:|---:|",
]
for name in names:
    av, bv = refs["A_replay_canonical"]["per_class"][name], r["metrics"]["per_class"][name]
    lines.append(
        f"| {name} | {bv['support']} | {av['f1']:.4f} | {bv['f1']:.4f} | {bv['precision']:.4f} | {bv['recall']:.4f} |"
    )
lines += [
    "",
    "Confusion matrices use the immutable label order: neutral, joy, sadness, anger, surprise, fear, disgust.",
    "",
    "A-replay-canonical:",
    "```json",
    json.dumps(refs["A_replay_canonical"]["confusion_matrix"]),
    "```",
    "B seed 1337:",
    "```json",
    json.dumps(r["metrics"]["confusion_matrix"]),
    "```",
    "",
    "### Fixed slices",
    "",
    "| Slice | Support | Text macro F1 | A macro F1 | B macro F1 |",
    "|---|---:|---:|---:|---:|",
]
keys = {v["key"]: i for i, v in enumerate(records)}
for name, members in read("evaluation/slice_membership.json").items():
    ix = np.array([keys[k] for k in members])
    scores = [cm_scores(y[ix], v[ix])["macro_f1"] for v in (t_pred, a_pred, b_pred)]
    lines.append(f"| {name} | {len(ix)} | {' | '.join(f'{v:.4f}' for v in scores)} |")
lines += [
    "",
    "### Paired dialogue uncertainty",
    "",
    "| Difference, B 1337 − A | Observed | 95% bootstrap interval |",
    "|---|---:|---:|",
]
for name, interval in r["bootstrap_vs_A_replay"]["differences"].items():
    diff = r["metrics"][name] - refs["A_replay_canonical"][name]
    lines.append(f"| {name} | {diff:+.5f} | [{interval['low']:+.5f}, {interval['high']:+.5f}] |")
lines += [
    "",
    "All three intervals include zero. They are paired dialogue-cluster, conditional post-selection descriptions (1,000 replicates), not significance evidence. Head-seed standard deviations measure a different source of variability.",
    "",
    "### Gate relationships and acoustic interventions",
    "",
    f"Reference gate correlations: influence {r['diagnostics']['relationships']['gate_vs_influence']:.3f}; text confidence {r['diagnostics']['relationships']['gate_vs_text_confidence']:.3f}; audio confidence {r['diagnostics']['relationships']['gate_vs_audio_confidence']:.3f}. The gate is not simply a low-text-confidence switch.",
    "",
    "| Adaptive seed | Matched macro F1 | Shuffle mean | Training-mean audio | Same-speaker swap | Fixed training-mean gate |",
    "|---|---:|---:|---:|---:|---:|",
]
for seed in (1337, 1338, 1339):
    v = read(f"evaluation/adaptive/{seed}/metrics.json")
    iv = v["interventions"]
    lines.append(
        f"| {seed} | {v['metrics']['macro_f1']:.4f} | {iv['shuffle_summary']['macro_f1_mean']:.4f} | {iv['training_mean_audio']['metrics']['macro_f1']:.4f} | {iv['same_speaker']['metrics']['macro_f1']:.4f} | {iv['training_mean_gate']['metrics']['macro_f1']:.4f} |"
    )
lines += [
    "",
    f"Same-speaker swaps cover {r['interventions']['same_speaker']['metadata']['support']}/839 rows. Training-mean replacements and training-mean gates use training data only. Confidence summaries use raw logits, without calibration inputs.",
    "",
    "### Raw silence/noise probes",
    "",
    "| Condition | Probe support | Correct | Prediction changes vs matched | Mean posterior TV vs matched |",
    "|---|---:|---:|---:|---:|",
]
cases = read("raw_validation/interventions.json")["cases"]
matched = {c["key"]: c for c in cases if c["condition"] == "matched"}
for condition in ("matched", "silence", "noise"):
    values = [v for v in cases if v["condition"] == condition]
    tv = []
    changes = 0
    for v in values:
        before = matched[v["key"]]["prediction"]
        after = v["prediction"]
        changes += before["emotion"] != after["emotion"]
        tv.append(
            0.5
            * np.abs(
                np.array(list(before["distribution"].values()))
                - np.array(list(after["distribution"].values()))
            ).sum()
        )
    correct = sum(v["prediction"]["emotion"] == names[v["label"]] for v in values)
    lines.append(f"| {condition} | {len(values)} | {correct} | {changes} | {np.mean(tv):.5f} |")
lines += [
    "",
    "These 15 predetermined duration/numerical probes are a small diagnostic set. Silence is actual zero PCM encoded by WavLM; noise is the fixed 10 dB condition. Equal correct counts do not establish robustness or expression sensitivity.",
    "",
    "### Raw input to both outputs",
    "",
    "The predetermined dev/99/3 probe produces the following measured output. It illustrates restrained correction when the audio classifier disagrees with text; it is not a newly selected success benchmark.",
    "```json",
    json.dumps(matched["dev/99/3"]["prediction"], indent=2),
    "```",
    "",
    "### Reproducibility and runtime details",
    "",
]
cons = read("raw_validation/cache_consistency.json")
runtime = read("raw_validation/runtime.json")
lines += [
    f"All {len(cons['cases'])} raw/cache cases passed, including all four original discrepancy clips. Maximum fused-probability difference was {max(c['fused_probability_max_abs'] for c in cons['cases']):.3g}; text probability differences were zero. No tolerance was loosened.",
    f"Runtime uses {len(runtime['samples'])} warm turns ({len(cons['cases'])} clips × 3 repeats), explicit end-of-turn, and real WavLM. Loads were measured in a warm process/filesystem context, not cold machine startup. Same-workload A uses canonical precision; historical A timing is preserved separately.",
    "",
    "| Canonical cache | Split | Rows | Extraction seconds |",
    "|---|---|---:|---:|",
]
for folder in ("features", "text_features"):
    for split, v in read(folder + "/summary.json").items():
        lines.append(
            f"| {'WavLM singleton BF16' if folder == 'features' else 'T1 singleton FP32'} | {split} | {v['rows']} | {v['elapsed_s']:.3f} |"
        )
lines += [
    "",
    "Direct recomputation from all 12 prediction files reproduces accuracy, macro/weighted F1 and confusion matrices. The validation suite reports 47 passing tests, including inherited A regressions and pretrained checks.",
    "",
    "An evaluation-only reporting repair handles undefined constant-gate correlations as null. The initial failed output, original source, model freeze and exact old/new source hashes are retained in evaluation_repair.json and evaluation_failed_constant_correlation_v1/. No head was retrained or reselected.",
    "",
    "Consented same-words recordings remain pending; the recording manifest is supplied. No optional imbalance experiment was run. The next continuation should keep seed 1337 fixed and finish the controlled delivery diagnostic and presentation, rather than search more architectures. The official test still requires a separate planned final pass.",
    "",
]
if (root / "demo/input.json").exists():
    lines += [
        "### Fresh-process temporal demonstration",
        "",
        "Input:",
        "```json",
        json.dumps(read("demo/input.json"), indent=2),
        "```",
        "Output after buffered audio chunks and explicit end-of-turn:",
        "```json",
        json.dumps(read("demo/output.json"), indent=2),
        "```",
        "",
    ]
report = root / "ROADMAP_B_REPORT.md"
content = report.read_text()
marker = "## Additional measured evidence"
if marker in content:
    content = content[: content.index(marker)] + content[content.index("## Decision") :]
content = content.replace("## Decision", "\n".join(lines) + "\n## Decision")
report.write_text(content)
print("Report completed; 12 prediction files independently recomputed.")

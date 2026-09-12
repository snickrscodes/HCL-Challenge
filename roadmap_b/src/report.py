"""Generate a concise report and complete machine-readable bundle from actual artifacts."""

import json
from pathlib import Path

from .constants import LABELS
from .utils import common_parser, load_config, read_json, write_json

MODEL_NAMES = ("majority", "text_current", "text_context", "audio", "concat", "late_fusion")


def generate_report(cfg):
    root = Path(cfg["output_root"])
    audit = read_json(Path(cfg["data_root"]) / "audit" / "summary.json")
    runtime = read_json(root / "runtime.json")
    selection, calibration = read_json(root / "selection.json"), read_json(root / "calibration.json")
    results = {}
    for split in ("dev_model", "dev_calib", "test"):
        if (root / "evaluation" / split / "late_fusion" / "metrics.json").exists():
            results[split] = {
                name: read_json(root / "evaluation" / split / name / "metrics.json") for name in MODEL_NAMES
            }
    bundle = {
        "config": cfg,
        "data_audit": audit,
        "runtime": runtime,
        "selection": selection,
        "calibration": calibration,
        "results": results,
        "status": "synthetic smoke only"
        if audit["fixture"]
        else ("final test complete" if "test" in results else "development only; test not evaluated"),
    }
    write_json(root / "results_bundle.json", bundle)
    lines = [
        "# Roadmap A report",
        "",
        f"Status: **{bundle['status']}**.",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(runtime["environment"], indent=2),
        "```",
        "",
        "## Data audit",
        "",
        f"Rows: {audit['counts']}. Valid media: {audit['valid_audio']}; invalid: {audit['invalid_audio']}.",
        f"Outliers: {audit['outliers']}. Duration quantiles (seconds): {audit['duration_quantiles_s']}.",
        audit["preprocessing"],
        "",
        "Failures and anomalies are retained in data audit JSONL. Audio-only scores use valid audio only; "
        "concat and fusion fall back to calibrated text for unavailable audio.",
        f"Dev dialogues: {len(audit['dev_split']['dev_model'])} model / {len(audit['dev_split']['dev_calib'])} calibration. "
        "Explicit IDs are saved in audit/dev_dialogues.json.",
        "",
        "## Models",
        "",
        f"Text: {cfg['text']['model']} @ {cfg['text']['revision']}.",
        f"Audio: {cfg['audio']['model']} @ {cfg['audio']['revision']}.",
        "T0: current only. T1: at most three preceding same-dialogue turns, relative speaker markers, "
        "current-only mean pooling, maximum 128 tokens. Old context removed first; target truncations logged.",
        "Frozen WavLM: mono 16 kHz, final-layer masked mean. Unequal-length waveforms are encoded separately "
        "to prevent group-normalization padding dependence. Audio head 768→256→7; concat head 1536→256→7 "
        "(tiny smoke fixtures use smaller encoders).",
        "",
        "```json",
        json.dumps(runtime["parameter_ledger"], indent=2),
        "```",
        "",
    ]
    for name in ("text_current", "text_context", "audio", "concat", "wavlm_extraction"):
        info = read_json(root / name / "environment.json")
        lines.append(
            f"- {name}: parameters {info.get('parameters')}; elapsed {info['elapsed_s']:.2f} s; "
            f"peak GPU allocated {info['peak_gpu_memory_bytes']} bytes."
        )
    for split, scores in results.items():
        lines.extend(
            [
                "",
                f"## Results: {split}",
                "",
                "| Model | Accuracy | Weighted F1 | Macro F1 | n |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, score in scores.items():
            lines.append(
                f"| {name} | {score['accuracy']:.4f} | {score['weighted_f1']:.4f} | {score['macro_f1']:.4f} | {score['n']} |"
            )
        for name, score in scores.items():
            lines.extend(
                [
                    "",
                    f"### {name}: per-class results",
                    "",
                    "| Class | Precision | Recall | F1 | Support |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for label in LABELS:
                item = score["per_class"][label]
                lines.append(
                    f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | {item['support']} |"
                )
            lines.extend(
                [
                    "",
                    "Confusion matrix: rows=true, columns=predicted; canonical label order.",
                    "",
                    "```json",
                    json.dumps(score["confusion_matrix"]),
                    "```",
                    "",
                ]
            )
    lines.extend(
        [
            "## Fusion",
            "",
            f"Matched text baseline: {selection['text']}. Raw-logit alpha: {selection['alpha']:.2f}.",
            f"Temperatures: text {calibration['text']:.4f}, audio {calibration['audio']:.4f}, fusion {calibration['fusion']:.4f}.",
            "Alpha uses dev_model; all temperatures use dev_calib after alpha is frozen.",
            "",
        ]
    )
    for split, scores in results.items():
        fused, text = scores["late_fusion"], scores[selection["text"]]
        lines.append(
            f"- {split}: macro-F1 difference {fused['macro_f1'] - text['macro_f1']:+.4f}; "
            f"changed {fused['predictions_changed']}, corrected {fused['text_wrong_fusion_correct']}, "
            f"harmed {fused['text_correct_fusion_wrong']}. This does not prove prosodic causality."
        )
    extraction = read_json(root / "wavlm_extraction" / "metrics.json")
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            runtime["definition"],
            "",
            "Observed warm latency (milliseconds):",
            "",
            "```json",
            json.dumps(runtime["summary_ms"], indent=2),
            "```",
            "",
            f"Load time: {runtime['model_load_s']:.3f} s. Median real-time factor: {runtime['median_real_time_factor']:.4f}.",
            "Extraction throughput:",
            "",
            "```json",
            json.dumps(extraction, indent=2),
            "```",
            "",
            "There is no predeclared latency SLA, VAD delay, generator TTFT, or streaming acoustic claim. "
            "The authored response is emitted with the final state. Samples and durations are in runtime.json.",
            "",
            "## Failures and limitations",
            "",
            "- Strong class imbalance; macro F1 and all minority-class scores remain necessary.",
            "- Friends speakers and episodes overlap across official splits; this is not a speaker-independent benchmark.",
            "- TV music, laughter, overlapping voices, and clip boundaries can provide non-emotional shortcuts.",
            "- Clip/alignment failures remain explicit; no failed or silent clip is relabeled neutral.",
            "- WavLM can exploit speaker, lexical content, or background. Improvements are complementarity evidence, not causal proof.",
            "- Ground-truth transcripts confer an advantage over a live speech-recognition system.",
            "- Buffering is incremental; acoustic computation is utterance-final.",
            "- Responses are authored examples, without MELD response supervision or a completed human-quality study.",
            "- Single seed and small calibration partition limit statistical certainty; dev_model is a selection set.",
            "",
            "## Decision",
            "",
        ]
    )
    development = results["dev_model"]
    fused, text = development["late_fusion"], development[selection["text"]]
    complementary = (
        selection["alpha"] > 0
        and fused["macro_f1"] > text["macro_f1"]
        and fused["text_wrong_fusion_correct"] > fused["text_correct_fusion_wrong"]
    )
    if audit["fixture"]:
        lines.append(
            "Roadmap A is NOT validated as a final fallback by synthetic fixtures. Do not proceed to Roadmap B until full MELD evidence exists."
        )
    elif "test" not in results:
        lines.append(
            "Roadmap A development run is complete; final fallback validation awaits frozen test evaluation. "
            + (
                "Audio complementarity warrants considering Roadmap B after validation."
                if complementary
                else "Do not proceed to Roadmap B: development evidence does not meet the complementarity rule."
            )
        )
    else:
        lines.append(
            "Roadmap A is valid as final fallback. "
            + (
                "Proceed to Roadmap B only as a new controlled experiment; positive development complementarity was observed."
                if complementary
                else "Do not proceed to Roadmap B: development evidence does not meet the complementarity rule."
            )
        )
    lines.extend(
        [
            "",
            "The continuation rule uses development evidence only: alpha > 0, improved macro F1, and more corrected than harmed examples. "
            "Test scores never trigger retuning. Parameter/runtime and data checks are necessary but do not establish deployment quality.",
            "",
        ]
    )
    (root / "ROADMAP_A_REPORT.md").write_text("\n".join(lines))
    return bundle


def main():
    args = common_parser(__doc__).parse_args()
    generate_report(load_config(args.config))


if __name__ == "__main__":
    main()

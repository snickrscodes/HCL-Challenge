"""Generate an evidence-based report; incomplete work is never reported as measured."""

import numpy as np

from .b_data import output, parser, settings
from .b_fusion import VARIANTS
from .utils import read_json, write_json


def decision(cfg, summary, references, results, raw_ok, tests_ok):
    a = references["A_replay_canonical"]
    m = summary["variants"]
    b = m["adaptive"]
    seeds = cfg["seeds"]
    runs = [results[f"adaptive/{s}"] for s in seeds]
    thresholds = cfg["advancement"]
    non_neutral = [r["paired_changes"]["A_replay_canonical"]["non_neutral"]["net_correct"] for r in runs]
    rare = [r["true_positives"]["fear_plus_disgust"] for r in runs]
    a_rare = summary["A_replay_true_positives"]["fear_plus_disgust"]
    shuffle = np.mean([r["interventions"]["shuffle_summary"]["macro_f1_mean"] for r in runs])
    mean_audio = np.mean([r["interventions"]["training_mean_audio"]["metrics"]["macro_f1"] for r in runs])
    gate_variable = all(
        r["diagnostics"]["gate"]["std"] >= thresholds["minimum_gate_std"]
        and r["diagnostics"]["gate"]["p90"] - r["diagnostics"]["gate"]["p10"]
        >= thresholds["minimum_gate_p90_p10"]
        for r in runs
    )
    criteria = {
        "mean_macro_gain": b["macro_f1"]["mean"] >= a["macro_f1"] + thresholds["macro_gain"],
        "positive_gain_in_at_least_two_seeds": sum(r["metrics"]["macro_f1"] > a["macro_f1"] for r in runs)
        >= thresholds["minimum_positive_seeds"],
        "weighted_f1_not_materially_worse": b["weighted_f1"]["mean"]
        >= a["weighted_f1"] - thresholds["max_weighted_loss"],
        "beats_constant_residual": b["macro_f1"]["mean"] > m["constant"]["macro_f1"]["mean"],
        "beats_matched_text_only": b["macro_f1"]["mean"] > m["text_only"]["macro_f1"]["mean"],
        "non_neutral_not_worse_mean_and_reference_seed": np.mean(non_neutral) >= 0 and non_neutral[0] >= 0,
        "fear_disgust_tp_not_worse_mean_and_reference_seed": np.mean(rare) >= a_rare and rare[0] >= a_rare,
        "matched_beats_shuffled_and_training_mean": b["macro_f1"]["mean"]
        >= max(shuffle, mean_audio) + thresholds["intervention_macro_margin"],
        "gate_not_effectively_constant": gate_variable,
        "raw_inference_fallback_parameter_ledger": bool(raw_ok),
        "tests_passed": bool(tests_ok),
    }
    criteria = {name: bool(value) for name, value in criteria.items()}
    return {
        "decision": "ADVANCE ROADMAP B" if all(criteria.values()) else "RETAIN ROADMAP A",
        "criteria": criteria,
        "thresholds": thresholds,
        "caution": "Engineering continuation criteria on selected development models, not significance tests",
        "reference_seed": 1337,
        "matched_reference": "A_replay_canonical",
        "intervention_macro_means": {"shuffle": float(shuffle), "training_mean": float(mean_audio)},
    }


def generate(cfg):
    root = output(cfg)
    report = [
        "# Roadmap B controlled experiment",
        "",
        "Question: can adaptive text-primary acoustic correction improve fixed fusion for a reason that requires matched audio?",
        "",
        "Roadmap A is preserved. Official test inference is forbidden in this workflow.",
        "",
    ]
    integrity_file = root / "integrity.json"
    policy_file = root / "canonical_policy.json"
    if integrity_file.exists():
        integrity = read_json(integrity_file)
        report += [
            "## Baseline integrity",
            "",
            f"Baseline lock SHA256: {integrity['lock_sha256']}.",
            f"Original archive SHA256: {integrity['archive_sha256']}.",
            "Exact split IDs and checkpoint hashes are in the supplied baseline lock; they were not regenerated.",
            "A-original and A-replay-canonical are separate references. Original alpha/temperatures remain fixed.",
            "",
        ]
    if policy_file.exists():
        policy = read_json(policy_file)
        replay = read_json(root / "numerical_replay/report.json")
        report += [
            "## Numerical policy",
            "",
            f"Canonical encoding: **{policy['encoding']}**, precision {policy['dtype']}.",
            f"Real equal-batch discrepancy found: {replay['meaningful_equal_batch_difference']}.",
            "Whole-utterance audio limit: 60 seconds. Longer decoded clips keep audio_valid=true but are acoustically ineligible; calibrated text fallback is used.",
            "No cropping, chunked pooling, label changes, or original-cache replacement.",
            "A separate user-approved numerical repair uses singleton FP32 text with the exact unchanged T1 weights. A-original remains the historical BF16-cache result; A-replay-canonical uses FP32 text and singleton BF16 audio with the original alpha and temperatures.",
            "Real text diagnostics found a maximum singleton/batch FP32 probability difference of 0.0000063. Compared with the original BF16 cache, 198/839 examples differed by more than 0.005 and six argmaxes changed. No performance scores selected this policy.",
            "",
            "| Probe | BF16 paired/single max embedding difference | Max fused probability difference |",
            "|---|---:|---:|",
        ]
        for row in replay["measurements"]:
            c = row["comparisons"]["singleton_bfloat16__paired_bfloat16"]
            report.append(
                f"| {row['key']} | {c['embedding_max_abs']:.6g} | {c['fusion_probabilities_max_abs']:.6g} |"
            )
        report += [
            "",
            "All BF16/FP32/cached pairwise comparisons, logits, probabilities, waveform hashes and runtime metadata are in numerical_replay/.",
            "",
        ]
    else:
        report += [
            "## Numerical policy",
            "",
            "**Pending real RTX 4090 waveform replay. No B training is authorized by the program yet.**",
            "The 60-second operational limit is predeclared separately from historical A configs.",
            "",
        ]
    report += [
        "## Architecture",
        "",
        "Frozen T1/WavLM/original heads supply h_t, h_a, z_t and z_a. Raw softmax confidence and entropy feed a 1540→32→1 sigmoid gate.",
        "u=tanh(W_r h_a+b_r); c=u−mean(u); delta=2c/max(1,max_abs(c)); z_B=z_t+availability*g*delta.",
        "The residual and final gate weights initialize to zero; final gate bias is −2. Initial logits equal text exactly, with nonzero residual gradients.",
        "Controls: centered bias, constant-gate acoustic residual, adaptive residual, and the same architecture with text replacing every acoustic input.",
        "All variants use identical eligible training identities and CE/AdamW. Seeds 1337/1338/1339 are fixed; 1337 is the reference, not a best-seed choice.",
        "",
    ]
    evaluation = root / "evaluation"
    final = None
    if (evaluation / "summary.json").exists():
        summary = read_json(evaluation / "summary.json")
        refs = read_json(evaluation / "references.json")
        results = {
            f"{v}/{s}": read_json(evaluation / v / str(s) / "metrics.json")
            for v in VARIANTS
            for s in cfg["seeds"]
        }
        report += [
            "## Development results",
            "",
            "| Model | Accuracy | Weighted F1 | Macro F1 |",
            "|---|---:|---:|---:|",
        ]
        for name, r in refs.items():
            report.append(f"| {name} | {r['accuracy']:.4f} | {r['weighted_f1']:.4f} | {r['macro_f1']:.4f} |")
        for name, r in summary["variants"].items():
            values = [
                f"{r[m]['mean']:.4f} ± {r[m]['std']:.4f}" for m in ("accuracy", "weighted_f1", "macro_f1")
            ]
            report.append(f"| {name}, three head seeds | {' | '.join(values)} |")
        report += [
            "",
            "Standard deviations describe head-seed variability, not encoder-seed robustness.",
            "",
            "| Variant | Seed | Best epoch | Macro F1 | Weighted F1 | Fit seconds |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, values in results.items():
            train = read_json(root / "runs" / name / "training.json")
            report.append(
                f"| {train['variant']} | {train['seed']} | {train['best_epoch']} | "
                f"{values['metrics']['macro_f1']:.4f} | {values['metrics']['weighted_f1']:.4f} | {train['elapsed_s']:.2f} |"
            )
        r = results["adaptive/1337"]
        report += [
            "",
            "## Reference-seed class effects against A-replay-canonical",
            "",
            "| Group | Changed | Corrected | Harmed | Net correct |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, c in r["paired_changes"]["A_replay_canonical"].items():
            report.append(
                f"| {name} | {c['changed']} | {c['corrected']} | {c['harmed']} | {c['net_correct']} |"
            )
        report += [
            "",
            f"Reference-seed fear true positives: {r['true_positives']['fear']}; disgust: {r['true_positives']['disgust']}; combined: {r['true_positives']['fear_plus_disgust']}.",
            "Full per-class metrics, confusion matrices, recovery/harm rates, fixed slice supports and seed-paired results are saved for every run.",
            "",
            "## Gate and posterior influence",
            "",
            "Audio influence is TV(softmax(z_B/T_B),softmax(z_t/T_B)), using the same temperature.",
            "Gate values are not causal attribution, a fraction of emotion, or compute-saving routing.",
            "",
            "| Diagnostic, seed 1337 | Mean | Median | p10 | p90 | p95 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name in ("gate", "delta_l2", "delta_max_abs", "audio_influence"):
            d = r["diagnostics"][name]
            report.append(
                f"| {name} | {d['mean']:.4f} | {d['median']:.4f} | {d['p10']:.4f} | {d['p90']:.4f} | {d['p95']:.4f} |"
            )
        report += [
            "",
            "Correlations with confidence and correction/harm outcomes are descriptive; complete ranges and saturation rates are saved.",
            "",
            "## Audio interventions, reference seed",
            "",
            "| Condition | Macro F1 |",
            "|---|---:|",
            f"| Matched | {r['metrics']['macro_f1']:.4f} |",
        ]
        for name in ("training_mean_audio", "same_speaker", "gate_zero", "training_mean_gate"):
            report.append(f"| {name} | {r['interventions'][name]['metrics']['macro_f1']:.4f} |")
        sh = r["interventions"]["shuffle_summary"]
        report += [
            f"| Ten shuffles, mean [range] | {sh['macro_f1_mean']:.4f} [{sh['macro_f1_min']:.4f}, {sh['macro_f1_max']:.4f}] |",
            "",
            "Shuffling also disrupts lexical/background correspondence; same-speaker swaps do not isolate prosody.",
            "Real silence/noise probes, when available, use actual WavLM on a predetermined subset, not zero features.",
            "",
            "## Calibration and uncertainty",
            "",
            "| Variant/seed | Temperature | Calibration NLL before → after | ECE before → after |",
            "|---|---:|---:|---:|",
        ]
        for name in results:
            c = read_json(evaluation / name / "calibration.json")
            report.append(
                f"| {name} | {c['temperature']:.4f} | {c['before']['nll']:.4f} → {c['after']['nll']:.4f} | "
                f"{c['before']['ece_15']:.4f} → {c['after']['ece_15']:.4f} |"
            )
        report += [
            "",
            "NLL fitting does not guarantee lower ECE. dev_calib selects no architecture or seed.",
            f"Paired dialogue-cluster bootstrap uses {cfg['evaluation']['bootstrap_replicates']} replicates. Intervals are conditional post-selection descriptions, not significance claims.",
            "",
        ]
        report += [
            "",
            "Complete bootstrap estimates and intervals: `evaluation/adaptive/1337/metrics.json`.",
            "",
        ]
        teacher = read_json(root / "teacher_train_statistics.json")
        old_teacher = read_json(root / "A_original_teacher_train_statistics.json")
        report += [
            f"Frozen teacher training errors: canonical FP32 {teacher['errors']}; original BF16 {old_teacher['errors']}. These are in-sample errors, not generalization estimates.",
            "",
        ]
        raw_file = root / "raw_validation/cache_consistency.json"
        fallback_file = root / "raw_validation/fallback.json"
        ledger_file = root / "raw_validation/parameter_ledger.json"
        raw_ok = (
            raw_file.exists()
            and fallback_file.exists()
            and ledger_file.exists()
            and read_json(raw_file)["passed"]
            and read_json(fallback_file)["passed"]
            and read_json(ledger_file)["within_6b"]
        )
        tests_file = root / "validation.json"
        tests_ok = tests_file.exists() and read_json(tests_file)["passed"]
        final = decision(cfg, summary, refs, results, raw_ok, tests_ok)
        write_json(root / "advancement.json", final)
    else:
        report += [
            "## Development results",
            "",
            "**Not run. No B results, calibration, or advancement claim are available.**",
            "",
        ]
    runtime_file = root / "raw_validation/runtime.json"
    report += ["## Runtime and resources", ""]
    if runtime_file.exists():
        runtime = read_json(runtime_file)
        ledger = read_json(root / "raw_validation/parameter_ledger.json")
        report += [
            f"GPU: {runtime['environment']['gpu']}; model load {runtime['model_load_s']:.3f} seconds.",
            f"Loaded required parameters: {ledger['total_required_parameters']:,}.",
            f"Median real-time factor: {runtime['median_rtf']:.5f}; peak allocated GPU bytes: {runtime['peak_gpu_allocated_bytes']:,}.",
            f"Peak sampled process RAM bytes: {runtime['peak_sampled_process_rss_bytes']:,}.",
            "",
            "| Stage | p50 ms | p95 ms |",
            "|---|---:|---:|",
        ]
        for name, v in runtime["summary_ms"].items():
            report.append(f"| {name} | {v['p50']:.3f} | {v['p95']:.3f} |")
        if "A_same_workload_runtime" in runtime:
            report += ["", "| Same-workload stage | A p50/p95 ms | B p50/p95 ms |", "|---|---:|---:|"]
            for stage, av in runtime["A_same_workload_runtime"]["summary_ms"].items():
                bv = runtime["summary_ms"][stage]
                report.append(
                    f"| {stage} | {av['p50']:.3f} / {av['p95']:.3f} | {bv['p50']:.3f} / {bv['p95']:.3f} |"
                )
        report += [
            "",
            "Timing starts after explicit end-of-turn and excludes recording, disk read, endpointing and ASR. WavLM is not streaming.",
            "",
        ]
    else:
        report += [
            "Pending actual RTX 4090 raw-input measurements; no latency or memory values are invented.",
            "",
        ]
    report += ["## Same-words/different-delivery diagnostic", ""]
    delivery_file = root / "delivery_diagnostic.json"
    if delivery_file.exists():
        delivery = read_json(delivery_file)
        for r in delivery["recordings"]:
            p = r["prediction"]
            report.append(
                f"- {r['pair_id']} / intended {r['intended_delivery']}: {p['emotion']} ({p['confidence']:.3f}); {p['response']}"
            )
        report += [
            "",
            "Intended delivery is not objective ground truth. All successes and failures are retained; no tuning uses these clips.",
            "",
        ]
    else:
        report += [
            "Pending consented recordings. The recording manifest is provided; no synthetic recording is presented as human evidence.",
            "",
        ]
    report += [
        "## Limitations",
        "",
        "MELD is imbalanced TV dialogue with overlapping speakers, lexical information, laughter/music and imperfect clip boundaries. The encoders may exploit these features instead of prosody.",
        "Teacher training outputs are in-sample. Repeated development selection and three cheap head seeds do not establish population or encoder-seed robustness.",
        "No ASR, VAD, LLM, vision, RL, new encoder training, or optional imbalance search was added.",
        "",
        "## Decision",
        "",
    ]
    if final:
        for name, passed in final["criteria"].items():
            report.append(f"- {'PASS' if passed else 'FAIL/UNSATISFIED'}: {name}")
        report += ["", final["caution"], "", final["decision"]]
    else:
        report += [
            "The experiment is incomplete; retention is provisional because required evidence is missing, not because an unrun gate failed.",
            "",
            "RETAIN ROADMAP A",
        ]
    (root / "ROADMAP_B_REPORT.md").parent.mkdir(parents=True, exist_ok=True)
    (root / "ROADMAP_B_REPORT.md").write_text("\n".join(report) + "\n")
    return final


def main():
    args = parser(__doc__).parse_args()
    generate(settings(args))


if __name__ == "__main__":
    main()

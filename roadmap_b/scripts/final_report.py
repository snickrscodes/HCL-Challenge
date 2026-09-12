"""Render the submission report from preserved evidence; never run inference or fit anything."""

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(__doc__)
p.add_argument("--baseline-root", required=True)
p.add_argument("--b-root", default="artifacts/roadmap_b")
p.add_argument("--submission-root", default="artifacts/submission")
a = p.parse_args()
base = Path(a.baseline_root)
b = Path(a.b_root)
out = Path(a.submission_root)
out.mkdir(parents=True, exist_ok=True)


def read(path):
    return json.loads(Path(path).read_text())


refs = read(b / "evaluation/references.json")
s = read(b / "evaluation/summary.json")
audit = read(base / "data/processed/audit/summary.json")
complete = (out / "official_test/COMPLETED.json").exists()
delivery = (out / "delivery_results/summary.json").exists()
report = [
    "# Text meaning and vocal evidence in a character-robot interaction",
    "",
    "**Final measured report.**"
    if complete
    else "**Preparation draft: official test has not been completed. Human diagnostic and final freeze status are reported explicitly below.**",
    "",
    "## 1. Challenge and scope",
    "",
    "This local Text + Audio prototype accepts a transcript, raw waveform and causal conversational history, then emits a seven-class MELD emotion state and a short authored response. The question is whether selective acoustic correction helps a strong text interpretation, rather than whether adding model complexity raises a leaderboard score. No novel architecture claim is made.",
    "",
    "## 2. Real-time interaction",
    "",
    "Real time means utterance-final causal inference in a warm loaded session: PCM chunks are buffered, a final transcript and explicit end-of-turn trigger computation, and only current/past turns are used. WavLM itself is not streaming. Per-turn latency excludes recording time, unmeasured endpointing, ASR, and disk reads. Cold initialization is reported separately.",
    "",
    "## 3. MELD audit",
    "",
    f"Pinned annotation revision: `{audit['annotation_revision']}`. Official counts: train 9,989, development 1,109, test 2,610. Decoded valid audio: {audit['valid_audio']:,}; invalid: {audit['invalid_audio']}; recorded outliers: {audit['outliers']}.",
    "Missing dev/110/7 and corrupt train/125/3 remain as explicit unavailable-audio rows. Mono float WAV at 16 kHz; no denoising, normalization or aggressive trimming. Decoded duration median 2.475 s, p95 7.893 s, p99 11.772 s. The two previously audited test clips above 60 seconds use text fallback under the operational policy; their labels did not select that threshold.",
    "The current pinned Unicode annotation CSV is authoritative. An older nested training CSV has matching keys/labels but 2,697 differing rows, primarily encoding corruption; it was not used. Development IDs remain fixed: 86 dev_model dialogues (839 rows) and 28 dev_calib dialogues (270 rows).",
    "",
    "## 4. Model progression",
    "",
    "Frozen T1 is RoBERTa-base with three relative speaker markers and at most three preceding same-dialogue turns, 128 tokens, current-span pooling. WavLM-base-plus is frozen with valid-frame mean pooling. Original text/audio heads remain unchanged.",
    "",
    "![Development progression](figures/A_progression.png)",
    "",
    "## 5. Roadmap A findings",
    "",
    "Causal context improves the current-only text baseline. Audio alone contains signal but is weaker. Concat adds little. Fixed raw-logit fusion at alpha 0.40 makes 46 corrections and 26 harms against the original text baseline on dev_model. These counts are conditional development diagnostics, not prosodic causality. A-original remains immutable.",
    "",
    "## 6. Numerical reproducibility",
    "",
    "Real RTX4090 replay reproduced the original equal-length BF16 WavLM cache exactly for four specified clips, but singleton encoding differed materially on short clips (maximum embedding difference 0.06205; downstream fused probability difference 0.02570). B therefore uses separate singleton BF16 audio caches and the same runtime policy.",
    "An independently diagnosed RoBERTa BF16 batch dependence prompted an explicitly approved FP32 text repair using unchanged T1 weights. FP32 singleton/batch probability discrepancy was at most about 0.0000063. Every historical A cache/result remains preserved. A-replay-canonical differs from A-original on two development decisions despite coincident aggregate F1/accuracy.",
    "Final numerical contract: singleton FP32 text, singleton BF16 audio, whole-utterance maximum 60 seconds, explicit missing/corrupt/duration-limit text fallback. Audio validity and acoustic availability are separate; no crop, chunk pooling, row deletion or relabeling.",
    "",
    "## 7. Controlled Roadmap B experiment",
    "",
    "The selected model is adaptive residual seed 1337, predeclared before results. New neural parameters: 54,728. Gate input is [h_t,h_a,raw text confidence/entropy,raw audio confidence/entropy]. A 1540→32→1 sigmoid gate scales a centered, per-coordinate bounded audio logit residual:",
    "`u=tanh(W_r h_a+b_r); c=u-mean(u); delta=2c/max(1,max_abs(c)); z_B=z_t+availability*g*delta`.",
    "Zero residual initialization begins exactly at text. Bounds, centering, first-step gradient flow, frozen encoders, zero gate, missing audio and checkpoint round trips are tested. Four controls × three head seeds use the same eligible training identities, ordinary CE, AdamW, LR .001, batch 128, max 20 epochs, patience 3, dev_model macro-F1 selection. There is no architecture/loss sweep. Scalar temperatures are fitted afterward on dev_calib.",
    "",
    "| Model | Accuracy | Weighted F1 | Macro F1 |",
    "|---|---:|---:|---:|",
]
for name in ("text", "A_replay_canonical"):
    r = refs[name]
    report.append(f"| {name} | {r['accuracy']:.4f} | {r['weighted_f1']:.4f} | {r['macro_f1']:.4f} |")
for name, v in s["variants"].items():
    report.append(
        f"| {name}, head seeds mean ± SD | "
        + " | ".join(
            f"{v[m]['mean']:.4f} ± {v[m]['std']:.4f}" for m in ("accuracy", "weighted_f1", "macro_f1")
        )
        + " |"
    )
report += [
    "",
    "![Controlled capacity comparison](figures/B_controls.png)",
    "",
    "![Matched audio interventions](figures/C_interventions.png)",
    "",
    "![All class effects](figures/D_classes.png)",
    "",
    "![Fixed confidence slices](figures/E_confidence_slices.png)",
    "",
    "![Gate and actual influence](figures/F_gate_influence.png)",
    "",
    "B passes the predeclared development engineering rule. This is a class tradeoff, not fewer overall errors: seed 1337 corrects 28 A errors and harms 30 A-correct decisions, net −2. Non-neutral net is +4; fear+disgust true positives rise from 10 to 14, entirely through disgust. Sadness declines and fear detections do not improve.",
    "Matched-audio macro F1 averages .4999, compared with .4907 under shuffling and .4936 under training-mean replacement. These interventions change lexical, speaker and background correspondence as well as expression. The gate is neither attribution nor compute-saving routing.",
    "The reference-seed paired dialogue bootstrap macro-F1 difference interval is [−.01919, +.04746] (1,000 replicates); all reported metric-difference intervals include zero. These are conditional post-selection descriptions, separate from head-seed variability. Teacher training outputs are in-sample (1,340 canonical errors; 1,341 historical errors).",
    "",
    "## 8. Controlled vocal delivery",
    "",
]
if delivery:
    d = read(out / "delivery_results/summary.json")
    report += [
        f"Collection: {d['collection']['recordings']} recordings, {d['collection']['phrases']} phrases, speakers {d['collection']['speakers']}, independent listeners {d['collection']['complete_listeners']}.",
        d["interpretation"],
        f"Descriptive counts: `{json.dumps(d['counts'])}`.",
        "See [the full delivery diagnostic](delivery_results/DELIVERY_DIAGNOSTIC.md) for all successes, failures, unclear judgments and listener procedure.",
    ]
else:
    report += [
        "**Pending real consented recordings and blinded listener ratings.** The fixed protocol has 10 phrases × 3 deliveries × 2 speakers (60 recordings), including two stable-control phrases. There is no synthetic substitute for human evidence. The recording kit stores exact transcripts, intentions, timestamps, waveform hashes and consent. The separate listener page hides intentions and model outputs. Completion, not a favorable model result, is required before final freeze."
    ]
report += ["", "## 9. Official MELD test", ""]
if complete:
    t = read(out / "official_test/summary.json")
    report += [
        f"One frozen pass: {t['overall_support']} overall rows; {t['audio_eligible_support']} audio-eligible rows. Unavailable: `{json.dumps(t['unavailable_rows'])}`.",
        "",
        "| Model | Accuracy | Weighted F1 | Macro F1 | NLL | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in t["models"].items():
        m = r["metrics"]
        report.append(
            f"| {name} | {m['accuracy']:.4f} | {m['weighted_f1']:.4f} | {m['macro_f1']:.4f} | {m['nll']:.4f} | {m['ece_15']:.4f} |"
        )
    report += [
        "",
        f"Production recommendation: **{t['production_recommendation']}**. Predeclared checks: `{json.dumps(t['predeclared_recommendation_checks'])}`.",
        f"Paired dialogue uncertainty: `{json.dumps(read(out / 'official_test/B_vs_A_bootstrap.json')['differences'])}`.",
        "All per-class precision/recall/F1/support, confusion matrices, corrections/harms, non-neutral effects and label-independent confidence slices are retained in official_test/. No post-test training or tuning is permitted.",
    ]
else:
    report += [
        "**Not run.** No official test scores are implied by the development results. The final runner requires FINAL_FREEZE.json, completed human diagnostic, current tests/raw validation and complete presentation, and creates an exclusive STARTED marker before inference. The predeclared final comparison is causal text, fixed A canonical, B adaptive 1337, constant residual 1337 and matched text-only residual 1337, with audio-only eligible-subset diagnostics. No A-original batched BF16 test representation is introduced."
    ]
report += ["", "## 10. Latency and resources", ""]
runtime_file = out / "preflight/raw_validation/runtime.json"
rt = read(runtime_file if runtime_file.exists() else b / "raw_validation/runtime.json")
report += [
    f"Hardware: {rt['environment']['gpu']}. Warm raw-input workload: {len(rt['samples'])} turns. Measured loaded perception parameters: 218,698,059; peak allocated GPU bytes {rt['peak_gpu_allocated_bytes']:,}; process memory {rt['peak_sampled_process_rss_bytes']:,}; median RTF {rt['median_rtf']:.5f}.",
    "",
    "| Stage | Warm p50 ms | Warm p95 ms |",
    "|---|---:|---:|",
]
for name, r in rt["summary_ms"].items():
    report.append(f"| {name} | {r['p50']:.3f} | {r['p95']:.3f} |")
report += [
    "",
    "The preserved fresh-process example took 1.839 seconds for its first prediction. Warm loaded-model measurements must not be advertised as cold startup. Model load and first-call kernel setup are separate from successive-turn interaction. No recording time, ASR, unmeasured endpointing or excluded disk I/O is included in the per-turn claim.",
    "",
    "## 11. Response and demonstration",
    "",
    "An authored deterministic response policy follows perception and considers transcript/function, final emotion and uncertainty. MELD contains no suitable robot-response supervision; response quality is not scored by BLEU/ROUGE. Responses are concise and non-diagnostic, but this small phrase policy is not established as a robust general conversational or distress-handling system.",
    "The trace-replay demo separates user-facing emotion/distribution/response from text/audio states, gate, actual posterior TV influence, availability, revisions and timings. It includes a useful correction, little-change case, harmful case, missing audio, causal context and the predetermined S01/p01 delivery set when available. Outcome-defined development examples are labeled retrospective showcases, not confirmatory evidence. Incorrect outputs remain visible.",
    "",
    "## 12. Limitations",
    "",
    "MELD is imbalanced television dialogue with shared cast identities, background laughter/music, overlapping speech and imperfect clip boundaries. WavLM can use lexical or scene information; no prosodic-causality claim is justified. Gold transcripts are an advantage over live ASR. Minority counts are small. Only head-seed variability was measured. Development selection and in-sample teacher features can overfit. Natural controlled recordings are out of domain and descriptive, not a new benchmark.",
    "",
    "## 13. Intentional omissions and ownership",
    "",
    "No new perception architecture, encoder, loss sweep, ASR, VAD research, vision, language-model generator, RL, diarization, TTS or streaming acoustic computation was added. MELD media are not redistributed. RoBERTa, WavLM, PyTorch/Transformers, NumPy/SciPy/scikit-learn, FFmpeg, Matplotlib and browser APIs are external components. Project code, tests, scripts, authored responses and reports were AI-assisted; their correctness and measured interpretation remain the project owner’s responsibility. See EXTERNAL_COMPONENTS.md and pinned environment files.",
    "",
    "## 14. Production recommendation",
    "",
]
report += [
    t["production_recommendation"]
    if complete
    else "B is the selected development candidate; A remains the permanent fallback. The final production recommendation awaits the already planned, frozen test pass. No additional model search is authorized."
]
file = out / ("FINAL_REPORT.md" if complete else "FINAL_REPORT_DRAFT.md")
file.write_text("\n".join(report) + "\n")
print(file)

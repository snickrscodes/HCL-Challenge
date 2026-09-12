"""Export an explicit public evidence allowlist; never modify source artifacts.

Run from roadmap_b with the preserved A snapshot available separately:
python scripts/export_public_evidence.py --baseline-root /path/to/roadmap_a_source
The destination must be new. Numerical values are not rounded/recomputed.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def public_paths(text):
    # Path-only redaction in separate exports, including JSON object keys.
    text = re.sub(r"/home/[^/\s\"'`]+", "<LOCAL_HOME>", text)
    text = text.replace("/workspace/", "<COMPUTE_WORKSPACE>/")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, default=Path("evidence"))
    args = parser.parse_args()
    a, project, out = args.baseline_root, args.project_root, args.output_root
    if out.exists():
        raise SystemExit("Refusing to overwrite an existing public evidence export")
    b = project / "artifacts/roadmap_b"
    s = project / "artifacts/submission"
    entries = []

    def export(source, destination, projection=None):
        raw = source.read_bytes()
        text = None
        if projection is not None:
            text = json.dumps(projection(json.loads(raw)), indent=2, ensure_ascii=False) + "\n"
        elif source.suffix in {".json", ".jsonl", ".yaml", ".md", ".txt", ".csv", ".svg"}:
            text = raw.decode("utf-8")
        data = public_paths(text).encode("utf-8") if text is not None else raw
        target = out / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        source_alias = (
            ("A_original/" + str(source.relative_to(a)))
            if source.is_relative_to(a)
            else ("B_continuation/" + str(source.relative_to(project)))
        )
        entries.append(
            {
                "file": str(destination),
                "source": source_alias,
                "source_sha256": sha(raw),
                "public_sha256": sha(data),
                "bytes": len(data),
                "transformation": "explicit integrity projection; local paths redacted"
                if projection
                else (
                    "local path labels redacted; numerical values unchanged"
                    if data != raw
                    else "byte-identical"
                ),
            }
        )
        assert source.read_bytes() == raw, "Source evidence changed during export"

    for name in [
        "ROADMAP_A_REPORT.md",
        "results_bundle.json",
        "calibration.json",
        "selection.json",
        "parameter_ledger.json",
        "runtime.json",
    ]:
        export(a / "artifacts" / name, Path("roadmap_a") / name)
    for model in ["text_current", "text_context", "audio", "concat", "late_fusion", "wavlm_extraction"]:
        for name in ["metrics.json", "environment.json", "config.yaml"]:
            export(a / "artifacts" / model / name, Path("roadmap_a") / model / name)
    for name in ["summary.json", "dev_dialogues.json", "outliers.jsonl", "failures.jsonl"]:
        export(a / "data/processed/audit" / name, Path("data_audit") / name)
    for name in [
        "ROADMAP_B_REPORT.md",
        "advancement.json",
        "training_summary.json",
        "model_freeze.json",
        "canonical_policy.json",
        "text_policy.json",
        "resolved_config.yaml",
        "final_integrity.json",
        "teacher_train_statistics.json",
        "A_original_teacher_train_statistics.json",
        "test_discipline.json",
        "prediction_recomputation.json",
        "evaluation_repair.json",
        "validation.json",
        "raw_probe_manifest.json",
    ]:
        export(b / name, Path("roadmap_b") / name)
    for folder in ["evaluation", "raw_validation"]:
        for source in sorted((b / folder).rglob("*")):
            if source.is_file() and source.suffix in {".json", ".jsonl"}:
                export(source, Path("roadmap_b") / source.relative_to(b))
    for name in ["report.json", "text_fp32_diagnostics.json", "text_batch_diagnostics.json"]:
        export(b / "numerical_replay" / name, Path("roadmap_b/numerical_replay") / name)
    export(s / "FINAL_INFERENCE_POLICY.json", Path("policy/FINAL_INFERENCE_POLICY.json"))
    for source in sorted((s / "figures").iterdir()):
        if source.suffix in {".svg", ".png", ".csv", ".txt", ".json"}:
            export(source, Path("figures") / source.name)
    smoke = s / "controlled_delivery_provisional_smoke"
    for name in [
        "CONTROLLED_DELIVERY_PROVISIONAL_SMOKE.md",
        "engineering_readout.json",
        "summary.json",
        "pairs.json",
        "sets.json",
        "intended_mapping.json",
        "uncalibrated_pairs.json",
        "validation.json",
        "NEXT_STEPS.md",
    ]:
        export(smoke / name, Path("controlled_delivery_provisional") / name)
    export(
        smoke / "integrity.json",
        Path("controlled_delivery_provisional/model_integrity.json"),
        lambda obj: {k: obj[k] for k in ["environment", "A_integrity", "model_files_sha256"]},
    )
    export(
        smoke / "source/delivery_smoke_readout.py",
        Path("controlled_delivery_provisional/source/delivery_smoke_readout.py.txt"),
    )
    manifest = {
        "scope": "Public development evidence export; not a final evaluation freeze",
        "policy": "Explicit allowlist. Originals untouched. No raw media, human metadata/consent/ratings, weights, features, or official-test predictions. Path redactions are display-only; these exports are not runtime artifact replacements.",
        "numeric_policy": "No scientific results recomputed, rounded, or changed by this exporter.",
        "files": entries,
    }
    (out / "PUBLIC_EXPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(entries)} files, {sum(row['bytes'] for row in entries):,} bytes")


if __name__ == "__main__":
    main()

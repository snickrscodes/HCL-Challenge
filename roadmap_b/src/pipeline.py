"""Small sequential workflow; final test evaluation is a separate frozen operation."""

import hashlib
import json
from pathlib import Path


from . import calibrate, evaluate, extract_audio, train_audio, train_fusion, train_text
from .constants import LABELS
from .data import validate_rows
from .audio import read_waveform
from .utils import key, row_sort, audio_file
import yaml
import soundfile as sf
from .utils import (
    assert_unfrozen,
    common_parser,
    digest,
    environment,
    load_config,
    read_json,
    read_rows,
    setup,
    write_json,
)


def preflight(cfg):
    summary = read_json(Path(cfg["data_root"]) / "audit" / "summary.json")
    rows = read_rows(Path(cfg["data_root"]) / "manifest.jsonl")
    validate_rows(rows, None if summary["fixture"] else cfg["data"]["expected_counts"])
    if not summary["fixture"]:
        for section in ("text", "audio"):
            revision = cfg[section]["revision"]
            if not isinstance(revision, str) or len(revision) != 40:
                raise ValueError("Use exact pretrained commit revisions")
    manifest_by_key = {key(row): row for row in rows}
    if rows != sorted(rows, key=row_sort):
        raise ValueError("Manifest is not numerically ordered")
    for split in ("train", "dev_model", "dev_calib", "test"):
        partition = read_rows(Path(cfg["data_root"]) / "splits" / f"{split}.jsonl")
        validate_rows(partition)
        if partition != sorted(partition, key=row_sort):
            raise ValueError("Partition is not numerically ordered")
        if any(manifest_by_key.get(key(row)) != row for row in partition):
            raise ValueError("Partition records differ from the processed manifest")
        source = "dev" if split.startswith("dev_") else split
        if any(row["split"] != source for row in partition):
            raise ValueError("Cross-split examples in partition")
        if split in ("train", "test") and {key(row) for row in partition} != {
            key(row) for row in rows if row["split"] == split
        }:
            raise ValueError("Incomplete official train/test partition")
    valid_count = sum(row["audio_valid"] for row in rows)
    if valid_count != summary["valid_audio"] or len(rows) - valid_count != summary["invalid_audio"]:
        raise ValueError("Audio availability counts differ from audit")
    for row in rows:
        if row["audio_valid"]:
            info = sf.info(audio_file(cfg, row))
            if (
                row["sample_rate"] != 16000
                or info.samplerate != 16000
                or info.channels != 1
                or info.frames != row["num_samples"]
            ):
                raise ValueError("Processed WAV metadata changed")
            read_waveform(audio_file(cfg, row))  # finite samples and minimum receptive field
    sets = {}
    for name in ("dev_model", "dev_calib"):
        values = read_rows(Path(cfg["data_root"]) / "splits" / f"{name}.jsonl")
        sets[name] = {r["dialogue_id"] for r in values}
        if sets[name] != set(summary["dev_split"][name]):
            raise ValueError("Development dialogue IDs differ from the saved policy")
        if not values or any(r["split"] != "dev" for r in values):
            raise ValueError("Development split contains non-dev or no examples")
    if sets["dev_model"] & sets["dev_calib"]:
        raise ValueError("Development dialogue leakage")
    if sets["dev_model"] | sets["dev_calib"] != {r["dialogue_id"] for r in rows if r["split"] == "dev"}:
        raise ValueError("Development partitions do not exhaust official dev")
    dev_keys = {
        key(row)
        for name in ("dev_model", "dev_calib")
        for row in read_rows(Path(cfg["data_root"]) / "splits" / f"{name}.jsonl")
    }
    if dev_keys != {key(row) for row in rows if row["split"] == "dev"}:
        raise ValueError("Development partitions omit official utterances")
    if not summary["valid_audio"]:
        raise ValueError("No valid audio; fix preprocessing before training")
    write_json(Path(cfg["output_root"]) / "preflight.json", environment(cfg))


def freeze_payload(cfg):
    for name in (
        "text_current",
        "text_context",
        "audio",
        "concat",
        "late_fusion",
        "calibration",
        "wavlm_extraction",
    ):
        resolved = yaml.safe_load((Path(cfg["output_root"]) / name / "config.yaml").read_text())
        if resolved != cfg:
            raise ValueError(f"Configuration differs from the completed {name} run")
    files = []
    for root in (Path(cfg["checkpoint_root"]), Path(cfg["data_root"]) / "splits"):
        files.extend(path for path in root.rglob("*") if path.is_file())
    files.extend(Path("src").glob("*.py"))
    files += [
        Path(cfg["data_root"]) / "manifest.jsonl",
        Path(cfg["output_root"]) / "selection.json",
        Path(cfg["output_root"]) / "calibration.json",
    ]
    return {
        "config": cfg,
        "labels": list(LABELS),
        "selection_rule": "dev_model macro F1",
        "files_sha256": {str(path): digest(path) for path in sorted(files)},
        "configuration_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
    }


def run(cfg, final_test=False):
    from .benchmark import benchmark
    from .report import generate_report

    setup(cfg["seed"])
    preflight(cfg)
    root = Path(cfg["output_root"])
    if final_test:
        if (root / "final_test_started.json").exists():
            raise RuntimeError("Final test was already started. No automatic test reruns.")
        for required in ("selection.json", "calibration.json", "runtime.json"):
            if not (root / required).exists():
                raise ValueError("Complete and inspect the development workflow before --final-test")
        payload = freeze_payload(cfg)
        if (root / "freeze.json").exists() and read_json(root / "freeze.json") != payload:
            raise ValueError("Frozen configuration/files changed")
        write_json(root / "freeze.json", payload)

        # Test audio is touched only after the complete development
        # configuration has been frozen. This does not use test labels.
        test_audio = extract_audio.extract_cache(cfg, ("test",))
        write_json(root / "final_test_audio_extraction.json", test_audio)

        # From this point onward test labels/metrics are being consumed.
        write_json(
            root / "final_test_started.json",
            {"status": "started; do not retune", "labels": list(LABELS)},
        )
        evaluate.evaluate_split(cfg, "test")
        write_json(root / "final_test_complete.json", {"status": "complete"})
    else:
        assert_unfrozen(cfg)
        evaluate.majority(cfg)
        train_text.train(cfg, "text_current")
        train_text.train(cfg, "text_context")
        extract_audio.extract(cfg)
        train_audio.train(cfg)
        train_fusion.train(cfg)
        calibrate.calibrate(cfg)
        for split in ("dev_model", "dev_calib"):
            evaluate.evaluate_split(cfg, split)
        benchmark(cfg)
    generate_report(cfg)


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.preflight_only:
        preflight(cfg)
    else:
        run(cfg, args.final_test)


if __name__ == "__main__":
    main()

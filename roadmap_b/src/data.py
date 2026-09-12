"""Validate official annotations, retain failures, and preconvert media."""

import csv
import hashlib
import json
import math
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .audio import convert_media
from .constants import LABELS, LABEL_TO_ID, SAMPLE_RATE
from .utils import common_parser, key, load_config, row_sort, setup, write_json, write_rows

OFFICIAL_SPLITS = ("train", "dev", "test")


def annotation_rows(path, split):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    parsed = []
    for row in rows:
        emotion = row["Emotion"].strip()
        parsed.append(
            {
                "split": split,
                "dialogue_id": int(row["Dialogue_ID"]),
                "utterance_id": int(row["Utterance_ID"]),
                "speaker": row["Speaker"],
                "text": row["Utterance"],
                "emotion": emotion,
                "label": LABEL_TO_ID[emotion],
                "annotation_start": row.get("StartTime"),
                "annotation_end": row.get("EndTime"),
                "season": row.get("Season"),
                "episode": row.get("Episode"),
            }
        )
    validate_rows(parsed)
    return sorted(parsed, key=row_sort)


def validate_rows(rows, expected=None):
    keys = [key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate split/dialogue/utterance key")
    for row in rows:
        if row["split"] not in OFFICIAL_SPLITS or row["label"] != LABEL_TO_ID[row["emotion"]]:
            raise ValueError("Invalid split or label mapping")
        if row["dialogue_id"] < 0 or row["utterance_id"] < 0 or not row["text"].strip():
            raise ValueError("Invalid ID or empty transcript")
    if expected is not None and dict(Counter(row["split"] for row in rows)) != expected:
        raise ValueError("Annotation counts do not match pinned contract")


def canonical_hash(rows):
    return hashlib.sha256(json.dumps(sorted(rows, key=row_sort), sort_keys=True).encode()).hexdigest()


def grouped_dev_split(rows, seed, fraction=0.5, candidates=256):
    groups = defaultdict(list)
    for row in rows:
        groups[int(row["dialogue_id"])].append(row["label"])
    ids = np.array(sorted(groups))
    if len(ids) < 2:
        raise ValueError("Development split needs at least two dialogues")
    n = min(len(ids) - 1, max(1, round(len(ids) * fraction)))
    total = np.bincount([row["label"] for row in rows], minlength=len(LABELS))
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(candidates):
        calibration = sorted(rng.choice(ids, n, replace=False).tolist())
        counts = np.bincount(
            [label for ident in calibration for label in groups[ident]], minlength=len(LABELS)
        )
        score = float(np.mean(((counts - fraction * total) / np.maximum(total, 1)) ** 2))
        score += float(((counts.sum() - fraction * total.sum()) / total.sum()) ** 2)
        choice = (score, calibration)
        if best is None or choice < best:
            best = choice
    calib = best[1]
    return {
        "dev_model": sorted(set(ids.tolist()) - set(calib)),
        "dev_calib": calib,
        "seed": seed,
        "calib_fraction": fraction,
        "balance_score": best[0],
    }


def timestamp_seconds(value):
    if not value:
        return None
    fields = value.replace(",", ".").split(":")
    if len(fields) != 3:
        raise ValueError("Malformed timestamp")
    hour, minute, second = map(float, fields)
    if (
        not all(math.isfinite(value) for value in (hour, minute, second))
        or hour < 0
        or not 0 <= minute < 60
        or not 0 <= second < 60
    ):
        raise ValueError("Malformed timestamp")
    return hour * 3600 + minute * 60 + second


def find_annotation(root, split):
    paths = sorted(root.rglob(f"{split}_sent_emo.csv"))
    if len(paths) != 1:
        raise ValueError(f"Expected one {split}_sent_emo.csv under raw root; found {paths}")
    return paths[0]


def media_index(root):
    index = defaultdict(list)
    for path in root.rglob("*"):
        if path.suffix.lower() not in {".mp4", ".wav", ".mkv", ".avi", ".webm"}:
            continue
        relative = path.relative_to(root)
        parents = [part.lower() for part in relative.parts[:-1]]
        splits = [
            split
            for split in OFFICIAL_SPLITS
            if any(part == split or part.startswith(split + "_") or part == split + "s" for part in parents)
        ]
        if len(splits) == 1:
            index[(splits[0], path.stem)].append(path)
    return index


def prepare_dataset(raw_root, output_root, cfg, fixture=False):
    raw_root, output_root = Path(raw_root).resolve(), Path(output_root)
    if (output_root / "manifest.jsonl").exists():
        raise ValueError("Processed manifest already exists; use a fresh output root")
    annotations, hashes = [], {}
    for split in OFFICIAL_SPLITS:
        supplied = find_annotation(raw_root, split)
        rows = annotation_rows(supplied, split)
        hashes[split] = canonical_hash(rows)
        if not fixture:
            revision = cfg["data"]["annotation_revision"]
            destination = output_root / "audit" / "annotations" / f"{split}_sent_emo.csv"
            destination.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/declare-lab/MELD/{revision}/data/MELD/{split}_sent_emo.csv"
            urllib.request.urlretrieve(url, destination)
            if hashes[split] != canonical_hash(annotation_rows(destination, split)):
                raise ValueError(f"{split} annotations differ from pinned official revision")
        annotations.extend(rows)
    validate_rows(annotations, None if fixture else cfg["data"]["expected_counts"])
    dev = [row for row in annotations if row["split"] == "dev"]
    policy = grouped_dev_split(
        dev, cfg["seed"], cfg["data"]["calib_fraction"], cfg["data"]["split_candidates"]
    )
    index = media_index(raw_root)
    failures, outliers, manifest = [], [], []
    for row in sorted(annotations, key=row_sort):
        row = dict(row)
        stem = f"dia{row['dialogue_id']}_utt{row['utterance_id']}"
        candidates = index.get((row["split"], stem), [])
        destination = Path("audio") / row["split"] / f"{stem}.wav"
        row.update(
            source_media=None,
            audio_path=None,
            sample_rate=None,
            num_samples=None,
            duration_s=None,
            audio_valid=False,
        )
        try:
            start, end = timestamp_seconds(row["annotation_start"]), timestamp_seconds(row["annotation_end"])
            if start is not None and end is not None:
                duration = end - start
                row["annotation_duration_s"] = duration
                if duration <= 0 or duration > cfg["data"]["outlier_seconds"]:
                    outliers.append({"key": key(row), "kind": "annotation_duration", "value": duration})
        except ValueError as error:
            outliers.append({"key": key(row), "kind": "timestamp", "error": str(error)})
        try:
            if len(candidates) != 1:
                raise ValueError(f"Expected exactly one media candidate, found {len(candidates)}")
            row["source_media"] = candidates[0].relative_to(raw_root).as_posix()
            wave = convert_media(candidates[0], output_root / destination)
            duration = len(wave) / SAMPLE_RATE
            row.update(
                audio_path=destination.as_posix(),
                sample_rate=SAMPLE_RATE,
                num_samples=len(wave),
                duration_s=duration,
                audio_valid=True,
            )
            if duration > cfg["data"]["outlier_seconds"]:
                outliers.append({"key": key(row), "kind": "decoded_duration", "value": duration})
            if np.max(np.abs(wave)) == 0:
                outliers.append({"key": key(row), "kind": "digital_silence"})
            annotated = row.get("annotation_duration_s")
            if annotated is not None and abs(annotated - duration) > max(1, duration * 0.25):
                outliers.append(
                    {
                        "key": key(row),
                        "kind": "duration_mismatch",
                        "annotation_s": annotated,
                        "decoded_s": duration,
                    }
                )
        except (ValueError, OSError, RuntimeError) as error:
            failures.append({"key": key(row), "source_media": row["source_media"], "error": str(error)})
        manifest.append(row)
    write_rows(output_root / "manifest.jsonl", manifest)
    for split in ("train", "test"):
        write_rows(output_root / "splits" / f"{split}.jsonl", [r for r in manifest if r["split"] == split])
    for name in ("dev_model", "dev_calib"):
        write_rows(
            output_root / "splits" / f"{name}.jsonl",
            [r for r in manifest if r["split"] == "dev" and r["dialogue_id"] in policy[name]],
        )
    write_json(output_root / "audit" / "dev_dialogues.json", policy)
    write_rows(output_root / "audit" / "failures.jsonl", failures)
    write_rows(output_root / "audit" / "outliers.jsonl", outliers)
    duration = [r["duration_s"] for r in manifest if r["audio_valid"]]
    summary = {
        "fixture": fixture,
        "annotation_revision": cfg["data"]["annotation_revision"],
        "annotation_hashes": hashes,
        "preprocessing_version": 1,
        "counts": dict(Counter(r["split"] for r in manifest)),
        "valid_audio": sum(r["audio_valid"] for r in manifest),
        "invalid_audio": len(failures),
        "class_counts": {
            s: {label: sum(r["split"] == s and r["emotion"] == label for r in manifest) for label in LABELS}
            for s in OFFICIAL_SPLITS
        },
        "duration_quantiles_s": dict(
            zip(
                ("min", "p50", "p90", "p95", "p99", "max"),
                np.quantile(duration, [0, 0.5, 0.9, 0.95, 0.99, 1]).tolist(),
            )
        )
        if duration
        else {},
        "total_audio_s": sum(duration),
        "outliers": len(outliers),
        "dev_split": policy,
        "preprocessing": "FFmpeg float WAV, mono 16000 Hz, no normalization/trim/denoising; failures retained",
    }
    write_json(output_root / "audit" / "summary.json", summary)
    return summary


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", default="data/processed")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup(cfg["seed"])
    print(json.dumps(prepare_dataset(args.raw_root, args.output_root, cfg), indent=2))


if __name__ == "__main__":
    main()

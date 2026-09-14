"""C1 v2: only expand existing dev/calibration speakers to ten non-test sentences."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np

from .c1_data import CONTRACT_PATH, normalize_text, sha, write_json

DEFAULT_ROOT = Path("artifacts/roadmap_c1_v2")


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Immutable v2 artifact differs: {path}")
        return
    write_json(path, value)


def summarize(rows, labels):
    eligible = [r for r in rows if r["supervision_eligible"]]
    unique = Counter(r["plurality"] for r in eligible if r["plurality"] is not None)
    mass = np.array([r["target"] for r in eligible], dtype=float).sum(0) if eligible else np.zeros(6)
    counts = (
        np.array([r["vote_counts"] for r in eligible], dtype=int).sum(0)
        if eligible
        else np.zeros(6, dtype=int)
    )
    return {
        "rows": len(rows),
        "soft_ce_rows": len(eligible),
        "ties": sum(r["plurality"] is None for r in eligible),
        "unique_plurality": {label: unique[i] for i, label in enumerate(labels)},
        "soft_vote_mass": dict(zip(labels, mass.tolist())),
        "raw_vote_counts": dict(zip(labels, counts.tolist())),
        "speaker_rows": dict(sorted(Counter(r["speaker"] for r in eligible).items())),
        "sentence_rows": dict(sorted(Counter(r["sentence"] for r in eligible).items())),
    }


def amend(v1, root):
    v1, root = Path(v1), Path(root)
    cfg = json.loads(CONTRACT_PATH.read_text())
    prior = json.loads((v1 / "DATA_AUDIT.json").read_text())
    assert prior["status"] == "DATA_PROTOCOL_BLOCKED"
    assert prior["contract_sha256"] == sha(CONTRACT_PATH)
    lock = json.loads((v1 / "C1_SPLIT_LOCK.json").read_text())
    assert sha(v1 / "manifest.jsonl") == lock["manifest_sha256"]
    assert sha(v1 / "C1_SPLIT_IDS.json") == lock["split_sha256"]
    old_split = json.loads((v1 / "C1_SPLIT_IDS.json").read_text())
    split = copy.deepcopy(old_split)
    ten = old_split["sentences"]["train"] + old_split["sentences"]["dev"]
    assert len(set(ten)) == 10 and set(ten).isdisjoint(old_split["sentences"]["external_test"])
    split["sentences"]["dev"] = ten
    split["sentences"]["calib"] = ten
    split["algorithm"] = (
        "Preserved v1 speaker/train/test assignments; authorized v2 dev/calib expansion to ten non-test sentences"
    )
    rows = [json.loads(line) for line in (v1 / "manifest.jsonl").read_text().splitlines()]
    amended = copy.deepcopy(rows)
    moved = []
    for before, row in zip(rows, amended):
        for part in ("dev", "calib"):
            if row["speaker"] in split["speakers"][part] and row["sentence"] in ten:
                row["partition"] = part
                row["exclusion_reasons"] = [
                    r for r in row["exclusion_reasons"] if r != "unused_speaker_sentence_cross_product"
                ]
                row["supervision_eligible"] = not row["exclusion_reasons"]
        if row["partition"] != before["partition"]:
            assert before["partition"] == "unused_cross_product" and row["partition"] in ("dev", "calib")
            moved.append({"id": row["id"], "from": before["partition"], "to": row["partition"]})
        for name in (
            "id",
            "speaker",
            "sentence",
            "target",
            "vote_counts",
            "plurality",
            "max_vote",
            "source_query_key",
        ):
            assert row[name] == before[name]
        if before["partition"] in ("train", "external_test"):
            assert row == before
        if row["id"] in cfg["quarantine_ids"]:
            assert not row["supervision_eligible"] and cfg["quarantine_reason"] in row["exclusion_reasons"]
    assert len({r["id"] for r in amended}) == len(amended)
    partitions = {
        p: [r for r in amended if r["partition"] == p] for p in ("train", "dev", "calib", "external_test")
    }
    checks = {}
    for i, p in enumerate(partitions):
        for q in list(partitions)[i + 1 :]:
            s = {r["speaker"] for r in partitions[p]} & {r["speaker"] for r in partitions[q]}
            assert not s
            text = {normalize_text(r["text"]) for r in partitions[p]} & {
                normalize_text(r["text"]) for r in partitions[q]
            }
            assert q != "external_test" or not text
            checks[f"{p}/{q}"] = {
                "speaker_overlap": [],
                "normalized_transcript_overlap_count": len(text),
                "text_overlap_expected": q != "external_test",
            }
    support = {p: summarize(rr, cfg["labels"]) for p, rr in partitions.items() if p != "external_test"}
    failures = [
        {
            "partition": p,
            "class": label,
            "observed": x["unique_plurality"][label],
            "required": cfg["support_minimum"][p],
        }
        for p, x in support.items()
        for label in cfg["labels"]
        if x["unique_plurality"][label] < cfg["support_minimum"][p]
    ]
    slices = {
        p: {
            kind: summarize(
                [r for r in partitions[p] if (r["sentence"] in old_split["sentences"]["train"]) == seen],
                cfg["labels"],
            )
            for kind, seen in [("seen_sentences", True), ("unseen_sentences", False)]
        }
        for p in ("dev", "calib")
    }
    parent_hashes = {str(p.relative_to(v1)): sha(p) for p in sorted(v1.rglob("*")) if p.is_file()}
    root.mkdir(parents=True, exist_ok=True)
    immutable_json(root / "V1_PRESERVATION.json", parent_hashes)
    new_cfg = {
        **cfg,
        "version": "c1-dual-execution-v2",
        "amendment": {
            "parent_contract_sha256": sha(CONTRACT_PATH),
            "only_change": "existing dev/calib speakers use all ten non-external-test sentences",
            "train_identities_preserved": True,
            "external_test_identities_preserved": True,
            "speaker_assignments_preserved": True,
            "targets_quarantines_thresholds_models_preserved": True,
            "dev_calib_generalization": "speaker-disjoint; mixed seen/unseen sentences",
            "external_test_generalization": "jointly speaker-and-sentence-held-out",
        },
    }
    immutable_json(root / "C1_EXPERIMENT_CONTRACT.json", new_cfg)
    immutable_json(root / "C1_SPLIT_IDS.json", split)
    immutable_json(root / "PARTITION_DELTA.json", moved)
    manifest = root / "metadata_manifest.jsonl"
    content = "".join(json.dumps(r, allow_nan=False) + "\n" for r in amended)
    if manifest.exists() and manifest.read_text() != content:
        raise ValueError("Amended manifest differs")
    manifest.write_text(content)
    report = {
        "status": "DATA_PROTOCOL_BLOCKED" if failures else "METADATA_FEASIBLE_MEDIA_AUDIT_REQUIRED",
        "support": support,
        "support_failures": failures,
        "seen_unseen_slices": slices,
        "partition_checks": checks,
        "counts": dict(Counter(r["partition"] for r in amended)),
        "added_dev_rows": sum(r["to"] == "dev" for r in moved),
        "added_calib_rows": sum(r["to"] == "calib" for r in moved),
        "train_rows_identical": True,
        "external_test_rows_identical": True,
        "all_targets_identical": True,
        "all_mismatch_quarantines_preserved": True,
        "identical_audio_check": "pending media acquisition; no waveform hashes available yet",
        "v1_preservation_sha256": sha(root / "V1_PRESERVATION.json"),
        "v1_audit_sha256": sha(v1 / "DATA_AUDIT.json"),
        "original_annotation_revision": cfg["revision"],
    }
    immutable_json(root / "METADATA_AUDIT.json", report)
    if not failures:
        freeze = {
            "version": "c1-protocol-v2",
            "candidate_inference_performed": False,
            "metadata_gate": "PASS",
            "media_gate": "PENDING",
            "files_sha256": {
                name: sha(root / name)
                for name in [
                    "C1_EXPERIMENT_CONTRACT.json",
                    "C1_SPLIT_IDS.json",
                    "PARTITION_DELTA.json",
                    "metadata_manifest.jsonl",
                    "METADATA_AUDIT.json",
                    "V1_PRESERVATION.json",
                ]
            },
        }
        immutable_json(root / "C1_PROTOCOL_FREEZE.json", freeze)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v1-root", type=Path, default=Path("artifacts/roadmap_c1"))
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = p.parse_args()
    r = amend(args.v1_root, args.root)
    print(json.dumps(r, indent=2))
    if r["support_failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

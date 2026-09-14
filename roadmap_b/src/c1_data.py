"""Pinned CREMA-D acquisition and deterministic, label-independent C1 feasibility audit."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import urllib.request

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "c1/C1_EXPERIMENT_CONTRACT.json"


def contract():
    return json.loads(CONTRACT_PATH.read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def reject_lfs_pointer(path):
    with Path(path).open("rb") as f:
        if f.read(128).startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError("Git LFS pointer is not waveform media")


def acquire(root):
    root = Path(root)
    cfg = contract()
    names = [
        "README.md",
        "LICENSE.txt",
        "SentenceFilenames.csv",
        "processedResults/tabulatedVotes.csv",
        "tabulateVotesV2.r",
        "readTabulatedVotes.R",
    ]
    records = []
    for name in names:
        url = f"https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/{cfg['revision']}/{name}"
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # Never replace an existing acquisition silently.
        if not path.exists():
            data = urllib.request.urlopen(url, timeout=60).read()
            temp = path.with_name(path.name + ".partial")
            temp.write_bytes(data)
            temp.replace(path)
        records.append({"path": name, "url": url, "sha256": sha(path), "bytes": path.stat().st_size})
    record = {
        "revision": cfg["revision"],
        "files": records,
        "media_downloaded": False,
        "personal_information_submitted": False,
        "registration_requested_by_readme": True,
    }
    target = root / "ACQUISITION.json"
    if target.exists():
        old = json.loads(target.read_text())
        if old != record:
            raise ValueError("Existing acquisition identity differs; inspect before proceeding")
    else:
        write_json(target, record)
    return record


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def normalize_text(text):
    return " ".join(
        "".join(
            c
            for c in unicodedata.normalize("NFKC", text).lower()
            if not unicodedata.category(c).startswith("P")
        ).split()
    )


def sentences_from_readme(text):
    # Exact sentence block, not the following actor emotion codes.
    section = text.split("Actors spoke from a selection of 12 sentences (")[1].split(
        "The sentences were presented"
    )[0]
    found = re.findall(r"^- (.+?)\s*\(([A-Z]{3})\)\.?\s*$", section, re.MULTILINE)
    if len(found) != 12:
        raise ValueError("Cannot reconcile the pinned twelve sentence definitions")
    return {code: phrase.rstrip(".") for phrase, code in found}


def join_voice_votes(files, votes, cfg):
    by_num, by_name = {}, {}
    for row in files:
        num, name = int(row["Stimulus_Number"]), row["Filename"]
        if num in by_num or name in by_name:
            raise ValueError("Duplicate canonical stimulus identity")
        by_num[num] = name
        by_name[name] = num
    joined, seen, modality_counts, examples = {}, set(), Counter(), []
    for row in votes:
        encoded = int(row[""])
        modality, num = divmod(encoded, 100000)
        if modality not in (1, 2, 3) or num not in by_num:
            raise ValueError("Invalid query-type/stimulus key")
        if encoded in seen:
            raise ValueError("Duplicate annotation query-type/clip key")
        seen.add(encoded)
        if row["fileName"] != by_num[num]:
            raise ValueError("R query-type/clip mapping disagrees with filename")
        modality_counts[modality] += 1
        if modality != cfg["query_type"]:
            continue
        counts = [int(row[k]) for k in cfg["count_columns"]]
        if min(counts) < 0 or sum(counts) != int(row["numResponses"]):
            raise ValueError("Invalid/non-reconciling vote counts")
        name = by_num[num]
        total = sum(counts)
        peak = max(counts)
        unique = counts.count(peak) == 1 and total > 0
        joined[name] = {
            "vote_counts": counts,
            "vote_total": total,
            "target": [c / total for c in counts] if total else None,
            "plurality": counts.index(peak) if unique else None,
            "max_vote": peak / total if total else None,
            "source_query_key": encoded,
            "stimulus_number": num,
        }
        if len(examples) < 5:
            examples.append(
                {
                    "raw_query_key": encoded,
                    "derived_query_type": modality,
                    "derived_stimulus_number": num,
                    "canonical_filename": name,
                    "source_counts": {k: int(row[k]) for k in ["A", "D", "F", "H", "N", "S"]},
                    "reordered_counts": counts,
                    "num_responses": total,
                }
            )
    return (
        by_name,
        joined,
        {
            "modality_row_counts": {str(k): v for k, v in modality_counts.items()},
            "mapping_examples": examples,
            "duplicate_query_keys": 0,
            "duplicate_joined_filenames": 0,
            "mapping_source": "tabulateVotesV2.r: 100000*queryType + clipNum; queryType 1 voice",
        },
    )


def split_ids(speakers, sentences, cfg):
    speakers = sorted(
        set(speakers), key=lambda x: hashlib.sha256((cfg["speaker_prefix"] + x).encode()).hexdigest()
    )
    sentences = sorted(
        set(sentences), key=lambda x: hashlib.sha256((cfg["sentence_prefix"] + x).encode()).hexdigest()
    )
    if len(speakers) != 91 or len(sentences) != 12:
        raise ValueError("Pinned 91-speaker/12-sentence inventory does not reconcile")
    speaker_sets = dict(
        zip(
            ["train", "dev", "calib", "external_test"],
            [speakers[:64], speakers[64:73], speakers[73:82], speakers[82:]],
        )
    )
    text_sets = {
        "train": sentences[:8],
        "dev": sentences[8:10],
        "calib": sentences[8:10],
        "external_test": sentences[10:],
    }
    return {
        "speakers": speaker_sets,
        "sentences": text_sets,
        "algorithm": "SHA256 of specified prefix + canonical string ID; no label input",
    }


def row_partition(speaker, sentence, split):
    for part, ids in split["speakers"].items():
        if speaker in ids and sentence in split["sentences"][part]:
            return part
    return "unused_cross_product"


def triplet_pools(rows, cfg):
    # Construct only from eligible training rows; caller cannot smuggle held-out rows.
    if any(r["partition"] != "train" or not r["supervision_eligible"] for r in rows):
        raise ValueError("Triplets require eligible external training rows only")
    confident = [
        r for r in rows if r["plurality"] is not None and r["max_vote"] >= cfg["triplet"]["confidence_min"]
    ]
    pools = {}
    for r in confident:
        pos = [
            x["id"]
            for x in confident
            if x["speaker"] != r["speaker"]
            and x["sentence"] != r["sentence"]
            and x["plurality"] == r["plurality"]
        ]
        neg = [
            x["id"]
            for x in confident
            if x["speaker"] == r["speaker"]
            and x["sentence"] == r["sentence"]
            and x["plurality"] != r["plurality"]
            and (not r.get("audio_sha256") or x.get("audio_sha256") != r["audio_sha256"])
        ]
        if pos and neg:
            pools[r["id"]] = {"positive": pos, "negative": neg, "class": r["plurality"]}
    return pools


def audit_metadata(acquisition, output):
    acquisition, output = Path(acquisition), Path(output)
    cfg = contract()
    acquired = json.loads((acquisition / "ACQUISITION.json").read_text())
    if acquired["revision"] != cfg["revision"]:
        raise ValueError("Wrong acquisition revision")
    for r in acquired["files"]:
        if sha(acquisition / r["path"]) != r["sha256"]:
            raise ValueError("Acquisition hash mismatch")
    files = read_csv(acquisition / "SentenceFilenames.csv")
    by_name, joined, join_audit = join_voice_votes(
        files, read_csv(acquisition / "processedResults/tabulatedVotes.csv"), cfg
    )
    sentence_text = sentences_from_readme((acquisition / "README.md").read_text())
    speakers = [name.split("_")[0] for name in by_name]
    sentences = [name.split("_")[1] for name in by_name]
    split = split_ids(speakers, sentences, cfg)
    normalized = defaultdict(list)
    for code, text in sentence_text.items():
        normalized[normalize_text(text)].append(code)
    if any(len(codes) > 1 for codes in normalized.values()):
        raise ValueError("Normalized duplicate sentences would leak between text partitions")
    rows = []
    for name in sorted(by_name):
        speaker, sentence, intended, level = name.split("_")
        row = {
            "id": name,
            "speaker": speaker,
            "sentence": sentence,
            "text": sentence_text[sentence],
            "actor_intended_metadata_only": intended,
            "level_metadata_only": level,
            "partition": row_partition(speaker, sentence, split),
            "media_audit": "not_started",
        }
        row.update(
            joined.get(
                name,
                {"vote_counts": None, "vote_total": 0, "target": None, "plurality": None, "max_vote": None},
            )
        )
        reasons = []
        if name in cfg["quarantine_ids"]:
            reasons.append(cfg["quarantine_reason"])
        if row["partition"] == "unused_cross_product":
            reasons.append("unused_speaker_sentence_cross_product")
        if row["vote_total"] == 0:
            reasons.append("missing_or_zero_voice_votes")
        row["exclusion_reasons"] = reasons
        row["supervision_eligible"] = not reasons
        rows.append(row)
    support = {}
    failures = []
    for part, minimum in cfg["support_minimum"].items():
        rr = [r for r in rows if r["partition"] == part and r["supervision_eligible"]]
        counts = Counter(r["plurality"] for r in rr if r["plurality"] is not None)
        support[part] = {
            "soft_ce_rows": len(rr),
            "unique_plurality": {label: counts[i] for i, label in enumerate(cfg["labels"])},
            "ties": sum(r["plurality"] is None for r in rr),
            "required_per_class": minimum,
        }
        for i, label in enumerate(cfg["labels"]):
            if counts[i] < minimum:
                failures.append(
                    {
                        "partition": part,
                        "class": label,
                        "observed_upper_bound": counts[i],
                        "required": minimum,
                        "shortfall": minimum - counts[i],
                    }
                )
    train = [r for r in rows if r["partition"] == "train" and r["supervision_eligible"]]
    pools = triplet_pools(train, cfg)
    classes = Counter(x["class"] for x in pools.values())
    triplet = {
        "anchors_before_media_dedup": len(pools),
        "classes": {cfg["labels"][k]: v for k, v in classes.items()},
        "meets_metadata_upper_bound": len(pools) >= cfg["triplet"]["minimum_anchors"]
        and len(classes) >= cfg["triplet"]["minimum_classes"],
        "audio_hash_distinctness_verified": False,
    }
    cells = Counter((r["speaker"], r["sentence"], r["actor_intended_metadata_only"]) for r in rows)
    all_emotions = sorted({r["actor_intended_metadata_only"] for r in rows})
    incomplete = [
        {"speaker": s, "sentence": t, "actor_intended_metadata_only": e}
        for s in sorted(set(speakers))
        for t in sorted(set(sentences))
        for e in all_emotions
        if not cells[s, t, e]
    ]
    state = "DATA_PROTOCOL_BLOCKED" if failures else "METADATA_FEASIBLE_MEDIA_AUDIT_REQUIRED"
    result = {
        "status": state,
        "counts": {
            "canonical_rows": len(rows),
            "speakers": len(set(speakers)),
            "sentences": len(set(sentences)),
            "voice_rows": len(joined),
            "partitions": dict(Counter(r["partition"] for r in rows)),
            "exclusion_reasons": dict(Counter(x for r in rows for x in r["exclusion_reasons"])),
        },
        "support": support,
        "support_failures": failures,
        "triplet": triplet,
        "join": join_audit,
        "missing_voice_ids": sorted(set(by_name) - set(joined)),
        "incomplete_actor_sentence_emotion_cells": len(incomplete),
        "normalized_transcript_collisions": [],
        "media_audit": {
            "status": "not_started",
            "reason": "annotation support gate precedes media acquisition",
            "checks_pending": [
                "decode",
                "channels",
                "sample_rate",
                "duration",
                "finite_samples",
                "clipping",
                "audio_hashes",
                "cross_partition_audio_duplicates",
            ],
        },
        "interpretation": "Counts are upper bounds before media failures/dedup; media validation cannot repair deficient class support.",
        "contract_sha256": sha(CONTRACT_PATH),
        "acquisition_sha256": sha(acquisition / "ACQUISITION.json"),
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [
        ("DATA_AUDIT.json", result),
        ("C1_SPLIT_IDS.json", split),
        ("incomplete_cells.json", incomplete),
        ("triplet_metadata_pools.json", pools),
    ]:
        target = output / name
        if target.exists() and json.loads(target.read_text()) != value:
            raise ValueError(f"Refusing to overwrite differing audit: {name}")
        write_json(target, value)
    manifest = output / "manifest.jsonl"
    content = "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)
    if manifest.exists() and manifest.read_text() != content:
        raise ValueError("Existing manifest differs")
    manifest.write_text(content)
    split_lock = {
        "contract_sha256": sha(CONTRACT_PATH),
        "split_sha256": sha(output / "C1_SPLIT_IDS.json"),
        "manifest_sha256": sha(manifest),
        "stage": "metadata_only_no_candidate_features",
        "test_predictions": False,
    }
    write_json(output / "C1_SPLIT_LOCK.json", split_lock)
    return result


def require_feasible(audit):
    if audit["status"] != "DATA_READY":
        raise ValueError(f"C1 execution prohibited before complete data feasibility: {audit['status']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["acquire", "audit"])
    p.add_argument("--root", type=Path, default=Path("artifacts/roadmap_c1"))
    args = p.parse_args()
    if args.stage == "acquire":
        value = acquire(args.root / "acquisition")
    else:
        value = audit_metadata(args.root / "acquisition", args.root)
    print(json.dumps(value, indent=2))
    if value.get("status") == "DATA_PROTOCOL_BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

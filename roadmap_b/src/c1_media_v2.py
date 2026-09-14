"""Pinned CREMA-D audio-only acquisition/audit for the locked C1 v2 intersections."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
import soundfile as sf

from .audio import prepare_waveform
from .c1_data import reject_lfs_pointer, sha, triplet_pools, write_json
from .c1_protocol_v2 import immutable_json, summarize


def verify_protocol(root):
    root = Path(root)
    lock = json.loads((root / "C1_PROTOCOL_FREEZE.json").read_text())
    for name, expected in lock["files_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError(f"Frozen protocol changed: {name}")
    cfg = json.loads((root / "C1_EXPERIMENT_CONTRACT.json").read_text())
    assert cfg["version"] == "c1-dual-execution-v2"
    return cfg


def fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as r:
                return r.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def fetch_clip(root, row, cfg):
    rid = row["id"]
    media = root / "media"
    original = media / "original" / f"{rid}.wav"
    pointer = media / "pointers" / f"{rid}.txt"
    record = media / "records" / f"{rid}.json"
    for p in (original, pointer, record):
        p.parent.mkdir(parents=True, exist_ok=True)
    raw_url = f"https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/{cfg['revision']}/AudioWAV/{rid}.wav"
    media_url = f"https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/{cfg['revision']}/AudioWAV/{rid}.wav"
    try:
        if not pointer.exists():
            pointer.write_bytes(fetch(raw_url))
        lines = pointer.read_text().splitlines()
        assert lines[0] == "version https://git-lfs.github.com/spec/v1"
        expected = next(x.split("sha256:")[1] for x in lines if x.startswith("oid sha256:"))
        size = int(next(x.split()[1] for x in lines if x.startswith("size ")))
        if not original.exists():
            data = fetch(media_url)
            if len(data) != size or hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("LFS content identity differs")
            temp = original.with_suffix(".partial")
            temp.write_bytes(data)
            temp.replace(original)
        reject_lfs_pointer(original)
        assert original.stat().st_size == size and sha(original) == expected
    except Exception as exc:
        return {"id": rid, "acquisition_error": f"{type(exc).__name__}: {exc}", "source_url": media_url}
    # Reuse immutable completed media records only after validating all retained bytes.
    if record.exists():
        prior = json.loads(record.read_text())
        if prior["source_sha256"] != expected:
            raise ValueError("Media record source changed")
        if prior.get("audio_path") and sha(root / prior["audio_path"]) != prior["processed_sha256"]:
            raise ValueError("Processed audio changed")
        return prior
    result = {
        "id": rid,
        "source_url": media_url,
        "pointer_url": raw_url,
        "source_sha256": expected,
        "source_bytes": size,
        "pointer_sha256": sha(pointer),
        "audio_valid": False,
        "audio_available": False,
    }
    try:
        audio, rate = sf.read(original, dtype="float32", always_2d=True)
        result.update(
            source_rate=rate,
            source_channels=audio.shape[1],
            source_frames=len(audio),
            source_duration_s=len(audio) / rate,
            finite_samples=bool(np.isfinite(audio).all()),
            peak=float(np.abs(audio).max()) if audio.size else None,
            clipped_sample_fraction=float((np.abs(audio) >= 32767 / 32768).mean()) if audio.size else None,
        )
        wave = prepare_waveform(audio, rate)
        result.update(
            audio_valid=True,
            sample_rate=16000,
            num_samples=len(wave),
            duration_s=len(wave) / 16000,
            pcm_sha256=hashlib.sha256(wave.astype("<f4").tobytes()).hexdigest(),
            audio_available=len(wave) <= cfg["audio_max_s"] * 16000,
            unavailable_reason=None if len(wave) <= cfg["audio_max_s"] * 16000 else "duration_limit",
        )
        dest = media / "mono16k" / f"{rid}.wav"
        dest.parent.mkdir(exist_ok=True)
        temp = dest.with_suffix(".partial")
        sf.write(temp, wave, 16000, format="WAV", subtype="FLOAT")
        temp.replace(dest)
        verified, sr = sf.read(dest, dtype="float32")
        assert sr == 16000 and verified.ndim == 1 and np.array_equal(wave, verified)
        result.update(audio_path=str(dest.relative_to(root)), processed_sha256=sha(dest))
    except Exception as exc:
        result.update(unavailable_reason="invalid_audio", decode_error=f"{type(exc).__name__}: {exc}")
    immutable_json(record, result)
    return result


def audit(root, workers=8):
    root = Path(root)
    cfg = verify_protocol(root)
    if (root / "DATA_FREEZE.json").exists():
        verify_data(root)
        return json.loads((root / "DATA_AUDIT.json").read_text())
    original = [json.loads(x) for x in (root / "metadata_manifest.jsonl").read_text().splitlines()]
    selected = [r for r in original if r["partition"] != "unused_cross_product"]
    results = {}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_clip, root, r, cfg): r["id"] for r in selected}
        for i, f in enumerate(as_completed(futures), 1):
            result = f.result()
            results[result["id"]] = result
            if i % 100 == 0:
                print(f"media {i}/{len(selected)} elapsed={time.perf_counter() - started:.1f}s", flush=True)
    errors = [r for r in results.values() if "acquisition_error" in r]
    if errors:
        result = {
            "status": "ACQUISITION_INCOMPLETE",
            "errors": errors,
            "completed": len(selected) - len(errors),
            "required": len(selected),
        }
        write_json(root / "acquisition_progress.json", result)
        return result
    duplicate_groups = []
    for field in ("source_sha256", "pcm_sha256"):
        groups = defaultdict(list)
        for row in selected:
            if results[row["id"]].get(field):
                groups[results[row["id"]][field]].append(row)
        for digest, rr in groups.items():
            if len(rr) > 1:
                duplicate_groups.append(
                    {
                        "kind": field,
                        "sha256": digest,
                        "ids": [r["id"] for r in rr],
                        "partitions": sorted({r["partition"] for r in rr}),
                        "cross_partition": len({r["partition"] for r in rr}) > 1,
                    }
                )
    cross_ids = {rid for group in duplicate_groups if group["cross_partition"] for rid in group["ids"]}
    rows = []
    for source in selected:
        row = {**source, **results[source["id"]]}
        row["exclusion_reasons"] = list(source["exclusion_reasons"])
        if not row["audio_available"]:
            row["exclusion_reasons"].append(row["unavailable_reason"])
        if row["id"] in cross_ids:
            row["exclusion_reasons"].append("cross_partition_identical_audio")
        row["supervision_eligible"] = not row["exclusion_reasons"]
        row["audio_sha256"] = row.get("pcm_sha256")
        rows.append(row)
    support = {
        p: summarize([r for r in rows if r["partition"] == p], cfg["labels"])
        for p in ("train", "dev", "calib")
    }
    failures = [
        {
            "partition": p,
            "class": label,
            "observed": v["unique_plurality"][label],
            "required": cfg["support_minimum"][p],
        }
        for p, v in support.items()
        for label in cfg["labels"]
        if v["unique_plurality"][label] < cfg["support_minimum"][p]
    ]
    train = [r for r in rows if r["partition"] == "train" and r["supervision_eligible"]]
    pools = triplet_pools(train, cfg)
    cls = Counter(p["class"] for p in pools.values())
    triplet_ok = (
        len(pools) >= cfg["triplet"]["minimum_anchors"] and len(cls) >= cfg["triplet"]["minimum_classes"]
    )
    durations = np.array([r["duration_s"] for r in rows if r["audio_valid"]])
    result = {
        "status": "DATA_PROTOCOL_BLOCKED" if failures else "DATA_READY",
        "support": support,
        "support_failures": failures,
        "triplets": {
            "executable": triplet_ok,
            "anchors": len(pools),
            "classes": {cfg["labels"][k]: v for k, v in cls.items()},
            "hash_distinctness_verified": True,
        },
        "counts": {
            "assigned_rows": len(rows),
            "valid_audio": sum(r["audio_valid"] for r in rows),
            "eligible_supervision": sum(r["supervision_eligible"] for r in rows),
            "quarantined_mismatch": sum(cfg["quarantine_reason"] in r["exclusion_reasons"] for r in rows),
            "external_test_rows": sum(r["partition"] == "external_test" for r in rows),
        },
        "duration_seconds": dict(
            zip(["min", "p50", "p95", "max"], np.quantile(durations, [0, 0.5, 0.95, 1]).tolist())
        ),
        "source_sample_rates": dict(Counter(str(r.get("source_rate")) for r in rows)),
        "source_channels": dict(Counter(str(r.get("source_channels")) for r in rows)),
        "clips_with_full_scale_samples": sum((r.get("clipped_sample_fraction") or 0) > 0 for r in rows),
        "duplicate_groups": duplicate_groups,
        "elapsed_seconds": time.perf_counter() - started,
        "protocol_freeze_sha256": sha(root / "C1_PROTOCOL_FREEZE.json"),
        "acquisition_scope": "Assigned CREMA-D WAVs only; no video or human recordings; external test waveform audit only, no model inference",
    }
    (root / "manifest.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    write_json(root / "DATA_AUDIT.json", result)
    write_json(root / "triplet_pools.json", pools)
    if not failures:
        immutable_json(
            root / "DATA_FREEZE.json",
            {
                "protocol_sha256": sha(root / "C1_PROTOCOL_FREEZE.json"),
                "files_sha256": {
                    n: sha(root / n) for n in ["manifest.jsonl", "DATA_AUDIT.json", "triplet_pools.json"]
                },
                "identity_rule": "Original train/test assignments and all targets retained; media exclusions explicit, no row moves",
            },
        )
    return result


def verify_data(root):
    root = Path(root)
    verify_protocol(root)
    lock = json.loads((root / "DATA_FREEZE.json").read_text())
    assert lock["protocol_sha256"] == sha(root / "C1_PROTOCOL_FREEZE.json")
    for name, digest in lock["files_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError("Frozen audited data changed")
    result = json.loads((root / "DATA_AUDIT.json").read_text())
    assert result["status"] == "DATA_READY"
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("artifacts/roadmap_c1_v2"))
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    r = audit(args.root, args.workers)
    print(json.dumps(r, indent=2))
    if r["status"] != "DATA_READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

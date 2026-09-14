#!/usr/bin/env python3
"""Bounded local raw/temporal verification and one fixed matched benchmark."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio import read_waveform
from src.b_inference import RoadmapB
from src.c1_transfer_runtime import RefinedC1Dual
from src.context import histories
from src.evaluate import probabilities
from src.ta_assets import sha256
from src.ta_interface import SystemPredictor, SystemSession
from src.ta_runtime import PortableTADual, portable_config
from src.utils import sync


def clean_state(state):
    return {k: v for k, v in state.items() if k != "latency_ms"}


def compare(left, right):
    p = np.array(list(left["meld_state"]["distribution"].values()))
    q = np.array(list(right["meld_state"]["distribution"].values()))
    a, b = left["delivery_evidence"], right["delivery_evidence"]
    return {
        "B_state_exact": clean_state(left["meld_state"]) == clean_state(right["meld_state"]),
        "B_probability_max_abs": float(np.max(np.abs(p - q))),
        "D2_probability_max_abs": float(
            np.max(
                np.abs(
                    np.array(list(a["distribution"].values())) - np.array(list(b["distribution"].values()))
                )
            )
        ),
        "D2_vector_max_abs": float(
            np.max(np.abs(np.array(a["normalized_128d"]) - np.array(b["normalized_128d"])))
        ),
    }


def aggregate(records):
    if not records:
        return {"support": 0}
    ms = np.array([r["milliseconds"] for r in records])
    rtf = np.array([r["milliseconds"] / (1000 * r["duration_s"]) for r in records])
    return {
        "support": len(records),
        "p50_ms": float(np.median(ms)),
        "p95_ms": float(np.quantile(ms, 0.95)),
        "max_ms": float(ms.max()),
        "p50_rtf": float(np.median(rtf)),
        "p95_rtf": float(np.quantile(rtf, 0.95)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset-root", required=True)
    p.add_argument("--evidence-root", required=True, type=Path, help="Retained original roadmap_b/artifacts")
    p.add_argument(
        "--baseline-root", required=True, type=Path, help="Retained frozen A reference; validation only"
    )
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--benchmark", action="store_true")
    args = p.parse_args()
    if args.output_root.exists():
        p.error("Output root exists; never rerun a completed block for favorable timing")
    args.output_root.mkdir(parents=True)
    started = time.time()
    record = {
        "status": "INCOMPLETE",
        "hardware": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "numerical_scope": "local same-GPU singleton FP32 text/BF16 WavLM/FP32 summaries and heads",
        "cross_4090_cache_equivalence": "NOT_CLAIMED; known cross-GPU BF16 failure retained",
        "official_test_or_confirmation_inference": False,
    }
    original_source = args.evidence_root.parent
    keys = json.loads((args.evidence_root / "roadmap_b/raw_probe_manifest.json").read_text())["keys"]
    rows = [
        json.loads(x)
        for x in (args.baseline_root / "data/processed/splits/dev_model.jsonl").read_text().splitlines()
    ]
    byid = {f"{r['split']}/{r['dialogue_id']}/{r['utterance_id']}": r for r in rows}
    hist = histories(rows, 3)
    waves = {
        k: read_waveform(
            args.evidence_root / "ta_finalization_v1/validation_data" / (k.replace("/", "_") + ".wav")
        )
        for k in keys
    }
    release = json.loads(
        (Path(__file__).resolve().parents[1] / "evidence/ta_system_closure/RELEASE.json").read_text()
    )
    record["selected_policy"] = release["default_policy"]
    record["release_manifest_sha256"] = sha256(
        Path(__file__).resolve().parents[1] / "evidence/ta_system_closure/RELEASE.json"
    )
    previous = os.getcwd()
    try:
        os.chdir(original_source)
        legacy = RefinedC1Dual(
            args.evidence_root / "roadmap_c1_v2", args.baseline_root, condition="D2", clock=True
        )
    finally:
        os.chdir(previous)
    references = {
        k: legacy.predict_turn(byid[k]["text"], waves[k], 16000, hist[k], speaker=byid[k]["speaker"])
        for k in keys
    }
    legacy.clock.close()
    del legacy
    gc.collect()
    torch.cuda.empty_cache()
    model = PortableTADual(args.asset_root)
    raw = []
    for k in keys:
        r = byid[k]
        calls = model.tap.calls
        got = model.predict_turn(r["text"], waves[k], 16000, hist[k], speaker=r["speaker"])
        checks = compare(references[k], got)
        assert (
            checks["B_state_exact"]
            and checks["D2_probability_max_abs"] <= 1e-7
            and checks["D2_vector_max_abs"] <= 1e-7
        )
        assert model.tap.calls == calls + 1
        # Same-environment cached-head replay, not the historical 4090 feature cache.
        cached = model.tap.last_summaries.detach().cpu().clone()
        value = model.head(cached.to(model.b.device))
        prob = probabilities(value["logits"][0].detach().cpu(), model.temperature)
        error = float(
            np.max(np.abs(prob - np.array(list(got["delivery_evidence"]["distribution"].values()))))
        )
        assert error <= 1e-7
        assert abs(np.linalg.norm(got["delivery_evidence"]["normalized_128d"]) - 1) < 1e-5
        raw.append(
            {
                "id": k,
                "duration_s": len(waves[k]) / 16000,
                "pcm_sha256": hashlib.sha256(waves[k].astype("<f4").tobytes()).hexdigest(),
                **checks,
                "same_environment_cached_D2_max_abs": error,
                "one_pass": True,
            }
        )
    record["raw_probes"] = raw
    probe = keys[0]
    r = byid[probe]
    availability = []
    for name, wave in [
        ("missing", None),
        ("corrupt", np.full(400, np.nan, np.float32)),
        ("long", np.zeros(60 * 16000 + 1, np.float32)),
        ("silent", np.zeros(16000, np.float32)),
    ]:
        calls = model.tap.calls
        got = model.predict_turn(r["text"], wave, 16000 if wave is not None else None, [], speaker="user")
        available = got["delivery_evidence"]["available"]
        assert available == (name == "silent")
        assert model.tap.calls == calls + int(available)
        assert got["meld_state"]["audio_valid"] == (name in ("long", "silent"))
        if not available:
            assert (
                got["delivery_evidence"]["normalized_128d"] is None
                and got["delivery_evidence"]["distribution"] is None
            )
        availability.append(
            {
                "case": name,
                "audio_valid": got["meld_state"]["audio_valid"],
                "audio_available": available,
                "reason": got["delivery_evidence"]["unavailable_reason"],
                "one_pass_or_skip": True,
            }
        )
    record["availability"] = availability
    # Same waveform, different observed histories: audio evidence stays isolated.
    first = model.predict_turn(r["text"], waves[probe], 16000, [], speaker=r["speaker"])
    second = model.predict_turn(r["text"], waves[probe], 16000, hist[probe], speaker=r["speaker"])
    assert first["delivery_evidence"]["distribution"] == second["delivery_evidence"]["distribution"]
    assert first["delivery_evidence"]["normalized_128d"] == second["delivery_evidence"]["normalized_128d"]
    record["context_isolation"] = {
        "D2_exact_across_histories": True,
        "history_length": len(hist[probe]),
        "B_distribution_changes": first["meld_state"]["distribution"] != second["meld_state"]["distribution"],
    }
    adapter = SystemPredictor(model, release, policy=release["default_policy"])
    session = SystemSession(adapter)
    session.start()
    temporal = []
    for k in ("dev/99/3", "dev/99/4", "dev/99/5", "dev/99/6"):
        row = byid[k]
        session.begin_turn(speaker=row["speaker"])
        wave = waves.get(k)
        if wave is not None:
            for offset in range(0, len(wave), 1600):
                session.push_audio(wave[offset : offset + 1600])
        session.set_transcript(row["text"])
        snap = session.end_turn().to_dict()
        assert snap["response"] == snap["interaction"]["response"]
        adapter.acknowledge_emitted(snap["turn_id"])
        temporal.append(
            {
                "id": k,
                "history_length": len(snap["observed_context"]),
                "audio_available": snap["acoustic_evidence"]["available"],
                "route_id": snap["interaction"]["route_id"],
                "state_reached_response": True,
            }
        )
    record["temporal_replay"] = temporal
    # Real tap failure after a successful turn must not preserve old summaries.
    forward = model.tap.encoder.backbone.forward

    def fail(*a, **kw):
        raise RuntimeError("intentional validation failure")

    model.tap.encoder.backbone.forward = fail
    try:
        try:
            model.predict_turn(r["text"], waves[probe], 16000, [], speaker=r["speaker"])
        except RuntimeError as exc:
            assert "intentional" in str(exc)
        else:
            raise AssertionError("failure injection did not fail")
    finally:
        model.tap.encoder.backbone.forward = forward
    assert model.tap.last_summaries is None
    got = model.predict_turn(r["text"], None, None, [], speaker=r["speaker"])
    assert not got["delivery_evidence"]["available"] and got["delivery_evidence"]["normalized_128d"] is None
    record["failure_cleanup"] = "PASS"
    record["parameter_ledger"] = model.parameter_ledger()
    record["runtime_source"] = str(Path(sys.modules["src.ta_runtime"].__file__).resolve())
    model.close()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if args.benchmark:
        workload = json.loads((args.evidence_root / "roadmap_c1_v2/RUNTIME_WORKLOAD.json").read_text())
        sequence = workload["sequence"]
        assert sequence == [keys[i % len(keys)] for i in range(200)] and workload["warmups"] == 30
        blocks = {}
        all_records = []
        for mode in ("B", "legacy_B_D2", "selected_system"):
            gc.collect()
            torch.cuda.empty_cache()
            if mode == "B":
                current = RoadmapB(portable_config(args.asset_root, "cuda"))
            elif mode == "legacy_B_D2":
                try:
                    os.chdir(original_source)
                    current = RefinedC1Dual(
                        args.evidence_root / "roadmap_c1_v2", args.baseline_root, condition="D2", clock=True
                    )
                finally:
                    os.chdir(previous)
            else:
                current = PortableTADual(args.asset_root)
                wrapper = SystemPredictor(current, release, policy=release["default_policy"])

            def call(k):
                row = byid[k]
                target = wrapper if mode == "selected_system" else current
                return target.predict_turn(row["text"], waves[k], 16000, hist[k], speaker=row["speaker"])

            for i in range(30):
                call(keys[i % len(keys)])
            samples = []
            for k in sequence:
                sync(torch.device("cuda"))
                t = time.perf_counter()
                call(k)
                sync(torch.device("cuda"))
                samples.append(
                    {
                        "mode": mode,
                        "id": k,
                        "duration_s": len(waves[k]) / 16000,
                        "milliseconds": (time.perf_counter() - t) * 1000,
                    }
                )
            strata = {}
            for low, high, label in [
                (0, 0.25, "under_250ms"),
                (0.25, 1, "250ms_to_1s"),
                (1, 3, "1_to_3s"),
                (3, 6, "3_to_6s"),
                (6, 12, "6_to_12s"),
                (12, 60, "12_to_60s"),
                (60, float("inf"), "over_60s"),
            ]:
                strata[label] = aggregate([x for x in samples if low <= x["duration_s"] < high])
            first_block = aggregate(samples[:30])
            last_block = aggregate(samples[-30:])
            blocks[mode] = {
                "all": aggregate(samples),
                "at_most_12s": aggregate([x for x in samples if x["duration_s"] <= 12]),
                "strata": strata,
                "first30": first_block,
                "last30": last_block,
                "drift_last_first_p50_ratio": last_block["p50_ms"] / first_block["p50_ms"],
            }
            all_records.extend(samples)
            if mode == "legacy_B_D2":
                current.clock.close()
            if mode == "selected_system":
                current.close()
                del wrapper
            del current
        (args.output_root / "BENCHMARK_SAMPLES.json").write_text(json.dumps(all_records, indent=2) + "\n")
        record["benchmark"] = {
            "order": ["B", "legacy_B_D2", "selected_system"],
            "warmups_per_mode": 30,
            "turns_per_mode": 200,
            "boundary": "predecoded waveform + observed transcript/history to full returned state/response; GPU synchronized; disk decode/load excluded",
            "blocks": blocks,
            "nonstationarity": "Single sequential block; drift ratios reported, no best-of reruns; hardware-local evidence only",
            "absolute_p95_100ms": blocks["selected_system"]["at_most_12s"]["p95_ms"] <= 100,
            "incremental_p95_10ms": blocks["selected_system"]["at_most_12s"]["p95_ms"]
            - blocks["B"]["at_most_12s"]["p95_ms"]
            <= 10,
            "rtf_p95_0_1": blocks["selected_system"]["all"]["p95_rtf"] <= 0.1,
        }
    record["status"] = "PASS"
    record["elapsed_s"] = time.time() - started
    (args.output_root / "RAW_VALIDATION.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in record.items() if k not in ("raw_probes", "parameter_ledger")}, indent=2)
    )


if __name__ == "__main__":
    main()

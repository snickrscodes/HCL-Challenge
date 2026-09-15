#!/usr/bin/env python3
"""Raw15 and saved1109 stateful regression; no training or test-set inference."""

# ruff: noqa: E402 -- checkout-local imports follow explicit source-path setup
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch

from src.audio import read_waveform
from src.b_data import load_inputs
from src.b_evaluate import distributions
from src.b_train import load_model, predict
from src.constants import LABELS
from src.context import histories
from src.interaction import StatefulSession
from src.interaction.replay import ReplayRecorder, replay
from src.responses import respond
from src.ta_assets import sha256
from src.ta_interface import SystemPredictor
from src.ta_policy import select_response
from src.ta_release import canonical
from src.ta_runtime import PortableTADual, portable_config
from src.utils import key, sync


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def difference(a, b):
    if a is None or b is None:
        require(a is None and b is None, "Evidence availability differs")
        return 0.0
    if isinstance(a, dict):
        require(set(a) == set(b), "Evidence namespaces differ")
        a, b = [a[k] for k in sorted(a)], [b[k] for k in sorted(b)]
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def compare(old, new, old_logits=None, new_logits=None):
    state, evidence = old["meld_state"], old["acoustic_evidence"]
    actual, acoustic = new["B_output"], new["D2_output"]

    def clean(value):
        return {k: v for k, v in value.items() if k != "latency_ms"}

    result = {
        "B_state_exact": clean(state) == clean(actual),
        "B_probability_max_abs": difference(state["distribution"], actual["distribution"]),
        "D2_evidence_exact": evidence == acoustic,
        "D2_probability_max_abs": difference(evidence["distribution"], acoustic["distribution"]),
        "D2_vector_max_abs": difference(evidence["normalized_128d"], acoustic["normalized_128d"]),
        "response_exact": old["interaction"] == new["proposed_action"],
        "context_exact": old["observed_context"]
        == [{"text": r["text"], "speaker": r["speaker"]} for r in new["context"]],
    }
    if old_logits is not None:
        result["B_logits_exact"] = bool(torch.equal(old_logits, new_logits))
        result["B_logit_max_abs"] = float((old_logits - new_logits).abs().max())
    require(all(v for k, v in result.items() if k.endswith("exact")), str(result))
    require(all(v == 0 for k, v in result.items() if k.endswith("max_abs")), str(result))
    return result


def timed_event(samples, callback, *args):
    started = time.perf_counter()
    result = callback(*args)
    if samples is not None:
        samples.append((time.perf_counter() - started) * 1000)
    return result


def prepare(model, release, row, wave, history=(), *, name="raw", events=None, corrupt=False):
    session = StatefulSession(model, release, session_id=name, clock=lambda: 0.0, history=history)
    timed_event(events, session.start)
    participant = timed_event(events, session.register_participant, row["speaker"])
    turn = timed_event(events, session.start_turn, participant)
    timed_event(events, session.finalize_text, turn, row["text"])
    if corrupt:
        timed_event(events, session.mark_audio_unavailable, turn)
    elif wave is not None:
        for offset in range(0, len(wave), 1600):
            timed_event(events, session.observe_audio_chunk, turn, wave[offset : offset + 1600])
    return session, turn


def quantiles(values):
    if not values:
        return {"support": 0}
    return {
        "support": len(values),
        "p50_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "max_ms": float(np.max(values)),
    }


def raw_validation(model, release, rows, keys, waves, report):
    byid, history = {key(r): r for r in rows}, histories(rows, 3)
    old = SystemPredictor(model, release, policy="evidence_only")
    logits = {}

    def capture(_module, _args, result):
        logits["value"] = result["logits"].detach().cpu().clone()

    hook = model.b.residual.register_forward_hook(capture)
    try:
        records, event_times = [], []
        for k in keys:
            row = byid[k]
            calls = model.tap.calls
            reference = old.predict_turn(
                row["text"], waves[k], 16000, history[k], speaker=row["speaker"]
            ).to_dict()
            reference_logits = logits["value"]
            require(model.tap.calls == calls + 1, "Old shared WavLM count differs")
            session, turn = prepare(
                model,
                release,
                row,
                waves[k],
                history[k],
                name="raw-" + k.replace("/", "-"),
                events=event_times,
            )
            calls = model.tap.calls
            try:
                snap = session.end_turn(turn).to_dict()
                checks = compare(reference, snap, reference_logits, logits["value"])
                require(model.tap.calls == calls + 1, "Stateful shared WavLM count differs")
                require(session.get_response(turn) == reference["response"], "Response getter differs")
                records.append(
                    {
                        "id": k,
                        "duration_s": len(waves[k]) / 16000,
                        "pcm_sha256": hashlib.sha256(waves[k].astype("<f4").tobytes()).hexdigest(),
                        "one_shared_wavlm_execution": True,
                        **checks,
                    }
                )
            finally:
                session.close()
        report["raw_probes"] = records
        report["ordinary_event_handling"] = quantiles(event_times)
        row, availability = byid[keys[0]], []
        cases = [
            ("missing", None),
            ("corrupt", np.full(400, np.nan, np.float32)),
            ("over_limit", np.zeros(960001, np.float32)),
            ("valid_silence", np.zeros(16000, np.float32)),
        ]
        for name, wave in cases:
            reference = old.predict_turn(
                row["text"], wave, 16000 if wave is not None else None, [], speaker=row["speaker"]
            ).to_dict()
            reference_logits = logits["value"]
            session, turn = prepare(
                model, release, row, wave, name="availability-" + name, corrupt=name == "corrupt"
            )
            calls = model.tap.calls
            try:
                snap = session.end_turn(turn).to_dict()
                checks = compare(reference, snap, reference_logits, logits["value"])
                available = snap["D2_output"]["available"]
                require(available == (name == "valid_silence"), "Fallback availability differs")
                require(model.tap.calls == calls + int(available), "Fallback WavLM count differs")
                availability.append(
                    {
                        "case": name,
                        "audio_valid": snap["B_output"]["audio_valid"],
                        "audio_available": available,
                        "reason": snap["D2_output"]["unavailable_reason"],
                        **checks,
                    }
                )
            finally:
                session.close()
        report["availability"] = availability
        recorder = ReplayRecorder()
        session = StatefulSession(model, release, session_id="temporal", clock=lambda: 0.0, recorder=recorder)
        session.start()
        previous, participants, temporal = [], {}, []
        try:
            for k in ("dev/99/3", "dev/99/4", "dev/99/5", "dev/99/6"):
                row, wave = byid[k], waves.get(k)
                reference = old.predict_turn(
                    row["text"], wave, 16000 if wave is not None else None, previous, speaker=row["speaker"]
                ).to_dict()
                reference_logits = logits["value"]
                if row["speaker"] not in participants:
                    participants[row["speaker"]] = session.register_participant(row["speaker"])
                turn = session.start_turn(participants[row["speaker"]])
                session.finalize_text(turn, row["text"])
                if wave is not None:
                    for offset in range(0, len(wave), 1600):
                        session.observe_audio_chunk(turn, wave[offset : offset + 1600])
                snap = session.end_turn(turn).to_dict()
                checks = compare(reference, snap, reference_logits, logits["value"])
                session.record_robot_emitted(turn)
                previous = list(session.history)
                temporal.append(
                    {
                        "id": k,
                        "history_length": len(snap["context"]),
                        "audio_available": snap["D2_output"]["available"],
                        **checks,
                    }
                )
            expected_state = session.to_dict()
            replayed = replay(recorder.to_records(), model, release)
            try:
                require(replayed.to_dict() == expected_state, "Real-model completion-aware replay differs")
                report["model_replay"] = {
                    "status": "PASS",
                    "replayed_final_turns": 4,
                    "state_sha256": hashlib.sha256(canonical(expected_state).encode()).hexdigest(),
                    "scope": "Exact semantic state/snapshots/execution; measured timing excluded",
                }
            finally:
                replayed.close()
        finally:
            session.close()
        report["temporal_replay"] = temporal
        require([r["history_length"] for r in temporal] == [0, 1, 2, 3], "Temporal history differs")
        require(
            [r["audio_available"] for r in temporal] == [True, False, False, True],
            "Temporal availability differs",
        )
    finally:
        hook.remove()


class SavedPredictor:
    """Validation-only record replay; checks actual observed predictor arguments."""

    def __init__(self):
        self.row = self.history = self.value = None
        self.calls = 0

    def predict_turn(self, text, audio, sample_rate, history, *, speaker="user"):
        require(
            text == self.row["text"] and speaker == self.row["speaker"], "Saved transcript/speaker differs"
        )
        require(list(history) == list(self.history), "Saved finalized human context differs")
        require(audio is None, "Saved-feature replay must not invent raw acoustic observations")
        self.calls += 1
        return copy.deepcopy(self.value)


def saved_inventory(args, release, report):
    cfg = portable_config(args.asset_root, "cpu")
    cfg.update(baseline_root=str(args.baseline_root), output_root=str(args.evidence_root / "roadmap_b"))
    head, _ = load_model(args.asset_root / "b/runs/adaptive/1337/checkpoint.pt")
    head.requires_grad_(False)
    temperature = read_json(args.asset_root / "b/evaluation/adaptive/1337/calibration.json")["temperature"]
    text_temperature = read_json(args.asset_root / "a/artifacts/calibration.json")["text"]
    totals, allkeys = [], set()
    for split, name, expected in (
        ("dev_model", "predictions.jsonl", 839),
        ("dev_calib", "dev_calib_predictions.jsonl", 270),
    ):
        rows, inputs = load_inputs(cfg, split)
        raw = predict(head, inputs)
        probs = distributions(raw["logits"], inputs, temperature, text_temperature)
        oracle_path = args.evidence_root / "roadmap_b/evaluation/adaptive/1337" / name
        oracle = read_rows(oracle_path)
        require(len(rows) == len(oracle) == expected, "Frozen inventory count differs")
        require([key(r) for r in rows] == [r["key"] for r in oracle], "Oracle keys differ")
        require(not allkeys.intersection(key(r) for r in rows), "Development partitions overlap")
        allkeys.update(key(r) for r in rows)
        ref_p, ref_z = (np.asarray([r[n] for r in oracle]) for n in ("probabilities", "raw_logits"))
        require(np.array_equal(probs.argmax(-1), ref_p.argmax(-1)), "Frozen categories changed")
        require(
            inputs["available"].tolist() == [r["audio_available"] for r in oracle],
            "Saved availability changed",
        )
        history = histories(rows, 3)
        sessions, participants, callback, snapshots = {}, {}, SavedPredictor(), []
        try:
            for index, row in enumerate(rows):
                k, dialogue = key(row), str(row["dialogue_id"])
                available = bool(inputs["available"][index])
                emotion = LABELS[int(probs[index].argmax())]
                confidence = float(probs[index].max())
                state = {
                    "emotion": emotion,
                    "confidence": confidence,
                    "distribution": dict(zip(LABELS, probs[index].tolist())),
                    "audio_valid": bool(row["audio_valid"]),
                    "audio_available": available,
                    "audio_unavailable_reason": None if available else "missing_or_invalid",
                    "response": respond(row["text"], emotion, confidence),
                    "raw_logits": raw["logits"][index].tolist(),
                }
                evidence = {
                    "available": False,
                    "unavailable_reason": "saved_B_features_only",
                    "namespace": "crema6_audio_votes_v1",
                    "distribution": None,
                    "normalized_128d": None,
                    "provenance": {"source": "saved_B_regression"},
                }
                callback.row, callback.history = row, history[k]
                callback.value = {
                    "meld_state": state,
                    "delivery_evidence": evidence,
                    "latency_ms": {"total": 0.0},
                }
                reference = (
                    SystemPredictor(callback, release)
                    .predict_turn(row["text"], None, None, history[k], speaker=row["speaker"])
                    .to_dict()
                )
                if dialogue not in sessions:
                    sessions[dialogue] = StatefulSession(
                        callback, release, session_id=split + "-" + dialogue, clock=lambda: 0.0
                    )
                    sessions[dialogue].start()
                    participants[dialogue] = {}
                session = sessions[dialogue]
                if row["speaker"] not in participants[dialogue]:
                    participants[dialogue][row["speaker"]] = session.register_participant(row["speaker"])
                turn = session.start_turn(participants[dialogue][row["speaker"]])
                session.finalize_text(turn, row["text"])
                snap = session.end_turn(turn).to_dict()
                compare(reference, snap)
                require(
                    session.get_response(turn) == select_response(row["text"], state, evidence)["response"],
                    "Saved deterministic response differs",
                )
                session.record_robot_emitted(turn)
                snapshots.append(hashlib.sha256(canonical(snap).encode()).hexdigest())
        finally:
            for session in sessions.values():
                session.close()
        require(callback.calls == expected * 2, "Saved predictor call count differs")
        paths = [
            args.baseline_root / "data/processed/splits" / (split + ".jsonl"),
            oracle_path,
            args.evidence_root / "roadmap_b/features" / (split + ".pt"),
            args.evidence_root / "roadmap_b/text_features" / (split + ".pt"),
        ]
        totals.append(
            {
                "split": split,
                "rows": expected,
                "dialogues": len(sessions),
                "categories_preserved": expected,
                "responses_exact": expected,
                "adapter_exact": True,
                "audio_available": int(inputs["available"].sum()),
                "historical_probability_max_abs": float(np.max(np.abs(probs - ref_p))),
                "historical_logit_max_abs": float(np.max(np.abs(raw["logits"].numpy() - ref_z))),
                "snapshot_sequence_sha256": hashlib.sha256(canonical(snapshots).encode()).hexdigest(),
                "input_sha256": {str(p): sha256(p) for p in paths},
            }
        )
    require(len(allkeys) == 1109, "Complete frozen inventory not replayed")
    report["saved_inventory"] = {
        "status": "PASS",
        "rows": len(allkeys),
        "dialogues": sum(r["dialogues"] for r in totals),
        "scope": "Unchanged CPU B head on frozen 4090-origin dev features; exact state-adapter record replay",
        "raw_encoder_inference": False,
        "D2": "Unavailable: saved representation-only cache not used",
        "historical_numeric_comparison": "Diagnostic; strict category preservation; no historical JSON rounding",
        "same_environment_adapter_comparison": "Bit-exact; no numerical tolerance",
        "splits": totals,
    }


def benchmark(model, release, rows, keys, waves, report, output_root):
    byid, history = {key(r): r for r in rows}, histories(rows, 3)
    records, event_times, blocks = [], [], {}
    for mode in ("old", "stateful"):
        old = SystemPredictor(model, release)
        for iteration in range(230):
            k = keys[(iteration if iteration < 30 else iteration - 30) % len(keys)]
            row, wave = byid[k], waves[k]
            session = turn = None
            if mode == "stateful":
                session, turn = prepare(
                    model,
                    release,
                    row,
                    wave,
                    history[k],
                    name="benchmark-" + str(iteration),
                    events=event_times if iteration >= 30 else None,
                )
            try:
                sync(model.b.device)
                started = time.perf_counter()
                if mode == "old":
                    old.predict_turn(row["text"], wave, 16000, history[k], speaker=row["speaker"])
                else:
                    session.end_turn(turn)
                sync(model.b.device)
                elapsed = (time.perf_counter() - started) * 1000
                if iteration >= 30:
                    item = {"mode": mode, "id": k, "milliseconds": elapsed, "duration_s": len(wave) / 16000}
                    if session is not None:
                        item["state_timing"] = dict(session.last_timing)
                    records.append(item)
            finally:
                if session is not None:
                    session.close()
                old.snapshots.clear()
        current = [r for r in records if r["mode"] == mode]
        blocks[mode] = {
            "all": quantiles([r["milliseconds"] for r in current]),
            "at_most_12s": quantiles([r["milliseconds"] for r in current if r["duration_s"] <= 12]),
            "p95_rtf": float(
                np.quantile([r["milliseconds"] / (1000 * r["duration_s"]) for r in current], 0.95)
            ),
        }
    state_samples = [r["state_timing"] for r in records if r["mode"] == "stateful"]
    timing = {
        name: quantiles([r[name] for r in state_samples])
        for name in ("predictor_ms", "finalization_overhead_ms", "state_ready_to_predictor_ms", "total_ms")
    }
    event_summary = quantiles(event_times)
    write_json(output_root / "BENCHMARK_SAMPLES.json", records)
    report["benchmark"] = {
        "order": ["old", "stateful"],
        "warmups_per_mode": 30,
        "measured_turns_per_mode": 200,
        "blocks": blocks,
        "state_timings": timing,
        "boundary": "Identical raw15 cyclic workload; state ingestion outside final timing; GPU synchronized",
        "ordinary_events": event_summary,
        "event_p95_5ms": event_summary["p95_ms"] <= 5,
        "finalization_overhead_p95_5ms": timing["finalization_overhead_ms"]["p95_ms"] <= 5,
        "known_limitation": "Ultrashort-clip RTF remains; no cross-GPU speed or equivalence claim",
    }
    require(report["benchmark"]["event_p95_5ms"], "Event handling exceeds 5 ms p95")
    require(report["benchmark"]["finalization_overhead_p95_5ms"], "Finalization overhead exceeds 5 ms p95")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("asset-root", "baseline-root", "evidence-root", "output-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    for name in ("asset_root", "baseline_root", "evidence_root", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output_root.exists():
        parser.error("Output root already exists; preserve completed blocks")
    args.output_root.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "INCOMPLETE",
        "official_test_inference": False,
        "new_training_or_calibration": False,
        "optional_temporal_experiments": 0,
        "source": str(Path(__file__).resolve()),
        "source_sha256": sha256(__file__),
    }
    started, model = time.perf_counter(), None
    try:
        require(torch.cuda.is_available(), "Required local GPU is unavailable")
        device = torch.cuda.get_device_name()
        require("5080" in device and "Laptop" in device, "Acceptance requires local RTX 5080 Laptop")
        report["environment"] = {
            "gpu": device,
            "torch": torch.__version__,
            "python": sys.version,
            "cuda": torch.version.cuda,
            "offline": True,
            "cross_4090_raw_equivalence": "NOT_CLAIMED",
        }
        keys = read_json(args.evidence_root / "roadmap_b/raw_probe_manifest.json")["keys"]
        require(len(keys) == len(set(keys)) == 15, "Established raw15 inventory differs")
        rows = read_rows(args.baseline_root / "data/processed/splits/dev_model.jsonl")
        require(all(r["split"] == "dev" for r in rows), "Non-development input rejected")
        paths = {
            k: args.evidence_root / "ta_finalization_v1/validation_data" / (k.replace("/", "_") + ".wav")
            for k in keys
        }
        waves = {k: read_waveform(path) for k, path in paths.items()}
        report["raw_input_sha256"] = {k: sha256(path) for k, path in paths.items()}
        release = read_json(PROJECT / "evidence/ta_system_closure/RELEASE.json")
        model = PortableTADual(args.asset_root)
        report["asset_verification"] = model.asset_verification
        report["parameter_ledger"] = model.parameter_ledger()
        raw_validation(model, release, rows, keys, waves, report)
        saved_inventory(args, release, report)
        if args.benchmark:
            benchmark(model, release, rows, keys, waves, report, args.output_root)
        sources = {
            name: str(Path(module.__file__).resolve())
            for name, module in sys.modules.items()
            if name.startswith("src.") and getattr(module, "__file__", None)
        }
        require(
            all(Path(path).is_relative_to(PROJECT) for path in sources.values()),
            "Source imported from outside this checkout",
        )
        report["source_isolation"] = {
            "project_modules": len(sources),
            "all_from_checkout": True,
            "sha256": {name: sha256(path) for name, path in sources.items()},
        }
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if model is not None:
            model.close()
        report["elapsed_s"] = time.perf_counter() - started
        write_json(args.output_root / "STATEFUL_VALIDATION.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "raw_probes": len(report["raw_probes"]),
                "saved_rows": report["saved_inventory"]["rows"],
                "temporal_turns": len(report["temporal_replay"]),
                "output_root": str(args.output_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

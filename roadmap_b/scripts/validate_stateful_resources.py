#!/usr/bin/env python3
"""Bounded, model-free stateful retention and overhead acceptance.

Timing: 1,000 fixed turns without tracemalloc. Memory: a separate 10,000-turn
session with tracemalloc, plus 100 closed-session weak-reference checks.
Only synthetic dialogue/PCM and a fixed test predictor are used.
"""

# ruff: noqa: E402 -- checkout-local imports follow explicit source-path setup
import argparse
from collections import Counter, deque
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import tracemalloc
import weakref

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np

from src.interaction import StatefulSession
from src.interaction.state import Limits
from src.responses import respond

MEMORY_GROWTH_LIMIT_BYTES = 1024 * 1024
PERFORMANCE_TURNS = 1000
MEMORY_TURNS = 10000
CLEANUP_SESSIONS = 100
MANIFEST = {"release_id": "synthetic-resource-validation", "condition": "D2"}
PCM = np.linspace(-0.2, 0.2, 400, dtype=np.float32)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class FixtureFailure(RuntimeError):
    pass


class FixedPredictor:
    """Small deterministic fixture; counters have no retained input references."""

    numerical_scope = "synthetic fixture; no learned inference"

    def __init__(self):
        self.calls = 0
        self.close_calls = 0

    def predict_turn(self, text, audio, sample_rate, history, *, speaker="user"):
        self.calls += 1
        require(
            audio is not None and audio.shape == (400,) and sample_rate == 16000,
            "Synthetic input shape/rate changed",
        )
        require(
            len(history) <= 3 and speaker in ("external-a", "external-b"),
            "Synthetic history/participant contract changed",
        )
        if text == "Forced validation failure.":
            raise FixtureFailure("deliberate fixture failure")
        state = {
            "emotion": "neutral",
            "confidence": 0.7,
            "distribution": {
                "neutral": 0.7,
                "joy": 0.05,
                "surprise": 0.05,
                "anger": 0.05,
                "sadness": 0.05,
                "disgust": 0.05,
                "fear": 0.05,
            },
            "audio_valid": True,
            "audio_available": True,
            "audio_unavailable_reason": None,
            "response": respond(text, "neutral", 0.7),
        }
        evidence = {
            "available": True,
            "unavailable_reason": None,
            "namespace": "crema6_audio_votes_v1",
            "distribution": {
                "neutral": 0.5,
                "happy": 0.1,
                "sad": 0.1,
                "angry": 0.1,
                "fear": 0.1,
                "disgust": 0.1,
            },
            "normalized_128d": [1.0] + [0.0] * 127,
            "provenance": {"source": "synthetic_resource_fixture"},
        }
        return {"meld_state": state, "delivery_evidence": evidence, "latency_ms": {"total": 0.0}}

    def close(self):
        self.close_calls += 1


class Measurements:
    """Bounded measurement storage; never retain session, event or audio objects."""

    def __init__(self):
        self.events = deque(maxlen=8192)
        self.finalizations = deque(maxlen=1024)
        self.overheads = deque(maxlen=1024)
        self.event_count = 0
        self.finalization_count = 0

    def ordinary(self, fn, *args):
        started = time.perf_counter()
        try:
            return fn(*args)
        finally:
            self.events.append((time.perf_counter() - started) * 1000)
            self.event_count += 1

    def final(self, session, turn):
        started = time.perf_counter()
        try:
            return session.end_turn(turn)
        finally:
            self.finalizations.append((time.perf_counter() - started) * 1000)
            self.finalization_count += 1


def call(measurements, fn, *args):
    return measurements.ordinary(fn, *args) if measurements is not None else fn(*args)


def bounds(session, *, closed=False):
    limits, counts = session.limits, session.retained_counts()
    maximum = {
        "participants": limits.max_participants,
        "turns": limits.retained_turns,
        "snapshots": limits.retained_snapshots,
        "observations": limits.retained_observations,
        "execution": limits.retained_execution,
        "dialogue": limits.retained_turns,
        "dedup": limits.dedup_events,
        "history": 3,
        "audio_bytes": 0,
    }
    require(set(counts) == set(maximum), "Resource diagnostic schema changed")
    for name, count in counts.items():
        require(0 <= count <= maximum[name], f"Unbounded {name}: {count}")
        if closed and name != "dedup":
            require(count == 0, f"Closed session retains {name}: {count}")
    return counts


def create_session(model, name, measurements=None):
    session = StatefulSession(model, MANIFEST, session_id=name, clock=lambda: 0.0)
    call(measurements, session.start)
    participants = tuple(
        call(measurements, session.register_participant, key) for key in ("external-a", "external-b")
    )
    return session, participants


def one_turn(session, participants, index, measurements=None):
    turn = call(measurements, session.start_turn, participants[index % 2])
    call(measurements, session.observe_text_fragment, turn, "Observed ")
    call(measurements, session.observe_text_fragment, turn, "turn")
    call(measurements, session.observe_audio_chunk, turn, PCM)
    if index % 10 == 0:
        call(measurements, session.cancel_turn, turn)
        require(session.retained_counts()["audio_bytes"] == 0, "Cancelled turn retains PCM")
        return "cancelled"
    failing = index % 17 == 0
    text = "Forced validation failure." if failing else "Observed turn."
    call(measurements, session.finalize_text, turn, text)
    try:
        snapshot = measurements.final(session, turn) if measurements else session.end_turn(turn)
    except FixtureFailure:
        require(failing, "Unscheduled fixture failure")
        require(session.retained_counts()["audio_bytes"] == 0, "Failed turn retains PCM")
        return "failed"
    require(not failing and snapshot is not None, "Expected failure or final snapshot missing")
    if measurements is not None:
        measurements.overheads.append(session.last_timing["finalization_overhead_ms"])
    require(snapshot.to_dict()["stage"] == "final", "Finalization returned partial snapshot")
    call(measurements, session.record_robot_emitted, turn)
    require(session.retained_counts()["audio_bytes"] == 0, "Successful turn retains PCM")
    return "complete"


def quantiles(values):
    values = list(values)
    require(bool(values), "No timing samples")
    return {
        "support": len(values),
        "p50_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "max_ms": float(np.max(values)),
    }


def performance(model):
    require(not tracemalloc.is_tracing(), "Timing must run without allocation tracing")
    measurements, outcomes = Measurements(), Counter()
    session, participants = create_session(model, "performance", measurements)
    try:
        for index in range(1, PERFORMANCE_TURNS + 1):
            outcomes[one_turn(session, participants, index, measurements)] += 1
            bounds(session)
        counts = bounds(session)
    finally:
        session.close()
    closed = bounds(session, closed=True)
    result = {
        "turns": PERFORMANCE_TURNS,
        "tracemalloc_enabled": False,
        "ordinary_event_count": measurements.event_count,
        "ordinary_events": quantiles(measurements.events),
        "finalization_count": measurements.finalization_count,
        "finalization_wall": quantiles(measurements.finalizations),
        "successful_finalization_overhead": quantiles(measurements.overheads),
        "measurement_storage_limits": {"events": 8192, "finalizations": 1024, "overheads": 1024},
        "outcomes": dict(outcomes),
        "retained_at_end": counts,
        "retained_after_close": closed,
    }
    result["ordinary_event_p95_5ms"] = result["ordinary_events"]["p95_ms"] <= 5.0
    result["finalization_p95_5ms"] = result["finalization_wall"]["p95_ms"] <= 5.0
    result["overhead_p95_5ms"] = result["successful_finalization_overhead"]["p95_ms"] <= 5.0
    return result


def memory_stress(model):
    gc.collect()
    tracemalloc.start()
    checkpoints, outcomes = {}, Counter()
    session, participants = create_session(model, "memory")
    try:
        for index in range(1, MEMORY_TURNS + 1):
            outcomes[one_turn(session, participants, index)] += 1
            counts = bounds(session)
            if index in (1000, 5000, 10000):
                gc.collect()
                live, peak = tracemalloc.get_traced_memory()
                checkpoints[str(index)] = {"live_bytes": live, "peak_bytes": peak, "retained_counts": counts}
        session.close()
        after_close = bounds(session, closed=True)
        gc.collect()
        close_live, close_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    growth = checkpoints["10000"]["live_bytes"] - checkpoints["1000"]["live_bytes"]
    late_growth = checkpoints["10000"]["live_bytes"] - checkpoints["5000"]["live_bytes"]
    result = {
        "turns": MEMORY_TURNS,
        "checkpoints": checkpoints,
        "outcomes": dict(outcomes),
        "live_growth_1000_to_10000_bytes": growth,
        "live_growth_5000_to_10000_bytes": late_growth,
        "prospective_growth_limit_bytes": MEMORY_GROWTH_LIMIT_BYTES,
        "measurement": "tracemalloc live Python allocations after gc.collect; no per-turn samples retained",
        "counts_checked_after_every_turn": True,
        "after_close_counts": after_close,
        "after_close_live_bytes": close_live,
        "after_close_peak_bytes": close_peak,
        "live_growth_bounded": growth <= MEMORY_GROWTH_LIMIT_BYTES
        and late_growth <= MEMORY_GROWTH_LIMIT_BYTES,
    }
    require(all(outcomes[k] > 0 for k in ("complete", "cancelled", "failed")), "Stress outcomes missing")
    return result


def repeated_cleanup(model):
    refs = []
    for index in range(CLEANUP_SESSIONS):
        session, participants = create_session(model, "cleanup-" + str(index))
        require(one_turn(session, participants, 1) == "complete", "Cleanup fixture did not finalize")
        session.close()
        bounds(session, closed=True)
        refs.append(weakref.ref(session))
        del session, participants
    gc.collect()
    live = sum(ref() is not None for ref in refs)
    require(live == 0, "Closed session objects remain retained")
    require(model.close_calls == 0, "Session cleanup closed the caller-owned shared predictor")
    require(
        set(vars(model)) == {"calls", "close_calls"}, "Session attached retained state to shared predictor"
    )
    return {
        "sessions": CLEANUP_SESSIONS,
        "live_session_weakrefs": live,
        "shared_predictor_close_calls": model.close_calls,
        "shared_predictor_has_no_retained_inputs": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    # Exclusive creation prevents accidental replacement of earlier measurements.
    with args.output.open("x") as destination:
        report = {
            "status": "INCOMPLETE",
            "source": str(Path(__file__).resolve()),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "synthetic_only": True,
            "learned_model_execution": False,
            "fixture_predictor_executed": True,
            "limits": asdict(Limits()),
        }
        started = time.perf_counter()
        model = FixedPredictor()
        try:
            report["performance"] = performance(model)
            report["memory"] = memory_stress(model)
            report["repeated_cleanup"] = repeated_cleanup(model)
            report["fixture_predictor_calls"] = model.calls
            require(model.calls == 10000, "Unexpected predictor executions in fixed workload")
            for gate in ("ordinary_event_p95_5ms", "finalization_p95_5ms", "overhead_p95_5ms"):
                require(report["performance"][gate], "Prospective performance gate failed: " + gate)
            require(
                report["memory"]["live_growth_bounded"],
                "Live retained memory exceeds prospective 1 MiB noise bound",
            )
            report["status"] = "PASS"
        except Exception as error:
            report["status"] = "FAIL"
            report["failure"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            report["elapsed_s"] = time.perf_counter() - started
            json.dump(report, destination, indent=2, allow_nan=False)
            destination.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "ordinary_event_p95_ms": report["performance"]["ordinary_events"]["p95_ms"],
                "finalization_p95_ms": report["performance"]["finalization_wall"]["p95_ms"],
                "memory_growth_bytes": report["memory"]["live_growth_1000_to_10000_bytes"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

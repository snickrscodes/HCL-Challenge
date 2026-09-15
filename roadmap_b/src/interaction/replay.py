"""Explicit bounded replay with separate observation and execution namespaces."""

from dataclasses import asdict
import json
from pathlib import Path
import time

from .events import ObservationEvent, canonical
from .session import StatefulSession
from .state import Limits


class ReplayRecorder:
    """Caller-owned debug artifact. Overflow disables export, never truncates silently."""

    def __init__(self, *, max_bytes=Limits().max_replay_bytes):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("Positive replay byte budget required")
        self.max_bytes, self.bytes_recorded, self.overflowed = max_bytes, 0, False
        self._records = []

    def _append(self, value):
        if self.overflowed:
            return
        encoded = canonical(value)
        size = len(encoded.encode("utf-8")) + 1
        if self.bytes_recorded + size > self.max_bytes:
            self.overflowed = True
            return
        self._records.append(encoded)
        self.bytes_recorded += size

    def bind(self, session_id, limits, manifest_sha256):
        if self._records or self.overflowed:
            raise ValueError("Recorder belongs to exactly one session")
        self._append(
            {
                "schema": "stateful-replay-v1",
                "namespace": "configuration",
                "session_id": session_id,
                "limits": asdict(limits),
                "manifest_sha256": manifest_sha256,
            }
        )

    def observe(self, event):
        self._append({"namespace": "observations", "event": event.to_dict()})

    def complete(self, turn_id, event_seq, generation, as_of_event_seq, disposition):
        self._append(
            {
                "namespace": "execution",
                "turn_id": turn_id,
                "event_seq": event_seq,
                "generation": generation,
                "as_of_event_seq": as_of_event_seq,
                "disposition": disposition,
            }
        )

    def to_records(self):
        if self.overflowed:
            raise ValueError("Replay recording budget exhausted; complete replay unavailable")
        return [json.loads(s) for s in self._records]

    def write(self, path):
        write_events(path, self.to_records(), max_bytes=self.max_bytes)


def write_events(path, events, *, max_bytes=Limits().max_replay_bytes):
    total = 0
    with Path(path).open("x", encoding="utf-8") as stream:
        for event in events:
            value = event.to_dict() if isinstance(event, ObservationEvent) else event
            line = canonical(value) + "\n"
            total += len(line.encode("utf-8"))
            if total > max_bytes:
                raise ValueError("Replay exceeds explicit byte budget")
            stream.write(line)


def read_events(path, *, max_bytes=Limits().max_replay_bytes):
    total = 0
    with Path(path).open("rb") as stream:
        while line := stream.readline(max_bytes - total + 1):
            total += len(line)
            if total > max_bytes:
                raise ValueError("Replay exceeds explicit byte budget")
            value = json.loads(line)
            yield ObservationEvent.from_dict(value) if value.get("schema") == "stateful-events-v1" else value


def replay(events, model, manifest, *, limits=None):
    session, pending, recorded = None, None, False
    for record in events:
        if session is None and isinstance(record, dict):
            if (
                set(record) != {"schema", "namespace", "session_id", "limits", "manifest_sha256"}
                or record["schema"] != "stateful-replay-v1"
                or record["namespace"] != "configuration"
            ):
                raise ValueError("Replay configuration required")
            session = StatefulSession(
                model, manifest, session_id=record["session_id"], limits=Limits(**record["limits"])
            )
            if session._adapter.manifest_sha256 != record["manifest_sha256"]:
                raise ValueError("Replay release identity differs")
            recorded = True
            continue
        if recorded:
            if not isinstance(record, dict):
                raise ValueError("Recorded replay requires namespaced records")
            if record.get("namespace") == "observations" and set(record) == {"namespace", "event"}:
                event = ObservationEvent.from_dict(record["event"])
                value = session.ingest(event, _defer_final=True)
                if isinstance(value, tuple):
                    if pending is not None:
                        raise ValueError("Overlapping recorded inference")
                    pending = value
            elif record.get("namespace") == "execution":
                if (
                    set(record)
                    != {"namespace", "turn_id", "event_seq", "generation", "as_of_event_seq", "disposition"}
                    or pending is None
                ):
                    raise ValueError("Unexpected execution boundary")
                turn, event, generation, *_ = pending
                if (
                    record["turn_id"],
                    record["event_seq"],
                    record["generation"],
                    record["as_of_event_seq"],
                ) != (turn.turn_id, event.event_seq, generation, session._seq):
                    raise ValueError("Execution boundary does not match request")
                if record["disposition"] == "complete":
                    if turn.inference_status != "pending" or turn.generation != generation or session._closed:
                        raise ValueError("Completed execution was cancelled")
                    if session._run_final(pending, time.perf_counter()) is None:
                        raise ValueError("Recorded successful result became stale")
                else:
                    session._replay_disposition(pending, record["disposition"])
                pending = None
            else:
                raise ValueError("Unknown replay namespace")
        else:
            if not isinstance(record, ObservationEvent):
                raise ValueError("ObservationEvent required for sequential replay")
            if session is None:
                session = StatefulSession(model, manifest, session_id=record.session_id, limits=limits)
            session.ingest(record)
    if session is None or not session._started or pending is not None:
        raise ValueError("Empty or incomplete replay execution")
    return session

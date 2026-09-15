"""Immutable receipt events. Model estimates are never observation events."""

from dataclasses import dataclass
import base64
import hashlib
import json
import math

import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ObservationEvent:
    session_id: str
    event_seq: int
    event_id: str
    monotonic_timestamp: float
    event_type: str
    participant_id: str | None = None
    turn_id: str | None = None
    payload_json: str = "{}"
    pcm_bytes: bytes = b""
    source: str = "application"
    availability: str = "observed"
    provenance_json: str = "{}"

    @classmethod
    def create(
        cls,
        session_id,
        event_seq,
        event_type,
        *,
        timestamp=0.0,
        participant_id=None,
        turn_id=None,
        payload=None,
        pcm=None,
        source="application",
        availability="observed",
        provenance=None,
    ):
        raw = b""
        if pcm is not None:
            wave = np.asarray(pcm)
            if wave.ndim != 1 or not np.issubdtype(wave.dtype, np.floating):
                raise ValueError("PCM must be mono floating point")
            if not np.isfinite(wave).all():
                raise ValueError("PCM must be finite")
            wave = wave.astype("<f4")
            if not np.isfinite(wave).all():
                raise ValueError("PCM cannot overflow float32")
            raw = wave.tobytes()
        return cls(
            session_id,
            event_seq,
            f"{session_id}:{event_seq}",
            timestamp,
            event_type,
            participant_id,
            turn_id,
            canonical({} if payload is None else payload),
            raw,
            source,
            availability,
            canonical({} if provenance is None else provenance),
        )

    def metadata(self):
        return {
            "schema": "stateful-events-v1",
            "event_id": self.event_id,
            "session_id": self.session_id,
            "event_seq": self.event_seq,
            "monotonic_timestamp": self.monotonic_timestamp,
            "event_type": self.event_type,
            "participant_id": self.participant_id,
            "turn_id": self.turn_id,
            "payload": json.loads(self.payload_json),
            "source": self.source,
            "availability": self.availability,
            "provenance": json.loads(self.provenance_json),
            "pcm_samples": len(self.pcm_bytes) // 4,
            "pcm_sha256": hashlib.sha256(self.pcm_bytes).hexdigest() if self.pcm_bytes else None,
        }

    def fingerprint(self):
        return hashlib.sha256(canonical(self.metadata()).encode()).hexdigest()

    def to_dict(self):
        return {**self.metadata(), "pcm_base64": base64.b64encode(self.pcm_bytes).decode("ascii")}

    @classmethod
    def from_dict(cls, value):
        allowed = {
            "schema",
            "event_id",
            "session_id",
            "event_seq",
            "monotonic_timestamp",
            "event_type",
            "participant_id",
            "turn_id",
            "payload",
            "source",
            "availability",
            "provenance",
            "pcm_samples",
            "pcm_sha256",
            "pcm_base64",
        }
        if not isinstance(value, dict) or set(value) != allowed or value["schema"] != "stateful-events-v1":
            raise ValueError("Unsupported event schema")
        raw = base64.b64decode(value["pcm_base64"], validate=True)
        event = cls(
            value["session_id"],
            value["event_seq"],
            value["event_id"],
            value["monotonic_timestamp"],
            value["event_type"],
            value["participant_id"],
            value["turn_id"],
            canonical(value["payload"]),
            raw,
            value["source"],
            value["availability"],
            canonical(value["provenance"]),
        )
        if event.metadata() != {k: v for k, v in value.items() if k != "pcm_base64"}:
            raise ValueError("PCM identity mismatch")
        return event

    def validate_envelope(self, max_id_chars, max_text_chars, max_chunk_samples):
        for value in (self.session_id, self.event_id, self.event_type, self.source, self.availability):
            if not isinstance(value, str) or not value or len(value) > max_id_chars + 32:
                raise ValueError("Invalid event identity or source")
        for value in (self.participant_id, self.turn_id):
            if value is not None and (not isinstance(value, str) or not value or len(value) > max_id_chars):
                raise ValueError("Invalid participant/turn identity")
        if type(self.event_seq) is not int or self.event_seq < 1:
            raise ValueError("Positive integer event sequence required")
        if self.event_id != f"{self.session_id}:{self.event_seq}":
            raise ValueError("Event ID must be sequence-bound")
        if type(self.monotonic_timestamp) not in (int, float) or not math.isfinite(self.monotonic_timestamp):
            raise ValueError("Finite monotonic timestamp required")
        if not isinstance(self.pcm_bytes, bytes) or len(self.pcm_bytes) % 4:
            raise ValueError("Immutable float32 PCM bytes required")
        if len(self.pcm_bytes) > max_chunk_samples * 4:
            raise ValueError("Audio chunk exceeds bounded event size")
        if not isinstance(self.payload_json, str) or len(self.payload_json) > 8 * max_text_chars:
            raise ValueError("Event payload too large")
        if not isinstance(self.provenance_json, str) or len(self.provenance_json) > 4096:
            raise ValueError("Event provenance too large")
        payload, provenance = json.loads(self.payload_json), json.loads(self.provenance_json)
        if not isinstance(payload, dict) or not isinstance(provenance, dict):
            raise ValueError("Event payload/provenance must be objects")
        canonical(payload)
        canonical(provenance)
        return payload

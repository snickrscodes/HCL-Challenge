"""Role-specific immutable actions around the existing utterance-final session."""

import hashlib
import time

import numpy as np

from .context import Turn
from .session import Session
from .ta_policy import select_response
from .ta_release import SnapshotPredictor, TurnSnapshot, canonical


class SystemPredictor(SnapshotPredictor):
    def __init__(self, model, manifest, *, policy="evidence_only"):
        super().__init__(model, manifest)
        self.policy = policy

    def predict_turn(self, text, audio, sample_rate, history, *, speaker="user"):
        started = time.perf_counter()
        result = self.model.predict_turn(text, audio, sample_rate, history, speaker=speaker)
        action = select_response(text, result["meld_state"], result["delivery_evidence"], policy=self.policy)
        observed = [{"text": turn.text, "speaker": turn.speaker} for turn in history[-3:]]
        timing = dict(result["latency_ms"])
        timing["system_total"] = (time.perf_counter() - started) * 1000
        payload = {
            "turn_id": f"turn-{len(self.snapshots) + 1:06d}",
            "transcript": text,
            "speaker_id": speaker,
            "observed_context": observed,
            "observed_context_sha256": hashlib.sha256(canonical(observed).encode()).hexdigest(),
            "context_policy": "last_three_observed_human_turns; no inferred-state/action feedback",
            "mode": "ROLE_SPECIFIC" if self.policy == "role_specific_h1" else "EVIDENCE_ONLY",
            "meld_state": result["meld_state"],
            "acoustic_evidence": result["delivery_evidence"],
            "interaction": action,
            "response": action["response"],
            "response_status": "proposed; emitted only when caller acknowledges",
            "timing": timing,
            "provenance": {
                "release_id": self.manifest["release_id"],
                "condition": "D2",
                "manifest_sha256": hashlib.sha256(canonical(self.manifest).encode()).hexdigest(),
            },
        }
        snapshot = TurnSnapshot(canonical(payload))
        self.snapshots.append(snapshot)
        return snapshot


class SystemSession(Session):
    """Existing PCM/history contract with explicit decode failure and aborted-turn cleanup."""

    def begin_turn(self, speaker=None):
        super().begin_turn(speaker)
        self.decode_failed = False

    def mark_decode_failed(self):
        if not self.active or self.chunks:
            raise RuntimeError("Decode failure must be marked before adding PCM")
        self.decode_failed = True

    def push_audio(self, chunk):
        if getattr(self, "decode_failed", False):
            raise RuntimeError("Cannot append PCM after a decode failure")
        return super().push_audio(chunk)

    def end_turn(self):
        if not self.active or self.transcript is None:
            raise RuntimeError("An active turn and transcript are required")
        audio = np.concatenate(self.chunks) if self.chunks else None
        if self.decode_failed:
            # Existing prepare_waveform rejects it at the shared fallback boundary.
            audio = np.full(400, np.nan, dtype=np.float32)
        try:
            prediction = self.predictor.predict_turn(
                self.transcript,
                audio,
                self.sample_rate if audio is not None else None,
                self.history,
                speaker=self.speaker,
            )
            self.history.append(Turn(self.transcript, self.speaker))
            self.history = self.history[-3:]
            return prediction
        finally:
            self.active, self.chunks, self.transcript, self.decode_failed = False, [], None, False

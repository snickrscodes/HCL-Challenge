"""Stateless adapter around the exact frozen final T+A computation."""

import hashlib
import json
import time

from ..ta_policy import select_response
from .events import canonical


class FinalPerceptionAdapter:
    def __init__(self, model, manifest):
        if (
            not isinstance(manifest, dict)
            or not isinstance(manifest.get("release_id"), str)
            or not manifest["release_id"]
        ):
            raise ValueError("Explicit release identity required")
        numerical = getattr(model, "numerical_scope", "test_or_external_predictor")
        if not isinstance(numerical, str) or len(numerical) > 4096:
            raise ValueError("Invalid numerical provenance")
        self.numerical_scope = numerical
        self.model = model
        self.manifest_json = canonical(manifest)
        self.manifest_sha256 = hashlib.sha256(self.manifest_json.encode()).hexdigest()

    def predict_turn(self, text, audio, sample_rate, history, *, speaker):
        started = time.perf_counter()
        result = self.model.predict_turn(text, audio, sample_rate, history, speaker=speaker)
        predictor_ms = (time.perf_counter() - started) * 1000
        action = select_response(
            text, result["meld_state"], result["delivery_evidence"], policy="evidence_only"
        )
        state = {k: v for k, v in result["meld_state"].items() if k != "latency_ms"}
        payload = json.loads(
            canonical(
                {"B_output": state, "D2_output": result["delivery_evidence"], "proposed_action": action}
            )
        )
        return payload, {"predictor_ms": predictor_ms, "model": result.get("latency_ms", {})}

    def provenance(self):
        return {
            "release_id": json.loads(self.manifest_json)["release_id"],
            "manifest_sha256": self.manifest_sha256,
            "condition": "D2",
            "numerical_scope": self.numerical_scope,
            "speaker_association": "EXTERNALLY_SUPPLIED",
        }

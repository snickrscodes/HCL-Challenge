"""Minimal immutable evidence-only turn snapshot around the unchanged legacy Session."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .session import Session


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class TurnSnapshot:
    payload_json: str

    def to_dict(self):
        return json.loads(self.payload_json)


class SnapshotPredictor:
    def __init__(self, model, manifest):
        self.model, self.manifest = model, dict(manifest)
        self.snapshots = []
        self.emitted_actions = []

    def predict_turn(self, text, audio, sample_rate, history, *, speaker="user"):
        observed = [{"text": turn.text, "speaker": turn.speaker} for turn in history]
        result = self.model.predict_turn(text, audio, sample_rate, history, speaker=speaker)
        payload = {
            "turn_id": f"turn-{len(self.snapshots) + 1:06d}",
            "transcript": text,
            "speaker_id": speaker,
            "observed_context": observed,
            "observed_context_sha256": hashlib.sha256(canonical(observed).encode()).hexdigest(),
            "context_policy": "legacy_last_three_observed_human_turns; no inferred-state/action feedback",
            "mode": "EVIDENCE_ONLY",
            "meld_state": result["meld_state"],
            "acoustic_evidence": result["delivery_evidence"],
            "response": result["meld_state"]["response"],
            "response_status": "proposed; emitted only when caller acknowledges",
            "timing": result["latency_ms"],
            "provenance": {
                "release_id": self.manifest["release_id"],
                "condition": self.manifest["condition"],
                "manifest_sha256": hashlib.sha256(canonical(self.manifest).encode()).hexdigest(),
            },
        }
        snapshot = TurnSnapshot(canonical(payload))
        self.snapshots.append(snapshot)
        return snapshot

    def acknowledge_emitted(self, turn_id):
        matches = [s.to_dict() for s in self.snapshots if s.to_dict()["turn_id"] == turn_id]
        if len(matches) != 1:
            raise ValueError("Unknown completed turn")
        if any(json.loads(x)["turn_id"] == turn_id for x in self.emitted_actions):
            raise ValueError("Response already acknowledged")
        self.emitted_actions.append(canonical({"turn_id": turn_id, "response": matches[0]["response"]}))


def verify_release(manifest, c1_root, baseline_root):
    from .c1_data import sha
    from .c1_transfer_runtime import __file__ as runtime_source

    if manifest["mode"] != "EVIDENCE_ONLY" or manifest["condition"] != "D2" or manifest["seed"] != 1337:
        raise ValueError("Only explicitly frozen D2 evidence-only release supported")
    if (
        sha(runtime_source) != manifest["refined_runtime_sha256"]
        or sha(__file__) != manifest["release_runtime_sha256"]
    ):
        raise ValueError("Runtime source identity mismatch")
    root = Path(c1_root)
    for name, expected in manifest["c1_required_files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError(f"Missing or changed frozen component: {name}")
    from .c1_workflow_v2 import bconfig
    from .b_data import verify_baseline, output
    from .b_evaluate import verify_freeze

    cfg = bconfig(baseline_root)
    verify_baseline(cfg)
    verify_freeze(cfg)
    for name, expected in manifest["B_required_files"].items():
        if sha(output(cfg) / name) != expected:
            raise ValueError(f"B release identity mismatch: {name}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--c1-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--audio")
    parser.add_argument("--chunk-samples", type=int, default=1600)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.chunk_samples < 1:
        raise ValueError("Positive chunk size required")
    manifest = json.loads(Path(args.manifest).read_text())
    verify_release(manifest, args.c1_root, args.baseline_root)
    from .c1_transfer_runtime import RefinedC1Dual
    from .audio import read_waveform

    model = RefinedC1Dual(args.c1_root, args.baseline_root, condition="D2", clock=True)
    adapter = SnapshotPredictor(model, manifest)
    session = Session(adapter)
    session.start()
    session.begin_turn()
    decode_failed = False
    if args.audio:
        try:
            wave = read_waveform(args.audio)
        except (ValueError, RuntimeError, OSError):
            decode_failed = True
        else:
            for start in range(0, len(wave), args.chunk_samples):
                session.push_audio(wave[start : start + args.chunk_samples])
    if decode_failed:
        # Signal invalid input to the existing C1 preprocessing/fallback boundary.
        # No substitute audio is encoded; prepare_waveform rejects this sentinel.
        import numpy as np

        snapshot = adapter.predict_turn(
            args.text, np.full(400, np.nan, dtype=np.float32), 16000, [], speaker="user"
        )
    else:
        session.set_transcript(args.text)
        snapshot = session.end_turn()
    result = snapshot.to_dict()
    if args.output:
        with Path(args.output).open("x") as f:
            f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    # Console has actually emitted the authored response as part of this result.
    adapter.acknowledge_emitted(result["turn_id"])


if __name__ == "__main__":
    main()

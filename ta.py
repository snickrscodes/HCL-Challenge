#!/usr/bin/env python3
"""Local T+A inference from a supplied transcript and optional waveform."""

import argparse
import json
import os
from pathlib import Path
import sys

SOURCE_ROOT = Path(__file__).resolve().parent / "roadmap_b"
sys.path.insert(0, str(SOURCE_ROOT))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset-root", required=True, help="Prepared, hash-verified external T+A assets")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Observed human transcript (no ASR)")
    group.add_argument(
        "--replay", type=Path, help="JSON object with a turns array; audio paths relative to this file"
    )
    p.add_argument("--audio", type=Path, help="Waveform for the single --text turn")
    p.add_argument("--chunk-samples", type=int, default=1600)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument(
        "--policy",
        choices=("evidence_only", "role_specific_h1"),
        default=None,
        help="Override documented default; role_specific_h1 may be research-only",
    )
    p.add_argument("--output", type=Path, help="Save structured output to a new file; never overwrite")
    return p


def read_turns(args):
    if args.chunk_samples < 1:
        raise ValueError("--chunk-samples must be positive")
    if args.output and args.output.exists():
        raise ValueError("--output already exists; choose a new path")
    if args.replay:
        if args.audio:
            raise ValueError("--audio belongs to --text; put each audio path in the replay file")
        value = json.loads(args.replay.read_text())
        if not isinstance(value, dict) or not isinstance(value.get("turns"), list) or not value["turns"]:
            raise ValueError("Replay must contain a nonempty turns array")
        turns = []
        for item in value["turns"]:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise ValueError("Each replay turn needs a string text field")
            if item.get("role", "human") != "human":
                raise ValueError("Replay history may contain only observed human turns")
            speaker = item.get("speaker", "user")
            if not isinstance(speaker, str) or not speaker:
                raise ValueError("Each speaker must be a nonempty string")
            audio = item.get("audio")
            if audio is not None and not isinstance(audio, str):
                raise ValueError("Audio must be a file path or null")
            turns.append(
                {
                    "text": item["text"],
                    "speaker": speaker,
                    "audio": (args.replay.resolve().parent / audio).resolve() if audio else None,
                }
            )
        return turns
    return [{"text": args.text, "speaker": "user", "audio": args.audio}]


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    model = None
    try:
        turns = read_turns(args)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from src.audio import read_waveform
        from src.ta_runtime import PortableTADual
        from src.ta_interface import SystemPredictor, SystemSession
        import src.ta_runtime as runtime_source

        if Path(runtime_source.__file__).resolve().parent != SOURCE_ROOT / "src":
            raise ValueError("Runtime imported from outside this checkout")
        release = json.loads((SOURCE_ROOT / "evidence/ta_system_closure/RELEASE.json").read_text())
        policy = args.policy or release["default_policy"]
        model = PortableTADual(args.asset_root, device=args.device, clock=True)
        adapter = SystemPredictor(model, release, policy=policy)
        session = SystemSession(adapter)
        session.start()
        snapshots = []
        for turn in turns:
            session.begin_turn(speaker=turn["speaker"])
            if turn["audio"]:
                try:
                    wave = read_waveform(turn["audio"])
                except (ValueError, RuntimeError, OSError):
                    session.mark_decode_failed()
                else:
                    for start in range(0, len(wave), args.chunk_samples):
                        session.push_audio(wave[start : start + args.chunk_samples])
            session.set_transcript(turn["text"])
            snapshots.append(session.end_turn().to_dict())
        result = {
            "release_scope": release["scope"],
            "policy": policy,
            "policy_status": release["policy_status"][policy],
            "numerical_scope": model.numerical_scope,
            "parameters": model.parameter_ledger(),
            "turns": snapshots,
        }
        rendered = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            with args.output.open("x") as stream:
                stream.write(rendered + "\n")
        print(rendered)
        for snapshot in snapshots:
            adapter.acknowledge_emitted(snapshot["turn_id"])
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        p.error(str(exc))
    finally:
        if model is not None:
            model.close()


if __name__ == "__main__":
    main()

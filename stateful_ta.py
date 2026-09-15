#!/usr/bin/env python3
"""Offline causal T+A sessions from supplied transcripts, PCM, or explicit event logs."""

import argparse
import json
import os
from pathlib import Path
import sys

SOURCE_ROOT = Path(__file__).resolve().parent / "roadmap_b"
sys.path.insert(0, str(SOURCE_ROOT))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset-root", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--replay", type=Path, help="Existing human-turn replay JSON")
    group.add_argument("--events", type=Path, help="Explicit stateful-events-v1 JSONL debug artifact")
    p.add_argument("--audio", type=Path)
    p.add_argument("--chunk-samples", type=int, default=1600)
    p.add_argument("--session-id", default="cli-session")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--output", type=Path, help="Write a new result file; never overwrite")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    model = session = None
    try:
        if not 1 <= args.chunk_samples <= 65536:
            raise ValueError("--chunk-samples must be between 1 and 65536")
        if args.output and args.output.exists():
            raise ValueError("Output already exists")
        if args.events and args.audio:
            raise ValueError("Audio belongs to text or turn replay, not event replay")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from src.audio import read_waveform
        from src.interaction import StatefulSession
        from src.interaction.replay import read_events, replay
        from src.ta_runtime import PortableTADual
        from ta import read_turns

        release = json.loads((SOURCE_ROOT / "evidence/ta_system_closure/RELEASE.json").read_text())
        turns = None if args.events else read_turns(args)
        if turns is not None and len(turns) > 64:
            raise ValueError("CLI replay supports at most 64 turns; use the bounded session API for live use")
        model = PortableTADual(args.asset_root, device=args.device)
        if args.events:
            session = replay(read_events(args.events), model, release)
            result = {"mode": "explicit_event_replay", "state": session.to_dict()}
        else:
            session = StatefulSession(model, release, session_id=args.session_id)
            session.start()
            snapshots = []
            turn_ids = []
            for item in turns:
                participant = session.register_participant(item["speaker"])
                turn = session.start_turn(participant)
                if item["audio"]:
                    try:
                        wave = read_waveform(item["audio"])
                    except (ValueError, RuntimeError, OSError):
                        session.mark_audio_unavailable(turn)
                    else:
                        for offset in range(0, len(wave), args.chunk_samples):
                            session.observe_audio_chunk(turn, wave[offset : offset + args.chunk_samples])
                session.finalize_text(turn, item["text"])
                snapshots.append(session.end_turn(turn).to_dict())
                turn_ids.append(turn)
            result = {
                "mode": "stateful_utterance_final",
                "policy": "evidence_only",
                "speaker_association": "EXTERNALLY_SUPPLIED",
                "turns": snapshots,
            }
        result["parameters"] = model.parameter_ledger()
        result["numerical_scope"] = model.numerical_scope
        rendered = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered + "\n")
        print(rendered)
        if not args.events:
            # Console output has actually emitted these final authored responses.
            for turn in turn_ids:
                session.record_robot_emitted(turn)
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        p.error(str(exc))
    finally:
        if session is not None and not session.to_dict()["closed"]:
            session.close()
        if model is not None:
            model.close()


if __name__ == "__main__":
    main()

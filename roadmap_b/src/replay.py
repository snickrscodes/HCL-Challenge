"""Replay a waveform in chunks and emit one final state plus authored response."""

import json

import numpy as np

from .audio import read_waveform
from .constants import SAMPLE_RATE
from .context import Turn
from .inference import RoadmapA
from .session import Session
from .utils import common_parser, load_config, read_json, setup


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--audio", help="Omit for the calibrated text-only fallback")
    parser.add_argument("--text", required=True)
    parser.add_argument("--history", help="JSON list of objects with text and speaker")
    parser.add_argument("--speaker", default="user")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup(cfg["seed"])
    history = [Turn(**turn) for turn in read_json(args.history)] if args.history else []
    session = Session(RoadmapA(cfg), SAMPLE_RATE, args.speaker, history)
    session.start()
    session.begin_turn()
    if args.audio:
        audio = read_waveform(args.audio)
        for chunk in np.array_split(audio, max(1, int(np.ceil(len(audio) / 3200)))):
            session.push_audio(chunk)
    session.set_transcript(args.text)
    print(json.dumps(session.end_turn().to_dict(), allow_nan=False))


if __name__ == "__main__":
    main()

"""Fresh-process raw demonstration on one predetermined ordinary development probe."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio import read_waveform
from src.b_data import output, parser, rows_for, settings, waveform_path
from src.b_inference import RoadmapB
from src.context import histories
from src.session import Session
from src.utils import digest, key, write_json

args = parser(__doc__).parse_args()
cfg = settings(args)
# Fixed duration-quantile probe chosen for input-flow demonstration, not based on outcomes.
rows = rows_for(cfg, "dev_model")
row = next(r for r in rows if key(r) == "dev/54/8")
history = histories(rows, 3)[key(row)]
wave_path = waveform_path(cfg, row)
wave = read_waveform(wave_path)
root = output(cfg) / "demo"
root.mkdir(exist_ok=False)
write_json(
    root / "input.json",
    {
        "key": key(row),
        "text": row["text"],
        "speaker": row["speaker"],
        "history": [{"text": t.text, "speaker": t.speaker} for t in history],
        "waveform_sha256": digest(wave_path),
        "num_samples": len(wave),
        "sample_rate": 16000,
        "audio_source": str(wave_path),
        "scope": "Predetermined dev probe, fresh-process raw inference; no feature cache",
    },
)
model = RoadmapB(cfg)
session = Session(model, history=history)
session.start()
session.begin_turn(speaker=row["speaker"])
for start in range(0, len(wave), 4000):
    session.push_audio(wave[start : start + 4000])
session.set_transcript(row["text"])
result = session.end_turn()
write_json(root / "output.json", result)
assert result["response"] and len(result["distribution"]) == 7
print(json.dumps(result, indent=2))

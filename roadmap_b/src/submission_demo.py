"""Predetermined development showcases using the unchanged raw Session path."""

import json
import re
import time
from pathlib import Path

import numpy as np

from .audio import read_waveform
from .b_data import parser, rows_for, settings, verify_baseline, waveform_path
from .b_inference import RoadmapB
from .context import histories
from .session import Session
from .utils import key, read_json, read_rows, write_json

RULES = {
    "version": 1,
    "scope": "Retrospective development showcases, not confirmatory subsets or tuning data",
    "order": "First numeric dialogue/utterance key satisfying each rule",
    "media": "Valid 0.5–8-second clips; transcript <=160 characters; exclude explicit self-harm/profanity from public presentation only",
    "audio_helps": "B correct while canonical text and fixed A are wrong",
    "audio_changes_little": "Text correct, raw confidence >=.8, B agrees, actual B influence <.02",
    "failure": "B wrong while fixed A is correct",
    "missing_audio": "Use the audio-helps transcript/history with audio omitted",
    "causal_history": "First current-T0-wrong/context-T1-correct candidate where unchanged raw T1 differs with history removed",
    "same_words": "S01/p01, all three intended deliveries, regardless of model behavior; no selection for a flattering result",
    "playback": "Actual buffered chunk calls paced at 250 ms; UI replays measured events and slows the final stages for legibility",
}


def build(cfg, dest, delivery_results=None):
    verify_baseline(cfg)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    write_json(dest / "selection_rule.json", RULES)
    b = Path(cfg["output_root"])
    base = Path(cfg["baseline_root"])
    rows = rows_for(cfg, "dev_model")
    hist = histories(rows, 3)
    bykey = {key(r): r for r in rows}
    bp = {r["key"]: r for r in read_rows(b / "evaluation/adaptive/1337/predictions.jsonl")}
    ap = {r["key"]: r for r in read_rows(b / "evaluation/A_replay_canonical_predictions.jsonl")}
    t0 = {
        r["key"]: r for r in read_rows(base / "artifacts/evaluation/dev_model/text_current/predictions.jsonl")
    }
    t1 = {
        r["key"]: r for r in read_rows(base / "artifacts/evaluation/dev_model/text_context/predictions.jsonl")
    }
    ordered = sorted(
        (
            r
            for r in rows
            if r["audio_valid"]
            and 0.5 <= r["duration_s"] <= 8
            and len(r["text"]) <= 160
            and not re.search(r"kill myself|kill yourself|suicide|\bfuck\b", r["text"], re.I)
        ),
        key=lambda r: (int(r["dialogue_id"]), int(r["utterance_id"])),
    )
    selected = {}
    for r in ordered:
        k = key(r)
        v = bp[k]
        y = r["label"]
        text = int(np.argmax(v["text_logits"]))
        a = int(np.argmax(ap[k]["probabilities"]))
        if "audio_helps" not in selected and v["prediction"] == y and text != y and a != y:
            selected["audio_helps"] = k
        if (
            "audio_changes_little" not in selected
            and text == y
            and v["prediction"] == text
            and v["text_max_probability"] >= 0.8
            and v["audio_influence"] < 0.02
        ):
            selected["audio_changes_little"] = k
        if "failure" not in selected and v["prediction"] != y and a == y:
            selected["failure"] = k
    if len(selected) != 3:
        raise ValueError(
            "A predeclared demonstration category has no eligible case; report this instead of inventing one."
        )
    model = RoadmapB(cfg)
    first = bykey[selected["audio_helps"]]
    model.predict_turn(
        first["text"],
        read_waveform(waveform_path(cfg, first)),
        16000,
        hist[key(first)],
        speaker=first["speaker"],
    )
    context_check = None
    for row in ordered:
        k = key(row)
        if (
            np.argmax(t0[k]["probabilities"]) == row["label"]
            or np.argmax(t1[k]["probabilities"]) != row["label"]
        ):
            continue
        wave = read_waveform(waveform_path(cfg, row))
        full = model.predict_turn(row["text"], wave, 16000, hist[k], speaker=row["speaker"])
        none = model.predict_turn(row["text"], wave, 16000, [], speaker=row["speaker"])
        if full["text_prediction"] != none["text_prediction"]:
            selected["causal_history"] = k
            context_check = {"with_history": full, "without_history": none}
            break
    if context_check is None:
        raise ValueError("No context-sensitive case found under the declared rule.")
    selected["missing_audio"] = selected["audio_helps"]
    cases = {}
    for name, k in selected.items():
        row = bykey[k]
        wave = None if name == "missing_audio" else read_waveform(waveform_path(cfg, row))
        session = Session(model, history=hist[k])
        session.start()
        session.begin_turn(speaker=row["speaker"])
        events = []
        began = time.perf_counter()
        if wave is not None:
            for index, start in enumerate(range(0, len(wave), 4000)):
                chunk = wave[start : start + 4000]
                time.sleep(len(chunk) / 16000)
                session.push_audio(chunk)
                events.append(
                    {
                        "event": "audio_chunk",
                        "index": index,
                        "samples": len(chunk),
                        "t_ms": 1000 * (time.perf_counter() - began),
                    }
                )
        session.set_transcript(row["text"])
        end_turn = 1000 * (time.perf_counter() - began)
        value = session.end_turn()
        events.append({"event": "end_turn", "t_ms": end_turn})
        elapsed = end_turn
        for stage in ("text", "audio", "fusion_response"):
            elapsed += value["latency_ms"][stage]
            events.append({"event": stage, "t_ms": elapsed})
        cases[name] = {
            "key": k,
            "transcript": row["text"],
            "speaker": row["speaker"],
            "history": [{"speaker": t.speaker, "text": t.text} for t in hist[k]],
            "label": row["emotion"],
            "events": events,
            "state": value,
            "A_distribution": ap[k]["probabilities"],
            "context_comparison": context_check if name == "causal_history" else None,
        }
    if delivery_results and (Path(delivery_results) / "outputs.json").exists():
        outputs = read_json(Path(delivery_results) / "outputs.json")
        chosen = {k: v for k, v in outputs.items() if k.startswith("S01_p01_")}
        if len(chosen) != 3:
            raise ValueError("Expected all three predeclared S01/p01 deliveries.")
        cases["same_words"] = {
            "diagnostic": chosen,
            "scope": "Predeclared S01/p01, no selection on model results",
        }
    payload = {
        "rules": RULES,
        "cases": cases,
        "same_words_status": "complete"
        if "same_words" in cases
        else "pending real recordings and listener pass",
    }
    write_json(dest / "trace.json", payload)
    write_json(
        dest / "manifest.json",
        {
            "cases": list(cases),
            "keys": selected,
            "same_words_status": payload["same_words_status"],
            "no_test_examples": True,
        },
    )
    template = (Path(__file__).resolve().parents[1] / "recording_kit/demo.html").read_text()
    (dest / "index.html").write_text(
        template.replace("__TRACE_JSON__", json.dumps(payload).replace("<", "\\u003c"))
    )
    verify_baseline(cfg)


def main():
    p = parser(__doc__)
    p.add_argument("--demo-output", default="artifacts/submission/demo")
    p.add_argument("--delivery-results", default="artifacts/submission/delivery_results")
    args = p.parse_args()
    build(settings(args), args.demo_output, args.delivery_results)


if __name__ == "__main__":
    main()

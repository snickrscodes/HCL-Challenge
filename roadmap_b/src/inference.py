"""Production raw-waveform inference; never imports experimental feature caches."""

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .audio import FrozenAudio, prepare_waveform
from .constants import LABELS
from .context import collate_text, encode_turn
from .evaluate import probabilities
from .models import TextClassifier, load_head
from .responses import respond
from .utils import autocast, counts, device_for, read_json, sync


@dataclass
class Prediction:
    emotion: str
    confidence: float
    distribution: dict
    audio_available: bool
    text_prediction: str
    audio_prediction: str | None
    fusion_alpha: float
    response: str
    latency_ms: dict
    current_truncated: bool

    def to_dict(self):
        return asdict(self)


class RoadmapA:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = device_for(cfg)
        self.selection = read_json(Path(cfg["output_root"]) / "selection.json")
        self.calibration = read_json(Path(cfg["output_root"]) / "calibration.json")
        self.text, self.tokenizer, self.text_metadata = TextClassifier.load(
            Path(cfg["checkpoint_root"]) / self.selection["text"]
        )
        self.text.to(self.device).eval()
        self.audio = FrozenAudio(cfg, local=True)
        self.audio_head = load_head(Path(cfg["checkpoint_root"]) / "audio").to(self.device).eval()

    def parameter_ledger(self):
        components = {
            "selected_text": counts(self.text),
            "frozen_wavlm": counts(self.audio),
            "audio_head": counts(self.audio_head),
        }
        total = sum(item["total"] for item in components.values()) + 4
        return {
            "components": components,
            "learned_scalars": {"alpha": 1, "temperatures": 3},
            "total_required_parameters": total,
            "within_6b": total <= 6_000_000_000,
            "note": "Includes frozen parameters and all diagnostic temperatures; unused alternate text/concat are separate experiments.",
        }

    @torch.inference_mode()
    def predict_turn(self, text, audio, sample_rate, history, *, speaker="user"):
        sync(self.device)
        started = time.perf_counter()
        item = encode_turn(
            self.tokenizer,
            text,
            history,
            speaker,
            self.text_metadata["max_length"],
            self.text_metadata["history_turns"],
        )
        batch = {k: v.to(self.device) for k, v in collate_text([item], self.tokenizer.pad_token_id).items()}
        with autocast(self.cfg, self.device):
            zt, _ = self.text(**batch)
        sync(self.device)
        text_end = time.perf_counter()
        available = audio is not None
        za = None
        if available:
            wave = prepare_waveform(audio, sample_rate)
            h = self.audio([wave]).to(self.device)
            za = self.audio_head(h.float())
            sync(self.device)
        audio_end = time.perf_counter()
        text_logits = zt.float().cpu().numpy()[0]
        if available:
            audio_logits = za.float().cpu().numpy()[0]
            alpha = self.selection["alpha"]
            logits = (1 - alpha) * text_logits + alpha * audio_logits
            prob = probabilities(logits, self.calibration["fusion"])
            audio_prediction = LABELS[int(audio_logits.argmax())]
        else:
            prob = probabilities(text_logits, self.calibration["text"])
            audio_prediction = None
        index = int(np.argmax(prob))
        response = respond(text, LABELS[index], float(prob[index]))
        return Prediction(
            LABELS[index],
            float(prob[index]),
            dict(zip(LABELS, map(float, prob))),
            available,
            LABELS[int(text_logits.argmax())],
            audio_prediction,
            self.selection["alpha"],
            response,
            {
                "text": (text_end - started) * 1000,
                "audio": (audio_end - text_end) * 1000,
                "fusion_response": (time.perf_counter() - audio_end) * 1000,
                "total": (time.perf_counter() - started) * 1000,
            },
            item["current_truncated"],
        )

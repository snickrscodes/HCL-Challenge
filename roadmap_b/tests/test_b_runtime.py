import numpy as np
import torch

from src.b_fusion import ResidualFusion
from src.b_inference import RoadmapB
from src.evaluate import probabilities
from src.session import Session


class Text(torch.nn.Module):
    def forward(self, input_ids, attention_mask, current_mask):
        h = torch.ones(len(input_ids), 768) * (input_ids * current_mask).sum(-1)[:, None] / 1000
        z = torch.arange(7).float()[None].expand(len(input_ids), -1) / 4
        return z, h


class Audio(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, waves):
        self.calls += 1
        return torch.stack([torch.full((768,), float(w.mean())) for w in waves])


def predictor(tokenizer):
    p = RoadmapB.__new__(RoadmapB)
    p.device = torch.device("cpu")
    p.cfg = {"runtime": {"dtype": "float32"}}
    p.policy = {"encoding": "singleton", "max_audio_seconds": 60, "dtype": "float32"}
    p.text, p.audio = Text(), Audio()
    p.tokenizer, p.text_metadata = tokenizer, {"max_length": 128, "history_turns": 3}
    p.audio_head = torch.nn.Linear(768, 7).eval()
    p.residual = ResidualFusion().eval()
    p.temperature, p.calibration = 1.2, {"text": 1.6}
    return p


def test_raw_fallback_duration_and_session(tokenizer):
    p = predictor(tokenizer)
    missing, internal = p.predict_turn("Yeah, sure.", None, None, [], return_internal=True)
    expected = probabilities(internal["zt"][0], p.calibration["text"])
    assert np.array_equal(list(missing["distribution"].values()), expected)
    assert torch.equal(internal["zt"], internal["logits"])
    long = p.predict_turn("Yeah, sure.", np.ones(960001, np.float32), 16000, [])
    assert long["audio_valid"] and not long["audio_available"]
    assert long["audio_unavailable_reason"] == "duration_limit"
    assert p.audio.calls == 0
    session = Session(p)
    session.start()
    session.begin_turn()
    session.push_audio(np.ones(800, np.float32))
    session.set_transcript("Yeah, sure.")
    result = session.end_turn()
    assert result["audio_available"] and p.audio.calls == 1
    assert result["response"] and len(result["distribution"]) == 7
    assert np.isclose(sum(result["distribution"].values()), 1)
    session.reset()
    assert session.history == [] and not session.active


def test_canonical_A_replay_uses_original_alpha_and_text_fallback(tokenizer):
    p = predictor(tokenizer)
    p.residual = None
    p.selection = {"alpha": 0.4}
    p.calibration["fusion"] = 1.08
    p.temperature = 1.08
    value, raw = p.predict_turn("Sure.", np.ones(1000, np.float32), 16000, [], return_internal=True)
    assert torch.equal(raw["logits"], 0.6 * raw["zt"] + 0.4 * raw["za"])
    missing, raw = p.predict_turn("Sure.", None, None, [], return_internal=True)
    assert torch.equal(raw["logits"], raw["zt"])
    assert np.array_equal(list(missing["distribution"].values()), probabilities(raw["zt"][0], 1.6))

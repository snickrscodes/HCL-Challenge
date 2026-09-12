import numpy as np

from src.calibrate import fit_temperature, late_logits
from src.evaluate import metrics, probabilities, recompute, save_evaluation
from src.train_fusion import select_alpha


def test_raw_logit_fusion_and_temperature():
    rng = np.random.default_rng(4)
    text, audio = rng.normal(size=(20, 7)), rng.normal(size=(20, 7))
    y = np.arange(20) % 7
    assert np.array_equal(late_logits(text, audio, 0), text)
    assert np.array_equal(late_logits(text, audio, 1), audio)
    alpha, _ = select_alpha(text, audio, y, np.ones(20, bool), 0.1)
    fused = late_logits(text, audio, alpha)
    assert np.allclose(fused, (1 - alpha) * text + alpha * audio)
    temperature = fit_temperature(fused, y)
    assert temperature > 0
    prob = probabilities(fused, temperature)
    assert np.allclose(prob.sum(1), 1)
    assert np.array_equal(prob.argmax(1), fused.argmax(1))
    assert metrics(y, prob)["nll"] <= metrics(y, probabilities(fused))["nll"] + 1e-8


def test_missing_audio_does_not_affect_alpha():
    text = np.eye(7) * 3
    audio = np.roll(text, 1, 1) * 100
    alpha, _ = select_alpha(text, audio, np.arange(7), np.zeros(7, bool), 0.01)
    assert alpha == 0


def test_metric_roundtrip(tmp_path):
    rows = [
        {"split": "dev", "dialogue_id": 1, "utterance_id": i, "label": i, "audio_valid": True}
        for i in range(7)
    ]
    prob = np.eye(7)[np.zeros(7, dtype=int)]
    result = save_evaluation(tmp_path, rows, prob)
    assert result == recompute(tmp_path / "predictions.jsonl")
    assert result["accuracy"] == 1 / 7
    assert result["macro_f1"] == 0.25 / 7

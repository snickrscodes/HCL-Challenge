import json

import numpy as np
import pytest

from src.delivery import majority, readiness
from src.final_evaluate import claim_test_run, freeze, score_test
from src.recording_kit import WEB


def test_majority_does_not_invent_listener_agreement():
    assert majority(["anger", "joy"]) is None
    assert majority(["anger", "anger", "joy"]) == "anger"
    assert majority([]) is None


def test_missing_recordings_remain_incomplete(tmp_path):
    (tmp_path / "protocol.json").write_bytes((WEB / "phrases.json").read_bytes())
    status = readiness(tmp_path)
    assert not status["complete"] and status["recordings"] == 0
    assert status["expected_recordings"] == 60


def test_official_test_start_marker_is_exclusive(tmp_path):
    folder = claim_test_run(tmp_path, "frozen-hash")
    assert json.loads((folder / "STARTED.json").read_text())["freeze_sha256"] == "frozen-hash"
    with pytest.raises(FileExistsError):
        claim_test_run(tmp_path, "another-hash")


def test_freeze_requires_actual_delivery_evidence(tmp_path, monkeypatch):
    from src import final_evaluate

    monkeypatch.setattr(final_evaluate, "write_policy", lambda *_: {})
    with pytest.raises(FileNotFoundError):
        freeze({}, tmp_path)
    assert not (tmp_path / "FINAL_FREEZE.json").exists()


def test_pure_test_scoring_keeps_unavailable_rows(tmp_path):
    # Synthetic seven-class fixtures exercise scoring only; no MELD files or model inference.
    from src.final_evaluate import FINAL_MODELS

    (tmp_path / "official_test").mkdir()
    rows = [{"dialogue_id": i // 7} for i in range(28)]
    values = []
    for i in range(28):
        p = np.full(7, 0.02)
        p[i % 7] = 0.88
        values.append(
            {
                "key": f"fixture/{i // 7}/{i % 7}",
                "label": i % 7,
                "audio_available": i >= 2,
                "audio_unavailable_reason": "duration_limit" if i < 2 else None,
                "text_logits": np.log(p).tolist(),
                "audio_logits": np.log(p).tolist(),
                "audio_probabilities": p.tolist() if i >= 2 else None,
                "probabilities": {n: p.tolist() for n in FINAL_MODELS},
            }
        )
    policy = {
        "bootstrap": {"replicates": 1000, "seed": 1},
        "production_recommendation_rule": {
            "B_default_requires": {"macro_F1_gain_at_least": 0.005, "weighted_F1_loss_at_most": 0.005}
        },
    }
    score_test({}, tmp_path, rows, values, {"policy": policy})
    result = json.loads((tmp_path / "official_test/summary.json").read_text())
    assert result["overall_support"] == 28 and result["audio_eligible_support"] == 26
    assert result["production_recommendation"].startswith("A canonical")
    assert result["models"]["B_adaptive_1337"]["metrics"]["accuracy"] == 1

"""Small interpretation checks with synthetic probability vectors, never human evidence."""

import json

import numpy as np
import pytest

from scripts.controlled_delivery_smoke import CATEGORY, listener, pair_metrics, provisional_summary
from src.constants import LABELS


def distribution(label, amount=0.7):
    p = np.full(len(LABELS), (1 - amount) / (len(LABELS) - 1))
    p[LABELS.index(label)] = amount
    return p.tolist()


def test_direction_and_tv_are_symmetric_under_reversing_pair():
    a, b = distribution("joy"), distribution("anger")
    x = pair_metrics(a, b, "joy", "anger")
    y = pair_metrics(b, a, "anger", "joy")
    assert x["direction"] == y["direction"] == "aligned"
    assert x["directional_contrast_change"] == pytest.approx(y["directional_contrast_change"])
    assert x["posterior_tv"] == pytest.approx(0.65)
    assert x["both_targets_increase_for_their_own_delivery"]
    assert pair_metrics(b, a, "joy", "anger")["direction"] == "opposite"


def test_unmapped_and_stable_are_not_scored_as_directional_success():
    p = distribution("neutral")
    assert CATEGORY["warm appreciation"] is None
    assert CATEGORY["willing and friendly"] is None
    assert CATEGORY["relaxed acceptance"] is None
    for c in (None, "neutral"):
        x = pair_metrics(p, p, c, "neutral")
        assert x["directional_contrast_change"] is None
        assert x["posterior_tv"] == 0


def test_all_pairs_fixed_text_and_stable_control_retained():
    outputs = {}
    for phrase, cats, stable in [
        ("p01", ("joy", "anger", "sadness"), False),
        ("p09", ("neutral", "neutral", "neutral"), True),
    ]:
        for i, c in enumerate(cats):
            outputs[f"S01_{phrase}_d{i}"] = {
                "metadata": {
                    "speaker": "S01",
                    "phrase_id": phrase,
                    "text": phrase,
                    "history": [],
                    "stable_control": stable,
                },
                "intended_category": c,
                "text": distribution("neutral"),
                "audio": distribution(c),
                "A": distribution(c),
                "B": distribution(c),
            }
    out = provisional_summary(outputs)
    assert len(out["sets"]) == 2 and len(out["pairs"]) == 6
    assert out["counts"]["B"]["aligned"] == 3
    assert out["counts"]["B"]["stable_support"] == 3
    assert out["all_text_sets_invariant"]
    assert out["final_requirement_satisfied"] is False
    assert out["human_validated"] is False


def test_listener_action_refuses_pending_ratings_before_reading_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.controlled_delivery_smoke.readiness", lambda _: {"complete": False})
    with pytest.raises(ValueError, match="incomplete"):
        listener(tmp_path, tmp_path / "no_outputs", tmp_path / "final")
    assert not (tmp_path / "final").exists()


def test_listener_interpretation_does_not_load_an_encoder():
    import inspect

    source = inspect.getsource(listener)
    assert "RoadmapB(" not in source and "predict_turn(" not in source
    assert "summarize(root, clean)" in source
    assert "intended_category" not in source
    json.dumps(pair_metrics(distribution("joy"), distribution("anger"), "joy", "anger"), allow_nan=False)


def test_listener_reuses_saved_predictions_and_strips_oracle_fields(tmp_path, monkeypatch):
    import src.b_inference
    from src.utils import digest, write_json

    root, smoke, dest = tmp_path / "recordings", tmp_path / "smoke", tmp_path / "final"
    root.mkdir()
    smoke.mkdir()
    frozen = tmp_path / "frozen_identity.txt"
    frozen.write_text("unchanged fixture identity; not a model")
    outputs = {
        "fixture": {
            "metadata": {"text": "Okay."},
            "text": distribution("neutral"),
            "audio": distribution("anger"),
            "A": distribution("neutral"),
            "B": distribution("neutral"),
            "intended_category": "anger",
            "human_validated": False,
        }
    }
    write_json(smoke / "outputs.json", outputs)
    write_json(smoke / "recording_lock.json", {"fixture": "accepted-hash"})
    write_json(smoke / "intended_mapping.json", {"temporary": True})
    write_json(smoke / "integrity.json", {"model_files_sha256": {str(frozen): digest(frozen)}})
    write_json(
        smoke / "COMPLETED.json",
        {
            field: digest(smoke / name)
            for field, name in [
                ("outputs_sha256", "outputs.json"),
                ("recording_lock_sha256", "recording_lock.json"),
                ("integrity_sha256", "integrity.json"),
                ("mapping_sha256", "intended_mapping.json"),
            ]
        },
    )
    monkeypatch.setattr("scripts.controlled_delivery_smoke.readiness", lambda _: {"complete": True})
    monkeypatch.setattr(
        "scripts.controlled_delivery_smoke.accepted", lambda _: ([], {"fixture": "accepted-hash"})
    )

    def no_model(*args, **kwargs):
        raise AssertionError("Listener interpretation must not load any encoder")

    def interpreted(folder, clean):
        assert folder == root
        assert "intended_category" not in clean["fixture"]
        assert "human_validated" not in clean["fixture"]
        assert clean["fixture"]["B"] == outputs["fixture"]["B"]
        return {
            "interpretation": "synthetic probability fixture",
            "listener_procedure": "test stub",
            "collection": {"complete": True},
            "counts": {},
            "cases": [],
        }

    monkeypatch.setattr(src.b_inference, "RoadmapB", no_model)
    monkeypatch.setattr("scripts.controlled_delivery_smoke.summarize", interpreted)
    listener(root, smoke, dest)
    assert json.loads((dest / "integrity.json").read_text())["model_inference_rerun"] is False
    assert json.loads((dest / "outputs.json").read_text())["fixture"]["B"] == outputs["fixture"]["B"]
    (smoke / "outputs.json").write_text("{}")
    with pytest.raises(ValueError, match="saved evidence changed"):
        listener(root, smoke, tmp_path / "must_not_exist")
    assert not (tmp_path / "must_not_exist").exists()

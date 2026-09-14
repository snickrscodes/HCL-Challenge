"""Scoring and boundary fixtures independent of corpus outcomes."""

from copy import deepcopy
import numpy as np
import pytest
from scripts.ta_system_evaluate import (
    acceptance,
    actor_mean,
    checked_path,
    classification_metrics,
    conditional_target,
    probabilities,
    repeat_summary,
    route_rows,
    route_summary,
    verify_shuffle,
)


def row(key, actor, counts, values):
    return {
        "id": key,
        "actor": actor,
        "counts9": counts,
        "transcript": "Tell me more.",
        "B_state": {"emotion": "neutral", "confidence": 0.8},
        "models": {"D2": {"probabilities": values, "representation": [1.0] * 128, "raw_logits": [0.0] * 6}},
    }


def test_compatibility_keeps_unsupported_mass_and_ties():
    assert conditional_target([0, 3, 0, 0, 7, 0, 0, 0, 0]) is None
    assert conditional_target([0, 2, 0, 0, 4, 4, 0, 0, 0]).tolist() == [0, 0, 0, 0.5, 0.5, 0]
    with pytest.raises(ValueError):
        conditional_target([1] * 8)


def test_actor_weighting_not_record_pooled():
    assert actor_mean([1, 1, -1], ["a", "a", "b"]) == 0


@pytest.mark.parametrize(
    "relative", ["ta_closure_v1/confirmation/predictions/a.json", "test/a.pt", "../outside.json"]
)
def test_forbidden_paths(tmp_path, relative):
    with pytest.raises(ValueError):
        checked_path(tmp_path, relative)


@pytest.mark.parametrize("mapping", [{"a": "a"}, {"a": "b", "b": "b"}, {"a": "b", "b": "c"}])
def test_shuffle_is_complete_bijection(mapping):
    with pytest.raises(ValueError):
        verify_shuffle(mapping, ["a", "b"])


def test_signed_score_penalizes_wrong_route_and_abstention_zero():
    rows = [
        row("a", "01", [0, 0, 0, 0, 10, 0, 0, 0, 0], [0.04, 0.04, 0.04, 0.8, 0.04, 0.04]),
        row("b", "01", [0, 0, 0, 0, 0, 10, 0, 0, 0], [0.04, 0.04, 0.04, 0.8, 0.04, 0.04]),
        row("c", "02", [0, 0, 0, 0, 10, 0, 0, 0, 0], [0.8, 0.04, 0.04, 0.04, 0.04, 0.04]),
    ]
    before = deepcopy(rows)
    result = route_summary(rows, route_rows(rows))
    assert rows == before
    assert result["actor_equal_signed_routed_agreement"] == 0
    assert result["eligible_route_coverage"] == 2 / 3
    assert result["routed_mean_listener_agreement"] == 0.5
    assert result["unique_plurality_class_slices"]["fear"]["routes"]["vocal_angry"] == 1
    baseline = route_summary(rows, route_rows(rows, unavailable=True))
    assert baseline["eligible_route_count"] == 0
    assert baseline["actor_equal_signed_routed_agreement"] == 0


def test_shuffle_changes_only_evidence_assignment():
    rows = [
        row("a", "01", [0, 0, 0, 0, 10, 0, 0, 0, 0], [0.04, 0.04, 0.04, 0.8, 0.04, 0.04]),
        row("b", "02", [0, 0, 0, 0, 0, 10, 0, 0, 0], [0.04, 0.04, 0.04, 0.04, 0.8, 0.04]),
    ]
    before = deepcopy(rows)
    actions = route_rows(rows, mapping={"a": "b", "b": "a"})
    assert actions["a"]["route_id"] == "vocal_fear"
    assert actions["b"]["route_id"] == "vocal_angry"
    assert rows == before


def test_repeat_threshold_crossing_counts_even_with_same_top1():
    rows = [
        row("a", "01", [0, 0, 0, 0, 10, 0, 0, 0, 0], [0.04, 0.04, 0.04, 0.8, 0.04, 0.04]),
        row("b", "01", [0, 0, 0, 0, 10, 0, 0, 0, 0], [0.1, 0.1, 0.1, 0.5, 0.1, 0.1]),
    ]
    pairs = [
        {"left": "a", "right": "b", "actor": "01", "kind": "repeat", "listener_consistent_repeat_proxy": True}
    ]
    result = repeat_summary(rows, pairs, route_rows(rows))["stable_proxy"]
    assert result["route_changes"] == 1
    assert result["D2_top1_changes"] == 0
    assert result["route_to_baseline_transitions"] == 1
    assert np.isclose(result["per_actor"]["01"]["D2_TV_mean"], 0.3)


def test_calibration_bin_includes_confidence_one():
    result = classification_metrics([[1.0, 0.0], [0.0, 1.0]], [0, 1], 2)
    assert result["ece_15"] == 0
    assert result["macro_f1"] == 1
    assert result["nll"] == 0


@pytest.mark.parametrize("values", [[0.1] * 6, [float("nan")] * 6, [1.1, -0.1, 0, 0, 0, 0]])
def test_invalid_probabilities(values):
    with pytest.raises(ValueError):
        probabilities(values, 6)


def test_acceptance_never_waives_failed_repeat():
    result = {
        "working": {
            "candidate": {
                "actor_equal_signed_routed_agreement": 0.2,
                "routed_mean_listener_agreement": 0.8,
                "eligible_route_coverage": 0.3,
                "actors_with_eligible_route": 12,
            },
            "evidence_only": {"actor_equal_signed_routed_agreement": 0},
            "fixed_coupled_shuffle": {"actor_equal_signed_routed_agreement": -0.1},
        },
        "stability": {"stable_proxy": {"route_changes": 13}},
        "source_preservation": {"exact_preservation": True},
        "fixed_cases": {"passed": True},
        "integrity": {"passed": True},
    }
    decision = acceptance(result)
    assert decision["decision"] == "REJECT_CHANGE"
    assert decision["recommended_default"] == "evidence_only"
    assert decision["measured_requirements"]["stable_route_changes_at_most_12_of_67"] == "FAIL"

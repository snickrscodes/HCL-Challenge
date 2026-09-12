import numpy as np
import pytest
import torch

from src.b_data import eligibility, encode_canonical, frozen_policy, rows_for
from src.b_evaluate import bootstrap, derangement, distributions, fixed_slices, swapped, transition_stats
from src.b_inference import add_noise
from src.evaluate import probabilities
from test_b_fusion import inputs


def test_duration_and_decode_semantics():
    row = {"audio_valid": True, "duration_s": 60.0}
    assert eligibility(row) == (True, None)
    row["duration_s"] = 60.00001
    assert eligibility(row) == (False, "duration_limit")
    assert row["audio_valid"] is True
    row["audio_valid"] = False
    assert eligibility(row) == (False, "missing_or_invalid")


def test_canonical_singleton_calls():
    calls = []

    def encoder(waves):
        calls.append(len(waves))
        return torch.tensor([[len(w)] * 768 for w in waves])

    encode_canonical(encoder, [np.ones(400), np.ones(400)], {"encoding": "singleton"})
    assert calls == [1, 1]


def test_test_partition_forbidden():
    with pytest.raises(ValueError, match="test inference"):
        rows_for({}, "test")


def test_no_training_policy_without_real_report(tmp_path):
    with pytest.raises(FileNotFoundError):
        frozen_policy({"output_root": str(tmp_path)})


def test_missing_probabilities_are_preserved_text_fallback():
    x = inputs()
    x["available"][0] = False
    z = x["zt"] + 5 * torch.randn_like(x["zt"])
    p = distributions(z, x, 0.2, 1.6053)
    assert np.array_equal(p[0], probabilities(x["zt"][0], 1.6053))


def test_paired_audio_replacement_and_fixed_slice():
    x = inputs(20)
    mapping = derangement(np.arange(20), 50)
    assert np.all(mapping != np.arange(20))
    assert np.array_equal(mapping, derangement(np.arange(20), 50))
    replaced = swapped(x, mapping)
    assert torch.equal(replaced["ha"], x["ha"][mapping])
    assert torch.equal(replaced["za"], x["za"][mapping])
    assert torch.equal(replaced["zt"], x["zt"])
    masks = fixed_slices(x)
    assert all(len(m) == 20 for m in masks.values())


def test_transition_denominators():
    v = transition_stats([0, 1, 2, 3], [1, 1, 0, 0], [0, 0, 1, 0])
    assert v["corrected"] == 1 and v["harmed"] == 1
    assert v["wrong_to_different_wrong"] == 1
    assert v["correction_precision_among_changed"] == 1 / 3
    assert v["recovery_rate_among_reference_errors"] == 1 / 3
    assert v["harm_rate_among_reference_correct"] == 1


def test_dialogue_bootstrap_identical_predictors():
    rows = [{"dialogue_id": x} for x in (0, 0, 1, 1, 1, 2)]
    labels = np.array([0, 1, 2, 3, 4, 5])
    r = bootstrap(rows, labels, labels, labels, 1000, 4)
    assert all(v == {"low": 0, "high": 0} for v in r["differences"].values())


def test_noise_snr_and_real_silence_preservation():
    wave = np.random.default_rng(8).normal(0, 0.1, 16000).astype(np.float32)
    noisy, meta = add_noise(wave, 10, 4)
    snr = 10 * np.log10(np.mean(wave**2) / np.mean((noisy - wave) ** 2))
    assert snr == pytest.approx(10, abs=1e-4)
    silent, meta = add_noise(np.zeros_like(wave), 10, 4)
    assert not meta["noise_added"] and not silent.any()


def test_constant_float32_correlation_is_undefined_not_nan():
    from src.b_evaluate import correlation

    x = np.full(839, 0.12, dtype=np.float32)
    assert correlation(x, np.arange(839)) is None
    assert correlation(np.arange(839), x) is None
    assert correlation(np.arange(839), np.arange(839)) == pytest.approx(1.0)

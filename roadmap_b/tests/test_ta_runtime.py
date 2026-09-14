"""Portable loading and cleanup tests without backbone inference."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from src import ta_runtime as runtime


def test_absolute_config_is_independent_of_working_directory(tmp_path, monkeypatch):
    root = tmp_path / "model-assets"
    (root / "provenance").mkdir(parents=True)
    cfg = {"runtime": {"device": "cuda", "dtype": "bfloat16", "cpu_threads": 4}}
    (root / "provenance/roadmap_b.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path.parent)
    resolved = runtime.portable_config(root, "cpu")
    assert resolved["baseline_root"] == str(root / "a")
    assert resolved["output_root"] == str(root / "b")
    assert resolved["baseline_lock"] == str(root / "provenance/baseline_lock.json")
    assert resolved["runtime"]["device"] == "cpu"
    with pytest.raises(ValueError, match="explicit"):
        runtime.portable_config(root, "auto")


def test_missing_assets_fail_before_model_construction(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, "RoadmapB", lambda *a, **kw: calls.append(True))
    with pytest.raises(runtime.AssetIntegrityError, match="Missing asset root"):
        runtime.PortableTADual(tmp_path / "missing")
    assert calls == []


def test_load_d2_rejects_wrong_role_before_constructing_head(tmp_path):
    (tmp_path / "d2").mkdir()
    torch.save(
        {"condition": "D3", "seed": 1337, "namespace": "crema6_audio_votes_v1", "labels": []},
        tmp_path / "d2/checkpoint.pt",
    )
    with pytest.raises(runtime.AssetIntegrityError, match="identity"):
        runtime.load_d2(tmp_path, {"namespace": "crema6_audio_votes_v1", "labels": []}, "cpu")


def test_embedded_normalization_and_frozen_head_roundtrip(tmp_path):
    (tmp_path / "d2").mkdir()
    mean, scale = torch.zeros(13, 1536), torch.ones(13, 1)
    head = runtime.EvidenceHead("D2", mean, scale).eval()
    labels = ["neutral", "happy", "sad", "angry", "fear", "disgust"]
    checkpoint = {
        "condition": "D2",
        "seed": 1337,
        "namespace": "crema6_audio_votes_v1",
        "labels": labels,
        "contract_sha256": "contract",
        "state_dict": head.state_dict(),
    }
    torch.save(checkpoint, tmp_path / "d2/checkpoint.pt")
    (tmp_path / "d2/calibration.json").write_text(json.dumps({"temperature": 1.1}))
    value = {
        "namespace": "crema6_audio_votes_v1",
        "labels": labels,
        "historical_c1_contract_sha256": "contract",
        "temperatures": {"D2": 1.1},
        "embedded_normalization": {
            "mean_sha256": runtime.buffer_sha256(mean),
            "scale_sha256": runtime.buffer_sha256(scale),
        },
    }
    loaded, _, temperature = runtime.load_d2(tmp_path, value, "cpu")
    assert temperature == 1.1
    assert not any(p.requires_grad for p in loaded.parameters())
    x = torch.arange(13 * 1536, dtype=torch.float32).reshape(1, 13, 1536) / 1000
    torch.testing.assert_close(head(x)["logits"], loaded(x)["logits"], atol=0, rtol=0)
    value["embedded_normalization"]["mean_sha256"] = "changed"
    with pytest.raises(runtime.AssetIntegrityError, match="normalization"):
        runtime.load_d2(tmp_path, value, "cpu")


class FakeTap:
    def __init__(self):
        self.calls = 0
        self.last_summaries = None

    def clear(self):
        self.last_summaries = None


class FakeB:
    device = torch.device("cpu")

    def __init__(self, tap):
        self.tap = tap

    def predict_turn(self, text, audio, sample_rate, history, *, speaker):
        if audio is not None:
            self.tap.calls += 1
            self.tap.last_summaries = torch.zeros(1, 13, 1536)
        return {
            "audio_available": audio is not None,
            "audio_unavailable_reason": None if audio is not None else "missing_or_invalid",
            "latency_ms": {"text": 0.0, "fusion_response": 0.0},
        }


def test_inherited_prediction_clears_evidence_after_head_failure_then_missing():
    model = runtime.PortableTADual.__new__(runtime.PortableTADual)
    model.tap = FakeTap()
    model.b = FakeB(model.tap)
    model.clock = None
    model.namespace = "crema6_audio_votes_v1"
    model.provenance = (("condition", "D2"),)

    def fail(_):
        raise RuntimeError("injected head failure")

    model.head = fail
    with pytest.raises(RuntimeError, match="injected"):
        model.predict_turn("hello", np.zeros(400, dtype=np.float32), 16000, [])
    assert model.tap.last_summaries is None
    result = model.predict_turn("hello", None, None, [])
    assert not result["delivery_evidence"]["available"]
    assert result["delivery_evidence"]["distribution"] is None
    assert model.tap.calls == 1
    assert model.tap.last_summaries is None


def test_close_releases_hooks_and_clears_evidence():
    model = runtime.PortableTADual.__new__(runtime.PortableTADual)
    closed = []
    model.clock = SimpleNamespace(close=lambda: closed.append(True))
    model.tap = FakeTap()
    model.tap.last_summaries = "stale"
    model.close()
    model.close()
    assert closed == [True]
    assert model.tap.last_summaries is None

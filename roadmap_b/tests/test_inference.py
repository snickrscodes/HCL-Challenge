from pathlib import Path

import numpy as np
import pytest
import torch

from src.audio import read_waveform
from src.calibrate import late_logits
from src.context import histories
from src.evaluate import probabilities
from src.inference import RoadmapA
from src.pipeline import run
from src.session import Session
from src.smoke import run_smoke
from src.train_fusion import cache_logits
from src.utils import audio_file, key, read_json, split_rows


def test_full_32_example_pipeline_raw_inference_and_session(tmp_path):
    cfg = run_smoke(tmp_path / "smoke")
    bundle = read_json(Path(cfg["output_root"]) / "results_bundle.json")
    assert bundle["data_audit"]["fixture"] and sum(bundle["data_audit"]["counts"].values()) == 32
    assert set(bundle["results"]["test"]) == {
        "majority",
        "text_current",
        "text_context",
        "audio",
        "concat",
        "late_fusion",
    }
    assert (Path(cfg["output_root"]) / "ROADMAP_A_REPORT.md").exists()
    predictor = RoadmapA(cfg)
    rows = split_rows(cfg, "dev_model")
    row = rows[0]
    wave = read_waveform(audio_file(cfg, row))
    history = histories(rows)[key(row)]
    prediction = predictor.predict_turn(row["text"], wave, 16000, history, speaker=row["speaker"])
    zt, za, valid = cache_logits(cfg, "dev_model")
    expected = probabilities(
        late_logits(zt[0], za[0], predictor.selection["alpha"]), predictor.calibration["fusion"]
    )
    assert valid[0]
    assert np.allclose(list(prediction.distribution.values()), expected, atol=2e-6, rtol=2e-5)
    assert predictor.parameter_ledger()["within_6b"]
    assert prediction.audio_available and prediction.response
    assert all(p.grad is None for p in predictor.audio.parameters())
    # No audio encoder call is permitted for the missing-audio path.
    original = predictor.audio.forward
    predictor.audio.forward = lambda _: (_ for _ in ()).throw(AssertionError("Audio should not run"))
    missing = predictor.predict_turn(row["text"], None, None, history, speaker=row["speaker"])
    assert not missing.audio_available and missing.audio_prediction is None
    assert np.allclose(
        list(missing.distribution.values()), probabilities(zt[0], predictor.calibration["text"]), atol=2e-6
    )
    predictor.audio.forward = original
    # The full causal text checkpoint must also agree with its offline path.
    from src.models import TextClassifier
    from src.train_text import predict_text

    original_text = (predictor.text, predictor.tokenizer, predictor.text_metadata)
    ctx_model, ctx_tokenizer, ctx_meta = TextClassifier.load(Path(cfg["checkpoint_root"]) / "text_context")
    predictor.text, predictor.tokenizer, predictor.text_metadata = ctx_model.eval(), ctx_tokenizer, ctx_meta
    causal_cache = predict_text(cfg, ctx_model, ctx_tokenizer, rows, True)
    index = next(i for i, value in enumerate(rows) if histories(rows)[key(value)])
    target = rows[index]
    causal = predictor.predict_turn(
        target["text"], None, None, histories(rows)[key(target)], speaker=target["speaker"]
    )
    assert np.allclose(
        list(causal.distribution.values()),
        probabilities(causal_cache["logits"][index], predictor.calibration["text"]),
        atol=2e-6,
    )
    predictor.text, predictor.tokenizer, predictor.text_metadata = original_text
    session = Session(predictor, speaker=row["speaker"], history=history)
    session.start()
    session.begin_turn()
    for chunk in np.array_split(wave, 3):
        session.push_audio(chunk)
    session.set_transcript(row["text"])
    replay = session.end_turn()
    assert replay.distribution == prediction.distribution
    session.reset()
    assert session.history == history and not session.started and not session.active
    with pytest.raises(RuntimeError):
        session.end_turn()
    with pytest.raises(RuntimeError, match="already started"):
        run(cfg, final_test=True)
    fresh = read_json(Path(cfg["output_root"]).parent / "fresh_process_prediction.json")
    assert fresh["response"] and fresh["audio_available"]


def test_missing_is_distinct_from_invalid(tiny_cfg):
    # Audio validation itself rejects malformed data instead of inventing neutral.
    from src.audio import prepare_waveform

    with pytest.raises(ValueError):
        prepare_waveform(torch.full((800,), float("inf")), 16000)

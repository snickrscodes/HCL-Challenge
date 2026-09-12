import hashlib
import io
import json

import numpy as np
import pytest
import soundfile as sf

from src.recording_kit import WEB, listener_plan, planned, store_rating, store_recording


def protocol(tmp_path):
    data = (WEB / "phrases.json").read_bytes()
    (tmp_path / "protocol.json").write_bytes(data)
    return json.loads(data)


def sample(recording_id):
    stream = io.BytesIO()
    sf.write(
        stream, np.sin(np.arange(8000) / 13).astype(np.float32) * 0.1, 16000, format="WAV", subtype="FLOAT"
    )
    wav = stream.getvalue()
    return {
        "recording_id": recording_id,
        "consented": True,
        "words_confirmed": True,
        "recording_timestamp": "2026-09-12T01:00:00Z",
        "waveform_sha256": hashlib.sha256(wav).hexdigest(),
    }, wav


def test_fixed_human_protocol_has_60_unique_recordings(tmp_path):
    p = protocol(tmp_path)
    assert len(planned(p)) == 60 and len(p["phrases"]) == 10
    assert sum(x["stable_control"] for x in p["phrases"]) == 2
    assert all(x["history"] == [] for x in p["phrases"])
    with pytest.raises(ValueError, match="Finish all"):
        listener_plan(tmp_path, p)


def test_accept_requires_consent_hash_and_exact_words(tmp_path):
    p = protocol(tmp_path)
    meta, wav = sample(next(iter(planned(p))))
    for field in ("consented", "words_confirmed"):
        with pytest.raises(ValueError):
            store_recording(tmp_path, p, {**meta, field: False}, wav)
    with pytest.raises(ValueError, match="hashes differ"):
        store_recording(tmp_path, p, {**meta, "waveform_sha256": "bad"}, wav)
    value = store_recording(tmp_path, p, meta, wav)
    assert value["sample_rate"] == 16000 and value["num_samples"] == 8000
    assert not value["public_audio_release_consented"]
    with pytest.raises(FileExistsError):
        store_recording(tmp_path, p, meta, wav)


def test_blinded_plan_hides_intention_and_ratings_are_immutable(tmp_path):
    p = protocol(tmp_path)
    # Synthetic unit-test waveforms live only in pytest's temporary directory.
    for k in planned(p):
        meta, wav = sample(k)
        store_recording(tmp_path, p, meta, wav)
    public, private = listener_plan(tmp_path, p)
    assert len(public["clips"]) == 60 and len(public["pairs"]) == 60
    assert "intended_delivery" not in json.dumps(public) and "speaker" not in json.dumps(public)
    assert listener_plan(tmp_path, p)[1] == private
    value = {
        "listener_id": "L01",
        "independent_listener": True,
        "kind": "clip",
        "token": public["clips"][0]["token"],
        "category": "unclear_or_mixed",
        "timestamp": "2026-09-12T01:00:00Z",
    }
    with pytest.raises(ValueError):
        store_rating(tmp_path, p, {**value, "independent_listener": False})
    store_rating(tmp_path, p, value)
    with pytest.raises(FileExistsError):
        store_rating(tmp_path, p, value)

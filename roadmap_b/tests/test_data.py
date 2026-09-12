from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.constants import LABELS, LABEL_TO_ID
from src.data import grouped_dev_split, prepare_dataset, validate_rows
from src.smoke import make_media
from src.utils import key, read_json, read_rows, row_sort


def test_label_order_is_immutable():
    assert len(LABELS) == 7
    assert [LABEL_TO_ID[name] for name in LABELS] == list(range(7))
    with pytest.raises(TypeError):
        LABEL_TO_ID["new"] = 7


def test_preprocessing_keeps_failures_and_reconciles(tiny_cfg, tmp_path):
    raw = make_media(tmp_path / "raw", missing=True)
    summary = prepare_dataset(raw, tiny_cfg["data_root"], tiny_cfg, fixture=True)
    rows = read_rows(Path(tiny_cfg["data_root"]) / "manifest.jsonl")
    assert summary["counts"] == {"train": 16, "dev": 12, "test": 4}
    assert len(rows) == 32 and len({key(r) for r in rows}) == 32
    assert summary["invalid_audio"] == 2
    assert sum(not r["audio_valid"] for r in rows) == 2
    failures = read_rows(Path(tiny_cfg["data_root"]) / "audit" / "failures.jsonl")
    assert {r["key"] for r in failures} == {"train/0/0", "dev/0/0"}
    for row in rows:
        if row["audio_valid"]:
            wave, rate = sf.read(Path(tiny_cfg["data_root"]) / row["audio_path"])
            assert wave.ndim == 1 and rate == 16000 and np.isfinite(wave).all()
            assert row["num_samples"] == len(wave)
    validate_rows(rows, summary["counts"])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_rows(rows + rows[:1])
    with pytest.raises(ValueError, match="counts"):
        validate_rows(rows, {"train": 1})
    policy = read_json(Path(tiny_cfg["data_root"]) / "audit" / "dev_dialogues.json")
    dev = [r for r in rows if r["split"] == "dev"]
    assert policy == grouped_dev_split(
        dev,
        tiny_cfg["seed"],
        tiny_cfg["data"]["calib_fraction"],
        tiny_cfg["data"]["split_candidates"],
    )
    assert not set(policy["dev_model"]) & set(policy["dev_calib"])
    assert set(policy["dev_model"]) | set(policy["dev_calib"]) == {r["dialogue_id"] for r in dev}


def test_numeric_sort():
    rows = [{"split": "train", "dialogue_id": d, "utterance_id": u} for d, u in [(10, 2), (2, 10), (2, 2)]]
    assert [(r["dialogue_id"], r["utterance_id"]) for r in sorted(rows, key=row_sort)] == [
        (2, 2),
        (2, 10),
        (10, 2),
    ]


def test_nonfinite_timestamp_is_an_explicit_error():
    from src.data import timestamp_seconds

    for value in ("00:00:nan", "00:90:01", "nonsense"):
        with pytest.raises(ValueError, match="Malformed"):
            timestamp_seconds(value)

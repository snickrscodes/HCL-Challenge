import pytest
import torch
from transformers import AutoTokenizer

from src.context import add_markers
from src.smoke import make_tiny_models
from src.utils import load_config


@pytest.fixture
def tiny_cfg(tmp_path):
    torch.set_num_threads(2)
    cfg = load_config()
    cfg.update(
        output_root=str(tmp_path / "artifacts"),
        data_root=str(tmp_path / "processed"),
        feature_root=str(tmp_path / "features"),
        checkpoint_root=str(tmp_path / "checkpoints"),
    )
    return make_tiny_models(tmp_path, cfg)


@pytest.fixture
def tokenizer(tiny_cfg):
    value = AutoTokenizer.from_pretrained(tiny_cfg["text"]["model"])
    add_markers(value)
    return value

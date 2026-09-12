import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.audio import FrozenAudio
from src.b_data import encode_canonical
from src.b_fusion import ResidualFusion
from src.b_train import forward
from src.models import TextClassifier, load_head
from src.utils import counts, load_config

pytestmark = [
    pytest.mark.pretrained,
    pytest.mark.skipif(
        os.environ.get("RUN_PRETRAINED") != "1" or not os.environ.get("B_BASELINE_ROOT"),
        reason="Provide RUN_PRETRAINED=1 and B_BASELINE_ROOT for frozen real-checkpoint integration",
    ),
]


def test_real_checkpoint_counts_and_singleton_repeat():
    base = Path(os.environ["B_BASELINE_ROOT"])
    cfg = load_config(base / "configs/roadmap_a.yaml")
    cfg["checkpoint_root"] = str(base / "checkpoints")
    text, _, _ = TextClassifier.load(base / "checkpoints/text_context")
    audio = FrozenAudio(cfg, local=True)
    head = load_head(base / "checkpoints/audio").eval()
    assert counts(text)["total"] == 124062727
    assert counts(audio)["total"] == 94381936
    assert counts(head)["total"] == 198663
    assert counts(ResidualFusion())["total"] == 54728
    assert 124062727 + 94381936 + 198663 + 54728 + 5 == 218698059
    wave = np.random.default_rng(14).normal(0, 0.05, 16000).astype(np.float32)
    once = encode_canonical(audio, [wave], {"encoding": "singleton"})
    repeated = encode_canonical(audio, [wave, wave], {"encoding": "singleton"})
    assert torch.allclose(once[0], repeated[0], atol=2e-4, rtol=2e-4)
    assert torch.allclose(once[0], repeated[1], atol=2e-4, rtol=2e-4)
    assert not any(p.requires_grad for p in audio.parameters())
    # Only the new module enters the optimizer; old frozen head is an immutable feature provider.
    head.requires_grad_(False)
    before = {n: p.detach().clone() for n, p in head.named_parameters()}
    with torch.inference_mode():
        za = head(once.float())
    x = {
        "ht": torch.randn(1, 768),
        "ha": once.clone(),
        "zt": torch.randn(1, 7),
        "za": za.clone(),
        "available": torch.ones(1, dtype=torch.bool),
    }
    model = ResidualFusion()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    torch.nn.functional.cross_entropy(forward(model, x)["logits"], torch.tensor([3])).backward()
    opt.step()
    assert all(torch.equal(p, before[n]) for n, p in head.named_parameters())

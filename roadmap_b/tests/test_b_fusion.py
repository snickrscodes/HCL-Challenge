import pytest
import torch

from src.b_fusion import ResidualFusion, VARIANTS, bounded_center
from src.b_train import forward, load_model, predict
from src.constants import LABELS
from src.utils import save_tensor


def inputs(n=16):
    g = torch.Generator().manual_seed(15)
    return {
        "ht": torch.randn(n, 768, generator=g),
        "ha": torch.randn(n, 768, generator=g),
        "zt": torch.randn(n, 7, generator=g),
        "za": torch.randn(n, 7, generator=g),
        "available": torch.ones(n, dtype=torch.bool),
        "labels": torch.arange(n) % 7,
    }


@pytest.mark.parametrize("variant", VARIANTS)
def test_initialization_missing_and_zero(variant):
    model = ResidualFusion(variant).eval()
    x = inputs()
    assert torch.equal(forward(model, x)["logits"], x["zt"])
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
    assert torch.equal(forward(model, x, forced_gate=0)["logits"], x["zt"])
    x["available"][:] = False
    assert torch.equal(forward(model, x)["logits"], x["zt"])


def test_strict_bound_and_centering():
    for scale in (0, 1, 100, 1e6):
        delta = bounded_center(torch.randn(64, 7) * scale)
        assert torch.allclose(delta.sum(-1), torch.zeros(64), atol=2e-6)
        assert delta.abs().max() <= 2.0
    # Saturated unequal signs exercise centering that would violate a naive 2*tanh bound.
    z = torch.tensor([[100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0]])
    assert bounded_center(z).abs().max() <= 2


@pytest.mark.parametrize("variant", ("constant", "adaptive", "text_only"))
def test_initial_gradient_step_and_frozen_inputs(variant):
    model = ResidualFusion(variant)
    x = inputs()
    for name in ("ht", "ha", "zt", "za"):
        x[name].requires_grad_()
    before = model.residual.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss = torch.nn.functional.cross_entropy(forward(model, x)["logits"], x["labels"])
    loss.backward()
    assert model.residual.weight.grad.abs().sum() > 0
    assert all(x[name].grad is None for name in ("ht", "ha", "zt", "za"))
    optimizer.step()
    assert not torch.equal(before, model.residual.weight)


def test_parameter_counts_and_no_audio_control():
    adaptive, text = ResidualFusion("adaptive"), ResidualFusion("text_only")

    def count(m):
        return sum(p.numel() for p in m.parameters())

    assert count(adaptive.gate) == 49345
    assert count(adaptive.residual) == 5383
    assert count(adaptive) == count(text) == 54728
    assert count(ResidualFusion("constant")) == 5384
    assert count(ResidualFusion("bias")) == 7
    with torch.no_grad():
        text.residual.weight.normal_()
    x = inputs()
    p = predict(text, x)["logits"]
    x["ha"].normal_(100, 100)
    x["za"].normal_(100, 100)
    assert torch.equal(p, predict(text, x)["logits"])


@pytest.mark.parametrize("variant", VARIANTS)
def test_roundtrip(tmp_path, variant):
    model = ResidualFusion(variant)
    x = inputs()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss = torch.nn.functional.cross_entropy(forward(model, x)["logits"], x["labels"])
    loss.backward()
    optimizer.step()
    expected = predict(model, x)["logits"]
    path = tmp_path / "head.pt"
    save_tensor(path, {"state_dict": model.state_dict(), "spec": model.spec, "labels": list(LABELS)})
    restored, _ = load_model(path)
    assert torch.equal(expected, predict(restored, x)["logits"])

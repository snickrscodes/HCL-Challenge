import torch

from src.context import collate_text, encode_turn
from src.models import ClassificationHead, TextClassifier, load_head, save_head
from src.utils import counts


def test_text_optimizer_checkpoint_and_current_pool(tiny_cfg, tmp_path):
    model, tokenizer = TextClassifier.pretrained(tiny_cfg)
    item = encode_turn(tokenizer, "I feel joy today.", [], "A")
    batch = collate_text([item], tokenizer.pad_token_id)
    model.train()
    z, h = model(**batch)
    assert z.shape == (1, 7) and h.shape == (1, 32)
    loss = torch.nn.functional.cross_entropy(z, torch.tensor([1]))
    assert torch.isfinite(loss)
    before = model.head.weight.detach().clone()
    loss.backward()
    torch.optim.AdamW(model.parameters(), lr=0.001).step()
    assert not torch.equal(before, model.head.weight)
    model.eval()
    expected = model(**batch)[0].detach()
    model.save(tmp_path / "checkpoint", tokenizer, {"history_turns": 0, "max_length": 128, "context": False})
    loaded, _, _ = TextClassifier.load(tmp_path / "checkpoint")
    loaded.eval()
    assert torch.equal(expected, loaded(**batch)[0])
    assert counts(model) == counts(loaded)


def test_head_checkpoint_dimensions(tmp_path):
    head = ClassificationHead(768, 256).eval()
    x = torch.randn(2, 768)
    assert head(x).shape == (2, 7)
    assert counts(head)["total"] == 198663
    save_head(tmp_path / "audio", head, 768, 256)
    restored = load_head(tmp_path / "audio").eval()
    assert torch.equal(head(x), restored(x))
    assert ClassificationHead(1536, 256)(torch.randn(2, 1536)).shape == (2, 7)

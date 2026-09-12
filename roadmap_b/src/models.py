"""Only three learned modules: text classifier, audio head, concat head."""

from pathlib import Path

import torch
from transformers import AutoTokenizer, RobertaModel

from .constants import LABELS
from .context import add_markers
from .utils import load_tensor, save_tensor, write_json, read_json


class TextClassifier(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Linear(backbone.config.hidden_size, len(LABELS))

    def forward(self, input_ids, attention_mask, current_mask):
        if (current_mask.sum(1) == 0).any():
            raise ValueError("Missing current-utterance content mask")
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = current_mask[..., None]
        pooled = (hidden.float() * mask).sum(1) / mask.sum(1)
        return self.head(pooled), pooled

    @classmethod
    def pretrained(cls, cfg):
        tokenizer = AutoTokenizer.from_pretrained(cfg["text"]["model"], revision=cfg["text"]["revision"])
        add_markers(tokenizer)
        backbone = RobertaModel.from_pretrained(
            cfg["text"]["model"], revision=cfg["text"]["revision"], add_pooling_layer=False
        )
        backbone.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        return cls(backbone), tokenizer

    def save(self, path, tokenizer, metadata):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path / "encoder")
        tokenizer.save_pretrained(path / "tokenizer")
        save_tensor(path / "head.pt", self.head.state_dict())
        write_json(path / "metadata.json", {**metadata, "labels": list(LABELS)})

    @classmethod
    def load(cls, path):
        path = Path(path)
        metadata = read_json(path / "metadata.json")
        if tuple(metadata["labels"]) != LABELS:
            raise ValueError("Checkpoint label order mismatch")
        model = cls(
            RobertaModel.from_pretrained(path / "encoder", add_pooling_layer=False, local_files_only=True)
        )
        model.head.load_state_dict(load_tensor(path / "head.pt"))
        tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
        return model, tokenizer, metadata


class ClassificationHead(torch.nn.Sequential):
    def __init__(self, input_size, hidden_size=256):
        super().__init__(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, len(LABELS)),
        )


def save_head(path, head, input_size, hidden_size):
    path = Path(path)
    save_tensor(path / "head.pt", head.state_dict())
    write_json(
        path / "metadata.json", {"input_size": input_size, "hidden_size": hidden_size, "labels": list(LABELS)}
    )


def load_head(path):
    path = Path(path)
    meta = read_json(path / "metadata.json")
    if tuple(meta["labels"]) != LABELS:
        raise ValueError("Checkpoint label order mismatch")
    head = ClassificationHead(meta["input_size"], meta["hidden_size"])
    head.load_state_dict(load_tensor(path / "head.pt"))
    return head

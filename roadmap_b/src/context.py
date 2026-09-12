"""Causal packing with a separate mask for current-utterance content."""

from dataclasses import dataclass

import torch

from .constants import MARKERS
from .utils import key, row_sort


@dataclass(frozen=True)
class Turn:
    text: str
    speaker: str = "user"


def histories(rows, limit=3):
    result, groups = {}, {}
    for row in sorted(rows, key=row_sort):
        group = (row["split"], int(row["dialogue_id"]))
        previous = groups.setdefault(group, [])
        result[key(row)] = list(previous[-limit:]) if limit else []
        previous.append(Turn(row["text"], row["speaker"]))
    return result


def add_markers(tokenizer):
    tokenizer.add_special_tokens({"additional_special_tokens": list(MARKERS)})


def encode_turn(tokenizer, text, history, speaker, max_length=128, history_turns=3):
    current = tokenizer.encode(text, add_special_tokens=False)
    if not current:
        raise ValueError("Current transcript must contain at least one token")
    # BOS + CURRENT + current content + EOS. Markers never enter the pooling mask.
    capacity = max_length - 3
    if capacity < 1:
        raise ValueError("max_length must be at least four")
    truncated = len(current) > capacity
    current = current[:capacity]
    parts = []
    for turn in history[-history_turns:] if history_turns else []:
        marker = MARKERS[0] if turn.speaker == speaker else MARKERS[1]
        parts.append(
            [tokenizer.convert_tokens_to_ids(marker)]
            + tokenizer.encode(turn.text, add_special_tokens=False)
            + [tokenizer.eos_token_id]
        )
    while parts and sum(map(len, parts)) + len(current) + 3 > max_length:
        parts.pop(0)
    prefix = [tokenizer.bos_token_id] + [token for part in parts for token in part]
    prefix.append(tokenizer.convert_tokens_to_ids(MARKERS[2]))
    ids = prefix + current + [tokenizer.eos_token_id]
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "current_mask": [0] * len(prefix) + [1] * len(current) + [0],
        "current_truncated": truncated,
    }


def collate_text(items, pad_token_id):
    width = max(len(item["input_ids"]) for item in items)
    batch = {}
    for name in ("input_ids", "attention_mask", "current_mask"):
        pad = pad_token_id if name == "input_ids" else 0
        batch[name] = torch.tensor([item[name] + [pad] * (width - len(item[name])) for item in items])
    return batch

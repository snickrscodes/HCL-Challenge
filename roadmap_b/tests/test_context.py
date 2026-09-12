from src.constants import MARKERS
from src.context import Turn, encode_turn, histories


def test_no_future_dialogue_or_split_crossing():
    rows = [
        {"split": split, "dialogue_id": dialogue, "utterance_id": utterance, "text": text, "speaker": "A"}
        for split, dialogue, utterance, text in [
            ("train", 2, 2, "future"),
            ("train", 2, 0, "first"),
            ("train", 2, 1, "second"),
            ("train", 10, 0, "different dialogue"),
            ("dev", 2, 0, "different split"),
        ]
    ]
    values = histories(rows)
    assert [t.text for t in values["train/2/1"]] == ["first"]
    assert values["train/10/0"] == values["dev/2/0"] == []
    assert [t.text for t in values["train/2/2"]] == ["first", "second"]


def test_target_priority_markers_and_pooling(tokenizer):
    text = "I feel joy today."
    target_ids = tokenizer.encode(text, add_special_tokens=False)
    item = encode_turn(tokenizer, text, [Turn("no " * 30, "B"), Turn("yes", "A")], "A", max_length=16)
    assert [i for i, m in zip(item["input_ids"], item["current_mask"]) if m] == target_ids
    assert tokenizer.convert_tokens_to_ids(MARKERS[0]) in item["input_ids"]
    assert tokenizer.convert_tokens_to_ids(MARKERS[1]) not in item["input_ids"]
    assert not item["current_truncated"]
    long_item = encode_turn(tokenizer, "I " * 50, [Turn("no", "B")], "A", max_length=12)
    assert long_item["current_truncated"] and len(long_item["input_ids"]) == 12
    assert sum(long_item["current_mask"]) == 9


def test_current_only_ignores_history(tokenizer):
    a = encode_turn(tokenizer, "yes", [Turn("no", "A")], "A", history_turns=0)
    b = encode_turn(tokenizer, "yes", [], "A", history_turns=0)
    assert a == b

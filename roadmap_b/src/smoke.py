"""A 32-example synthetic integration fixture. Never reports MELD model quality."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel, WavLMConfig, WavLMModel

from .constants import LABELS
from .data import prepare_dataset
from .pipeline import run
from .utils import common_parser, load_config, write_json


def make_tiny_models(root, cfg):
    root = Path(root)
    torch.manual_seed(cfg["seed"])
    text_path, audio_path = root / "tiny_text", root / "tiny_audio"
    vocab = {
        word: i
        for i, word in enumerate(
            [
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "<mask>",
                "I",
                "feel",
                "today",
                ".",
                "That",
                "sounds",
                "fine",
                "really",
                "What",
                "happened",
                "?",
                "yes",
                "no",
            ]
            + list(LABELS)
        )
    }
    raw_tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw_tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    tokenizer.save_pretrained(text_path)
    text = RobertaModel(
        RobertaConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=132,
            bos_token_id=0,
            pad_token_id=1,
            eos_token_id=2,
        ),
        add_pooling_layer=False,
    )
    text.save_pretrained(text_path)
    audio = WavLMModel(
        WavLMConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            conv_dim=[8] * 7,
            conv_kernel=[10, 3, 3, 3, 3, 2, 2],
            conv_stride=[5, 2, 2, 2, 2, 2, 2],
            num_conv_pos_embeddings=16,
            num_conv_pos_embedding_groups=4,
            num_buckets=16,
            max_bucket_distance=32,
            mask_time_prob=0,
            apply_spec_augment=False,
        )
    )
    audio.save_pretrained(audio_path)
    cfg["text"].update(
        model=str(text_path), revision=None, batch_size=4, accumulation_steps=1, epochs=1, patience=1
    )
    cfg["audio"].update(model=str(audio_path), revision=None, hidden_size=32, head_hidden=16, batch_size=4)
    cfg["heads"].update(epochs=2, patience=1, batch_size=8)
    cfg["fusion"].update(concat_hidden=16)
    cfg["runtime"].update(device="cpu", dtype="float32", workers=0, benchmark_turns=3)
    return cfg


def make_media(root, counts=None, missing=False):
    root = Path(root)
    counts = counts or {"train": 16, "dev": 12, "test": 4}
    rng = np.random.default_rng(123)
    for split, count in counts.items():
        directory = root / (split + "_splits")
        directory.mkdir(parents=True, exist_ok=True)
        with (root / f"{split}_sent_emo.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "Dialogue_ID",
                    "Utterance_ID",
                    "Speaker",
                    "Utterance",
                    "Emotion",
                    "StartTime",
                    "EndTime",
                    "Season",
                    "Episode",
                ],
            )
            writer.writeheader()
            for i in range(count):
                dialogue, utterance = i // 3, i % 3
                label = LABELS[i % len(LABELS)]
                writer.writerow(
                    {
                        "Dialogue_ID": dialogue,
                        "Utterance_ID": utterance,
                        "Speaker": "A" if i % 2 else "B",
                        "Utterance": f"I feel {label} today.",
                        "Emotion": label,
                        "StartTime": "00:00:00,000",
                        "EndTime": "00:00:00,180",
                        "Season": 1,
                        "Episode": 1,
                    }
                )
                if missing and split in ("train", "dev") and i == 0:
                    continue
                length = 2400 + i % 4 * 320
                wave = (rng.normal(0, 0.03, length) + 0.08 * np.sin(np.arange(length) * 0.04)).astype(
                    np.float32
                )
                sf.write(directory / f"dia{dialogue}_utt{utterance}.wav", wave, 16000, subtype="FLOAT")
    return root


def run_smoke(root, base_config="configs/roadmap_a.yaml", pretrained=False):
    root = Path(root)
    if root.exists():
        raise ValueError("Use a fresh smoke output directory")
    root.mkdir(parents=True)
    torch.set_num_threads(2)
    cfg = load_config(base_config)
    cfg.update(
        output_root=str(root / "artifacts"),
        data_root=str(root / "processed"),
        feature_root=str(root / "features"),
        checkpoint_root=str(root / "checkpoints"),
    )
    if pretrained:
        cfg["text"].update(batch_size=2, accumulation_steps=1, epochs=1, patience=1)
        cfg["heads"].update(epochs=2, patience=1, batch_size=8)
        cfg["runtime"]["benchmark_turns"] = 3
    else:
        cfg = make_tiny_models(root, cfg)
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    raw = make_media(root / "raw", missing=True)
    prepare_dataset(raw, cfg["data_root"], cfg, fixture=True)
    run(cfg)
    run(cfg, final_test=True)
    wave = Path(cfg["data_root"]) / "audio" / "dev" / "dia0_utt1.wav"
    command = [
        sys.executable,
        "-m",
        "src.replay",
        "--config",
        str(config_path),
        "--audio",
        str(wave),
        "--text",
        "I feel joy today.",
    ]
    process = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    prediction = json.loads(process.stdout)
    write_json(root / "fresh_process_prediction.json", prediction)
    if not prediction["audio_available"] or not prediction["response"]:
        raise AssertionError("Raw-input replay did not reach both outputs")
    return cfg


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--output", default=".smoke/run")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    cfg = run_smoke(args.output, args.config, args.pretrained)
    print(json.dumps({"status": "synthetic integration passed", "artifacts": cfg["output_root"]}))


if __name__ == "__main__":
    main()

"""Cache only pooled, frozen WavLM embeddings for experiments."""

import time
from pathlib import Path

import torch

from .audio import FrozenAudio, read_waveform
from .utils import (
    LOG,
    link_checkpoint,
    audio_file,
    common_parser,
    counts,
    finish_run,
    key,
    load_config,
    save_tensor,
    setup,
    split_rows,
    start_run,
    sync,
    write_json,
    write_rows,
)


def extract_cache(cfg, splits, encoder=None):
    """Extract frozen pooled WavLM caches for the requested partitions."""
    encoder = FrozenAudio(cfg) if encoder is None else encoder
    if encoder.backbone.config.hidden_size != cfg["audio"]["hidden_size"]:
        raise ValueError("Audio hidden size differs from configuration")

    summaries = {}
    for split in splits:
        rows = split_rows(cfg, split)
        features = torch.zeros(len(rows), encoder.backbone.config.hidden_size)
        indices = sorted(
            (i for i, row in enumerate(rows) if row["audio_valid"]),
            key=lambda i: rows[i]["num_samples"],
        )

        sync(encoder.device)
        started = time.perf_counter()

        for offset in range(0, len(indices), cfg["audio"]["batch_size"]):
            batch = indices[offset : offset + cfg["audio"]["batch_size"]]
            features[batch] = encoder([read_waveform(audio_file(cfg, rows[i])) for i in batch])

        sync(encoder.device)
        elapsed = time.perf_counter() - started

        LOG.info(
            "WavLM %s: %d valid utterances in %.2f seconds",
            split,
            len(indices),
            elapsed,
        )

        summaries[split] = {
            "elapsed_s": elapsed,
            "valid_utterances": len(indices),
            "utterances_per_s": len(indices) / elapsed,
            "audio_seconds_per_s": (sum(rows[i]["duration_s"] for i in indices) / elapsed),
        }

        save_tensor(
            Path(cfg["feature_root"]) / "wavlm_base_plus" / f"{split}.pt",
            {
                "keys": [key(row) for row in rows],
                "features": features,
                "audio_valid": torch.tensor([row["audio_valid"] for row in rows]),
                "revision": cfg["audio"]["revision"],
                "summary": summaries[split],
            },
        )

    return summaries


def extract(cfg, splits=("train", "dev_model", "dev_calib")):
    """Development extraction run; official test is intentionally excluded."""
    path, started = start_run(cfg, "wavlm_extraction")
    encoder = FrozenAudio(cfg)

    if encoder.backbone.config.hidden_size != cfg["audio"]["hidden_size"]:
        raise ValueError("Audio hidden size differs from configuration")

    encoder.backbone.save_pretrained(Path(cfg["checkpoint_root"]) / "wavlm" / "encoder")
    write_json(
        Path(cfg["checkpoint_root"]) / "wavlm" / "metadata.json",
        {
            "model": cfg["audio"]["model"],
            "revision": cfg["audio"]["revision"],
            "pooling": "masked final-layer mean; exact-length batches",
        },
    )
    link_checkpoint(path, Path(cfg["checkpoint_root"]) / "wavlm")

    summaries = extract_cache(cfg, splits, encoder=encoder)

    write_json(path / "metrics.json", summaries)
    write_rows(path / "train_log.jsonl", [])
    write_rows(path / "predictions.jsonl", [])
    finish_run(cfg, path, started, {"parameters": counts(encoder)})
    return summaries


def main():
    args = common_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    setup(cfg["seed"])
    extract(cfg)


if __name__ == "__main__":
    main()

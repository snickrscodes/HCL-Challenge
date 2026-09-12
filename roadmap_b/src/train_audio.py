"""Train the small audio head; shared head fitting also serves concatenation."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .evaluate import metrics, probabilities, save_evaluation
from .models import ClassificationHead, load_head, save_head
from .utils import (
    LOG,
    link_checkpoint,
    common_parser,
    counts,
    finish_run,
    key,
    load_config,
    load_tensor,
    read_json,
    write_json,
    setup,
    split_rows,
    start_run,
    write_rows,
)


def aligned_cache(cache, rows):
    if cache["keys"] != [key(row) for row in rows]:
        raise ValueError("Feature cache keys/order differ from manifest")
    if "audio_valid" in cache and cache["audio_valid"].tolist() != [row["audio_valid"] for row in rows]:
        raise ValueError("Feature cache audio availability differs from manifest")
    if not torch.isfinite(cache["features"]).all():
        raise ValueError("Non-finite feature cache")
    return cache


def fit_head(cfg, name, train_x, dev_x, train_rows, dev_rows, hidden):
    path, started = start_run(cfg, name)
    setup(cfg["seed"])
    # Cached heads are deliberately trained on CPU: tiny matrices, no encoder memory.
    head = ClassificationHead(train_x.shape[1], hidden)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["heads"]["learning_rate"], weight_decay=0.01)
    train_y = torch.tensor([row["label"] for row in train_rows])
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=cfg["heads"]["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    best, stale, logs, best_epoch = -1.0, 0, [], None
    checkpoint = Path(cfg["checkpoint_root"]) / name
    link_checkpoint(path, checkpoint)
    for epoch in range(cfg["heads"]["epochs"]):
        head.train()
        total = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite head loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(y)
        head.eval()
        with torch.inference_mode():
            probs = probabilities(head(dev_x))
        values = metrics([row["label"] for row in dev_rows], probs)
        LOG.info(
            "%s epoch %d: macro F1 %.4f, weighted F1 %.4f",
            name,
            epoch + 1,
            values["macro_f1"],
            values["weighted_f1"],
        )
        logs.append({"epoch": epoch + 1, "train_loss": total / len(train_rows), "dev": values})
        write_rows(path / "train_log.jsonl", logs)
        if values["macro_f1"] > best:
            best, stale, best_epoch = values["macro_f1"], 0, epoch + 1
            save_head(checkpoint, head, train_x.shape[1], hidden)
            save_evaluation(path, dev_rows, probs)
        else:
            stale += 1
        if stale >= cfg["heads"]["patience"]:
            break
    finish_run(
        cfg,
        path,
        started,
        {
            "parameters": counts(head),
            "best_epoch": best_epoch,
            "best_checkpoint": str(checkpoint),
            "training_examples": len(train_rows),
            "evaluation_examples": len(dev_rows),
        },
    )
    return load_head(checkpoint)


def train(cfg):
    all_rows, caches = {}, {}
    for split in ("train", "dev_model"):
        rows = split_rows(cfg, split)
        cache = aligned_cache(
            load_tensor(Path(cfg["feature_root"]) / "wavlm_base_plus" / f"{split}.pt"), rows
        )
        valid = cache["audio_valid"]
        if not valid.any():
            raise ValueError("No valid audio examples")
        caches[split] = cache["features"][valid]
        all_rows[split] = [row for row, keep in zip(rows, valid) if keep]
    head = fit_head(
        cfg,
        "audio",
        caches["train"],
        caches["dev_model"],
        all_rows["train"],
        all_rows["dev_model"],
        cfg["audio"]["head_hidden"],
    )

    with torch.inference_mode():
        predicted = head.eval()(caches["dev_model"]).argmax(1)
    unique = sorted(predicted.unique().tolist())
    health = {
        "predicted_classes": unique,
        "nontrivial": len(unique) > 1,
        "note": "Inspect per-class scores and train/dev curves before fusion.",
    }
    write_json(Path(cfg["output_root"]) / "audio" / "health.json", health)
    fixture = read_json(Path(cfg["data_root"]) / "audit" / "summary.json")["fixture"]
    if not fixture and not health["nontrivial"]:
        raise RuntimeError(
            "Audio head collapsed to a single class. Inspect saved audio diagnostics before fusion."
        )
    return head


def main():
    args = common_parser(__doc__).parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()

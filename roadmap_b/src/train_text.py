"""Fine-tune current-only or causal-context RoBERTa; select on dev_model macro F1."""

import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .context import collate_text, encode_turn, histories
from .evaluate import metrics, probabilities, save_evaluation
from .models import TextClassifier
from .utils import (
    LOG,
    link_checkpoint,
    autocast,
    common_parser,
    counts,
    device_for,
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


def encoded_rows(cfg, rows, tokenizer, context):
    history = histories(rows, cfg["text"]["history_turns"] if context else 0)
    items = [
        encode_turn(
            tokenizer,
            row["text"],
            history[key(row)],
            row["speaker"],
            cfg["text"]["max_length"],
            cfg["text"]["history_turns"] if context else 0,
        )
        for row in rows
    ]
    return items


@torch.inference_mode()
def predict_text(cfg, model, tokenizer, rows, context):
    device = device_for(cfg)
    model.to(device).eval()
    items = encoded_rows(cfg, rows, tokenizer, context)
    logits, features = [], []
    for start in range(0, len(items), cfg["text"]["batch_size"]):
        batch = {
            k: v.to(device)
            for k, v in collate_text(
                items[start : start + cfg["text"]["batch_size"]], tokenizer.pad_token_id
            ).items()
        }
        with autocast(cfg, device):
            z, h = model(**batch)
        logits.append(z.float().cpu())
        features.append(h.float().cpu())
    return {
        "keys": [key(row) for row in rows],
        "logits": torch.cat(logits),
        "features": torch.cat(features),
        "labels": torch.tensor([r["label"] for r in rows]),
        "current_truncations": sum(item["current_truncated"] for item in items),
    }


def train(cfg, name):
    context = name == "text_context"
    path, started = start_run(cfg, name)
    setup(cfg["seed"])
    device = device_for(cfg)
    model, tokenizer = TextClassifier.pretrained(cfg)
    model.to(device)
    train_rows, dev_rows = split_rows(cfg, "train"), split_rows(cfg, "dev_model")
    items = encoded_rows(cfg, train_rows, tokenizer, context)
    labels = torch.tensor([r["label"] for r in train_rows])
    loader = DataLoader(
        list(range(len(items))),
        batch_size=cfg["text"]["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg["seed"]),
        num_workers=cfg["runtime"]["workers"],
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["text"]["learning_rate"], weight_decay=0.01)
    accumulation = cfg["text"]["accumulation_steps"]
    total_steps = math.ceil(len(loader) / accumulation) * cfg["text"]["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(
            (step + 1) / max(1, total_steps * 0.06),
            max(0.0, (total_steps - step) / max(1, total_steps * 0.94)),
        ),
    )
    best, stale, logs = -1.0, 0, []
    checkpoint = Path(cfg["checkpoint_root"]) / name
    link_checkpoint(path, checkpoint)
    best_epoch = None
    for epoch in range(cfg["text"]["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for step, indices in enumerate(loader):
            batch = {
                k: v.to(device)
                for k, v in collate_text([items[i] for i in indices.tolist()], tokenizer.pad_token_id).items()
            }
            with autocast(cfg, device):
                z, _ = model(**batch)
                loss = torch.nn.functional.cross_entropy(z.float(), labels[indices].to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite text loss")
            # Correct scaling for a short final accumulation group.
            group_size = min(accumulation, len(loader) - (step // accumulation) * accumulation)
            (loss / group_size).backward()
            loss_sum += float(loss.detach()) * len(indices)
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        prediction = predict_text(cfg, model, tokenizer, dev_rows, context)
        dev_metrics = metrics([r["label"] for r in dev_rows], probabilities(prediction["logits"]))
        LOG.info(
            "%s epoch %d: macro F1 %.4f, weighted F1 %.4f",
            name,
            epoch + 1,
            dev_metrics["macro_f1"],
            dev_metrics["weighted_f1"],
        )
        logs.append({"epoch": epoch + 1, "train_loss": loss_sum / len(items), "dev": dev_metrics})
        write_rows(path / "train_log.jsonl", logs)
        if dev_metrics["macro_f1"] > best:
            best, stale, best_epoch = dev_metrics["macro_f1"], 0, epoch + 1
            model.save(
                checkpoint,
                tokenizer,
                {
                    "context": context,
                    "history_turns": cfg["text"]["history_turns"] if context else 0,
                    "max_length": cfg["text"]["max_length"],
                    "revision": cfg["text"]["revision"],
                },
            )
            write_json(path / "metrics.json", dev_metrics)
            save_evaluation(path, dev_rows, probabilities(prediction["logits"]))
        else:
            stale += 1
        if stale >= cfg["text"]["patience"]:
            break
    sync(device)
    finish_run(
        cfg,
        path,
        started,
        {
            "parameters": counts(model),
            "best_epoch": best_epoch,
            "best_checkpoint": str(checkpoint),
            "learning_rate": cfg["text"]["learning_rate"],
            "batch_size": cfg["text"]["batch_size"],
            "effective_batch_size": cfg["text"]["batch_size"] * accumulation,
            "current_truncations_train": sum(x["current_truncated"] for x in items),
        },
    )
    return checkpoint


def extract_text(cfg, name, split):
    model, tokenizer, metadata = TextClassifier.load(Path(cfg["checkpoint_root"]) / name)
    if metadata["max_length"] != cfg["text"]["max_length"] or (
        metadata["context"] and metadata["history_turns"] != cfg["text"]["history_turns"]
    ):
        raise ValueError("Text context configuration differs from the trained checkpoint")
    start = time.perf_counter()
    result = predict_text(cfg, model, tokenizer, split_rows(cfg, split), metadata["context"])
    result["extraction_s"] = time.perf_counter() - start
    save_tensor(Path(cfg["feature_root"]) / name / f"{split}.pt", result)
    return result


def main():
    parser = common_parser(__doc__)
    parser.add_argument("--mode", choices=("current", "context"), required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, "text_" + args.mode)


if __name__ == "__main__":
    main()

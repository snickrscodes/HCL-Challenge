"""One controlled 4-variant x 3-seed fit, after the real numerical replay."""

import time
from pathlib import Path

import torch
import yaml

from .b_data import frozen_policy, load_inputs, output, parser, settings, verify_baseline
from .b_fusion import ResidualFusion, VARIANTS
from .constants import LABELS
from .evaluate import metrics, probabilities
from .utils import counts, digest, environment, key, load_tensor, save_tensor, setup, write_json, write_rows


def forward(model, inputs, indices=None, forced_gate=None):
    names = ("ht", "ha", "zt", "za", "available")
    args = [inputs[n] if indices is None else inputs[n][indices] for n in names]
    return model(*args, forced_gate=forced_gate)


@torch.inference_mode()
def predict(model, inputs, forced_gate=None):
    model.eval()
    return {k: v.detach().cpu() for k, v in forward(model, inputs, forced_gate=forced_gate).items()}


def load_model(path):
    ckpt = load_tensor(path)
    if tuple(ckpt["labels"]) != LABELS:
        raise ValueError("B checkpoint label order differs")
    model = ResidualFusion(**ckpt["spec"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.eval(), ckpt


def teacher_statistics(rows, inputs):
    p = probabilities(inputs["zt"])
    y = inputs["labels"].numpy()
    pred = p.argmax(-1)
    result = {
        "scope": "In-sample outputs of the frozen teacher; not out-of-fold stacking features",
        "metrics": metrics(y, p),
        "errors": int((pred != y).sum()),
        "per_class": {},
    }
    for i, name in enumerate(LABELS):
        mask = y == i
        correct = int(((pred == y) & mask).sum())
        result["per_class"][name] = {
            "support": int(mask.sum()),
            "correct": correct,
            "incorrect": int(mask.sum()) - correct,
            "accuracy": float((pred[mask] == y[mask]).mean()),
            "mean_max_probability": float(p[mask].max(-1).mean()),
            "mean_true_label_probability": float(p[mask, i].mean()),
        }
    return result


def fit_one(cfg, variant, seed, train, dev, train_rows, dev_rows, root):
    setup(seed)
    model = ResidualFusion(variant, **cfg["fusion"])
    spec = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"]
    )
    eligible = train["available"].nonzero().flatten()
    if not len(eligible):
        raise ValueError("No eligible training examples")
    generator = torch.Generator().manual_seed(seed)
    root.mkdir(parents=True, exist_ok=False)
    (root / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    started = time.perf_counter()
    best, stale, logs, best_epoch = -1.0, 0, [], None
    for epoch in range(1, spec["epochs"] + 1):
        model.train()
        loss_sum = 0.0
        order = eligible[torch.randperm(len(eligible), generator=generator)]
        for index in order.split(spec["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            result = forward(model, train, index)
            loss = torch.nn.functional.cross_entropy(result["logits"], train["labels"][index])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite B training loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
        pred = predict(model, dev)
        values = metrics(dev["labels"], probabilities(pred["logits"]))
        logs.append({"epoch": epoch, "train_loss": loss_sum / len(eligible), "dev_model": values})
        write_rows(root / "train_log.jsonl", logs)
        print(f"{variant} seed {seed} epoch {epoch}: macro={values['macro_f1']:.4f}", flush=True)
        if values["macro_f1"] > best:
            best, best_epoch, stale = values["macro_f1"], epoch, 0
            save_tensor(
                root / "checkpoint.pt",
                {
                    "state_dict": model.state_dict(),
                    "spec": model.spec,
                    "labels": list(LABELS),
                    "seed": seed,
                    "best_epoch": epoch,
                },
            )
        else:
            stale += 1
        if stale >= spec["patience"]:
            break
    loaded, _ = load_model(root / "checkpoint.pt")
    pred = predict(loaded, dev)
    values = metrics(dev["labels"], probabilities(pred["logits"]))
    train_gate = predict(loaded, train)["gate"][train["available"]]
    write_json(root / "metrics.json", values)
    write_rows(
        root / "predictions_raw.jsonl",
        [
            {"key": key(row), "label": row["label"], "logits": z.tolist(), "prediction": int(z.argmax())}
            for row, z in zip(dev_rows, pred["logits"])
        ],
    )
    metadata = {
        "variant": variant,
        "seed": seed,
        "best_epoch": best_epoch,
        "elapsed_s": time.perf_counter() - started,
        "parameters": counts(loaded),
        "eligible_training_rows": len(eligible),
        "training_keys_sha256": __import__("hashlib")
        .sha256("\n".join(key(train_rows[i]) for i in eligible.tolist()).encode())
        .hexdigest(),
        "training_mean_gate": float(train_gate.mean()),
        "environment": environment({**cfg, "seed": seed}),
        "training_device": "cpu; cached frozen representations",
    }
    write_json(root / "training.json", metadata)
    return metadata


def matrix(cfg):
    integrity = verify_baseline(cfg)
    policy = frozen_policy(cfg)
    root = output(cfg)
    if (root / "training_started.json").exists():
        raise RuntimeError("The predeclared matrix was already started; no automatic sweep or overwrite")
    train_rows, train = load_inputs(cfg, "train")
    dev_rows, dev = load_inputs(cfg, "dev_model")
    write_json(root / "teacher_train_statistics.json", teacher_statistics(train_rows, train))
    original_teacher = load_tensor(Path(cfg["baseline_root"]) / "features/text_context/train.pt")
    assert original_teacher["keys"] == [key(r) for r in train_rows]
    write_json(
        root / "A_original_teacher_train_statistics.json",
        teacher_statistics(train_rows, {**train, "zt": original_teacher["logits"]}),
    )
    write_json(
        root / "training_started.json",
        {
            "config": cfg,
            "integrity": integrity,
            "policy": policy,
            "variants": list(VARIANTS),
            "seeds": cfg["seeds"],
            "deployment_seed": 1337,
        },
    )
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    results = []
    for variant in VARIANTS:
        for seed in cfg["seeds"]:
            results.append(
                fit_one(
                    cfg, variant, seed, train, dev, train_rows, dev_rows, root / "runs" / variant / str(seed)
                )
            )
    files = {str(p.relative_to(root)): digest(p) for p in sorted((root / "runs").rglob("checkpoint.pt"))}
    if len(files) != 12:
        raise AssertionError("Expected exactly twelve frozen head checkpoints")
    write_json(
        root / "model_freeze.json",
        {
            "checkpoints_sha256": files,
            "reference_seed": 1337,
            "selection": "per-run dev_model macro F1; no seed selection",
            "policy_sha256": digest(root / "canonical_policy.json"),
            "text_policy_sha256": digest(root / "text_policy.json"),
            "cache_sha256": {
                str(p.relative_to(root)): digest(p)
                for folder in ("features", "text_features")
                for p in sorted((root / folder).glob("*.pt"))
            },
            "resolved_config_sha256": digest(root / "resolved_config.yaml"),
            "baseline_lock_sha256": digest(cfg["baseline_lock"]),
            "implementation_sha256": {p.name: digest(p) for p in sorted(Path("src").glob("b_*.py"))},
        },
    )
    write_json(root / "training_summary.json", results)
    verify_baseline(cfg)


def main():
    args = parser(__doc__).parse_args()
    matrix(settings(args))


if __name__ == "__main__":
    main()

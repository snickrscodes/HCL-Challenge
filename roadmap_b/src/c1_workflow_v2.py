"""Fixed C1 v2 extraction/training/calibration/freeze and one external-test pass."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import log_softmax, xlogy
import torch

from .audio import FrozenAudio, read_waveform
from .b_data import encoder_config, parser as bparser, settings, verify_baseline
from .b_evaluate import verify_freeze as verify_b
from .c1_data import sha, write_json
from .c1_media_v2 import verify_data, verify_protocol
from .c1_metrics import metrics, fit_temperature
from .c1_model import (
    AudioSummaryTap,
    EvidenceHead,
    epoch_order,
    fit_normalization,
    sample_triplets,
    soft_ce,
    triplet_loss,
)
from .c1_protocol_v2 import immutable_json
from .evaluate import probabilities


def read_rows(root, part):
    if part not in ("train", "dev", "calib", "external_test"):
        raise ValueError("Official MELD test/foreign partition forbidden")
    return [
        json.loads(x)
        for x in (Path(root) / "manifest.jsonl").read_text().splitlines()
        if json.loads(x)["partition"] == part and json.loads(x)["supervision_eligible"]
    ]


def tensor_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"Refusing to replace tensor {path}")
    temp = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temp)
    torch.load(temp, map_location="cpu", weights_only=True)
    temp.replace(path)


def source_files():
    here = Path(__file__).parent
    names = [
        "c1_data.py",
        "c1_model.py",
        "c1_metrics.py",
        "c1_protocol_v2.py",
        "c1_media_v2.py",
        "c1_workflow_v2.py",
        "c1_runtime_v2.py",
    ]
    return {n: sha(here / n) for n in names}


def bconfig(baseline_root):
    args = bparser("frozen backbone").parse_args(["--baseline-root", str(baseline_root)])
    return settings(args)


def extract(root, part, baseline_root, *, external_authorized=False):
    root = Path(root)
    verify_data(root)
    if part not in ("train", "dev", "calib"):
        if part != "external_test" or not external_authorized:
            raise ValueError("Only guarded custom external test allowed; official MELD test forbidden")
        verify_model_freeze(root)
    rows = read_rows(root, part)
    cfg = bconfig(baseline_root)
    verify_baseline(cfg)
    verify_b(cfg)
    weights = Path(baseline_root) / "checkpoints/wavlm/encoder/model.safetensors"
    if not weights.exists():
        candidates = list((weights.parent).glob("*.safetensors")) + list(
            weights.parent.glob("pytorch_model*.bin")
        )
        assert len(candidates) == 1
        weights = candidates[0]
    identity = {
        "dataset": "CREMA-D",
        "partition": part,
        "ids": [r["id"] for r in rows],
        "waveform_sha256": [r["processed_sha256"] for r in rows],
        "checkpoint_sha256": sha(weights),
        "data_freeze_sha256": sha(root / "DATA_FREEZE.json"),
        "precision": "singleton BF16; per-layer reductions FP32; population std",
        "dimensions": [13, 1536],
        "head_summary_source_sha256": sha(Path(__file__).parent / "c1_model.py"),
    }
    path = root / "features" / f"{part}.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert saved["complete"] and saved["identity"] == identity
        assert (
            saved["summaries"].shape == (len(rows), 13, 1536)
            and saved["summaries"].dtype == torch.float32
            and torch.isfinite(saved["summaries"]).all()
        )
        return saved
    encoder = FrozenAudio(encoder_config(cfg), local=True)
    tap = AudioSummaryTap(encoder)
    assert not any(p.requires_grad for p in tap.parameters())
    values = []
    started = time.perf_counter()
    for i, row in enumerate(rows):
        p = root / row["audio_path"]
        assert sha(p) == row["processed_sha256"]
        tap([read_waveform(p)])
        values.append(tap.last_summaries[0])
        if (i + 1) % 500 == 0:
            print(f"extract {part} {i + 1}/{len(rows)}", flush=True)
    result = {
        "complete": True,
        "identity": identity,
        "summaries": torch.stack(values),
        "elapsed_seconds": time.perf_counter() - started,
        "backbone_calls": tap.calls,
    }
    tensor_save(path, result)
    write_json(
        root / "features" / f"{part}.json",
        {
            "identity": identity,
            "elapsed_seconds": result["elapsed_seconds"],
            "backbone_calls": tap.calls,
            "cache_sha256": sha(path),
        },
    )
    return result


def load_features(root, part):
    rows = read_rows(root, part)
    value = torch.load(Path(root) / "features" / f"{part}.pt", map_location="cpu", weights_only=True)
    assert (
        value["complete"]
        and value["identity"]["dataset"] == "CREMA-D"
        and value["identity"]["partition"] == part
    )
    assert value["identity"]["ids"] == [r["id"] for r in rows]
    assert value["identity"]["data_freeze_sha256"] == sha(Path(root) / "DATA_FREEZE.json")
    x = value["summaries"]
    assert x.shape == (len(rows), 13, 1536) and x.dtype == torch.float32 and torch.isfinite(x).all()
    return rows, x, torch.tensor([r["target"] for r in rows], dtype=torch.float32)


@torch.inference_mode()
def predict(head, x):
    head.eval()
    outputs = []
    latent = []
    device = next(head.parameters()).device
    for batch in x.split(256):
        result = head(batch.to(device))
        outputs.append(result["logits"].cpu())
        latent.append(result["representation"].cpu())
    return torch.cat(outputs).numpy(), torch.cat(latent).numpy()


def load_head(root, condition, seed, device="cpu"):
    checkpoint = torch.load(
        Path(root) / "runs" / condition / str(seed) / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    assert (
        checkpoint["condition"] == condition
        and checkpoint["seed"] == seed
        and checkpoint["namespace"] == "crema6_audio_votes_v1"
    )
    state = checkpoint["state_dict"]
    model = EvidenceHead(condition, state["mean"], state["scale"])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), checkpoint


def evaluate_rows(root, condition, seed, partition, head, x, rows, temperature, destination):
    z, d = predict(head, x)
    y = np.array([r["target"] for r in rows])
    before = metrics(z, y)
    after = metrics(z, y, temperature)
    groups = {}
    for field in ("speaker", "sentence"):
        groups[field] = {}
        for val in sorted({r[field] for r in rows}):
            mask = np.array([r[field] == val for r in rows])
            groups[field][val] = {
                "before": metrics(z[mask], y[mask]),
                "after": metrics(z[mask], y[mask], temperature),
            }
    train_sentences = set(json.loads((Path(root) / "C1_SPLIT_IDS.json").read_text())["sentences"]["train"])
    slices = {}
    for name, seen in [("seen_sentences", True), ("unseen_sentences", False)]:
        mask = np.array([(r["sentence"] in train_sentences) == seen for r in rows])
        if mask.any():
            slices[name] = {
                "before": metrics(z[mask], y[mask]),
                "after": metrics(z[mask], y[mask], temperature),
            }
    result = {
        "condition": condition,
        "seed": seed,
        "partition": partition,
        "before": before,
        "after": after,
        "groups": groups,
        "seen_unseen": slices,
    }
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "metrics.json", result)
    p = probabilities(z, temperature)
    records = [
        {
            "id": r["id"],
            "speaker": r["speaker"],
            "sentence": r["sentence"],
            "namespace": "crema6_audio_votes_v1",
            "target": r["target"],
            "plurality": r["plurality"],
            "raw_logits": zi.tolist(),
            "probabilities": pi.tolist(),
            "prediction": int(pi.argmax()),
            "waveform_sha256": r["processed_sha256"],
        }
        for r, zi, pi in zip(rows, z, p)
    ]
    (destination / "predictions.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in records)
    )
    # Private retained representation cache, no human recordings.
    tensor_save(
        destination / "representations.pt",
        {"ids": [r["id"] for r in rows], "normalized_128d": torch.from_numpy(d)},
    )
    return result


def train(root, baseline_root):
    root = Path(root)
    cfg = verify_protocol(root)
    audit = verify_data(root)
    if (root / "C1_MODEL_FREEZE.json").exists():
        raise ValueError("Training after freeze forbidden")
    bc = bconfig(baseline_root)
    a_before = verify_baseline(bc)
    b_before = verify_b(bc)
    train_rows, x, y = load_features(root, "train")
    dev_rows, dx, dy = load_features(root, "dev")
    calib_rows, cx, cy = load_features(root, "calib")
    pools = json.loads((root / "triplet_pools.json").read_text())
    index = {r["id"]: i for i, r in enumerate(train_rows)}
    assert set(pools) <= set(index)
    norms = {}
    for condition in ("D0", "D1", "D2"):
        path = root / "normalization" / f"{condition}.pt"
        mean, scale = fit_normalization(x, "train", condition)
        value = {
            "mean": mean,
            "scale": scale,
            "train_ids_sha256": hashlib.sha256("\n".join(index).encode()).hexdigest(),
            "feature_cache_sha256": sha(root / "features/train.pt"),
            "variance": "population",
            "scale_rule": "one scalar per layer; floor 1e-5",
        }
        if path.exists():
            old = torch.load(path, weights_only=True)
            assert torch.equal(old["mean"], mean) and torch.equal(old["scale"], scale)
        else:
            tensor_save(path, value)
        norms[condition] = (mean, scale)
    norms["D3"] = norms["D2"]
    device = torch.device("cuda")
    assert torch.cuda.is_available() and "4090" in torch.cuda.get_device_name()
    torch.set_num_threads(4)
    allx = x.to(device)
    ally = y.to(device)
    summaries = []
    for condition in cfg["conditions"]:
        if condition == "D3" and not audit["triplets"]["executable"]:
            continue
        for seed in cfg["seeds"]:
            dest = root / "runs" / condition / str(seed)
            if (dest / "run.json").exists():
                completed = json.loads((dest / "run.json").read_text())
                assert completed["checkpoint_sha256"] == sha(dest / "checkpoint.pt")
                summaries.append(completed)
                continue
            if dest.exists():
                raise ValueError(f"Incomplete run needs explicit operational review: {dest}")
            dest.mkdir(parents=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = EvidenceHead(condition, *norms[condition]).to(device)
            initial = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
            initial_sha = hashlib.sha256(b"".join(t.numpy().tobytes() for t in initial.values())).hexdigest()
            opt = torch.optim.AdamW(
                model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
            )
            assert {id(p) for group in opt.param_groups for p in group["params"]} == {
                id(p) for p in model.parameters()
            }
            best = float("inf")
            bad = 0
            best_state = None
            logs = []
            started = time.perf_counter()
            ce_orders = []
            for epoch in range(1, cfg["training"]["max_epochs"] + 1):
                model.train()
                order = epoch_order(len(x), seed, epoch)
                ce_orders.append(hashlib.sha256(order.tobytes()).hexdigest())
                ce_sum = 0.0
                pair_sum = 0.0
                triplets = 0
                zero = 0
                sampled_classes = Counter()
                for batch_index, start in enumerate(range(0, len(order), cfg["training"]["batch_size"])):
                    ids = order[start : start + cfg["training"]["batch_size"]]
                    idx = torch.as_tensor(ids, device=device)
                    opt.zero_grad(set_to_none=True)
                    out = model(allx[idx])
                    ce = soft_ce(out["logits"], ally[idx])
                    pair = ce.new_zeros(())
                    sampled = (
                        sample_triplets([train_rows[i]["id"] for i in ids], pools, seed, epoch, batch_index)
                        if condition == "D3"
                        else []
                    )
                    if sampled:
                        positions = {train_rows[i]["id"]: j for j, i in enumerate(ids)}
                        anchors = torch.tensor([positions[a] for a, p, n in sampled], device=device)
                        extra = torch.tensor(
                            [index[p] for a, p, n in sampled] + [index[n] for a, p, n in sampled],
                            device=device,
                        )
                        # Retained vectors precede dropout. Preserve CE dropout RNG exactly across D2/D3.
                        with torch.random.fork_rng(devices=[device.index or 0]):
                            dd = model(allx[extra])["representation"]
                        pair = triplet_loss(
                            out["representation"][anchors], dd[: len(sampled)], dd[len(sampled) :]
                        )
                        sampled_classes.update(train_rows[index[a]]["plurality"] for a, p, n in sampled)
                    else:
                        zero += 1
                    loss = ce + cfg["triplet"]["weight"] * pair
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
                    opt.step()
                    ce_sum += float(ce.detach()) * len(ids)
                    pair_sum += float(pair.detach()) * len(sampled)
                    triplets += len(sampled)
                z, _ = predict(model, dx)
                nll = metrics(z, dy.numpy())["soft_nll"]
                log = {
                    "epoch": epoch,
                    "ce": ce_sum / len(x),
                    "triplet_loss": pair_sum / triplets if triplets else 0.0,
                    "triplet_count": triplets,
                    "zero_triplet_batches": zero,
                    "eligible_anchors": len(pools) if condition == "D3" else 0,
                    "sampled_classes": {cfg["labels"][int(k)]: v for k, v in sampled_classes.items()},
                    "dev_soft_nll": nll,
                    "ce_order_sha256": ce_orders[-1],
                }
                logs.append(log)
                (dest / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in logs))
                if nll < best:
                    best = nll
                    best_epoch = epoch
                    best_state = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                if bad >= cfg["training"]["patience"]:
                    break
            duration = time.perf_counter() - started
            model.load_state_dict(best_state)
            checkpoint = {
                "condition": condition,
                "seed": seed,
                "namespace": cfg["namespace"],
                "labels": cfg["labels"],
                "best_epoch": best_epoch,
                "best_dev_soft_nll": best,
                "state_dict": best_state,
                "contract_sha256": sha(root / "C1_EXPERIMENT_CONTRACT.json"),
                "train_cache_sha256": sha(root / "features/train.pt"),
                "initial_state_sha256": initial_sha,
            }
            tensor_save(dest / "checkpoint.pt", checkpoint)
            cz, _ = predict(model, cx)
            temperature = fit_temperature(cz, cy.numpy(), "calib")
            write_json(dest / "calibration.json", temperature)
            dev = evaluate_rows(
                root, condition, seed, "dev", model, dx, dev_rows, temperature["temperature"], dest / "dev"
            )
            evaluate_rows(
                root,
                condition,
                seed,
                "calib",
                model,
                cx,
                calib_rows,
                temperature["temperature"],
                dest / "calib",
            )
            run = {
                "condition": condition,
                "seed": seed,
                "best_epoch": best_epoch,
                "epochs": len(logs),
                "training_seconds": duration,
                "initial_state_sha256": initial_sha,
                "ce_order_sha256": ce_orders,
                "checkpoint_sha256": sha(dest / "checkpoint.pt"),
                "normalization": f"normalization/{'D2' if condition == 'D3' else condition}.pt",
                "parameters": sum(p.numel() for p in model.parameters()),
                "normalization_buffers": sum(t.numel() for t in model.buffers()),
                "calibration": temperature,
                "dev_before": dev["before"],
                "dev_after": dev["after"],
            }
            write_json(dest / "run.json", run)
            summaries.append(run)
            print(
                f"{condition}/{seed} epoch={best_epoch} dev_nll={best:.6f} T={temperature['temperature']:.4f}",
                flush=True,
            )
    for seed in cfg["seeds"]:
        dd = {r["condition"]: r for r in summaries if r["seed"] == seed}
        if "D3" in dd:
            assert dd["D2"]["initial_state_sha256"] == dd["D3"]["initial_state_sha256"]
            n = min(len(dd["D2"]["ce_order_sha256"]), len(dd["D3"]["ce_order_sha256"]))
            assert dd["D2"]["ce_order_sha256"][:n] == dd["D3"]["ce_order_sha256"][:n]
    assert verify_baseline(bc) == a_before and verify_b(bc) == b_before
    write_json(
        root / "training_summary.json",
        {
            "status": "COMPLETE" if len(summaries) == 12 else "PARTIAL_MATRIX_TRIPLET_SUPPORT_INSUFFICIENT",
            "runs": summaries,
            "reference": {"condition": "D3", "seed": 1337},
            "B_unchanged": True,
            "source_sha256": source_files(),
        },
    )
    return summaries


def freeze(root):
    root = Path(root)
    cfg = verify_protocol(root)
    verify_data(root)
    summary = json.loads((root / "training_summary.json").read_text())
    conditions = (
        cfg["conditions"]
        if json.loads((root / "DATA_AUDIT.json").read_text())["triplets"]["executable"]
        else cfg["conditions"][:-1]
    )
    assert {(r["condition"], r["seed"]) for r in summary["runs"]} == {
        (c, s) for c in conditions for s in cfg["seeds"]
    }
    assert source_files() == summary["source_sha256"], (
        "Training implementation changed; requires operational review before freeze"
    )
    names = [
        "C1_EXPERIMENT_CONTRACT.json",
        "C1_PROTOCOL_FREEZE.json",
        "C1_AMENDMENT_LOCK.json",
        "C1_PROTOCOL_ADDENDUM.md",
        "DATA_FREEZE.json",
        "DATA_AUDIT.json",
        "manifest.jsonl",
        "training_summary.json",
        "triplet_pools.json",
    ]
    names += [
        str(p.relative_to(root))
        for folder in ("runs", "normalization", "features")
        for p in (root / folder).rglob("*")
        if p.is_file()
    ]
    result = {
        "version": "c1-model-v2",
        "conditions": conditions,
        "seeds": cfg["seeds"],
        "reference": {"condition": "D3", "seed": 1337},
        "selection_complete": True,
        "files_sha256": {n: sha(root / n) for n in names},
        "source_sha256": source_files(),
        "official_MELD_test": "forbidden",
        "external_test": "one planned pass only",
        "human_transfer": "pending",
    }
    immutable_json(root / "C1_MODEL_FREEZE.json", result)
    return result


def verify_model_freeze(root, *, runtime=False):
    root = Path(root)
    if not runtime:
        verify_data(root)
    value = json.loads((root / "C1_MODEL_FREEZE.json").read_text())
    assert value["version"] == "c1-model-v2" and value["selection_complete"]
    assert value["reference"] == {"condition": "D3", "seed": 1337} and value["seeds"] == [1337, 1338, 1339]
    assert source_files() == value["source_sha256"]
    runtime_required = {
        "C1_EXPERIMENT_CONTRACT.json",
        "C1_PROTOCOL_FREEZE.json",
        "C1_AMENDMENT_LOCK.json",
        "C1_PROTOCOL_ADDENDUM.md",
        "runs/D3/1337/checkpoint.pt",
        "runs/D3/1337/calibration.json",
        "normalization/D2.pt",
    }
    if runtime and not runtime_required <= set(value["files_sha256"]):
        raise ValueError("Incomplete deployment provenance")
    for name, digest in value["files_sha256"].items():
        if runtime and name not in runtime_required:
            continue
        p = (root / name).resolve()
        assert p.is_relative_to(root.resolve()) and sha(p) == digest
    return value


def per_row_losses(z, y, temperature):
    p = probabilities(z, temperature)
    logp = log_softmax(np.asarray(z) / temperature, axis=1)
    m = (p + y) / 2
    return {
        "soft_nll": -(y * logp).sum(1),
        "brier": ((p - y) ** 2).sum(1),
        "jsd": 0.5 * ((xlogy(p, p) - xlogy(p, m)).sum(1) + (xlogy(y, y) - xlogy(y, m)).sum(1)),
    }


def external_test(root, baseline_root):
    root = Path(root)
    frozen = verify_model_freeze(root)
    marker = root / "C1_EXTERNAL_TEST_STARTED.json"
    if marker.exists():
        raise ValueError("External test already started; no automatic repeat or reselection")
    immutable_json(
        marker,
        {
            "freeze_sha256": sha(root / "C1_MODEL_FREEZE.json"),
            "start_unix": time.time(),
            "conditions": frozen["conditions"],
            "seeds": frozen["seeds"],
        },
    )
    extract(root, "external_test", baseline_root, external_authorized=True)
    rows, x, y = load_features(root, "external_test")
    results = []
    losses = {}
    for condition in frozen["conditions"]:
        for seed in frozen["seeds"]:
            model, _ = load_head(root, condition, seed, "cuda")
            temp = json.loads((root / "runs" / condition / str(seed) / "calibration.json").read_text())[
                "temperature"
            ]
            result = evaluate_rows(
                root,
                condition,
                seed,
                "external_test",
                model,
                x,
                rows,
                temp,
                root / "external_test" / condition / str(seed),
            )
            results.append(result)
            z, _ = predict(model, x)
            losses[condition, seed] = per_row_losses(z, y.numpy(), temp)
    cfg = verify_protocol(root)
    groups = [
        np.array([i for i, r in enumerate(rows) if r["speaker"] == s])
        for s in sorted({r["speaker"] for r in rows})
    ]
    rng = np.random.default_rng(cfg["metrics"]["bootstrap_seed"])
    draws = rng.integers(0, len(groups), (cfg["metrics"]["bootstrap_replicates"], len(groups)))
    freq = np.stack([(draws == i).sum(1) for i in range(len(groups))], 1)
    n = np.array([len(g) for g in groups])
    denom = freq @ n
    comparisons = {}
    for a, b in [("D1", "D0"), ("D2", "D1"), ("D3", "D2"), ("D3", "D0")]:
        if a not in frozen["conditions"]:
            continue
        mm = {}
        for metric in ("soft_nll", "brier", "jsd"):
            delta = np.stack([losses[b, s][metric] - losses[a, s][metric] for s in frozen["seeds"]])
            avg = delta.mean(0)
            sums = np.array([avg[g].sum() for g in groups])
            dist = freq @ sums / denom
            mm[metric] = {
                "definition": "control loss minus candidate loss; positive favors candidate",
                "seed_paired_improvements": delta.mean(1).tolist(),
                "mean_improvement": float(avg.mean()),
                "head_seed_std": float(delta.mean(1).std(ddof=1)),
                "speaker_cluster_percentile_95": np.quantile(dist, [0.025, 0.975]).tolist(),
                "replicates": len(draws),
            }
        comparisons[f"{a}-{b}"] = mm
    gate = {"status": "NOT_EVALUATED_D3_UNAVAILABLE"}
    if "D3" in frozen["conditions"]:
        d0 = np.array([losses["D0", s]["soft_nll"].mean() for s in frozen["seeds"]])
        d3 = np.array([losses["D3", s]["soft_nll"].mean() for s in frozen["seeds"]])
        gain = float((d0.mean() - d3.mean()) / d0.mean())
        positive = int((d0 > d3).sum())
        jsd_ok = np.mean([losses["D3", s]["jsd"].mean() for s in frozen["seeds"]]) <= np.mean(
            [losses["D0", s]["jsd"].mean() for s in frozen["seeds"]]
        )
        gate = {
            "status": "EXTERNAL_CORPUS_CRITERION_PASSED"
            if gain >= 0.02 and positive >= 2 and jsd_ok
            else "EXTERNAL_CORPUS_CRITERION_FAILED",
            "postcalibrated_nll_relative_gain": gain,
            "positive_seed_pairs": positive,
            "mean_jsd_no_worse": bool(jsd_ok),
            "human_transfer": "PENDING",
            "no_control_promotion": True,
        }
    report = {
        "support": len(rows),
        "speakers": len(groups),
        "sentences": sorted({r["sentence"] for r in rows}),
        "results": results,
        "comparisons": comparisons,
        "gate": gate,
        "uncertainty": "Paired speaker-cluster bootstrap conditional on the fixed small speaker/sentence partition; seed variation reported separately",
        "freeze_sha256": sha(root / "C1_MODEL_FREEZE.json"),
    }
    write_json(root / "external_test_summary.json", report)
    immutable_json(
        root / "C1_EXTERNAL_TEST_COMPLETE.json",
        {
            "started_sha256": sha(marker),
            "summary_sha256": sha(root / "external_test_summary.json"),
            "no_reselection": True,
        },
    )
    verify_model_freeze(root)
    return gate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["extract", "train", "freeze", "external-test"])
    p.add_argument("--root", type=Path, default=Path("artifacts/roadmap_c1_v2"))
    p.add_argument("--baseline-root", type=Path, required=True)
    args = p.parse_args()
    if args.stage == "extract":
        for part in ("train", "dev", "calib"):
            extract(args.root, part, args.baseline_root)
    elif args.stage == "train":
        train(args.root, args.baseline_root)
    elif args.stage == "freeze":
        freeze(args.root)
    else:
        print(json.dumps(external_test(args.root, args.baseline_root), indent=2))


if __name__ == "__main__":
    main()

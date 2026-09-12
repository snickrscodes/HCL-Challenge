"""Integrity, real 4090 numerical replay, and versioned canonical audio caches."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .audio import FrozenAudio, read_waveform
from .constants import LABELS
from .evaluate import probabilities
from .models import load_head
from .utils import (
    digest,
    environment,
    key,
    load_config,
    load_tensor,
    read_json,
    read_rows,
    save_tensor,
    write_json,
)

SPLITS = ("train", "dev_model", "dev_calib")
REPLAY_KEYS = ("dev/99/3", "dev/108/5", "dev/62/4", "dev/72/0")


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/roadmap_b.yaml")
    p.add_argument("--baseline-root")
    p.add_argument("--audio-root", help="Processed root containing audio/, not the audio/ directory")
    p.add_argument("--output-root")
    return p


def settings(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    for name in ("baseline_root", "audio_root", "output_root"):
        if getattr(args, name, None):
            cfg[name] = getattr(args, name)
    for name in ("baseline_root", "baseline_lock", "output_root"):
        cfg[name] = str(Path(cfg[name]).resolve())
    if cfg["audio_root"]:
        cfg["audio_root"] = str(Path(cfg["audio_root"]).resolve())
    base, output = Path(cfg["baseline_root"]), Path(cfg["output_root"])
    if output == base or output.is_relative_to(base):
        raise ValueError("B output must be outside immutable Roadmap A")
    if tuple(cfg["seeds"]) != (1337, 1338, 1339) or cfg["reference_seed"] != 1337:
        raise ValueError("The predeclared head seeds are fixed")
    if cfg["fusion"] != {"width": 32, "bound": 2.0, "dropout": 0.1}:
        raise ValueError("Do not change the predeclared architecture")
    if cfg["policy"]["max_audio_seconds"] != 60:
        raise ValueError("The separate operational duration policy is fixed at 60 seconds")
    required_training = {
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "batch_size": 128,
        "epochs": 20,
        "patience": 3,
    }
    if cfg["training"] != required_training:
        raise ValueError("The predeclared training recipe cannot be changed in this pass")
    if cfg["evaluation"]["bootstrap_replicates"] < 1000:
        raise ValueError("Use at least 1000 dialogue-bootstrap replicates")
    if len(cfg["evaluation"]["shuffle_seeds"]) != 10:
        raise ValueError("Use exactly ten predeclared shuffle seeds")
    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    return cfg


def original_config(cfg):
    base = Path(cfg["baseline_root"])
    original = load_config(base / "configs/roadmap_a.yaml")
    for name in ("data_root", "feature_root", "checkpoint_root", "output_root"):
        original[name] = str(base / original[name])
    return original


def encoder_config(cfg):
    result = original_config(cfg)
    result["runtime"] = {**result["runtime"], **cfg["runtime"]}
    return result


def output(cfg):
    return Path(cfg["output_root"])


def verify_baseline(cfg):
    base = Path(cfg["baseline_root"])
    lock = read_json(cfg["baseline_lock"])
    for relative, expected in lock["baseline_files_sha256"].items():
        file = base / relative
        if not file.is_file() or digest(file) != expected:
            raise ValueError(f"Immutable baseline mismatch: {relative}")
    return {
        "lock_sha256": digest(cfg["baseline_lock"]),
        "archive_sha256": lock["archive_sha256"],
        "verified_files": len(lock["baseline_files_sha256"]),
    }


def rows_for(cfg, split):
    if split not in SPLITS:
        raise ValueError("Official test inference is forbidden in Roadmap B")
    return read_rows(Path(cfg["baseline_root"]) / "data/processed/splits" / f"{split}.jsonl")


def eligibility(row, maximum=60.0):
    if not row["audio_valid"]:
        return False, "missing_or_invalid"
    if row["duration_s"] > maximum:
        return False, "duration_limit"
    return True, None


def waveform_path(cfg, row):
    root = Path(cfg["audio_root"]) if cfg["audio_root"] else Path(cfg["baseline_root"]) / "data/processed"
    return root / row["audio_path"]


def probe_rows(cfg, count=None):
    values = [r for r in rows_for(cfg, "dev_model") if eligibility(r)[0]]
    values.sort(key=lambda r: (r["num_samples"], key(r)))
    n = min(count or cfg["evaluation"]["probe_count"], len(values))
    return [values[i] for i in np.unique(np.linspace(0, len(values) - 1, n).astype(int))]


def frozen_policy(cfg):
    p = read_json(output(cfg) / "canonical_policy.json")
    replay_file = output(cfg) / "numerical_replay/report.json"
    if not p["real_4090_replay_complete"] or p["replay_sha256"] != digest(replay_file):
        raise ValueError("Real replay evidence is missing or changed")
    if p["baseline_lock_sha256"] != digest(cfg["baseline_lock"]):
        raise ValueError("Baseline lock changed after replay")
    return p


def compare_outputs(a, b, policy):
    x, y = a["embedding"].float(), b["embedding"].float()
    d = (x - y).abs()
    result = {
        "embedding_max_abs": float(d.max()),
        "embedding_mean_abs": float(d.mean()),
        "embedding_cosine": float(torch.nn.functional.cosine_similarity(x[None], y[None]).item()),
        "embedding_allclose": bool(
            torch.allclose(x, y, atol=policy["embedding_atol"], rtol=policy["embedding_rtol"])
        ),
    }
    for name in ("audio_logits", "audio_probabilities", "fusion_probabilities"):
        va, vb = np.asarray(a[name]), np.asarray(b[name])
        result[name + "_max_abs"] = float(np.max(np.abs(va - vb)))
        result[name + "_argmax_changed"] = bool(va.argmax() != vb.argmax())
    return result


def replay(cfg):
    """No training. Requires real media, actual 4090, and locked checkpoint identity."""
    integrity = verify_baseline(cfg)
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name():
        raise RuntimeError("Required real numerical replay must run on the actual RTX 4090")
    base = Path(cfg["baseline_root"])
    if read_json(base / "data/processed/audit/summary.json")["fixture"]:
        raise ValueError("Synthetic data cannot settle the real audio policy")
    dest = output(cfg) / "numerical_replay"
    if dest.exists():
        raise ValueError("Replay evidence already exists; inspect it, do not overwrite")
    rows = rows_for(cfg, "dev_model")
    bykey = {key(r): r for r in rows}
    for k, length in zip(REPLAY_KEYS, (1024, 1024, 125611, 125611)):
        if bykey[k]["num_samples"] != length:
            raise ValueError(f"Unexpected real probe length: {k}")
    controls = [r for r in probe_rows(cfg, 8) if key(r) not in REPLAY_KEYS]
    targets = list(REPLAY_KEYS) + [key(r) for r in controls]
    ordered = sorted((r for r in rows if r["audio_valid"]), key=lambda r: r["num_samples"])
    original_batch = original_config(cfg)["audio"]["batch_size"]
    groups = {}
    for k in targets:
        i = next(i for i, r in enumerate(ordered) if key(r) == k)
        block = ordered[(i // original_batch) * original_batch : (i // original_batch + 1) * original_batch]
        groups[k] = [key(r) for r in block if r["num_samples"] == bykey[k]["num_samples"]]
    waves = {k: read_waveform(waveform_path(cfg, bykey[k])) for k in set(sum(groups.values(), []))}
    for k, w in waves.items():
        if len(w) != bykey[k]["num_samples"]:
            raise ValueError(f"Waveform length differs from locked manifest: {k}")
    audio_cache = load_tensor(base / "features/wavlm_base_plus/dev_model.pt")
    text_cache = load_tensor(base / "features/text_context/dev_model.pt")
    assert audio_cache["keys"] == text_cache["keys"] == [key(r) for r in rows]
    index = {k: i for i, k in enumerate(audio_cache["keys"])}
    ecfg = encoder_config(cfg)
    encoder = FrozenAudio(ecfg, local=True)
    head = load_head(base / "checkpoints/audio").eval()
    temps = read_json(base / "artifacts/calibration.json")
    alpha = read_json(base / "artifacts/selection.json")["alpha"]

    def values(k, embedding):
        with torch.inference_mode():
            za = head(embedding[None].float())[0].numpy()
        zt = text_cache["logits"][index[k]].numpy()
        return {
            "embedding": embedding.cpu(),
            "audio_logits": za.tolist(),
            "audio_probabilities": probabilities(za, temps["audio"]).tolist(),
            "fusion_probabilities": probabilities((1 - alpha) * zt + alpha * za, temps["fusion"]).tolist(),
        }

    tensors, measurements, meaningful, repeat_ok = {}, [], False, True
    for k in targets:
        modes = {"cached_original": values(k, audio_cache["features"][index[k]])}
        for dtype in ("bfloat16", "float32"):
            ecfg["runtime"]["dtype"] = dtype
            modes[f"singleton_{dtype}"] = values(k, encoder([waves[k]])[0])
            group = groups[k]
            pooled = encoder([waves[g] for g in group])
            modes[f"paired_{dtype}"] = values(k, pooled[group.index(k)])
            modes[f"repeat_singleton_{dtype}"] = values(k, encoder([waves[k]])[0])
        comparisons = {}
        names = list(modes)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                comparisons[a + "__" + b] = compare_outputs(modes[a], modes[b], cfg["policy"])
        paired = comparisons["singleton_bfloat16__paired_bfloat16"]
        if k in REPLAY_KEYS:
            meaningful |= (
                not paired["embedding_allclose"]
                or paired["fusion_probabilities_max_abs"] > cfg["policy"]["probability_atol"]
            )
        repeat_ok &= comparisons["singleton_bfloat16__repeat_singleton_bfloat16"]["embedding_allclose"]
        tensors[k] = modes
        measurements.append(
            {
                "key": k,
                "num_samples": len(waves[k]),
                "waveform_file_sha256": digest(waveform_path(cfg, bykey[k])),
                "pcm_float32_sha256": __import__("hashlib").sha256(waves[k].tobytes()).hexdigest(),
                "original_equal_length_group": groups[k],
                "comparisons": comparisons,
            }
        )
    dest.mkdir(parents=True)
    save_tensor(dest / "embeddings_and_outputs.pt", tensors)
    report = {
        "status": "complete",
        "scope": "real development media only; no training or test inference",
        "integrity": integrity,
        "environment": environment(ecfg),
        "precision_metadata": {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cpu_threads": torch.get_num_threads(),
            "recorded_modes": [
                "cached_original",
                "singleton_bfloat16",
                "paired_bfloat16",
                "singleton_float32",
                "paired_float32",
            ],
        },
        "measurements": measurements,
        "meaningful_equal_batch_difference": bool(meaningful),
        "singleton_repeat_stable": bool(repeat_ok),
        "tolerances": cfg["policy"],
        "fusion_text_source": "immutable cached raw T1 logits; original alpha and temperatures",
        "tensor_sha256": digest(dest / "embeddings_and_outputs.pt"),
    }
    write_json(dest / "report.json", report)
    if not repeat_ok:
        raise RuntimeError(
            "Singleton BF16 repeatability failed; do not train. Diagnose before choosing precision."
        )
    policy = {
        **cfg["policy"],
        "encoding": "singleton" if meaningful else "original_exact_length",
        "dtype": "bfloat16",
        "real_4090_replay_complete": True,
        "baseline_lock_sha256": integrity["lock_sha256"],
        "replay_sha256": digest(dest / "report.json"),
        "unavailable_behavior": "preserved calibrated T1 fallback",
        "audio_valid_semantics": "decode validity, separate from acoustic eligibility",
    }
    write_json(output(cfg) / "canonical_policy.json", policy)
    write_json(output(cfg) / "integrity.json", integrity)
    write_json(
        output(cfg) / "raw_probe_manifest.json",
        {
            "selected_before_training": True,
            "keys": [key(r) for r in probe_rows(cfg)],
            "noise_snr_db": cfg["evaluation"]["noise_snr_db"],
            "selection": "fixed duration quantiles",
        },
    )
    return report


def encode_canonical(encoder, waveforms, policy):
    if policy["encoding"] == "singleton":
        return torch.stack([encoder([w])[0] for w in waveforms])
    return encoder(waveforms)


def extract_text_canonical(cfg):
    """Approved precision repair: unchanged T1, singleton FP32, separate B caches."""
    from .models import TextClassifier
    from .train_text import predict_text

    integrity = verify_baseline(cfg)
    frozen_policy(cfg)
    dest = output(cfg) / "text_features"
    policy_file = output(cfg) / "text_policy.json"
    evidence = output(cfg) / "numerical_replay/text_fp32_diagnostics.json"
    if dest.exists() or policy_file.exists():
        raise ValueError("Canonical text caches/policy already exist; never overwrite")
    if not evidence.is_file():
        raise ValueError("Real numerical text evidence required before precision repair")
    policy = {
        "version": 1,
        "encoding": "singleton",
        "dtype": "float32",
        "baseline_lock_sha256": integrity["lock_sha256"],
        "evidence_sha256": digest(evidence),
        "authorization": "User approved separate FP32 B text caches after real batch diagnostics",
        "weights": "Exact unchanged selected causal T1; no training or new representation",
        "embedding_atol": 0.0002,
        "embedding_rtol": 0.0002,
        "probability_atol": 0.00002,
    }
    write_json(policy_file, policy)
    ecfg = encoder_config(cfg)
    ecfg["runtime"]["dtype"] = "float32"
    ecfg["text"]["batch_size"] = 1
    model, tokenizer, metadata = TextClassifier.load(Path(ecfg["checkpoint_root"]) / "text_context")
    model.eval().requires_grad_(False)
    assert metadata["context"] and metadata["history_turns"] == 3 and metadata["max_length"] == 128
    summaries = {}
    for split in SPLITS:
        started = time.perf_counter()
        value = predict_text(ecfg, model, tokenizer, rows_for(cfg, split), True)
        value["policy_sha256"] = digest(policy_file)
        save_tensor(dest / f"{split}.pt", value)
        summaries[split] = {
            "rows": len(value["keys"]),
            "elapsed_s": time.perf_counter() - started,
            "cache_sha256": digest(dest / f"{split}.pt"),
        }
    write_json(dest / "summary.json", summaries)
    probes = output(cfg) / "raw_probe_manifest.json"
    manifest = read_json(probes)
    manifest["keys"] = list(dict.fromkeys([*manifest["keys"], *REPLAY_KEYS]))
    manifest["selection"] = (
        "Fixed duration quantiles plus four original numerical-replay keys; before B training"
    )
    write_json(probes, manifest)
    verify_baseline(cfg)


def extract(cfg):
    verify_baseline(cfg)
    policy = frozen_policy(cfg)
    dest = output(cfg) / "features"
    if dest.exists():
        raise ValueError("Canonical caches already exist; never overwrite")
    ecfg = encoder_config(cfg)
    ecfg["runtime"]["dtype"] = policy["dtype"]
    encoder = FrozenAudio(ecfg, local=True)
    summaries = {}
    dest.mkdir(parents=True)
    for split in SPLITS:
        rows = rows_for(cfg, split)
        eligible = torch.tensor([eligibility(r)[0] for r in rows])
        start = time.perf_counter()
        if policy["encoding"] == "original_exact_length":
            features = load_tensor(Path(cfg["baseline_root"]) / f"features/wavlm_base_plus/{split}.pt")[
                "features"
            ]
        else:
            features = torch.zeros(len(rows), 768)
            for i, row in enumerate(rows):
                if eligible[i]:
                    wave = read_waveform(waveform_path(cfg, row))
                    if len(wave) != row["num_samples"]:
                        raise ValueError("Raw waveform changed")
                    features[i] = encode_canonical(encoder, [wave], policy)[0]
        if not torch.isfinite(features[eligible]).all():
            raise ValueError("Non-finite canonical features")
        save_tensor(
            dest / f"{split}.pt",
            {
                "keys": [key(r) for r in rows],
                "features": features,
                "audio_valid": torch.tensor([r["audio_valid"] for r in rows]),
                "audio_available": eligible,
                "policy_sha256": digest(output(cfg) / "canonical_policy.json"),
            },
        )
        summaries[split] = {
            "elapsed_s": time.perf_counter() - start,
            "eligible": int(eligible.sum()),
            "rows": len(rows),
            "cache_sha256": digest(dest / f"{split}.pt"),
        }
    write_json(dest / "summary.json", summaries)
    verify_baseline(cfg)


def load_inputs(cfg, split):
    rows = rows_for(cfg, split)
    base = Path(cfg["baseline_root"])
    t = load_tensor(output(cfg) / f"text_features/{split}.pt")
    if t["policy_sha256"] != digest(output(cfg) / "text_policy.json"):
        raise ValueError("Canonical text cache policy changed")
    a = load_tensor(output(cfg) / f"features/{split}.pt")
    if a["keys"] != t["keys"] or t["keys"] != [key(r) for r in rows]:
        raise ValueError("Cached identities changed")
    if a["policy_sha256"] != digest(output(cfg) / "canonical_policy.json"):
        raise ValueError("Canonical cache policy changed")
    expected_available = torch.tensor([eligibility(r)[0] for r in rows])
    if not torch.equal(a["audio_available"].bool(), expected_available):
        raise ValueError("Cache acoustic eligibility differs from the locked operational policy")
    for values in (t["features"], a["features"]):
        if values.shape != (len(rows), 768) or not torch.isfinite(values).all():
            raise ValueError("Invalid cached representation")
    if t["logits"].shape != (len(rows), len(LABELS)) or not torch.isfinite(t["logits"]).all():
        raise ValueError("Invalid cached text logits")
    head = load_head(base / "checkpoints/audio").eval()
    with torch.inference_mode():
        za = head(a["features"].float())
    return rows, {
        "ht": t["features"].float(),
        "ha": a["features"].float(),
        "zt": t["logits"].float(),
        "za": za,
        "available": a["audio_available"].bool(),
        "labels": torch.tensor([r["label"] for r in rows]),
    }


def main():
    p = parser(__doc__)
    p.add_argument("action", choices=("verify", "replay", "extract", "extract-text"))
    args = p.parse_args()
    cfg = settings(args)
    action = {
        "verify": verify_baseline,
        "replay": replay,
        "extract": extract,
        "extract-text": extract_text_canonical,
    }[args.action]
    result = action(cfg)
    print(json.dumps(result if result is not None else {"status": "complete"}, indent=2))


if __name__ == "__main__":
    main()

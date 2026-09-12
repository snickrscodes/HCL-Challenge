"""Configuration, provenance, serialization, and device helpers."""

import argparse
import contextlib
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import transformers
import yaml

from .constants import LABELS

LOG = logging.getLogger("roadmap_a")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def read_rows(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def key(row):
    return f"{row['split']}/{int(row['dialogue_id'])}/{int(row['utterance_id'])}"


def row_sort(row):
    return (row["split"], int(row["dialogue_id"]), int(row["utterance_id"]))


def load_config(path="configs/roadmap_a.yaml"):
    cfg = yaml.safe_load(Path(path).read_text())
    if "labels" in cfg and tuple(cfg["labels"]) != LABELS:
        raise ValueError("Configuration label order differs from constants")
    if cfg["audio"]["sample_rate"] != 16000:
        raise ValueError("Roadmap A requires 16 kHz")
    return cfg


def common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/roadmap_a.yaml")
    return parser


def setup(seed):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def device_for(cfg):
    name = cfg["runtime"]["device"]
    return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name)


def autocast(cfg, device):
    enabled = device.type == "cuda" and cfg["runtime"]["dtype"] == "bfloat16"
    if enabled and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configured BF16 is unsupported; use float32")
    return torch.autocast("cuda", dtype=torch.bfloat16) if enabled else contextlib.nullcontext()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def counts(model):
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment(cfg):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu.name if gpu else None,
        "gpu_memory_bytes": gpu.total_memory if gpu else None,
        "cpu": cpu_name(),
        "cpu_count": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "process_rss_bytes": psutil.Process().memory_info().rss,
        "seed": cfg["seed"],
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if gpu else 0,
    }


def start_run(cfg, name):
    assert_unfrozen(cfg)
    path = Path(cfg["output_root"]) / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    info = environment(cfg)
    manifest = Path(cfg["data_root"]) / "manifest.jsonl"
    info["preprocessing_manifest_sha256"] = digest(manifest) if manifest.exists() else None
    write_json(path / "environment.json", info)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return path, time.perf_counter()


def finish_run(cfg, path, started, metadata=None):
    info = environment(cfg)
    info["elapsed_s"] = time.perf_counter() - started
    manifest = Path(cfg["data_root"]) / "manifest.jsonl"
    info["preprocessing_manifest_sha256"] = digest(manifest) if manifest.exists() else None
    if metadata:
        info.update(metadata)
    write_json(path / "environment.json", info)


def split_rows(cfg, split):
    return read_rows(Path(cfg["data_root"]) / "splits" / f"{split}.jsonl")


def audio_file(cfg, row):
    return Path(cfg["data_root"]) / row["audio_path"]


def assert_unfrozen(cfg):
    if (Path(cfg["output_root"]) / "freeze.json").exists():
        raise RuntimeError("This experiment is frozen. Use a new output/checkpoint/feature root to retrain.")


def load_tensor(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def cpu_name():
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def link_checkpoint(run_path, checkpoint):
    link = Path(run_path) / "checkpoint"
    target = os.path.relpath(Path(checkpoint).resolve(), link.parent.resolve())
    if not link.exists():
        link.symlink_to(target, target_is_directory=True)

"""Portable loader for frozen T1 + B/adaptive/1337 + D2/1337.

Waveform preparation, backbone execution, state and evidence construction are
inherited unchanged. Only artifact resolution and D2-only loading are new.
"""

import hashlib
import json
from pathlib import Path

import torch
import yaml

from .b_inference import RoadmapB
from .c1_data import contract
from .c1_model import EvidenceHead
from .c1_runtime_v2 import BackboneClock
from .c1_transfer_runtime import RefinedC1Dual, RefinedSummaryTap
from .ta_assets import AssetIntegrityError, manifest, sha256, verify_asset_root


def portable_config(asset_root, device):
    if device not in ("cuda", "cpu"):
        raise ValueError("Select explicit device 'cuda' or 'cpu'; automatic fallback is not supported")
    root = Path(asset_root).expanduser().resolve()
    cfg = yaml.safe_load((root / "provenance/roadmap_b.yaml").read_text())
    cfg.update(
        baseline_root=str(root / "a"),
        baseline_lock=str(root / "provenance/baseline_lock.json"),
        output_root=str(root / "b"),
        audio_root=None,
    )
    cfg["runtime"] = {**cfg["runtime"], "device": device}
    return cfg


def buffer_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().astype("<f4").tobytes()).hexdigest()


def load_d2(asset_root, value, device):
    root = Path(asset_root)
    checkpoint = torch.load(root / "d2/checkpoint.pt", map_location="cpu", weights_only=True)
    if (
        checkpoint["condition"] != "D2"
        or checkpoint["seed"] != 1337
        or checkpoint["namespace"] != value["namespace"]
        or checkpoint["labels"] != value["labels"]
        or checkpoint["contract_sha256"] != value["historical_c1_contract_sha256"]
    ):
        raise AssetIntegrityError("D2 checkpoint identity or label namespace mismatch")
    state = checkpoint["state_dict"]
    for name in ("mean", "scale"):
        if buffer_sha256(state[name]) != value["embedded_normalization"][name + "_sha256"]:
            raise AssetIntegrityError("D2 embedded normalization differs")
    head = EvidenceHead("D2", state["mean"], state["scale"])
    head.load_state_dict(state, strict=True)
    head.to(device).eval().requires_grad_(False)
    calibration = json.loads((root / "d2/calibration.json").read_text())
    if calibration["temperature"] != value["temperatures"]["D2"]:
        raise AssetIntegrityError("D2 calibration temperature differs")
    return head, checkpoint, calibration["temperature"]


class PortableTADual(RefinedC1Dual):
    """Selected D2 alone: no D3/D0 load, cache, media or CWD dependency."""

    def __init__(self, asset_root, *, device="cuda", clock=True):
        self.asset_manifest = manifest()
        self.asset_verification = verify_asset_root(asset_root, value=self.asset_manifest)
        self.root = Path(self.asset_verification["asset_root"])
        cfg = portable_config(self.root, device)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; request --device cpu explicitly for a separate FP32 run")
        torch.set_num_threads(cfg["runtime"]["cpu_threads"])
        self.b = RoadmapB(cfg, variant="adaptive", seed=1337)
        self.tap = RefinedSummaryTap(self.b.audio)
        self.b.audio = self.tap
        self.head, self.checkpoint, self.temperature = load_d2(self.root, self.asset_manifest, self.b.device)
        self.namespace = self.checkpoint["namespace"]
        self.labels = tuple(self.checkpoint["labels"])
        if list(contract()["labels"]) != self.asset_manifest["labels"]:
            raise AssetIntegrityError("Inherited evidence label order differs from the selected manifest")
        self.numerical_scope = (
            "singleton FP32 text; singleton BF16 WavLM; FP32 summaries/head"
            if device == "cuda"
            else "explicit CPU FP32; no cross-device equivalence claim"
        )
        self.condition = "D2"
        self.provenance = tuple(
            {
                "condition": "D2",
                "seed": "1337",
                "model_sha256": sha256(self.root / "d2/checkpoint.pt"),
                "calibration_sha256": sha256(self.root / "d2/calibration.json"),
                "normalization_sha256": self.asset_manifest["embedded_normalization"]["original_file_sha256"],
                "normalization_storage": "exact mean/scale buffers embedded in selected D2 checkpoint",
                "protocol_sha256": sha256(self.root / "provenance/c1/C1_PROTOCOL_FREEZE.json"),
                "model_freeze_sha256": sha256(self.root / "provenance/c1/C1_MODEL_FREEZE.json"),
                "runtime_source_sha256": sha256(Path(__file__)),
                "numerical_scope": self.numerical_scope,
            }.items()
        )
        if self.b.temperature != self.asset_manifest["temperatures"]["B"] or (
            self.b.calibration["text"] != self.asset_manifest["temperatures"]["text_fallback"]
        ):
            raise AssetIntegrityError("B or text-fallback temperature differs")
        self.clock = BackboneClock(self.tap.encoder.backbone, self.b.device) if clock else None
        ledger = self.parameter_ledger()
        if ledger["total_learned_parameters"] != self.asset_manifest["learned_parameters"]:
            raise AssetIntegrityError("Loaded parameter ledger differs from the selected release")

    def close(self):
        if self.clock is not None:
            self.clock.close()
            self.clock = None
        self.tap.clear()

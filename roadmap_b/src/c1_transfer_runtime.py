"""Two profiled C1 runtime changes; frozen weights, arithmetic and B are preserved."""

import json
from pathlib import Path
import time

import torch

from .audio import masked_mean, prepare_waveform
from .c1_data import sha
from .c1_model import statistics
from .c1_runtime_v2 import C1Dual, TimedSummaryTap
from .c1_workflow_v2 import load_head
from .utils import autocast, sync


def grouped_statistics(hidden_states, mask):
    """Same reductions over frames, vectorized over the 13 independent layers."""
    if len(hidden_states) != 13 or any(h.shape != hidden_states[0].shape for h in hidden_states):
        raise ValueError("Expected all thirteen matching singleton hidden states")
    if hidden_states[0].shape[0] != 1:
        raise ValueError("Canonical singleton only")
    stacked = torch.cat(hidden_states, dim=0)
    return statistics(stacked, mask.expand(13, -1))[None]


class RefinedSummaryTap(TimedSummaryTap):
    """Group layer reductions; retain summaries on GPU for the GPU evidence head."""

    @torch.inference_mode()
    def forward(self, waveforms):
        self.clear()
        sync(self.encoder.device)
        started = time.perf_counter()
        try:
            if len(waveforms) != 1:
                raise ValueError("Canonical singleton only")
            wave = prepare_waveform(waveforms[0], 16000)
            if len(wave) > 60 * 16000:
                raise ValueError("Duration-ineligible audio")
            self.encoder.eval()
            values = torch.from_numpy(wave).unsqueeze(0).to(self.encoder.device)
            samples = torch.ones_like(values, dtype=torch.long)
            with autocast(self.encoder.cfg, self.encoder.device):
                result = self.encoder.backbone(values, attention_mask=samples, output_hidden_states=True)
            self.calls += 1
            lengths = self.encoder.backbone._get_feat_extract_output_lengths(
                torch.tensor([len(wave)], device=values.device)
            )
            mask = (
                torch.arange(result.last_hidden_state.shape[1], device=values.device)[None] < lengths[:, None]
            )
            # Exact legacy B operation remains in the same position and on CPU.
            legacy = masked_mean(result.last_hidden_state, mask).cpu()
            summaries = grouped_statistics(result.hidden_states, mask)
            if not torch.isfinite(summaries).all():
                raise ValueError("Nonfinite C1 summaries")
            self.last_summaries = summaries
            sync(self.encoder.device)
            self.elapsed_ms = (time.perf_counter() - started) * 1000
            return legacy
        except Exception:
            self.clear()
            raise


class RefinedC1Dual(C1Dual):
    """Explicit experimental condition; never changes B/default response behavior."""

    def __init__(self, root, baseline_root, *, condition="D3", clock=False):
        if condition not in ("D0", "D2", "D3"):
            raise ValueError("Only the prospective fixed pilot heads are allowed")
        super().__init__(root, baseline_root, clock=clock)
        self.tap = RefinedSummaryTap(self.tap.encoder)
        self.b.audio = self.tap
        frozen = json.loads((self.root / "C1_MODEL_FREEZE.json").read_text())
        paths = {
            "checkpoint": f"runs/{condition}/1337/checkpoint.pt",
            "calibration": f"runs/{condition}/1337/calibration.json",
            "normalization": f"normalization/{'D0' if condition == 'D0' else 'D2'}.pt",
        }
        for p in paths.values():
            if sha(self.root / p) != frozen["files_sha256"][p]:
                raise ValueError("Frozen condition identity changed")
        self.head, self.checkpoint = load_head(self.root, condition, 1337, self.b.device)
        self.head.requires_grad_(False)
        self.temperature = json.loads((self.root / paths["calibration"]).read_text())["temperature"]
        self.condition = condition
        provenance = dict(self.provenance)
        provenance.update(
            condition=condition,
            model_sha256=sha(self.root / paths["checkpoint"]),
            normalization_sha256=sha(self.root / paths["normalization"]),
            calibration_sha256=sha(self.root / paths["calibration"]),
            runtime_source_sha256=sha(Path(__file__)),
        )
        self.provenance = tuple(provenance.items())

    @torch.inference_mode()
    def predict_turn(self, *args, **kwargs):
        try:
            return super().predict_turn(*args, **kwargs)
        except Exception:
            self.tap.clear()
            raise

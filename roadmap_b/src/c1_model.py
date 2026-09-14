"""C1 mathematical building blocks. No trainer or changes to historical B modules."""

import numpy as np
import torch
from torch import nn

from .audio import masked_mean, prepare_waveform
from .c1_data import contract
from .utils import autocast


def statistics(hidden, mask):
    """Population mean/std in FP32; padded frames excluded, no positive std floor."""
    mean = masked_mean(hidden, mask)
    variance = ((hidden.float() - mean[:, None]).square() * mask[..., None]).sum(1) / mask.sum(
        1, keepdim=True
    )
    return torch.cat([mean, variance.clamp_min(0).sqrt()], dim=-1)


class AudioSummaryTap(nn.Module):
    """Single canonical backbone call returns legacy B mean and saves C1 summaries.

    The existing encoder is referenced once, never cloned. Clear between turns;
    the application must not read an old summary after a missing-audio fallback.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.last_summaries = None
        self.calls = 0

    def clear(self):
        self.last_summaries = None

    @torch.inference_mode()
    def forward(self, waveforms):
        if len(waveforms) != 1:
            raise ValueError("C1 canonical extraction is singleton")
        self.clear()
        wave = prepare_waveform(waveforms[0], 16000)
        if len(wave) / 16000 > contract()["audio_max_s"]:
            raise ValueError("Duration-ineligible audio must fall back before acoustic encoding")
        self.encoder.eval()
        values = torch.from_numpy(wave).unsqueeze(0).to(self.encoder.device)
        samples = torch.ones_like(values, dtype=torch.long)
        with autocast(self.encoder.cfg, self.encoder.device):
            result = self.encoder.backbone(values, attention_mask=samples, output_hidden_states=True)
        self.calls += 1
        lengths = self.encoder.backbone._get_feat_extract_output_lengths(
            torch.tensor([len(wave)], device=values.device)
        )
        mask = torch.arange(result.last_hidden_state.shape[1], device=values.device)[None] < lengths[:, None]
        legacy = masked_mean(result.last_hidden_state, mask).cpu()
        self.last_summaries = torch.stack([statistics(h, mask) for h in result.hidden_states], 1).cpu()
        if not torch.isfinite(self.last_summaries).all():
            raise ValueError("Nonfinite C1 summaries")
        return legacy


class EvidenceHead(nn.Module):
    def __init__(self, condition, mean, scale):
        super().__init__()
        if condition not in contract()["conditions"]:
            raise ValueError("Unknown C1 condition")
        expected = (768,) if condition == "D0" else (1536,) if condition == "D1" else (13, 1536)
        if tuple(mean.shape) != expected or tuple(scale.shape) != expected[:-1] + (1,):
            raise ValueError("Normalization shape mismatch")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Invalid normalization buffers")
        self.condition = condition
        self.register_buffer("mean", mean.float().clone())
        self.register_buffer("scale", scale.float().clone())
        self.mixture = nn.Parameter(torch.zeros(13)) if condition in ("D2", "D3") else None
        self.project = nn.Linear(expected[-1], 128)
        self.norm = nn.LayerNorm(128)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(128, 6)

    def forward(self, summaries):
        x = select_input(summaries, self.condition)
        x = (x - self.mean) / self.scale
        if self.mixture is not None:
            x = (x * self.mixture.softmax(0)[None, :, None]).sum(1)
        v = self.activation(self.norm(self.project(x)))
        d = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        logits = self.classifier(self.dropout(v))
        return {"logits": logits, "representation": d, "unnormalized": v}


def select_input(summaries, condition):
    if summaries.ndim != 3 or tuple(summaries.shape[1:]) != (13, 1536):
        raise ValueError("Expected [rows,13,1536] pooled summaries")
    if condition == "D0":
        return summaries[:, -1, :768]
    if condition == "D1":
        return summaries[:, -1]
    if condition not in ("D2", "D3"):
        raise ValueError("Unknown C1 condition")
    return summaries


def fit_normalization(summaries, partition, condition):
    if partition != "train":
        raise ValueError("Normalization may only fit external training rows")
    x = select_input(summaries.float(), condition)
    if len(x) == 0 or not torch.isfinite(x).all():
        raise ValueError("Invalid training summaries")
    mean = x.mean(0)
    scale = x.var(0, correction=0).mean(-1, keepdim=True).sqrt().clamp_min(1e-5)
    return mean, scale


def soft_ce(logits, targets):
    if logits.shape != targets.shape or logits.shape[-1] != 6:
        raise ValueError("C1 six-class shape mismatch")
    if (
        not torch.isfinite(targets).all()
        or (targets < 0).any()
        or not torch.allclose(targets.sum(-1), torch.ones_like(targets[:, 0]), atol=1e-6)
    ):
        raise ValueError("Invalid listener target")
    return -(targets * logits.log_softmax(-1)).sum(-1).mean()


def triplet_loss(anchor, positive, negative):
    if len(anchor) == 0:
        return anchor.sum() * 0
    return (0.2 + (anchor * negative).sum(-1) - (anchor * positive).sum(-1)).clamp_min(0).mean()


def epoch_order(n, seed, epoch):
    return np.random.default_rng(np.random.SeedSequence([seed, epoch, 0])).permutation(n)


def sample_triplets(batch_ids, pools, seed, epoch, batch_index):
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 1, batch_index]))
    return [
        (rid, str(rng.choice(pools[rid]["positive"])), str(rng.choice(pools[rid]["negative"])))
        for rid in batch_ids
        if rid in pools
    ]

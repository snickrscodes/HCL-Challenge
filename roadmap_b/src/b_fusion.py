"""Four controlled corrections sharing one implementation; no encoders here."""

import math

import torch
from torch import nn

from .constants import LABELS

VARIANTS = ("bias", "constant", "adaptive", "text_only")


def summaries(logits):
    p = logits.float().softmax(-1)
    entropy = -(p * p.clamp_min(1e-12).log()).sum(-1) / math.log(len(LABELS))
    return p.max(-1).values, entropy


def bounded_center(value, bound=2.0):
    u = value.tanh()
    centered = u - u.mean(-1, keepdim=True)
    return bound * centered / centered.abs().amax(-1, keepdim=True).clamp_min(1.0)


class ResidualFusion(nn.Module):
    def __init__(self, variant="adaptive", width=32, bound=2.0, dropout=0.1):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown controlled variant: {variant}")
        self.variant, self.bound = variant, bound
        self.spec = dict(variant=variant, width=width, bound=bound, dropout=dropout)
        if variant == "bias":
            self.bias = nn.Parameter(torch.zeros(len(LABELS)))
        else:
            self.residual = nn.Linear(768, len(LABELS))
            nn.init.zeros_(self.residual.weight)
            nn.init.zeros_(self.residual.bias)
            if variant == "constant":
                self.gate_logit = nn.Parameter(torch.tensor(-2.0))
            else:
                self.gate = nn.Sequential(
                    nn.Linear(1540, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1)
                )
                nn.init.zeros_(self.gate[-1].weight)
                nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, ht, ha, zt, za, available, forced_gate=None):
        # Frozen features are inputs; no gradients may propagate into A.
        ht, ha, zt, za = (x.detach().float() for x in (ht, ha, zt, za))
        if ht.shape[-1] != 768 or ha.shape != ht.shape or zt.shape[-1] != len(LABELS):
            raise ValueError("Invalid representation dimensions")
        if za.shape != zt.shape or len(zt) != len(ht):
            raise ValueError("Unaligned representation/logit dimensions")
        tc, te = summaries(zt)
        ac, ae = summaries(za)
        if self.variant == "bias":
            delta = (self.bias - self.bias.mean()).expand_as(zt)
            gate = torch.ones(len(zt), device=zt.device)
        else:
            residual_input = ht if self.variant == "text_only" else ha
            delta = bounded_center(self.residual(residual_input), self.bound)
            if self.variant == "constant":
                gate = self.gate_logit.sigmoid().expand(len(zt))
            else:
                other, oc, oe = (ht, tc, te) if self.variant == "text_only" else (ha, ac, ae)
                features = torch.cat((ht, other, tc[:, None], te[:, None], oc[:, None], oe[:, None]), -1)
                gate = self.gate(features).squeeze(-1).sigmoid()
        if forced_gate is not None:
            if not 0 <= float(forced_gate) <= 1:
                raise ValueError("Forced gate must be in [0,1]")
            gate = torch.full_like(gate, float(forced_gate))
        effective_gate = gate * available.to(zt.device).bool().float()
        correction = effective_gate[:, None] * delta
        logits = zt + correction
        return {
            "logits": logits,
            "gate": gate,
            "delta": delta,
            "correction": correction,
            "text_max_probability": tc,
            "audio_max_probability": ac,
            "text_entropy": te,
            "audio_entropy": ae,
        }

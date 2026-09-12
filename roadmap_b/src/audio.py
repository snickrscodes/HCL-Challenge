"""The shared waveform and frozen WavLM path. No experimental feature cache here."""

import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import WavLMModel

from .constants import SAMPLE_RATE
from .utils import autocast, device_for

MIN_SAMPLES = 400  # receptive field of the configured seven convolutional layers


def prepare_waveform(audio, sample_rate):
    if sample_rate is None or not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("A positive integer sample rate is required")
    if torch.is_tensor(audio):
        audio = audio.detach().cpu().numpy()
    wave = np.asarray(audio)
    if wave.ndim not in (1, 2):
        raise ValueError("Waveform must be samples or samples x channels")
    if not np.issubdtype(wave.dtype, np.floating):
        raise ValueError("PCM arrays must be floating point; decode integer WAV with read_waveform")
    if not np.isfinite(wave).all() or wave.size == 0:
        raise ValueError("Empty or non-finite waveform")
    wave = wave.astype(np.float32)
    if wave.ndim == 2:
        if not 1 <= wave.shape[1] <= 32:
            raise ValueError("Use samples x channels, not channels x samples")
        wave = wave.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
        wave = resample_poly(wave, SAMPLE_RATE // divisor, int(sample_rate) // divisor).astype(np.float32)
    if len(wave) < MIN_SAMPLES or not np.isfinite(wave).all():
        raise ValueError("Waveform is non-finite or shorter than WavLM's 400-sample receptive field")
    return np.ascontiguousarray(wave)


def read_waveform(path):
    wave, rate = sf.read(path, dtype="float32", always_2d=False)
    return prepare_waveform(wave, rate)


def convert_media(source, destination):
    """FFmpeg decodes/downmixes/resamples once; output is lossless float WAV."""
    ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_f32le",
            str(destination),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        destination.unlink(missing_ok=True)
        raise ValueError(proc.stderr.strip()[-2000:] or "FFmpeg failed")
    try:
        wave, rate = sf.read(destination, dtype="float32")
        if rate != SAMPLE_RATE or wave.ndim != 1:
            raise ValueError("Converted waveform violates mono 16 kHz contract")
        return prepare_waveform(wave, rate)
    except (ValueError, RuntimeError):
        destination.unlink(missing_ok=True)
        raise


def masked_mean(hidden, mask):
    if hidden.shape[:2] != mask.shape:
        raise ValueError("Frame mask shape mismatch")
    if (mask.sum(dim=1) == 0).any():
        raise ValueError("No valid frames")
    return (hidden.float() * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True)


class FrozenAudio(torch.nn.Module):
    def __init__(self, cfg, model=None, *, local=False):
        super().__init__()
        self.cfg = cfg
        self.backbone = (
            model
            if model is not None
            else WavLMModel.from_pretrained(
                Path(cfg["checkpoint_root"]) / "wavlm" / "encoder", local_files_only=True
            )
            if local
            else WavLMModel.from_pretrained(cfg["audio"]["model"], revision=cfg["audio"]["revision"])
        )
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.device = device_for(cfg)
        self.to(self.device)

    def train(self, mode=True):
        super().train(False)
        self.backbone.eval()
        return self

    @torch.inference_mode()
    def forward(self, waveforms):
        """Batch equal sample lengths only, preserving WavLM group-norm invariance.

        Unequal lengths are unpadded and encoded separately. Pooling still uses
        the exact convolution-derived frame mask. This is intentionally simpler
        than altering the pretrained feature extractor.
        """
        groups = defaultdict(list)
        for index, waveform in enumerate(waveforms):
            wave = prepare_waveform(waveform, SAMPLE_RATE)
            groups[len(wave)].append((index, wave))
        outputs = [None] * len(waveforms)
        self.eval()
        for length, group in groups.items():
            values = torch.from_numpy(np.stack([wave for _, wave in group])).to(self.device)
            # The official base-plus processor does not normalize waveform amplitude.
            sample_mask = torch.ones_like(values, dtype=torch.long)
            with autocast(self.cfg, self.device):
                hidden = self.backbone(values, attention_mask=sample_mask).last_hidden_state
            frame_lengths = self.backbone._get_feat_extract_output_lengths(
                torch.full((len(group),), length, device=self.device, dtype=torch.long)
            )
            mask = torch.arange(hidden.shape[1], device=self.device)[None] < frame_lengths[:, None]
            pooled = masked_mean(hidden, mask).cpu()
            if not torch.isfinite(pooled).all():
                raise ValueError("WavLM produced non-finite embeddings")
            for (index, _), embedding in zip(group, pooled):
                outputs[index] = embedding
        return torch.stack(outputs) if outputs else torch.empty(0, self.backbone.config.hidden_size)

    def from_padded(self, waveforms, lengths):
        return self([waveforms[i, : int(length)].detach().cpu().numpy() for i, length in enumerate(lengths)])

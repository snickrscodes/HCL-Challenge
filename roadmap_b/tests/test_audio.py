import os

import numpy as np
import pytest
import torch

from src.audio import FrozenAudio, masked_mean, prepare_waveform
from src.utils import load_config


def test_waveform_contract():
    stereo = np.ones((800, 2), np.float32)
    assert prepare_waveform(stereo, 8000).shape == (1600,)
    for wave in (np.array([], np.float32), np.array([np.nan] * 800, np.float32)):
        with pytest.raises(ValueError):
            prepare_waveform(wave, 16000)
    with pytest.raises(ValueError):
        prepare_waveform(np.ones(800, np.int16), 16000)
    assert np.max(prepare_waveform(np.zeros(800, np.float32), 16000)) == 0


def test_mask_excludes_padding():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [999.0, 999.0]]])
    assert torch.equal(masked_mean(hidden, torch.tensor([[1, 1, 0]])), torch.tensor([[2.0, 3.0]]))


def assert_padding_invariance(encoder, atol=1e-5, rtol=1e-5):
    rng = np.random.default_rng(1)
    target, short, long = [rng.normal(0, 0.1, size).astype(np.float32) for size in (6400, 3200, 9600)]
    alone = encoder([target])[0]
    for others in ([target, short], [target, long]):
        batch = torch.zeros(2, max(map(len, others)))
        for i, wave in enumerate(others):
            batch[i, : len(wave)] = torch.from_numpy(wave)
        padded = encoder.from_padded(batch, list(map(len, others)))[0]
        assert torch.allclose(alone, padded, atol=atol, rtol=rtol)
    assert torch.equal(alone, encoder([target])[0])
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())


def test_padding_and_frozen_eval(tiny_cfg):
    encoder = FrozenAudio(tiny_cfg)
    encoder.train()
    assert not encoder.training and not encoder.backbone.training
    assert_padding_invariance(encoder)


@pytest.mark.pretrained
@pytest.mark.skipif(
    os.environ.get("RUN_PRETRAINED") != "1", reason="Set RUN_PRETRAINED=1 to validate pinned real WavLM"
)
def test_official_wavlm_padding_invariance():
    cfg = load_config()
    cfg["runtime"]["dtype"] = "float32"
    assert_padding_invariance(FrozenAudio(cfg), atol=2e-5, rtol=2e-5)


@pytest.mark.pretrained
@pytest.mark.skipif(
    os.environ.get("RUN_PRETRAINED") != "1", reason="Set RUN_PRETRAINED=1 for BF16 numerical validation"
)
def test_official_bf16_stability():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 CUDA GPU unavailable")
    cfg = load_config()
    cfg["runtime"].update(device="cuda", dtype="float32")
    encoder = FrozenAudio(cfg)
    wave = np.random.default_rng(29).normal(0, 0.05, 16000).astype(np.float32)
    reference = encoder([wave])[0]
    cfg["runtime"]["dtype"] = "bfloat16"
    actual = encoder([wave])[0]
    assert torch.isfinite(actual).all()
    cosine = torch.nn.functional.cosine_similarity(reference[None], actual[None]).item()
    assert cosine > 0.995
    assert_padding_invariance(encoder, atol=2e-4, rtol=2e-4)


def test_mp4_decode_and_missing_audio_stream(tmp_path):
    import subprocess
    import imageio_ffmpeg
    from src.audio import convert_media

    executable = imageio_ffmpeg.get_ffmpeg_exe()
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x32:r=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=300:sample_rate=44100",
            "-t",
            "0.2",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(clip),
        ],
        check=True,
    )
    wave = convert_media(clip, tmp_path / "converted.wav")
    assert wave.ndim == 1 and len(wave) > 400 and np.isfinite(wave).all()
    video_only = tmp_path / "video_only.mp4"
    subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x32:r=10",
            "-t",
            "0.2",
            "-c:v",
            "mpeg4",
            str(video_only),
        ],
        check=True,
    )
    with pytest.raises(ValueError):
        convert_media(video_only, tmp_path / "invalid.wav")
    assert not (tmp_path / "invalid.wav").exists()

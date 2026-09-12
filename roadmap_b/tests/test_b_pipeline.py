"""Synthetic 32-example orchestration test; never uses real MELD training rows."""

import json
from pathlib import Path

import torch
import yaml

from src import b_evaluate, b_train
from src.b_fusion import VARIANTS
from src.b_report import generate
from src.constants import LABELS
from src.evaluate import probabilities, save_evaluation
from src.models import ClassificationHead, save_head
from src.utils import digest, save_tensor, write_json, write_rows


def test_synthetic_32_example_matrix_calibration_interventions_report(tmp_path, monkeypatch):
    cfg = yaml.safe_load(Path("configs/roadmap_b.yaml").read_text())
    base, out = tmp_path / "A", tmp_path / "B"
    cfg.update(baseline_root=str(base), output_root=str(out), baseline_lock=str(tmp_path / "lock.json"))
    cfg["training"].update(epochs=2, patience=1)
    write_json(cfg["baseline_lock"], {"fixture": True})
    write_json(out / "canonical_policy.json", {"fixture": True})
    write_json(out / "text_policy.json", {"fixture": True})
    torch.set_num_threads(2)
    torch.manual_seed(3)
    audio = ClassificationHead(768)
    save_head(base / "checkpoints/audio", audio, 768, 256)
    write_json(base / "artifacts/calibration.json", {"text": 1.6, "audio": 1.0, "fusion": 1.08})
    write_json(base / "artifacts/selection.json", {"alpha": 0.4, "text": "text_context"})
    for split, n in (("train", 32), ("dev_model", 21), ("dev_calib", 14)):
        rows = [
            {
                "split": "train" if split == "train" else "dev",
                "dialogue_id": i // 7,
                "utterance_id": i % 7,
                "speaker": "s" + str(i % 2),
                "text": "synthetic",
                "label": i % 7,
                "emotion": LABELS[i % 7],
                "audio_valid": i != 0,
                "duration_s": 1.0,
            }
            for i in range(n)
        ]
        keys = [f"{r['split']}/{r['dialogue_id']}/{r['utterance_id']}" for r in rows]
        # Split identities need only match their own canonical partitions in this synthetic fixture.
        ht, ha, zt = torch.randn(n, 768), torch.randn(n, 768), torch.randn(n, 7)
        write_rows(base / f"data/processed/splits/{split}.jsonl", rows)
        save_tensor(base / f"features/text_context/{split}.pt", {"keys": keys, "features": ht, "logits": zt})
        save_tensor(
            out / f"text_features/{split}.pt",
            {"keys": keys, "features": ht, "logits": zt, "policy_sha256": digest(out / "text_policy.json")},
        )
        save_tensor(
            out / f"features/{split}.pt",
            {
                "keys": keys,
                "features": ha,
                "audio_available": torch.tensor([r["audio_valid"] for r in rows]),
                "policy_sha256": digest(out / "canonical_policy.json"),
            },
        )
        if split == "dev_model":
            for model in ("text_context", "concat", "late_fusion"):
                save_evaluation(base / f"artifacts/evaluation/dev_model/{model}", rows, probabilities(zt))
    identity = {"lock_sha256": digest(cfg["baseline_lock"]), "archive_sha256": "synthetic"}
    monkeypatch.setattr(b_train, "verify_baseline", lambda _: identity)
    monkeypatch.setattr(b_train, "frozen_policy", lambda _: {"fixture": True})
    monkeypatch.setattr(b_evaluate, "verify_baseline", lambda _: identity)
    monkeypatch.setattr(b_evaluate, "frozen_policy", lambda _: {"fixture": True})
    b_train.matrix(cfg)
    assert len(list((out / "runs").rglob("checkpoint.pt"))) == 12
    b_evaluate.evaluate(cfg)
    summary = json.loads((out / "evaluation/summary.json").read_text())
    assert set(summary["variants"]) == set(VARIANTS)
    for variant in VARIANTS:
        for seed in cfg["seeds"]:
            result = json.loads((out / f"evaluation/{variant}/{seed}/metrics.json").read_text())
            assert result["temperature"] > 0
            assert result["metrics"]["n"] == 21
            assert result["bootstrap_vs_A_replay"]["replicates"] >= 1000
            assert result["interventions"]["same_speaker"]["metadata"]["support"] > 0
    # Remove the synthetic policy marker only for the incomplete-report branch, which should not
    # mistake synthetic orchestration for the required real numerical replay.
    (out / "canonical_policy.json").unlink()
    generate(cfg)
    report = (out / "ROADMAP_B_REPORT.md").read_text()
    assert report.rstrip().endswith("RETAIN ROADMAP A")
    assert "Pending actual RTX 4090" in report

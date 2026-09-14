"""Minimal C1-Dual wrapper: unchanged frozen B plus independent CREMA6 evidence."""

import argparse
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch

from .audio import prepare_waveform, read_waveform
from .b_data import output, rows_for, waveform_path
from .b_inference import RoadmapB
from .c1_data import contract, sha, write_json
from .c1_model import AudioSummaryTap
from .c1_workflow_v2 import bconfig, load_head, load_features, verify_model_freeze
from .context import Turn, histories
from .evaluate import probabilities
from .session import Session
from .utils import key, sync


@dataclass(frozen=True)
class DeliveryEvidence:
    available: bool
    reason: str | None
    namespace: str
    posterior: tuple[float, ...] | None
    representation: tuple[float, ...] | None
    entropy: float | None
    max_probability: float | None
    provenance: tuple[tuple[str, str], ...]

    def to_dict(self):
        labels = contract()["labels"]
        return {
            "available": self.available,
            "unavailable_reason": self.reason,
            "namespace": self.namespace,
            "distribution": dict(zip(labels, self.posterior)) if self.posterior is not None else None,
            "normalized_128d": list(self.representation) if self.representation is not None else None,
            "entropy": self.entropy,
            "max_probability": self.max_probability,
            "provenance": dict(self.provenance),
            "training_domain": "CREMA-D acted speech, aggregate voice-only listener votes; human transfer unvalidated",
        }


class BackboneClock:
    def __init__(self, backbone, device):
        self.device = device
        self.elapsed_ms = 0.0
        self.calls = 0
        self.handles = [
            backbone.register_forward_pre_hook(self.start),
            backbone.register_forward_hook(self.end),
        ]

    def start(self, module, args):
        sync(self.device)
        self.started = time.perf_counter()

    def end(self, module, args, result):
        sync(self.device)
        self.elapsed_ms = (time.perf_counter() - self.started) * 1000
        self.calls += 1

    def close(self):
        for handle in self.handles:
            handle.remove()


class TimedSummaryTap(AudioSummaryTap):
    def forward(self, waves):
        sync(self.encoder.device)
        started = time.perf_counter()
        result = super().forward(waves)
        sync(self.encoder.device)
        self.elapsed_ms = (time.perf_counter() - started) * 1000
        return result


class C1Dual:
    def __init__(self, root, baseline_root, *, clock=False):
        self.root = Path(root)
        frozen = verify_model_freeze(self.root, runtime=True)
        if "D3" not in frozen["conditions"]:
            raise ValueError("D3 unavailable; do not promote a control")
        self.b = RoadmapB(bconfig(baseline_root))
        self.tap = TimedSummaryTap(self.b.audio)
        self.b.audio = self.tap
        self.head, self.checkpoint = load_head(self.root, "D3", 1337, self.b.device)
        self.head.requires_grad_(False)
        calibration = self.root / "runs/D3/1337/calibration.json"
        self.temperature = json.loads(calibration.read_text())["temperature"]
        self.namespace = self.checkpoint["namespace"]
        self.labels = tuple(self.checkpoint["labels"])
        self.provenance = tuple(
            {
                "condition": "D3",
                "seed": "1337",
                "model_sha256": sha(self.root / "runs/D3/1337/checkpoint.pt"),
                "normalization_sha256": sha(self.root / "normalization/D2.pt"),
                "calibration_sha256": sha(calibration),
                "protocol_sha256": sha(self.root / "C1_PROTOCOL_FREEZE.json"),
                "model_freeze_sha256": sha(self.root / "C1_MODEL_FREEZE.json"),
            }.items()
        )
        self.clock = BackboneClock(self.tap.encoder.backbone, self.b.device) if clock else None

    @torch.inference_mode()
    def predict_turn(self, text, audio, sample_rate, history, *, speaker="user"):
        sync(self.b.device)
        started = time.perf_counter()
        self.tap.clear()
        invalid = None
        if audio is not None:
            try:
                audio = prepare_waveform(audio, sample_rate)
                sample_rate = 16000
            except ValueError:
                audio = None
                sample_rate = None
                invalid = "invalid_audio"
        calls = self.tap.calls
        state = self.b.predict_turn(text, audio, sample_rate, history, speaker=speaker)
        sync(self.b.device)
        head_start = time.perf_counter()
        if state["audio_available"]:
            assert self.tap.calls == calls + 1 and self.tap.last_summaries is not None
            value = self.head(self.tap.last_summaries.to(self.b.device))
            prob = probabilities(value["logits"][0].cpu(), self.temperature)
            latent = value["representation"][0].cpu().tolist()
            evidence = DeliveryEvidence(
                True,
                None,
                self.namespace,
                tuple(prob.tolist()),
                tuple(latent),
                float(-(prob * np.log(np.maximum(prob, 1e-300))).sum()),
                float(prob.max()),
                self.provenance,
            )
        else:
            assert self.tap.calls == calls and self.tap.last_summaries is None
            evidence = DeliveryEvidence(
                False,
                invalid or state["audio_unavailable_reason"],
                self.namespace,
                None,
                None,
                None,
                None,
                self.provenance,
            )
        result = {"meld_state": state, "delivery_evidence": evidence.to_dict()}
        sync(self.b.device)
        ended = time.perf_counter()
        wavlm = self.clock.elapsed_ms if self.clock and state["audio_available"] else None
        result["latency_ms"] = {
            "text": state["latency_ms"]["text"],
            "wavlm": wavlm,
            "summary_wrapper": max(0.0, self.tap.elapsed_ms - wavlm) if wavlm is not None else None,
            "new_head_evidence": (ended - head_start) * 1000,
            "B_state_response": state["latency_ms"]["fusion_response"],
            "total": (ended - started) * 1000,
        }
        return result

    def parameter_ledger(self):
        b = self.b.parameter_ledger()
        new = sum(p.numel() for p in self.head.parameters())
        parameters = [
            *self.b.text.parameters(),
            *self.tap.parameters(),
            *self.b.audio_head.parameters(),
            *self.b.residual.parameters(),
            *self.head.parameters(),
        ]
        unique = {id(p): p for p in parameters}
        neural = sum(p.numel() for p in unique.values())
        assert neural + 6 == b["total_required_parameters"] + new + 1 == 218895839
        return {
            "B": b,
            "additional_neural_parameters": new,
            "additional_temperature": 1,
            "loaded_unique_neural_parameters": neural,
            "total_learned_parameters": neural + 6,
            "normalization_buffer_elements": sum(t.numel() for t in self.head.buffers()),
            "normalization_buffer_bytes": sum(t.numel() * t.element_size() for t in self.head.buffers()),
            "shared_wavlm_instances": 1,
        }


def quantiles(records):
    result = {}
    if not records:
        return {"support": 0}
    for name in records[0]["latency_ms"]:
        values = [r["latency_ms"][name] for r in records if r["latency_ms"][name] is not None]
        if values:
            result[name] = {"p50": float(np.median(values)), "p95": float(np.quantile(values, 0.95))}
    rtf = [r["latency_ms"]["total"] / (1000 * r["duration_s"]) for r in records]
    return {
        "support": len(records),
        "latency_ms": result,
        "rtf": {"p50": float(np.median(rtf)), "p95": float(np.quantile(rtf, 0.95))},
    }


def validate_and_benchmark(root, baseline_root):
    root = Path(root)
    verify_model_freeze(root)
    if (root / "runtime.json").exists():
        raise ValueError("Preserve prior runtime; no automatic repeated benchmark")
    cfg = bconfig(baseline_root)
    rows = rows_for(cfg, "dev_model")
    by_key = {key(r): r for r in rows}
    hist = histories(rows, 3)
    keys = json.loads((output(cfg) / "raw_probe_manifest.json").read_text())["keys"]
    workload = [keys[i % len(keys)] for i in range(200)]
    waves = {rid: read_waveform(waveform_path(cfg, by_key[rid])) for rid in keys}
    write_json(
        root / "RUNTIME_WORKLOAD.json",
        {
            "keys": keys,
            "sequence": workload,
            "warmups": 30,
            "timed_turns": 200,
            "source_manifest_sha256": sha(output(cfg) / "raw_probe_manifest.json"),
            "selection": "Preserved B probe manifest; cyclic order, no outcome/latency filtering",
        },
    )
    measured = {}
    reference = {}
    raw_checks = []
    cold = {}
    for mode in ("B", "B+C1"):
        gc.collect()
        torch.cuda.empty_cache()
        started = time.perf_counter()
        model = RoadmapB(cfg) if mode == "B" else C1Dual(root, baseline_root, clock=True)
        device = model.device if mode == "B" else model.b.device
        sync(device)
        load_s = time.perf_counter() - started
        clock = BackboneClock(model.audio.backbone, device) if mode == "B" else model.clock
        first = keys[0]
        r = by_key[first]
        cold[mode] = model.predict_turn(r["text"], waves[first], 16000, hist[first], speaker=r["speaker"])
        for i in range(30):
            rid = keys[i % len(keys)]
            r = by_key[rid]
            model.predict_turn(r["text"], waves[rid], 16000, hist[rid], speaker=r["speaker"])
        torch.cuda.reset_peak_memory_stats()
        records = []
        for rid in workload:
            r = by_key[rid]
            n = clock.calls
            value = model.predict_turn(r["text"], waves[rid], 16000, hist[rid], speaker=r["speaker"])
            assert clock.calls == n + 1
            state = value if mode == "B" else value["meld_state"]
            if mode == "B":
                reference[rid] = state
                times = {
                    "text": state["latency_ms"]["text"],
                    "wavlm": clock.elapsed_ms,
                    "summary_wrapper": max(0.0, state["latency_ms"]["audio"] - clock.elapsed_ms),
                    "new_head_evidence": 0.0,
                    "B_state_response": state["latency_ms"]["fusion_response"],
                    "total": state["latency_ms"]["total"],
                }
            else:
                expected = reference[rid]
                assert state["emotion"] == expected["emotion"] and state["response"] == expected["response"]
                diff = max(
                    abs(state["distribution"][k] - expected["distribution"][k]) for k in state["distribution"]
                )
                assert diff <= cfg["policy"]["probability_atol"]
                raw_checks.append(
                    {"key": rid, "probability_max_abs": diff, "labels_and_response_identical": True}
                )
                times = value["latency_ms"]
            records.append(
                {
                    "key": rid,
                    "duration_s": len(waves[rid]) / 16000,
                    "latency_ms": times,
                    "rss_bytes": psutil.Process().memory_info().rss,
                }
            )
        measured[mode] = {
            "model_load_s": load_s,
            "records": records,
            "all": quantiles(records),
            "length_strata": {
                name: quantiles([r for r in records if lo < r["duration_s"] <= hi])
                for name, lo, hi in [("0-3s", 0, 3), ("3-6s", 3, 6), ("6-12s", 6, 12), ("12-60s", 12, 60)]
            },
            "up_to_12s": quantiles([r for r in records if r["duration_s"] <= 12]),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "peak_rss_bytes": max(r["rss_bytes"] for r in records),
        }
        if mode == "B+C1":
            ledger = model.parameter_ledger()
            write_json(root / "parameter_ledger.json", ledger)
            # Real raw/cache C1 check on a fixed eligible training clip; never human recordings.
            rr, xx, _ = load_features(root, "train")
            r = rr[0]
            value = model.predict_turn(r["text"], read_waveform(root / r["audio_path"]), 16000, [])
            with torch.inference_mode():
                z = model.head(xx[:1].to(device))["logits"][0].cpu()
            p = probabilities(z, model.temperature)
            actual = np.array(list(value["delivery_evidence"]["distribution"].values()))
            cache_diff = float(np.max(np.abs(p - actual)))
            assert cache_diff <= 2e-5
            availability = []
            for name, w, sr in [
                ("missing", None, None),
                ("duration_limit", np.zeros(61 * 16000, dtype=np.float32), 16000),
                ("silence", np.zeros(16000, dtype=np.float32), 16000),
                ("corrupt", np.array([np.nan], dtype=np.float32), 16000),
            ]:
                before = clock.calls
                v = model.predict_turn("Okay.", w, sr, [])
                should = name == "silence"
                assert (
                    v["delivery_evidence"]["available"] is should
                    and v["meld_state"]["audio_available"] is should
                )
                assert clock.calls - before == int(should)
                availability.append({"case": name, "state": v})
            # Incremental PCM compatibility and unchanged legacy human-only history.
            probe = by_key[keys[0]]
            session = Session(model, history=hist[keys[0]])
            session.start()
            session.begin_turn(speaker=probe["speaker"])
            for chunk in np.array_split(waves[keys[0]], 5):
                session.push_audio(chunk)
            session.set_transcript(probe["text"])
            demo = session.end_turn()
            assert isinstance(session.history[-1], Turn) and session.history[-1].text == probe["text"]
            write_json(
                root / "raw_dual_demo.json",
                {
                    "input": {
                        "key": keys[0],
                        "text": probe["text"],
                        "history": [t.__dict__ for t in hist[keys[0]]],
                        "audio_chunks": 5,
                        "waveform_sha256": sha(waveform_path(cfg, probe)),
                    },
                    "output": demo,
                },
            )
            write_json(
                root / "raw_compatibility.json",
                {
                    "raw_B_comparisons": raw_checks,
                    "C1_cache_raw_probability_max_abs": cache_diff,
                    "availability": availability,
                    "one_backbone_per_eligible_turn": True,
                    "legacy_history_preserved": True,
                },
            )
        clock.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    a = measured["B"]["up_to_12s"]
    c = measured["B+C1"]["up_to_12s"]
    p95 = c["latency_ms"]["total"]["p95"]
    increase = p95 - a["latency_ms"]["total"]["p95"]
    result = {
        "hardware": torch.cuda.get_device_name(),
        "modes": measured,
        "cold_first_turn": cold,
        "guardrails": {
            "total_p95_le100ms": p95 <= 100,
            "p95_increase_le10ms": increase <= 10,
            "p95_rtf_lt0_1": c["rtf"]["p95"] < 0.1,
            "p95_increase_ms": increase,
        },
        "timing_scope": "Warm loaded utterance-final; recording, endpointing, ASR and disk reads excluded. CPU/GPU transfers and output construction included. Synchronized backbone hooks in both modes. summary_wrapper includes legacy pooling, input preparation and transfers, not exclusively statistical arithmetic. B baseline summary_wrapper also includes original audio head. Cold model load/first turn separate.",
    }
    write_json(root / "runtime.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["predict", "validate"])
    p.add_argument("--root", type=Path, default=Path("artifacts/roadmap_c1_v2"))
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--audio", type=Path)
    p.add_argument("--text", default="Okay.")
    args = p.parse_args()
    if args.stage == "validate":
        validate_and_benchmark(args.root, args.baseline_root)
    else:
        audio = None
        if args.audio:
            try:
                audio = read_waveform(args.audio)
            except (ValueError, RuntimeError, OSError):
                audio = None
        model = C1Dual(args.root, args.baseline_root)
        print(
            json.dumps(
                model.predict_turn(args.text, audio, 16000 if audio is not None else None, []), indent=2
            )
        )


if __name__ == "__main__":
    main()

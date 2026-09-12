"""Raw waveform/text B inference, probe interventions and 4090 runtime validation."""

import time
import gc
from pathlib import Path

import numpy as np
import psutil
import torch

from .audio import prepare_waveform, read_waveform
from .b_data import (
    encode_canonical,
    encoder_config,
    frozen_policy,
    load_inputs,
    output,
    parser,
    settings,
    verify_baseline,
    waveform_path,
)
from .b_evaluate import distributions
from .b_evaluate import verify_freeze
from .b_train import load_model, predict
from .constants import LABELS
from .context import collate_text, encode_turn, histories
from .evaluate import probabilities
from .inference import RoadmapA
from .responses import respond
from .utils import autocast, counts, digest, environment, key, read_json, save_tensor, sync, write_json


class RoadmapB(RoadmapA):
    def __init__(self, cfg, variant="adaptive", seed=1337):
        self.bcfg = cfg
        self.policy = frozen_policy(cfg)
        super().__init__(encoder_config(cfg))
        self.text.requires_grad_(False)
        self.audio.requires_grad_(False)
        self.audio_head.requires_grad_(False)
        self.text_policy = read_json(output(cfg) / "text_policy.json")
        assert self.text_policy["dtype"] == "float32"
        self.residual = None
        self.temperature = self.calibration["fusion"]
        if variant != "A_replay":
            self.residual, _ = load_model(output(cfg) / "runs" / variant / str(seed) / "checkpoint.pt")
            self.residual.to(self.device).eval().requires_grad_(False)
            self.temperature = read_json(
                output(cfg) / "evaluation" / variant / str(seed) / "calibration.json"
            )["temperature"]
        self.variant, self.seed = variant, seed
        self.cfg["runtime"]["dtype"] = self.policy["dtype"]

    def parameter_ledger(self):
        components = {
            "selected_text": counts(self.text),
            "frozen_wavlm": counts(self.audio),
            "audio_head": counts(self.audio_head),
            "residual": counts(self.residual) if self.residual is not None else {"total": 0, "trainable": 0},
        }
        total = sum(c["total"] for c in components.values()) + 5
        return {
            "components": components,
            "learned_scalars": {
                "original_alpha": 1,
                "original_diagnostic_temperatures": 3,
                "B_temperature": 1,
            },
            "total_required_parameters": total,
            "within_6b": total <= 6_000_000_000,
            "note": "Conservative count retains all loaded A diagnostic scalars; no quantization discount",
        }

    @torch.inference_mode()
    def predict_turn(self, text, audio, sample_rate, history, *, speaker="user", return_internal=False):
        sync(self.device)
        started = time.perf_counter()
        item = encode_turn(
            self.tokenizer,
            text,
            history,
            speaker,
            self.text_metadata["max_length"],
            self.text_metadata["history_turns"],
        )
        batch = {k: v.to(self.device) for k, v in collate_text([item], self.tokenizer.pad_token_id).items()}
        with autocast({**self.cfg, "runtime": {**self.cfg["runtime"], "dtype": "float32"}}, self.device):
            zt, ht = self.text(**batch)
        zt, ht = zt.float(), ht.float()
        sync(self.device)
        text_end = time.perf_counter()
        audio_valid = audio is not None
        reason = "missing_or_invalid" if not audio_valid else None
        ha, za = torch.zeros_like(ht), torch.zeros_like(zt)
        available = False
        if audio_valid:
            wave = prepare_waveform(audio, sample_rate)
            if len(wave) / 16000 > self.policy["max_audio_seconds"]:
                reason = "duration_limit"
            else:
                ha = encode_canonical(self.audio, [wave], self.policy).to(self.device)
                za = self.audio_head(ha.float())
                available = True
        sync(self.device)
        audio_end = time.perf_counter()
        if self.residual is None:
            fused = (1 - self.selection["alpha"]) * zt + self.selection["alpha"] * za if available else zt
            result = {
                "logits": fused,
                "gate": torch.tensor([self.selection["alpha"] if available else 0.0]),
                "delta": torch.zeros_like(zt),
            }
        else:
            result = self.residual(ht, ha, zt, za, torch.tensor([available], device=self.device))
        logits = result["logits"].cpu()
        if available:
            p = probabilities(logits[0], self.temperature)
            influence = 0.5 * np.abs(p - probabilities(zt[0].cpu(), self.temperature)).sum()
        else:
            p = probabilities(zt[0].cpu(), self.calibration["text"])
            influence = 0.0
        emotion = LABELS[int(p.argmax())]
        response = respond(text, emotion, float(p.max()))
        sync(self.device)
        ended = time.perf_counter()
        final = {
            "emotion": emotion,
            "confidence": float(p.max()),
            "distribution": dict(zip(LABELS, p.tolist())),
            "audio_valid": audio_valid,
            "audio_available": available,
            "audio_unavailable_reason": reason,
            "text_prediction": LABELS[int(zt.argmax())],
            "audio_prediction": LABELS[int(za.argmax())] if available else None,
            "gate": float(result["gate"][0]) if available else 0.0,
            "audio_influence": float(influence),
            "response": response,
            "delta_l2": float(result["delta"].norm(dim=-1)[0]) if available else 0.0,
            "delta_max_abs": float(result["delta"].abs().max()) if available else 0.0,
            "current_truncated": item["current_truncated"],
            "latency_ms": {
                "text": (text_end - started) * 1000,
                "audio": (audio_end - text_end) * 1000,
                "fusion_response": (ended - audio_end) * 1000,
                "total": (ended - started) * 1000,
            },
        }
        if return_internal:
            return final, {
                "ht": ht.cpu(),
                "ha": ha.cpu(),
                "zt": zt.cpu(),
                "za": za.cpu(),
                "logits": logits,
                "available": torch.tensor([available]),
            }
        return final


def add_noise(wave, snr_db, seed):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(wave)).astype(np.float32)
    signal_rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
    if signal_rms == 0:
        return wave.copy(), {"noise_added": False, "reason": "input_digital_silence"}
    noise *= signal_rms / (10 ** (snr_db / 20) * np.sqrt(np.mean(noise.astype(np.float64) ** 2)))
    return wave + noise, {"noise_added": True, "snr_db": snr_db, "seed": seed}


def raw_validation(cfg):
    verify_baseline(cfg)
    verify_freeze(cfg)
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name():
        raise RuntimeError("Final raw validation/benchmark requires the actual RTX 4090")
    root = output(cfg)
    dest = root / "raw_validation"
    dest.mkdir(parents=True, exist_ok=False)
    all_rows, cached = load_inputs(cfg, "dev_model")
    cache_index = {key(r): i for i, r in enumerate(all_rows)}
    hist = histories(all_rows, 3)
    selected_keys = read_json(root / "raw_probe_manifest.json")["keys"]
    rows = [all_rows[cache_index[k]] for k in selected_keys]
    # Inputs are loaded before timing, matching A's recorded benchmark convention.
    waves = {key(r): read_waveform(waveform_path(cfg, r)) for r in rows}
    first = rows[0]
    a_started = time.perf_counter()
    a_model = RoadmapB(cfg, variant="A_replay")
    sync(a_model.device)
    a_load_s = time.perf_counter() - a_started
    a_model.predict_turn(first["text"], waves[key(first)], 16000, hist[key(first)], speaker=first["speaker"])
    torch.cuda.reset_peak_memory_stats()
    a_measurements = []
    a_rss = []
    for row in rows:
        for repeat in range(cfg["evaluation"]["benchmark_repeats"]):
            value = a_model.predict_turn(
                row["text"], waves[key(row)], 16000, hist[key(row)], speaker=row["speaker"]
            )
            a_measurements.append(
                {
                    "key": key(row),
                    "repeat": repeat,
                    "latency_ms": value["latency_ms"],
                    "duration_s": row["duration_s"],
                }
            )
            a_rss.append(psutil.Process().memory_info().rss)
    a_runtime = {
        "model_load_s": a_load_s,
        "samples": a_measurements,
        "scope": "A-replay canonical FP32 text/BF16 singleton audio on exactly the B probe workload; no A retuning",
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_sampled_process_rss_bytes": max(a_rss),
        "summary_ms": {
            n: {
                "p50": float(np.median([x["latency_ms"][n] for x in a_measurements])),
                "p95": float(np.quantile([x["latency_ms"][n] for x in a_measurements], 0.95)),
            }
            for n in ("text", "audio", "fusion_response", "total")
        },
    }
    write_json(dest / "A_same_workload_runtime.json", a_runtime)
    del a_model
    gc.collect()
    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    model = RoadmapB(cfg)
    sync(model.device)
    load_s = time.perf_counter() - t0
    model.predict_turn(first["text"], waves[key(first)], 16000, hist[key(first)], speaker=first["speaker"])
    torch.cuda.reset_peak_memory_stats()
    rss, measurements, probes, comparisons, raw_features = [], [], [], [], {}
    allclose = True
    for row in rows:
        k, i = key(row), cache_index[key(row)]
        result, raw = model.predict_turn(
            row["text"], waves[k], 16000, hist[k], speaker=row["speaker"], return_internal=True
        )
        subset = {n: v[i : i + 1] for n, v in cached.items()}
        offline = predict(model.residual.cpu(), subset)
        model.residual.to(model.device)
        cached_p = distributions(offline["logits"], subset, model.temperature, model.calibration["text"])[0]
        raw_p = np.array(list(result["distribution"].values()))
        emb_ok = bool(
            torch.allclose(
                raw["ha"][0],
                cached["ha"][i],
                atol=model.policy["embedding_atol"],
                rtol=model.policy["embedding_rtol"],
            )
        )
        prob_delta = float(np.max(np.abs(raw_p - cached_p)))
        text_ok = bool(
            torch.allclose(
                raw["ht"][0],
                cached["ht"][i],
                atol=model.text_policy["embedding_atol"],
                rtol=model.text_policy["embedding_rtol"],
            )
        )
        text_prob_delta = float(np.abs(probabilities(raw["zt"][0]) - probabilities(cached["zt"][i])).max())
        ok = (
            emb_ok
            and text_ok
            and text_prob_delta <= model.text_policy["probability_atol"]
            and prob_delta <= model.policy["probability_atol"]
        )
        allclose &= ok
        comparisons.append(
            {
                "key": k,
                "waveform_sha256": digest(waveform_path(cfg, row)),
                "embedding_allclose": emb_ok,
                "text_embedding_allclose": text_ok,
                "text_probability_max_abs": text_prob_delta,
                "fused_probability_max_abs": prob_delta,
                "argmax_changed": bool(raw_p.argmax() != cached_p.argmax()),
                "passed": ok,
            }
        )
        raw_features[k] = raw
        for repeat in range(cfg["evaluation"]["benchmark_repeats"]):
            timed = model.predict_turn(row["text"], waves[k], 16000, hist[k], speaker=row["speaker"])
            measurements.append(
                {
                    "key": k,
                    "repeat": repeat,
                    "duration_s": row["duration_s"],
                    "latency_ms": timed["latency_ms"],
                    "rtf": timed["latency_ms"]["total"] / 1000 / row["duration_s"],
                }
            )
            rss.append(psutil.Process().memory_info().rss)
        noisy, noise_meta = add_noise(waves[k], cfg["evaluation"]["noise_snr_db"], cfg["seed"] + i)
        conditions = [
            ("matched", waves[k], {}),
            ("silence", np.zeros_like(waves[k]), {}),
            ("noise", noisy, noise_meta),
        ]
        for condition, wave, metadata in conditions:
            value = model.predict_turn(row["text"], wave, 16000, hist[k], speaker=row["speaker"])
            probes.append(
                {
                    "key": k,
                    "label": row["label"],
                    "condition": condition,
                    "prediction": value,
                    "metadata": metadata,
                }
            )
    ledger = model.parameter_ledger()
    if not ledger["within_6b"]:
        raise AssertionError("Parameter budget exceeded")
    save_tensor(dest / "raw_features.pt", raw_features)
    write_json(
        dest / "cache_consistency.json",
        {"passed": bool(allclose), "tolerances": model.policy, "cases": comparisons},
    )
    write_json(
        dest / "interventions.json",
        {
            "scope": "Predetermined raw development probe set, not overall MELD performance",
            "noise_snr_db": cfg["evaluation"]["noise_snr_db"],
            "cases": probes,
        },
    )
    write_json(dest / "parameter_ledger.json", ledger)
    write_json(
        dest / "runtime.json",
        {
            "environment": environment(model.cfg),
            "model_load_s": load_s,
            "samples": measurements,
            "summary_ms": {
                n: {
                    "p50": float(np.median([x["latency_ms"][n] for x in measurements])),
                    "p95": float(np.quantile([x["latency_ms"][n] for x in measurements], 0.95)),
                }
                for n in ("text", "audio", "fusion_response", "total")
            },
            "median_rtf": float(np.median([m["rtf"] for m in measurements])),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_sampled_process_rss_bytes": max(rss),
            "excluded": ["recording", "disk read", "endpointing", "ASR"],
            "A_same_workload_runtime": a_runtime,
            "comparison_caution": "Same-workload A and B are measured sequentially; historical A uses its original sample",
            "A_original_runtime": read_json(Path(cfg["baseline_root"]) / "artifacts/runtime.json")[
                "summary_ms"
            ],
        },
    )
    # Verify fallback on a real transcript; no zero features are passed as available audio.
    fallback, internal = model.predict_turn(
        first["text"], None, None, hist[key(first)], speaker=first["speaker"], return_internal=True
    )
    expected = probabilities(internal["zt"][0], model.calibration["text"])
    assert np.array_equal(np.array(list(fallback["distribution"].values())), expected)
    write_json(dest / "fallback.json", {"passed": True, "prediction": fallback})
    verify_baseline(cfg)
    if not allclose:
        raise RuntimeError(
            "Raw/cache numerical consistency failed; inspect report without loosening tolerances"
        )


def delivery_diagnostic(cfg, manifest):
    manifest = Path(manifest).resolve()
    records = read_json(manifest)["recordings"]
    if not records:
        raise ValueError("Provide real consented recordings; no generated benchmark claims")
    groups = {}
    for row in records:
        groups.setdefault(row["pair_id"], []).append(row)
    if any(len(group) < 2 or len({r["text"] for r in group}) != 1 for group in groups.values()):
        raise ValueError("Every delivery pair needs at least two recordings with exactly the same words")
    model = RoadmapB(cfg)
    outputs = []
    for row in records:
        if row.get("speaker_consented") is not True:
            raise ValueError("Every diagnostic recording needs explicit speaker consent")
        wave = read_waveform(manifest.parent / row["audio"])
        prediction = model.predict_turn(row["text"], wave, 16000, [], speaker=row["speaker"])
        outputs.append(
            {**row, "waveform_sha256": digest(manifest.parent / row["audio"]), "prediction": prediction}
        )
    dest = output(cfg) / "delivery_diagnostic.json"
    if dest.exists():
        raise ValueError("Do not overwrite or tune on delivery diagnostic results")
    write_json(
        dest,
        {
            "scope": "Consented same-words/different-delivery diagnostic, not a MELD benchmark",
            "manifest_sha256": digest(manifest),
            "recordings": outputs,
        },
    )


def main():
    p = parser(__doc__)
    p.add_argument("action", choices=("validate", "delivery", "predict"))
    p.add_argument("--manifest")
    p.add_argument("--audio")
    p.add_argument("--text")
    p.add_argument("--history")
    args = p.parse_args()
    cfg = settings(args)
    if args.action == "validate":
        raw_validation(cfg)
    elif args.action == "delivery":
        delivery_diagnostic(cfg, args.manifest)
    else:
        from .context import Turn
        from .session import Session

        history = [Turn(**r) for r in read_json(args.history)] if args.history else []
        session = Session(RoadmapB(cfg), history=history)
        session.start()
        session.begin_turn()
        if args.audio:
            for chunk in np.array_split(read_waveform(args.audio), 8):
                session.push_audio(chunk)
        session.set_transcript(args.text)
        print(__import__("json").dumps(session.end_turn(), indent=2))


if __name__ == "__main__":
    main()

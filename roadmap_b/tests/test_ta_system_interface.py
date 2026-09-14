import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.ta_interface import SystemPredictor, SystemSession
from src.ta_policy import select_response

ROOT = Path(__file__).resolve().parents[2]
CASES = json.loads((ROOT / "roadmap_b/evidence/ta_system_closure/POLICY_CASES.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_prospective_fixed_policy_cases(case):
    state, evidence = copy.deepcopy(case["state"]), copy.deepcopy(case["evidence"])
    result = select_response(case["text"], state, evidence, policy="role_specific_h1")
    assert result["route_id"] == case["expected_route"]
    assert state == case["state"] and evidence == case["evidence"]
    if result["route_id"] != "b_policy":
        assert result["response"] != result["reference_response"]
    assert select_response(case["text"], state, evidence)["route_id"] == "b_policy"


@pytest.mark.parametrize("mutation", ["namespace", "extra_surprise", "nan", "negative", "sum"])
def test_malformed_evidence_fails_closed(mutation):
    case = copy.deepcopy(CASES[0])
    evidence = case["evidence"]
    if mutation == "namespace":
        evidence["namespace"] = "MELD7"
    elif mutation == "extra_surprise":
        evidence["distribution"]["surprise"] = 0
    elif mutation == "nan":
        evidence["distribution"]["angry"] = float("nan")
    elif mutation == "negative":
        evidence["distribution"]["neutral"] = -0.1
    else:
        evidence["distribution"]["angry"] = 0.5
    with pytest.raises(ValueError):
        select_response(case["text"], case["state"], evidence, policy="role_specific_h1")


class Dummy:
    def __init__(self):
        self.fail = False
        self.calls = []

    def predict_turn(self, text, audio, sample_rate, history, *, speaker):
        self.calls.append((text, audio, sample_rate, list(history)))
        if self.fail:
            raise RuntimeError("Injected prediction failure")
        evidence = copy.deepcopy(CASES[0]["evidence"])
        evidence["available"] = audio is not None and bool(np.isfinite(audio).all())
        evidence["unavailable_reason"] = None if evidence["available"] else "invalid_or_missing"
        if not evidence["available"]:
            evidence["distribution"] = None
        return {
            "meld_state": {
                "emotion": "neutral",
                "confidence": 0.8,
                "distribution": {"neutral": 0.8},
                "response": "baseline",
            },
            "delivery_evidence": evidence,
            "latency_ms": {"total": 1},
        }


def session():
    model = Dummy()
    adapter = SystemPredictor(model, {"release_id": "test"}, policy="role_specific_h1")
    result = SystemSession(adapter)
    result.start()
    return model, adapter, result


def test_temporal_actions_do_not_become_history_and_snapshots_are_immutable():
    model, adapter, s = session()
    for i in range(5):
        s.begin_turn()
        s.push_audio(np.zeros(200, np.float32))
        s.push_audio(np.ones(200, np.float32))
        s.set_transcript(f"Observed {i}")
        snapshot = s.end_turn()
        d = snapshot.to_dict()
        assert [x["text"] for x in d["observed_context"]] == [
            f"Observed {j}" for j in range(max(0, i - 3), i)
        ]
        assert d["response"] == d["interaction"]["response"]
        assert d["response"] != d["meld_state"]["response"]
        assert d["interaction"]["route_id"] == "vocal_angry"
        d["response"] = "mutated"
        assert snapshot.to_dict()["response"] != "mutated"
        adapter.acknowledge_emitted(d["turn_id"])
        with pytest.raises(ValueError, match="already acknowledged"):
            adapter.acknowledge_emitted(d["turn_id"])
    assert len(adapter.emitted_actions) == 5
    assert len(model.calls[-1][1]) == 400
    assert len(s.history) == 3


def test_failure_aborts_buffers_without_promoting_history_and_next_turn_is_fresh():
    model, adapter, s = session()
    s.begin_turn()
    s.push_audio(np.ones(400, np.float32))
    s.set_transcript("failed")
    model.fail = True
    with pytest.raises(RuntimeError, match="Injected"):
        s.end_turn()
    assert not s.active and s.chunks == [] and s.transcript is None
    assert s.history == [] and adapter.snapshots == []
    model.fail = False
    s.begin_turn()
    s.set_transcript("recovered")
    value = s.end_turn().to_dict()
    assert value["turn_id"] == "turn-000001"
    assert model.calls[-1][1] is None
    assert value["interaction"]["route_id"] == "b_policy"


def test_corrupt_audio_is_distinct_from_missing_and_retains_observed_history():
    model, adapter, s = session()
    s.begin_turn()
    s.mark_decode_failed()
    s.set_transcript("corrupt recording")
    value = s.end_turn().to_dict()
    assert not np.isfinite(model.calls[-1][1]).all()
    assert not value["acoustic_evidence"]["available"]
    assert value["interaction"]["route_id"] == "b_policy"
    assert s.history[0].text == "corrupt recording"
    s.begin_turn()
    s.set_transcript("missing recording")
    s.end_turn()
    assert model.calls[-1][1] is None


def test_cli_help_is_cwd_independent_and_does_not_require_assets(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "ta.py"), "--help"], cwd=tmp_path, capture_output=True, text=True
    )
    assert proc.returncode == 0
    assert "--asset-root" in proc.stdout and "--replay" in proc.stdout


def cli_module():
    spec = importlib.util.spec_from_file_location("ta_cli_test", ROOT / "ta.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_replay_paths_and_human_history_contract(tmp_path):
    mod = cli_module()
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps({"turns": [{"text": "hello", "audio": "sample.wav"}]}))
    args = mod.parser().parse_args(["--asset-root", "assets", "--replay", str(replay)])
    assert mod.read_turns(args)[0]["audio"] == tmp_path / "sample.wav"
    replay.write_text(json.dumps({"turns": [{"text": "generated", "role": "assistant"}]}))
    with pytest.raises(ValueError, match="observed human"):
        mod.read_turns(args)
    args.chunk_samples = 0
    with pytest.raises(ValueError, match="positive"):
        mod.read_turns(args)


def test_output_preservation_checked_before_loading(tmp_path):
    mod = cli_module()
    target = tmp_path / "existing.json"
    target.write_text("retain")
    args = mod.parser().parse_args(["--asset-root", "absent", "--text", "hello", "--output", str(target)])
    with pytest.raises(ValueError, match="already exists"):
        mod.read_turns(args)
    assert target.read_text() == "retain"

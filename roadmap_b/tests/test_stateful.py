"""Causal state contracts; no learned models, external media or model downloads."""

from dataclasses import FrozenInstanceError
import hashlib
import json
import threading

import numpy as np
import pytest

from src.context import Turn
from src.interaction import Limits, ObservationEvent, PerceptionSnapshot, StatefulSession
from src.interaction.replay import ReplayRecorder, read_events, replay, write_events

LABELS = ("neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust")
LABELS6 = ("neutral", "happy", "sad", "angry", "fear", "disgust")
MANIFEST = {"release_id": "synthetic-state-contract", "condition": "D2"}
REJECTION = (ValueError, RuntimeError)


class FrozenModelDouble:
    """Deterministic actual-input function with intentionally volatile timing."""

    numerical_scope = "synthetic software fixture"

    def __init__(self):
        self.calls, self.callback, self.fail = [], None, False

    def predict_turn(self, text, audio, sample_rate, history, *, speaker):
        pcm = None if audio is None else np.array(audio, copy=True)
        self.calls.append((text, pcm, sample_rate, tuple(history), speaker))
        if self.callback:
            self.callback()
        if self.fail:
            raise RuntimeError("injected inference failure")
        valid = pcm is not None and bool(np.isfinite(pcm).all()) and len(pcm) >= 400
        available = valid and len(pcm) <= 960000
        reason = None if available else "duration_limit" if valid else "missing_or_invalid"
        encoded = json.dumps([text, speaker, [(t.text, t.speaker) for t in history]]).encode()
        if available:
            encoded += pcm.astype("<f4").tobytes()
        chosen = hashlib.sha256(encoded).digest()[0] % 7
        p = [0.01] * 7
        p[chosen] = 0.94
        timing = float(len(self.calls))
        return {
            "meld_state": {
                "emotion": LABELS[chosen],
                "confidence": 0.94,
                "distribution": dict(zip(LABELS, p)),
                "audio_valid": valid,
                "audio_available": available,
                "audio_unavailable_reason": reason,
                "response": "legacy authored reference",
                "latency_ms": {"total": timing},
            },
            "delivery_evidence": {
                "available": available,
                "unavailable_reason": "invalid_audio" if pcm is not None and not valid else reason,
                "namespace": "crema6_audio_votes_v1",
                "distribution": dict(zip(LABELS6, [0.02, 0.02, 0.02, 0.9, 0.02, 0.02]))
                if available
                else None,
                "normalized_128d": [0.125] * 128 if available else None,
                "provenance": {"condition": "D2", "model_sha256": "synthetic"},
            },
            "latency_ms": {"total": timing, "text": timing / 2},
        }


def session(**kwargs):
    model = kwargs.pop("model", None) or FrozenModelDouble()
    kwargs.setdefault("session_id", "s")
    s = StatefulSession(model, MANIFEST, clock=lambda: 0.0, **kwargs)
    s.start()
    return s, model


def begin(s, speaker="A"):
    return s.start_turn(s.register_participant(speaker))


def finish(s, text="Observed human words.", speaker="A", audio=None):
    turn = begin(s, speaker)
    if audio is not None:
        s.observe_audio_chunk(turn, audio)
    s.finalize_text(turn, text)
    return turn, s.end_turn(turn)


def state(s):
    return json.dumps(s.to_dict(), sort_keys=True, allow_nan=False)


def reject_unchanged(s, action):
    before = state(s)
    with pytest.raises(REJECTION):
        action()
    assert state(s) == before


def no_final(s, turn):
    latest = s.get_latest_perception(turn)
    assert latest is None or latest.to_dict()["stage"] != "final"


def prefix_events():
    make = ObservationEvent.create
    return [
        make("replay-s", 1, "SESSION_STARTED"),
        make(
            "replay-s",
            2,
            "PARTICIPANT_REGISTERED",
            participant_id="participant-000001",
            payload={"external_key": "A"},
        ),
        make("replay-s", 3, "HUMAN_TURN_STARTED", participant_id="participant-000001", turn_id="turn-000003"),
        make(
            "replay-s", 4, "TEXT_FRAGMENT_OBSERVED", turn_id="turn-000003", payload={"text": "Shared words"}
        ),
        make("replay-s", 5, "AUDIO_CHUNK_OBSERVED", turn_id="turn-000003", pcm=np.zeros(400, np.float32)),
    ]


def test_partial_checkpoint_deep_immutability_and_no_model_call():
    s, model = session()
    turn = begin(s)
    s.observe_text_fragment(turn, "first")
    old = s.get_latest_perception(turn)
    assert isinstance(old, PerceptionSnapshot)
    frozen = old.payload_json
    view = old.to_dict()
    assert view["stage"] == "partial"
    assert view.get("B_output") is None and view.get("D2_output") is None
    view["observed_input_summary"]["caller_mutation"] = True
    s.observe_text_fragment(turn, " second")
    s.observe_audio_chunk(turn, np.ones(400, np.float32))
    assert old.payload_json == frozen
    assert "second" not in frozen and "caller_mutation" not in old.payload_json
    assert model.calls == [] and s.history == ()
    with pytest.raises((FrozenInstanceError, AttributeError)):
        old.payload_json = "changed"
    reject_unchanged(s, lambda: s.get_response(turn))


def test_final_text_correction_and_audio_after_final_text():
    s, model = session()
    turn = begin(s)
    s.observe_text_fragment(turn, "incorrect partial")
    old = s.get_latest_perception(turn)
    frozen = old.payload_json
    s.finalize_text(turn, "correct observed final")
    reject_unchanged(s, lambda: s.observe_text_fragment(turn, "late fragment"))
    s.observe_audio_chunk(turn, np.zeros(400, np.float32))
    final = s.end_turn(turn)
    assert final.to_dict()["stage"] == "final"
    assert model.calls[-1][0] == "correct observed final"
    assert old.payload_json == frozen and "incorrect partial" in frozen
    assert s.history == (Turn("correct observed final", "A"),)


def test_whole_pcm_is_exact_across_chunking_and_caller_mutation():
    left, lm = session()
    right, rm = session()
    a, b = begin(left), begin(right)
    expected = np.linspace(-0.5, 0.5, 903, dtype=np.float32)
    chunk = expected[:217].copy()
    left.observe_audio_chunk(a, chunk)
    chunk[:] = 123
    left.observe_audio_chunk(a, expected[217:])
    left.finalize_text(a, "same text")
    right.finalize_text(b, "same text")
    right.observe_audio_chunk(b, expected)
    left.end_turn(a)
    right.end_turn(b)
    np.testing.assert_array_equal(lm.calls[-1][1], expected)
    np.testing.assert_array_equal(lm.calls[-1][1], rm.calls[-1][1])
    assert lm.calls[-1][2:] == rm.calls[-1][2:]


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((3, 2)),
        np.arange(4),
        np.array([np.nan]),
        np.array([np.inf]),
    ],
)
def test_invalid_pcm_rejection_is_atomic(invalid):
    s, model = session()
    turn = begin(s)
    reject_unchanged(s, lambda: s.observe_audio_chunk(turn, invalid))
    assert model.calls == []


@pytest.mark.parametrize("kind", ["missing", "corrupt", "silence", "over_limit"])
def test_audio_availability_and_release(kind):
    s, model = session()
    turn = begin(s)
    if kind == "corrupt":
        s.mark_audio_unavailable(turn)
    elif kind == "silence":
        s.observe_audio_chunk(turn, np.zeros(400, np.float32))
    elif kind == "over_limit":
        for _ in range(15):
            s.observe_audio_chunk(turn, np.zeros(64000, np.float32))
        s.observe_audio_chunk(turn, np.zeros(1, np.float32))
    s.finalize_text(turn, "Observed words.")
    value = s.end_turn(turn).to_dict()
    assert value["D2_output"]["available"] == (kind == "silence")
    assert (
        value["D2_output"]["unavailable_reason"]
        == {
            "missing": "missing_or_invalid",
            "corrupt": "invalid_audio",
            "silence": None,
            "over_limit": "duration_limit",
        }[kind]
    )
    if kind == "missing":
        assert model.calls[-1][1] is None
    if kind == "over_limit":
        assert len(model.calls[-1][1]) == 960001
    assert s.retained_counts()["audio_bytes"] == 0


def test_failure_is_observed_but_not_promoted_to_t1():
    s, model = session()
    turn = begin(s)
    s.observe_audio_chunk(turn, np.zeros(400, np.float32))
    s.finalize_text(turn, "actually observed before failure")
    model.fail = True
    with pytest.raises(RuntimeError, match="injected"):
        s.end_turn(turn)
    assert s.history == ()
    assert "actually observed before failure" in json.dumps(s.to_dict()["observations"])
    assert "failed" in json.dumps(s.to_dict()["execution"])
    no_final(s, turn)
    model.fail = False
    finish(s, "fresh successful turn")
    assert model.calls[-1][3] == ()
    assert s.history == (Turn("fresh successful turn", "A"),)


def test_three_successful_humans_across_participants_and_no_robot_feedback():
    s, model = session()
    expected = []
    for i, speaker in enumerate(["A", "B", "A", "B", "A", "B"]):
        text = f"human {i}"
        turn, snapshot = finish(s, text, speaker)
        assert model.calls[-1][3] == tuple(expected[-3:])
        assert model.calls[-1][4] == speaker
        expected.append(Turn(text, speaker))
        before = state(s)
        response = s.get_response(turn)
        assert response == snapshot.to_dict()["proposed_action"]["response"]
        assert state(s) == before
        s.record_robot_emitted(turn)
    assert s.history == tuple(expected[-3:])


def test_initial_context_remains_exact():
    initial = [Turn("earlier A", "A"), Turn("earlier B", "B")]
    s, model = session(history=initial)
    finish(s, "current A", "A")
    assert model.calls[-1][3] == tuple(initial)
    assert model.calls[-1][4] == "A"


def test_robot_draft_is_not_emission_and_emission_occurs_once():
    s, _ = session()
    turn, _ = finish(s)
    reject_unchanged(s, lambda: s.record_robot_emitted(turn, "private substituted draft"))
    s.record_robot_emitted(turn)
    reject_unchanged(s, lambda: s.record_robot_emitted(turn))
    assert "private substituted draft" not in state(s)


def test_prior_response_can_be_emitted_after_unrelated_later_observation():
    s, _ = session()
    turn, snap = finish(s, "earlier")
    finish(s, "later", "B")
    s.record_robot_emitted(turn, snap.to_dict()["proposed_action"]["response"])
    assert [t.text for t in s.history] == ["earlier", "later"]


def test_model_evidence_never_enters_observation_namespace_or_speaker_key():
    s, model = session()
    _, snap = finish(s, "human source words", audio=np.zeros(400, np.float32))
    observed = json.dumps(s.to_dict()["observations"], sort_keys=True)
    for forbidden in ("normalized_128d", "crema6_audio_votes_v1", "distribution", "model_sha256"):
        assert forbidden not in observed
    assert snap.to_dict()["D2_output"]["namespace"] == "crema6_audio_votes_v1"
    assert s.register_participant("A") == "participant-000001"
    assert model.calls[-1][4] == "A"
    assert s.history[0].text == "human source words"


def test_participant_and_session_isolation():
    left, lm = session(session_id="left")
    right, rm = session(session_id="right")
    a, b = left.register_participant("A"), left.register_participant("B")
    assert a != b and left.register_participant("A") == a
    assert right.register_participant("A") == "participant-000001"
    finish(left, "left-only text")
    finish(right, "right-only text")
    assert lm.calls[-1][3] == rm.calls[-1][3] == ()
    assert "left-only text" not in state(right) and "right-only text" not in state(left)


def test_cancelled_turn_rejects_finalization_chunks_and_emission():
    s, model = session()
    turn = begin(s)
    s.observe_text_fragment(turn, "cancelled observed fragment")
    s.cancel_turn(turn)
    for action in (
        lambda: s.finalize_text(turn, "late final"),
        lambda: s.end_turn(turn),
        lambda: s.observe_audio_chunk(turn, np.zeros(400, np.float32)),
        lambda: s.record_robot_emitted(turn),
    ):
        reject_unchanged(s, action)
    assert model.calls == [] and s.history == ()
    finish(s, "subsequent success")
    assert model.calls[-1][3] == ()


@pytest.mark.parametrize("operation", ["text", "audio", "cancel", "end"])
def test_finalized_turn_rejects_mutation(operation):
    s, _ = session()
    turn, _ = finish(s)
    actions = {
        "text": lambda: s.observe_text_fragment(turn, "future"),
        "audio": lambda: s.observe_audio_chunk(turn, np.zeros(400, np.float32)),
        "cancel": lambda: s.cancel_turn(turn),
        "end": lambda: s.end_turn(turn),
    }
    reject_unchanged(s, actions[operation])


def test_overlap_and_incomplete_end_reject_atomically():
    s, model = session()
    participant = s.register_participant("A")
    turn = s.start_turn(participant)
    reject_unchanged(s, lambda: s.start_turn(participant))
    reject_unchanged(s, lambda: s.end_turn(turn))
    assert model.calls == []


def test_decode_failure_before_pcm_and_no_pcm_after_decode_failure():
    s, _ = session()
    turn = begin(s)
    s.observe_audio_chunk(turn, np.zeros(400, np.float32))
    reject_unchanged(s, lambda: s.mark_audio_unavailable(turn))
    s.cancel_turn(turn)
    turn = begin(s)
    s.mark_audio_unavailable(turn)
    reject_unchanged(s, lambda: s.observe_audio_chunk(turn, np.zeros(400, np.float32)))


@pytest.mark.parametrize("disposition", ["cancel", "close"])
def test_inflight_completion_cannot_commit_after_cancellation_or_close(disposition):
    s, model = session()
    turn = begin(s)
    s.finalize_text(turn, "observed before interruption")
    model.callback = (lambda: s.cancel_turn(turn)) if disposition == "cancel" else s.close
    assert s.end_turn(turn) is None
    assert s.history == ()
    if disposition == "cancel":
        no_final(s, turn)
        model.callback = None
        finish(s, "new current turn")
        assert model.calls[-1][3] == ()


def test_pending_model_rejects_another_active_turn():
    s, model = session()
    participant = s.register_participant("A")
    turn = s.start_turn(participant)
    s.finalize_text(turn, "in flight")

    def during_prediction():
        reject_unchanged(s, lambda: s.start_turn(participant))

    model.callback = during_prediction
    assert s.end_turn(turn).to_dict()["stage"] == "final"


def test_retained_duplicate_idempotency_and_conflicting_duplicate():
    s = StatefulSession(FrozenModelDouble(), MANIFEST, session_id="replay-s", clock=lambda: 0.0)
    event = prefix_events()[0]
    s.ingest(event)
    before = state(s)
    s.ingest(event)
    assert state(s) == before
    conflict = ObservationEvent.create("replay-s", 1, "SESSION_STARTED", timestamp=1.0)
    reject_unchanged(s, lambda: s.ingest(conflict))


@pytest.mark.parametrize("mutation", ["foreign", "gap", "backwards", "nan", "unknown"])
def test_envelope_rejection_never_advances_state(mutation):
    s, _ = session()
    options = dict(
        session_id="s",
        event_seq=2,
        event_type="PARTICIPANT_REGISTERED",
        timestamp=0.0,
        participant_id="participant-000001",
        payload={"external_key": "A"},
    )
    if mutation == "foreign":
        options["session_id"] = "other"
    elif mutation == "gap":
        options["event_seq"] = 3
    elif mutation == "backwards":
        options["timestamp"] = -1.0
    elif mutation == "nan":
        options["timestamp"] = float("nan")
    elif mutation == "unknown":
        options["event_type"] = "MODEL_PREDICTED_SADNESS"
    reject_unchanged(s, lambda: s.ingest(ObservationEvent.create(**options)))
    assert s.register_participant("A") == "participant-000001"


def test_expired_duplicate_is_stale_and_never_reapplied():
    s = StatefulSession(
        FrozenModelDouble(), MANIFEST, session_id="replay-s", clock=lambda: 0.0, limits=Limits(dedup_events=2)
    )
    events = prefix_events()
    for event in events:
        s.ingest(event)
    reject_unchanged(s, lambda: s.ingest(events[0]))


def test_event_pcm_is_copied_to_immutable_bytes():
    pcm = np.arange(400, dtype=np.float32)
    event = ObservationEvent.create("s", 4, "AUDIO_CHUNK_OBSERVED", turn_id="turn-000003", pcm=pcm)
    frozen = event.pcm_bytes
    pcm[:] = -1
    np.testing.assert_array_equal(np.frombuffer(frozen, dtype="<f4"), np.arange(400, dtype=np.float32))
    assert isinstance(frozen, bytes)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        event.pcm_bytes = b"changed"


def test_shared_prefix_counterfactual_and_future_audio_text_isolation():
    left = StatefulSession(FrozenModelDouble(), MANIFEST, session_id="replay-s", clock=lambda: 0.0)
    right = StatefulSession(FrozenModelDouble(), MANIFEST, session_id="replay-s", clock=lambda: 0.0)
    for event in prefix_events():
        left.ingest(event)
        right.ingest(event)
        assert state(left) == state(right)
    turn = "turn-000003"
    a, b = left.get_latest_perception(turn), right.get_latest_perception(turn)
    frozen = a.payload_json
    assert frozen == b.payload_json
    for s, suffix, sign in ((left, "future A", 1), (right, "future B", -1)):
        s.observe_text_fragment(turn, suffix)
        s.observe_audio_chunk(turn, np.full(400, sign, np.float32))
        s.finalize_text(turn, "Shared words " + suffix)
        s.end_turn(turn)
    assert a.payload_json == b.payload_json == frozen


def test_jsonl_replay_preserves_pcm_identities_and_excludes_volatile_timing(tmp_path):
    events = prefix_events() + [
        ObservationEvent.create(
            "replay-s", 6, "TEXT_FINALIZED", turn_id="turn-000003", payload={"text": "Shared words complete"}
        ),
        ObservationEvent.create("replay-s", 7, "HUMAN_TURN_ENDED", turn_id="turn-000003"),
    ]
    path = tmp_path / "events.jsonl"
    write_events(path, events)
    loaded = list(read_events(path))
    assert [e.event_id for e in loaded] == [e.event_id for e in events]
    assert loaded[4].pcm_bytes == events[4].pcm_bytes
    model = FrozenModelDouble()
    first, second = replay(loaded, model, MANIFEST), replay(loaded, model, MANIFEST)
    assert (
        first.get_latest_perception("turn-000003").payload_json
        == second.get_latest_perception("turn-000003").payload_json
    )
    assert first.history == second.history
    assert first.get_response("turn-000003") == second.get_response("turn-000003")
    for namespace in ("observations", "perception", "execution"):
        assert first.to_dict()[namespace] == second.to_dict()[namespace]


def test_small_retention_limits_and_nonrepeating_turn_ids():
    limits = Limits(
        retained_turns=4,
        retained_snapshots=6,
        retained_observations=10,
        retained_execution=8,
        dedup_events=10,
        max_participants=2,
    )
    s, model = session(limits=limits)
    ids = set()
    for i in range(40):
        turn, _ = finish(s, f"bounded {i}", "A" if i % 2 else "B")
        assert turn not in ids
        ids.add(turn)
        s.record_robot_emitted(turn)
        counts = s.retained_counts()
        assert set(counts) == {
            "participants",
            "turns",
            "snapshots",
            "observations",
            "execution",
            "dialogue",
            "dedup",
            "history",
            "audio_bytes",
        }
        for key, maximum in (
            ("turns", 4),
            ("snapshots", 6),
            ("observations", 10),
            ("execution", 8),
            ("dedup", 10),
            ("dialogue", 4),
            ("participants", 2),
            ("history", 3),
            ("audio_bytes", 0),
        ):
            assert counts[key] <= maximum
        assert counts["history"] == len(s.history)
    assert model.calls[-1][3] == tuple(Turn(f"bounded {i}", "A" if i % 2 else "B") for i in range(36, 39))
    reject_unchanged(s, lambda: s.register_participant("third participant"))


def test_text_and_chunk_limits_reject_without_partial_mutation():
    s, _ = session(limits=Limits(max_text_chars=12, max_chunk_samples=400))
    turn = begin(s)
    for action in (
        lambda: s.observe_text_fragment(turn, "x" * 13),
        lambda: s.finalize_text(turn, "x" * 13),
        lambda: s.observe_audio_chunk(turn, np.zeros(401, np.float32)),
    ):
        reject_unchanged(s, action)
    s.observe_text_fragment(turn, "12345678")
    reject_unchanged(s, lambda: s.observe_text_fragment(turn, "12345"))


def test_close_cancels_incomplete_releases_content_and_rejects_future_events():
    s, _ = session()
    completed, snapshot = finish(s, "completed content")
    frozen = snapshot.payload_json
    turn = begin(s, "B")
    s.observe_text_fragment(turn, "incomplete content")
    s.observe_audio_chunk(turn, np.zeros(400, np.float32))
    s.close()
    assert s.history == ()
    exported = state(s)
    assert "completed content" not in exported and "incomplete content" not in exported
    for action in (
        lambda: s.start(),
        lambda: s.register_participant("C"),
        lambda: s.observe_text_fragment(turn, "late"),
        lambda: s.end_turn(turn),
        lambda: s.record_robot_emitted(completed),
    ):
        reject_unchanged(s, action)
    assert snapshot.payload_json == frozen


def test_exported_state_is_a_defensive_copy():
    s, _ = session()
    finish(s)
    before = state(s)
    view = s.to_dict()
    for namespace in ("observations", "perception", "execution", "participants"):
        view[namespace].clear()
    assert state(s) == before


def test_cancelled_inflight_guard_releases_before_new_generation():
    s, model = session()
    participant = s.register_participant("A")
    old = s.start_turn(participant)
    s.finalize_text(old, "older request")

    def supersede():
        s.cancel_turn(old)
        reject_unchanged(s, lambda: s.start_turn(participant))

    model.callback = supersede
    assert s.end_turn(old) is None
    assert s.history == ()
    no_final(s, old)
    model.callback = None
    newer, snapshot = finish(s, "newer completed request", "B")
    assert s.history == (Turn("newer completed request", "B"),)
    assert s.get_latest_perception(newer).payload_json == snapshot.payload_json
    no_final(s, old)


def test_duplicate_completed_end_event_never_executes_model_twice():
    model = FrozenModelDouble()
    s = StatefulSession(model, MANIFEST, session_id="replay-s", clock=lambda: 0.0)
    events = prefix_events() + [
        ObservationEvent.create(
            "replay-s", 6, "TEXT_FINALIZED", turn_id="turn-000003", payload={"text": "complete"}
        ),
        ObservationEvent.create("replay-s", 7, "HUMAN_TURN_ENDED", turn_id="turn-000003"),
    ]
    for event in events:
        s.ingest(event)
    assert len(model.calls) == 1
    before = state(s)
    s.ingest(events[-1])
    assert len(model.calls) == 1 and state(s) == before


def test_threaded_cancellation_blocks_reentry_until_owned_inference_returns():
    s, model = session()
    participant = s.register_participant("A")
    old = s.start_turn(participant)
    s.finalize_text(old, "observed before threaded cancellation")
    entered, release = threading.Event(), threading.Event()
    results, errors = [], []

    def blocked_prediction():
        entered.set()
        assert release.wait(timeout=5), "test did not release blocked prediction"

    def run_finalization():
        try:
            results.append(s.end_turn(old))
        except BaseException as exc:
            errors.append(exc)

    model.callback = blocked_prediction
    worker = threading.Thread(target=run_finalization, daemon=True)
    worker.start()
    try:
        assert entered.wait(timeout=5), "prediction did not reach its blocking point"
        s.cancel_turn(old)
        reject_unchanged(s, lambda: s.start_turn(participant))
        assert s.history == ()
        no_final(s, old)
        assert len(model.calls) == 1
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive(), "owned finalization did not stop"
    assert errors == [] and results == [None]
    assert s.history == ()
    no_final(s, old)
    model.callback = None
    fresh, snapshot = finish(s, "fresh after cancelled computation")
    assert fresh != old and snapshot.to_dict()["stage"] == "final"
    assert model.calls[-1][3] == ()
    assert s.history == (Turn("fresh after cancelled computation", "A"),)
    assert s.retained_counts()["audio_bytes"] == 0


def test_completion_record_replays_inflight_cancellation_before_future_success():
    recorder = ReplayRecorder()
    s, model = session(recorder=recorder)
    old = begin(s)
    s.observe_audio_chunk(old, np.ones(400, np.float32))
    s.finalize_text(old, "observed cancelled request")
    model.callback = lambda: s.cancel_turn(old)
    assert s.end_turn(old) is None
    model.callback = None
    future, snapshot = finish(s, "subsequent successful observation", "B")
    s.record_robot_emitted(future)
    replay_model = FrozenModelDouble()
    restored = replay(recorder.to_records(), replay_model, MANIFEST)
    assert restored.to_dict() == s.to_dict()
    assert restored.get_latest_perception(future).payload_json == snapshot.payload_json
    assert restored.history == (Turn("subsequent successful observation", "B"),)
    assert len(model.calls) == 2
    assert [call[0] for call in replay_model.calls] == ["subsequent successful observation"]
    no_final(restored, old)


def test_failure_completion_record_does_not_rerun_failed_model_request():
    recorder = ReplayRecorder()
    s, model = session(recorder=recorder)
    failed = begin(s)
    s.finalize_text(failed, "received before genuine execution failure")
    model.fail = True
    with pytest.raises(RuntimeError, match="injected"):
        s.end_turn(failed)
    model.fail = False
    future, snapshot = finish(s, "successful next observation", "B")
    replay_model = FrozenModelDouble()
    restored = replay(recorder.to_records(), replay_model, MANIFEST)
    assert restored.to_dict() == s.to_dict()
    assert restored.get_latest_perception(future).payload_json == snapshot.payload_json
    assert [call[0] for call in replay_model.calls] == ["successful next observation"]
    assert replay_model.calls[0][3] == ()
    assert "received before genuine execution failure" in json.dumps(restored.to_dict()["observations"])
    assert "failed" in json.dumps(restored.to_dict()["execution"])
    no_final(restored, failed)


def test_recorded_limits_and_completion_boundaries_survive_jsonl_roundtrip(tmp_path):
    limits = Limits(
        retained_turns=3,
        retained_snapshots=4,
        retained_observations=6,
        retained_execution=5,
        dedup_events=7,
        max_participants=2,
    )
    recorder = ReplayRecorder()
    s, _ = session(recorder=recorder, limits=limits)
    for index in range(8):
        turn, _ = finish(s, f"recorded bounded turn {index}", "A" if index % 2 else "B")
        s.record_robot_emitted(turn)
    path = tmp_path / "recorded-session.jsonl"
    recorder.write(path)
    restored = replay(read_events(path), FrozenModelDouble(), MANIFEST)
    assert restored.limits == limits
    assert restored.retained_counts() == s.retained_counts()
    assert restored.to_dict() == s.to_dict()
    assert restored.get_response(turn) == s.get_response(turn)
    assert restored.retained_counts()["turns"] == 3
    assert restored.retained_counts()["observations"] == 6
    assert restored.retained_counts()["audio_bytes"] == 0


def test_cancelled_request_that_then_raises_has_identical_recorded_replay():
    recorder = ReplayRecorder()
    s, model = session(recorder=recorder)
    old = begin(s)
    s.finalize_text(old, "observed before cancel and model failure")
    model.callback = lambda: s.cancel_turn(old)
    model.fail = True
    with pytest.raises(RuntimeError, match="injected"):
        s.end_turn(old)
    model.callback, model.fail = None, False
    finish(s, "fresh after interrupted failure", "B")
    replay_model = FrozenModelDouble()
    restored = replay(recorder.to_records(), replay_model, MANIFEST)
    assert restored.to_dict() == s.to_dict()
    assert [call[0] for call in replay_model.calls] == ["fresh after interrupted failure"]
    no_final(restored, old)


def stateful_cli_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "stateful_ta.py"
    spec = importlib.util.spec_from_file_location("stateful_cli_contract_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stateful_cli_help_is_cwd_independent_and_needs_no_assets(tmp_path):
    from pathlib import Path
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[2] / "stateful_ta.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=tmp_path, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0
    assert "--events" in result.stdout and "--session-id" in result.stdout
    assert "--asset-root" in result.stdout


@pytest.mark.parametrize("problem", ["existing_output", "chunk_zero", "chunk_big", "events_and_audio"])
def test_stateful_cli_invalid_inputs_fail_before_model_loading(problem, tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    module = stateful_cli_module()

    def unexpected_model_load(*args, **kwargs):
        pytest.fail("invalid CLI input must be rejected before constructing models")

    monkeypatch.setitem(sys.modules, "src.ta_runtime", SimpleNamespace(PortableTADual=unexpected_model_load))
    argv = ["--asset-root", str(tmp_path / "unprepared-assets"), "--text", "observed"]
    existing = tmp_path / "preserved.json"
    existing.write_text("preserve these bytes")
    if problem == "existing_output":
        argv.extend(["--output", str(existing)])
    elif problem == "chunk_zero":
        argv.extend(["--chunk-samples", "0"])
    elif problem == "chunk_big":
        argv.extend(["--chunk-samples", "65537"])
    else:
        argv = [
            "--asset-root",
            str(tmp_path / "unprepared-assets"),
            "--events",
            str(tmp_path / "events.jsonl"),
            "--audio",
            str(tmp_path / "input.wav"),
        ]
    with pytest.raises(SystemExit) as error:
        module.main(argv)
    assert error.value.code == 2
    assert existing.read_text() == "preserve these bytes"

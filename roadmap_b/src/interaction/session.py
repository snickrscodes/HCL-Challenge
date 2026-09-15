"""Causal, bounded interaction ownership around unchanged utterance-final inference."""

from collections import OrderedDict, deque
from dataclasses import asdict, replace
import hashlib
import json
import threading
import time

import numpy as np

from ..context import Turn
from .adapter import FinalPerceptionAdapter
from .events import ObservationEvent, canonical
from .state import Limits, ParticipantState, PerceptionSnapshot, TurnState


class StatefulSession:
    def __init__(
        self,
        model,
        manifest,
        *,
        session_id="session",
        clock=time.monotonic,
        limits=None,
        history=None,
        recorder=None,
    ):
        self.limits = limits or Limits()
        self._check_string(session_id, self.limits.max_id_chars, "session ID")
        self.session_id, self._clock = session_id, clock
        self._adapter = FinalPerceptionAdapter(model, manifest)
        self._lock = threading.RLock()
        self._seq, self._timestamp = 0, float("-inf")
        self._started, self._closed, self._inflight = False, False, False
        self._active = None
        self._initial_history = tuple(history or ())
        self._validate_history([{"text": t.text, "speaker": t.speaker} for t in self._initial_history])
        self._participants, self._turns = OrderedDict(), OrderedDict()
        self._snapshots, self._seen = OrderedDict(), OrderedDict()
        self._observations = deque(maxlen=self.limits.retained_observations)
        self._execution = deque(maxlen=self.limits.retained_execution)
        self._dialogue = deque(maxlen=self.limits.retained_turns)
        self._context = deque(maxlen=3)
        self._last_timing = {}
        self._recorder = recorder
        if recorder is not None:
            recorder.bind(session_id, self.limits, self._adapter.manifest_sha256)

    @staticmethod
    def _check_string(value, limit, name, *, empty=False):
        if not isinstance(value, str) or (not empty and not value) or len(value) > limit:
            raise ValueError(f"Invalid or oversized {name}")

    def _validate_history(self, history):
        if not isinstance(history, list) or len(history) > 3:
            raise ValueError("At most three observed initial human turns")
        for item in history:
            if not isinstance(item, dict) or set(item) != {"text", "speaker"}:
                raise ValueError("Initial history contains only observed human text/speaker")
            self._check_string(item["text"], self.limits.max_text_chars, "history text")
            self._check_string(item["speaker"], self.limits.max_id_chars, "history speaker")

    @property
    def history(self):
        with self._lock:
            return tuple(Turn(x["text"], x["speaker"]) for x in self._context)

    @property
    def last_timing(self):
        with self._lock:
            return json.loads(canonical(self._last_timing))

    def _event(self, kind, *, participant_id=None, turn_id=None, payload=None, pcm=None):
        with self._lock:
            return ObservationEvent.create(
                self.session_id,
                self._seq + 1,
                kind,
                timestamp=self._clock(),
                participant_id=participant_id,
                turn_id=turn_id,
                payload=payload,
                pcm=pcm,
            )

    def start(self):
        history = [{"text": t.text, "speaker": t.speaker} for t in self._initial_history]
        return self.ingest(self._event("SESSION_STARTED", payload={"history": history}))

    def register_participant(self, external_key, participant_id=None):
        with self._lock:
            if self._closed:
                raise RuntimeError("Session closed")
            for p in self._participants.values():
                if p.external_key == external_key:
                    if participant_id is not None and p.participant_id != participant_id:
                        raise ValueError("External key already associated")
                    return p.participant_id
            pid = participant_id or f"participant-{len(self._participants) + 1:06d}"
        self.ingest(
            self._event("PARTICIPANT_REGISTERED", participant_id=pid, payload={"external_key": external_key})
        )
        return pid

    def start_turn(self, participant_id):
        with self._lock:
            tid = f"turn-{self._seq + 1:06d}"
        self.ingest(self._event("HUMAN_TURN_STARTED", participant_id=participant_id, turn_id=tid))
        return tid

    def observe_text_fragment(self, turn_id, text):
        return self.ingest(self._event("TEXT_FRAGMENT_OBSERVED", turn_id=turn_id, payload={"text": text}))

    def finalize_text(self, turn_id, text):
        return self.ingest(self._event("TEXT_FINALIZED", turn_id=turn_id, payload={"text": text}))

    def observe_audio_chunk(self, turn_id, chunk):
        return self.ingest(self._event("AUDIO_CHUNK_OBSERVED", turn_id=turn_id, pcm=chunk))

    def mark_audio_unavailable(self, turn_id):
        return self.ingest(
            self._event("AUDIO_UNAVAILABLE", turn_id=turn_id, payload={"reason": "decode_failed"})
        )

    def end_turn(self, turn_id):
        return self.ingest(self._event("HUMAN_TURN_ENDED", turn_id=turn_id))

    def cancel_turn(self, turn_id):
        return self.ingest(self._event("TURN_CANCELLED", turn_id=turn_id))

    def close(self):
        return self.ingest(self._event("SESSION_ENDED"))

    def get_latest_perception(self, turn_id=None):
        with self._lock:
            if turn_id is None:
                return next(reversed(self._snapshots.values()), None) if self._snapshots else None
            turn = self._turns.get(turn_id)
            return self._snapshots.get(turn.latest_snapshot_id) if turn else None

    def get_response(self, turn_id):
        with self._lock:
            turn = self._turns.get(turn_id)
            if not turn or turn.inference_status != "complete" or turn.response is None:
                raise RuntimeError("A retained successful final snapshot is required")
            return turn.response

    def record_robot_emitted(self, turn_id, text=None):
        text = self.get_response(turn_id) if text is None else text
        return self.ingest(self._event("ROBOT_RESPONSE_EMITTED", turn_id=turn_id, payload={"text": text}))

    def _validate(self, e, payload):
        kind = e.event_type
        allowed = {
            "SESSION_STARTED": {"history"},
            "PARTICIPANT_REGISTERED": {"external_key"},
            "HUMAN_TURN_STARTED": set(),
            "TEXT_FRAGMENT_OBSERVED": {"text"},
            "TEXT_FINALIZED": {"text"},
            "AUDIO_CHUNK_OBSERVED": set(),
            "AUDIO_UNAVAILABLE": {"reason"},
            "HUMAN_TURN_ENDED": set(),
            "TURN_CANCELLED": set(),
            "ROBOT_RESPONSE_EMITTED": {"text"},
            "SESSION_ENDED": set(),
        }
        if kind not in allowed or (
            set(payload) != allowed[kind] and not (kind == "SESSION_STARTED" and not payload)
        ):
            raise ValueError("Unknown event type or invalid payload fields")
        if kind != "AUDIO_CHUNK_OBSERVED" and e.pcm_bytes:
            raise ValueError("Only audio observations may carry PCM")
        if self._closed:
            raise RuntimeError("Session closed")
        if kind == "SESSION_STARTED":
            if self._started or self._seq or e.turn_id or e.participant_id:
                raise RuntimeError("Session must start exactly once")
            self._validate_history(payload.get("history", []))
            return
        if not self._started:
            raise RuntimeError("Start session first")
        if kind == "PARTICIPANT_REGISTERED":
            self._check_string(e.participant_id, self.limits.max_id_chars, "participant ID")
            self._check_string(payload["external_key"], self.limits.max_id_chars, "external speaker key")
            if e.turn_id or e.participant_id in self._participants:
                raise ValueError("Participant already registered or invalid turn association")
            if any(p.external_key == payload["external_key"] for p in self._participants.values()):
                raise ValueError("External key already associated")
            if len(self._participants) >= self.limits.max_participants:
                raise ValueError("Participant capacity reached")
            return
        if kind == "SESSION_ENDED":
            if e.turn_id or e.participant_id:
                raise ValueError("Session end has no turn/participant association")
            return
        if kind == "HUMAN_TURN_STARTED":
            if self._active is not None or self._inflight:
                raise RuntimeError("Only one active or in-flight turn is supported")
            if e.participant_id not in self._participants or e.turn_id != f"turn-{e.event_seq:06d}":
                raise ValueError("Known participant and sequence-bound new turn ID required")
            return
        turn = self._turns.get(e.turn_id)
        if not turn:
            raise ValueError("Unknown or expired turn")
        if e.participant_id is not None and e.participant_id != turn.participant_id:
            raise ValueError("Turn participant cannot change")
        if kind == "ROBOT_RESPONSE_EMITTED":
            if turn.inference_status != "complete" or turn.emitted or payload["text"] != turn.response:
                raise RuntimeError("Only a matching unacknowledged final response can be emitted")
            return
        if kind == "TURN_CANCELLED":
            if turn.status == "cancelled" or turn.inference_status in ("complete", "failed"):
                raise RuntimeError("Only receiving or pending turns can be cancelled")
            return
        if turn.status != "receiving" or self._active != turn.turn_id:
            raise RuntimeError("Turn is not receiving")
        if kind in ("TEXT_FRAGMENT_OBSERVED", "TEXT_FINALIZED"):
            if turn.text_final is not None:
                raise RuntimeError("Text already finalized")
            self._check_string(
                payload["text"], self.limits.max_text_chars, "text", empty=kind != "TEXT_FINALIZED"
            )
            if (
                kind == "TEXT_FRAGMENT_OBSERVED"
                and len(turn.text_so_far) + len(payload["text"]) > self.limits.max_text_chars
            ):
                raise ValueError("Accumulated text exceeds limit")
        elif kind == "AUDIO_CHUNK_OBSERVED":
            if turn.audio_status == "invalid_audio":
                raise RuntimeError("Cannot add audio after decode failure")
            if not np.isfinite(np.frombuffer(e.pcm_bytes, dtype="<f4")).all():
                raise ValueError("PCM must be finite")
        elif kind == "AUDIO_UNAVAILABLE":
            if (
                turn.audio_sample_count
                or turn.audio_status == "invalid_audio"
                or payload["reason"] != "decode_failed"
            ):
                raise RuntimeError("Decode failure must precede PCM")
        elif kind == "HUMAN_TURN_ENDED" and turn.text_final is None:
            raise RuntimeError("Finalized transcript required")

    def ingest(self, event, *, _defer_final=False):
        started = time.perf_counter()
        if not isinstance(event, ObservationEvent):
            raise TypeError("ObservationEvent required")
        with self._lock:
            payload = event.validate_envelope(
                self.limits.max_id_chars, self.limits.max_text_chars, self.limits.max_chunk_samples
            )
            if event.session_id != self.session_id:
                raise ValueError("Foreign session event")
            fingerprint = event.fingerprint()
            if event.event_seq in self._seen:
                if self._seen[event.event_seq] != fingerprint:
                    raise ValueError("Conflicting duplicate event")
                return None
            if event.event_seq != self._seq + 1:
                raise ValueError("Stale or out-of-order event sequence")
            if event.monotonic_timestamp < self._timestamp:
                raise ValueError("Timestamp cannot rewind")
            self._validate(event, payload)
            if self._recorder is not None:
                self._recorder.observe(event)
            self._seq, self._timestamp = event.event_seq, event.monotonic_timestamp
            self._seen[event.event_seq] = fingerprint
            while len(self._seen) > self.limits.dedup_events:
                self._seen.popitem(last=False)
            self._observations.append(event.metadata())
            task = self._apply(event, payload)
            if task is None:
                turn = self._turns.get(event.turn_id)
                if turn and turn.status == "receiving":
                    snapshot = self._snapshot(turn, event, list(self._context))
                    self._retain_snapshot(turn, snapshot)
                    return snapshot
                return None
        return task if _defer_final else self._run_final(task, started)

    def _apply(self, e, payload):
        kind = e.event_type
        if kind == "SESSION_STARTED":
            self._started = True
            for i, item in enumerate(payload.get("history", [])):
                ref = f"initial-history-{i + 1}"
                self._context.append({**item, "turn_id": ref})
                self._dialogue.append({**item, "turn_id": ref, "role": "human", "event_seq": e.event_seq})
            self._initial_history = ()
        elif kind == "PARTICIPANT_REGISTERED":
            self._participants[e.participant_id] = ParticipantState(
                e.participant_id, payload["external_key"], e.event_seq
            )
        elif kind == "HUMAN_TURN_STARTED":
            turn = TurnState(e.turn_id, e.participant_id, e.monotonic_timestamp)
            self._turns[e.turn_id], self._active = turn, e.turn_id
            self._participants[e.participant_id] = replace(
                self._participants[e.participant_id], last_observed_event_seq=e.event_seq
            )
            while len(self._turns) > self.limits.retained_turns:
                old, _ = self._turns.popitem(last=False)
                for sid, snap in list(self._snapshots.items()):
                    if snap.to_dict()["turn_id"] == old:
                        del self._snapshots[sid]
        elif kind == "SESSION_ENDED":
            self._closed, self._active = True, None
            for turn in self._turns.values():
                turn.chunks.clear()
            for collection in (
                self._participants,
                self._turns,
                self._snapshots,
                self._observations,
                self._execution,
                self._dialogue,
                self._context,
            ):
                collection.clear()
            self._initial_history, self._last_timing = (), {}
            if not self._inflight:
                self._recorder = None
        else:
            turn = self._turns[e.turn_id]
            pid = turn.participant_id
            self._participants[pid] = replace(self._participants[pid], last_observed_event_seq=e.event_seq)
            if kind == "TEXT_FRAGMENT_OBSERVED":
                turn.text_so_far += payload["text"]
            elif kind == "TEXT_FINALIZED":
                turn.text_final = payload["text"]
            elif kind == "AUDIO_UNAVAILABLE":
                turn.audio_status = "invalid_audio"
            elif kind == "AUDIO_CHUNK_OBSERVED":
                turn.audio_sample_count += len(e.pcm_bytes) // 4
                turn.audio_digest.update(e.pcm_bytes)
                if turn.audio_sample_count > self.limits.max_audio_samples:
                    turn.audio_status = "duration_limit"
                    turn.chunks.clear()
                elif e.pcm_bytes:
                    turn.audio_status = "buffered"
                    turn.chunks.append(e.pcm_bytes)
            elif kind == "TURN_CANCELLED":
                turn.status, turn.inference_status = "cancelled", "cancelled"
                turn.ended_at = e.monotonic_timestamp
                turn.generation += 1
                turn.chunks.clear()
                self._active = None
                self._execution.append(
                    {"turn_id": turn.turn_id, "event_seq": e.event_seq, "disposition": "cancelled"}
                )
            elif kind == "ROBOT_RESPONSE_EMITTED":
                turn.emitted = True
                self._dialogue.append(
                    {
                        "role": "robot",
                        "text": payload["text"],
                        "turn_id": turn.turn_id,
                        "event_seq": e.event_seq,
                    }
                )
                self._execution.append(
                    {"turn_id": turn.turn_id, "event_seq": e.event_seq, "disposition": "emitted"}
                )
            elif kind == "HUMAN_TURN_ENDED":
                turn.status, turn.inference_status = "finalized", "pending"
                turn.ended_at, turn.generation = e.monotonic_timestamp, turn.generation + 1
                self._inflight = True
                speaker = self._participants[pid].external_key
                self._dialogue.append(
                    {
                        "role": "human",
                        "text": turn.text_final,
                        "speaker": speaker,
                        "participant_id": pid,
                        "turn_id": turn.turn_id,
                        "event_seq": e.event_seq,
                    }
                )
                self._execution.append(
                    {
                        "turn_id": turn.turn_id,
                        "event_seq": e.event_seq,
                        "generation": turn.generation,
                        "disposition": "requested",
                    }
                )
                if turn.audio_status == "invalid_audio":
                    audio = np.full(400, np.nan, dtype=np.float32)
                elif turn.audio_status == "duration_limit":
                    # Execution-only sentinel; unchanged duration policy skips WavLM.
                    audio = np.zeros(960001, dtype=np.float32)
                elif turn.chunks:
                    audio = np.frombuffer(b"".join(turn.chunks), dtype="<f4").copy()
                else:
                    audio = None
                turn.chunks.clear()
                return (turn, e, turn.generation, list(self._context), speaker, turn.text_final, audio)
        return None

    def _input_summary(self, turn):
        return {
            "text_so_far": turn.text_so_far,
            "text_final": turn.text_final,
            "audio_sample_count": turn.audio_sample_count,
            "audio_status": turn.audio_status,
            "audio_sha256": turn.audio_digest.hexdigest() if turn.audio_sample_count else None,
            "sample_rate": 16000,
            "started_at": turn.started_at,
            "ended_at": turn.ended_at,
        }

    def _snapshot(self, turn, event, context, result=None):
        body = {
            "schema": "stateful-perception-v1",
            "session_id": self.session_id,
            "turn_id": turn.turn_id,
            "participant_id": turn.participant_id,
            "as_of_event_seq": event.event_seq,
            "as_of_timestamp": event.monotonic_timestamp,
            "created_at": event.monotonic_timestamp,
            "stage": "final" if result else "partial",
            "observed_input_summary": self._input_summary(turn),
            "context": context,
            "context_policy": "last_three_successful_finalized_human_turns",
            "model_evidence_status": "computed" if result else "not_computed",
            "B_output": result["B_output"] if result else None,
            "D2_output": result["D2_output"] if result else None,
            "proposed_action": result["proposed_action"] if result else None,
            "availability_by_modality": {
                "text": "observed" if turn.text_final is not None or turn.text_so_far else "unavailable",
                "audio": (
                    "available"
                    if result["D2_output"]["available"]
                    else result["D2_output"].get("unavailable_reason")
                )
                if result
                else turn.audio_status,
            },
            "provenance": self._adapter.provenance(),
        }
        body["snapshot_id"] = hashlib.sha256(canonical(body).encode()).hexdigest()
        return PerceptionSnapshot(canonical(body))

    def _retain_snapshot(self, turn, snapshot):
        sid = snapshot.snapshot_id
        self._snapshots[sid] = snapshot
        turn.latest_snapshot_id = sid
        while len(self._snapshots) > self.limits.retained_snapshots:
            old, _ = self._snapshots.popitem(last=False)
            for item in self._turns.values():
                if item.latest_snapshot_id == old:
                    item.latest_snapshot_id = None

    def _run_final(self, task, started):
        turn, event, generation, context, speaker, text, audio = task
        try:
            ready = time.perf_counter()
            result, timing = self._adapter.predict_turn(
                text,
                audio,
                16000 if audio is not None else None,
                [Turn(x["text"], x["speaker"]) for x in context],
                speaker=speaker,
            )
            with self._lock:
                if self._closed or turn.generation != generation or turn.inference_status != "pending":
                    if not self._closed:
                        self._execution.append(
                            {
                                "turn_id": turn.turn_id,
                                "event_seq": event.event_seq,
                                "generation": generation,
                                "disposition": "discarded_stale",
                            }
                        )
                    self._record_completion(event, generation, "discarded_stale")
                    return None
                snapshot = self._snapshot(turn, event, context, result)
                self._retain_snapshot(turn, snapshot)
                turn.response = result["proposed_action"]["response"]
                turn.inference_status = "complete"
                self._context.append({"text": text, "speaker": speaker, "turn_id": turn.turn_id})
                self._execution.append(
                    {
                        "turn_id": turn.turn_id,
                        "event_seq": event.event_seq,
                        "generation": generation,
                        "disposition": "complete",
                    }
                )
                self._record_completion(event, generation, "complete")
                return snapshot
        except BaseException:
            with self._lock:
                if not self._closed and turn.generation == generation and turn.inference_status == "pending":
                    turn.inference_status = "failed"
                    self._execution.append(
                        {
                            "turn_id": turn.turn_id,
                            "event_seq": event.event_seq,
                            "generation": generation,
                            "disposition": "failed",
                        }
                    )
                    self._record_completion(event, generation, "failed")
                elif self._closed or turn.generation != generation:
                    if not self._closed:
                        self._execution.append(
                            {
                                "turn_id": turn.turn_id,
                                "event_seq": event.event_seq,
                                "generation": generation,
                                "disposition": "discarded_stale",
                            }
                        )
                    self._record_completion(event, generation, "discarded_stale")
            raise
        finally:
            with self._lock:
                turn.chunks.clear()
                self._inflight = False
                if self._closed:
                    self._recorder = None
                if self._active == turn.turn_id:
                    self._active = None
                if not self._closed and "timing" in locals():
                    total = (time.perf_counter() - started) * 1000
                    self._last_timing = {
                        **timing,
                        "total_ms": total,
                        "state_ready_to_predictor_ms": (ready - started) * 1000,
                        "finalization_overhead_ms": total - timing["predictor_ms"],
                    }

    def _record_completion(self, event, generation, disposition):
        if self._recorder is not None:
            self._recorder.complete(event.turn_id, event.event_seq, generation, self._seq, disposition)

    def _replay_disposition(self, task, disposition):
        """Apply recorded execution outcomes, never fabricate perception observations."""
        turn, event, generation, *_ = task
        with self._lock:
            if disposition == "failed":
                if self._closed or turn.inference_status != "pending" or turn.generation != generation:
                    raise ValueError("Recorded failure does not match pending request")
                turn.inference_status = "failed"
                self._execution.append(
                    {
                        "turn_id": turn.turn_id,
                        "event_seq": event.event_seq,
                        "generation": generation,
                        "disposition": "failed",
                    }
                )
            elif disposition == "discarded_stale":
                if not self._closed and turn.generation == generation and turn.inference_status == "pending":
                    raise ValueError("Recorded discard requires cancellation or close")
                if not self._closed:
                    self._execution.append(
                        {
                            "turn_id": turn.turn_id,
                            "event_seq": event.event_seq,
                            "generation": generation,
                            "disposition": "discarded_stale",
                        }
                    )
            else:
                raise ValueError("Unknown recorded execution disposition")
            turn.chunks.clear()
            self._inflight = False
            if self._active == turn.turn_id:
                self._active = None

    def retained_counts(self):
        with self._lock:
            return {
                "participants": len(self._participants),
                "turns": len(self._turns),
                "snapshots": len(self._snapshots),
                "observations": len(self._observations),
                "execution": len(self._execution),
                "dialogue": len(self._dialogue),
                "dedup": len(self._seen),
                "history": len(self._context),
                "audio_bytes": sum(sum(map(len, t.chunks)) for t in self._turns.values()),
            }

    def to_dict(self):
        with self._lock:
            state = {
                "session_id": self.session_id,
                "event_seq": self._seq,
                "started": self._started,
                "closed": self._closed,
                "active_turn": self._active,
                "participants": [asdict(p) for p in self._participants.values()],
                "turns": [
                    {
                        "turn_id": t.turn_id,
                        "participant_id": t.participant_id,
                        "status": t.status,
                        "inference_status": t.inference_status,
                        "emitted": t.emitted,
                        **self._input_summary(t),
                    }
                    for t in self._turns.values()
                ],
                "observations": list(self._observations),
                "observed_dialogue": list(self._dialogue),
                "perception": [s.to_dict() for s in self._snapshots.values()],
                "execution": list(self._execution),
                "human_context": list(self._context),
            }
            return json.loads(canonical(state))

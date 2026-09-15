"""Small owned state types; immutable public perception values."""

from dataclasses import dataclass, field
import hashlib
import json


@dataclass(frozen=True)
class Limits:
    sample_rate: int = 16000
    max_audio_samples: int = 960000
    max_chunk_samples: int = 65536
    max_text_chars: int = 16384
    max_id_chars: int = 128
    max_participants: int = 32
    retained_turns: int = 64
    retained_snapshots: int = 128
    retained_observations: int = 256
    retained_execution: int = 256
    dedup_events: int = 256
    max_replay_bytes: int = 67108864

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in vars(self).values()):
            raise ValueError("Retention limits must be positive integers")
        if self.sample_rate != 16000 or self.max_audio_samples != 960000:
            raise ValueError("Frozen audio contract is mono 16kHz and 60 seconds")


@dataclass(frozen=True)
class ParticipantState:
    participant_id: str
    external_key: str
    last_observed_event_seq: int


@dataclass
class TurnState:
    turn_id: str
    participant_id: str
    started_at: float
    status: str = "receiving"
    text_so_far: str = ""
    text_final: str | None = None
    ended_at: float | None = None
    audio_sample_count: int = 0
    audio_status: str = "missing"
    chunks: list = field(default_factory=list)
    audio_digest: object = field(default_factory=hashlib.sha256)
    latest_snapshot_id: str | None = None
    response: str | None = None
    emitted: bool = False
    inference_status: str = "not_requested"
    generation: int = 0


@dataclass(frozen=True)
class PerceptionSnapshot:
    payload_json: str

    @property
    def snapshot_id(self):
        return json.loads(self.payload_json)["snapshot_id"]

    def to_dict(self):
        return json.loads(self.payload_json)

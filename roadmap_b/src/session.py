"""Utterance-final interaction: PCM buffering is incremental; WavLM is not streaming."""

import numpy as np

from .context import Turn


class Session:
    def __init__(self, predictor, sample_rate=16000, speaker="user", history=None):
        self.predictor, self.sample_rate, self.speaker = predictor, sample_rate, speaker
        self.initial_history = list(history or [])
        self.reset()

    def reset(self):
        self.history = list(self.initial_history)
        self.started, self.active = False, False
        self.chunks, self.transcript = [], None

    def start(self):
        if self.started:
            raise RuntimeError("Session already started")
        self.started = True

    def begin_turn(self, speaker=None):
        if not self.started or self.active:
            raise RuntimeError("Start session and end the previous turn first")
        if speaker is not None:
            self.speaker = speaker
        self.active, self.chunks, self.transcript = True, [], None

    def push_audio(self, chunk):
        if not self.active:
            raise RuntimeError("No active turn")
        chunk = np.asarray(chunk)
        if chunk.ndim != 1 or not np.issubdtype(chunk.dtype, np.floating) or not np.isfinite(chunk).all():
            raise ValueError("Chunks must be finite mono floating-point PCM")
        if len(chunk):
            self.chunks.append(chunk.astype(np.float32, copy=True))

    def set_transcript(self, text):
        if not self.active:
            raise RuntimeError("No active turn")
        self.transcript = text

    def end_turn(self):
        if not self.active or self.transcript is None:
            raise RuntimeError("An active turn and transcript are required")
        audio = np.concatenate(self.chunks) if self.chunks else None
        prediction = self.predictor.predict_turn(
            self.transcript,
            audio,
            self.sample_rate if audio is not None else None,
            self.history,
            speaker=self.speaker,
        )
        self.history.append(Turn(self.transcript, self.speaker))
        self.history = self.history[-3:]
        self.active, self.chunks, self.transcript = False, [], None
        return prediction

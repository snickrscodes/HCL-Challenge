"""Local-only collection of consented PCM recordings and blinded listener judgments."""

import argparse
import base64
import hashlib
import io
import itertools
import json
import random
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import soundfile as sf

from .constants import LABELS

WEB = Path(__file__).resolve().parents[1] / "recording_kit"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def records(root):
    return [json.loads(p.read_text()) for p in sorted((root / "records").glob("*.json"))]


def planned(manifest):
    return {
        f"{speaker}_{p['phrase_id']}_{d['delivery_id']}": {
            "recording_id": f"{speaker}_{p['phrase_id']}_{d['delivery_id']}",
            "speaker": speaker,
            "phrase_id": p["phrase_id"],
            "text": p["text"],
            "history": p["history"],
            "stable_control": p["stable_control"],
            **d,
        }
        for speaker in manifest["speakers"]
        for p in manifest["phrases"]
        for d in p["deliveries"]
    }


def store_recording(root, manifest, meta, wav):
    item = planned(manifest).get(meta.get("recording_id"))
    if item is None or meta.get("consented") is not True or meta.get("words_confirmed") is not True:
        raise ValueError("Choose a planned recording, consent, and confirm the exact words.")
    if sha(wav) != meta.get("waveform_sha256"):
        raise ValueError("Browser/server waveform hashes differ.")
    waveform, rate = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
    if waveform.shape[1] != 1 or not np.isfinite(waveform).all() or not 0.1 <= len(waveform) / rate <= 30:
        raise ValueError("Recording must be finite mono PCM between 0.1 and 30 seconds.")
    if np.max(np.abs(waveform)) == 0:
        raise ValueError("Recording is digital silence; check microphone and record again.")
    timestamp = meta.get("recording_timestamp")
    if not isinstance(timestamp, str):
        raise ValueError("Recording timestamp is required.")
    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    folder = root / "records"
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / (item["recording_id"] + ".wav")
    # Accepting a take is the evidence boundary; accepted files cannot be replaced.
    with file.open("xb") as stream:
        stream.write(wav)
    value = {
        **item,
        "speaker_consented": True,
        "consent_scope": "local post-selection diagnostic",
        "consent_timestamp": meta.get("consent_timestamp"),
        "public_audio_release_consented": meta.get("public_audio_release_consented") is True,
        "words_confirmed": True,
        "recording_timestamp": timestamp,
        "accepted_timestamp": datetime.now(timezone.utc).isoformat(),
        "waveform_sha256": sha(wav),
        "num_samples": len(waveform),
        "sample_rate": rate,
        "duration_s": len(waveform) / rate,
        "audio": str(file.relative_to(root)),
        "peak_amplitude": float(np.max(np.abs(waveform))),
        "fraction_at_or_above_full_scale": float((np.abs(waveform) >= 1).mean()),
        "capture_settings": meta.get("capture_settings", {}),
        "human_recording": True,
    }
    (folder / (item["recording_id"] + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    return value


def listener_plan(root, manifest):
    values = records(root)
    if {r["recording_id"] for r in values} != set(planned(manifest)):
        raise ValueError("Finish all 60 accepted recordings before the blinded listener pass.")
    rng = random.Random(manifest["listener_seed"])
    rng.shuffle(values)
    clips = []
    tokens = {}
    for i, value in enumerate(values):
        token = f"clip_{i + 1:03d}"
        tokens[value["recording_id"]] = token
        clips.append({"token": token, "text": value["text"], "audio_url": "/audio/" + token})
    groups = {}
    for v in values:
        groups.setdefault((v["speaker"], v["phrase_id"]), []).append(v["recording_id"])
    pairs = []
    for group in groups.values():
        for a, b in itertools.combinations(sorted(group), 2):
            if rng.random() < 0.5:
                a, b = b, a
            pairs.append({"first": tokens[a], "second": tokens[b]})
    rng.shuffle(pairs)
    for i, v in enumerate(pairs):
        v["pair_token"] = f"pair_{i + 1:03d}"
    private = {
        "token_to_recording": {v: k for k, v in tokens.items()},
        "clips": clips,
        "pairs": pairs,
        "protocol_sha256": sha((root / "protocol.json").read_bytes()),
    }
    path = root / "listener_plan_private.json"
    if path.exists() and json.loads(path.read_text()) != private:
        raise ValueError("Blinded plan changed; preserve evidence and inspect.")
    if not path.exists():
        path.write_text(json.dumps(private, indent=2) + "\n")
    return {"clips": clips, "pairs": pairs, "labels": list(LABELS) + ["unclear_or_mixed"]}, private


def store_rating(root, manifest, value):
    public, _ = listener_plan(root, manifest)
    listener = value.get("listener_id", "")
    if not re.fullmatch(r"L[0-9]{2}", listener) or value.get("independent_listener") is not True:
        raise ValueError("Use an independent listener ID such as L01 and confirm independence.")
    kind = value.get("kind")
    if kind == "clip":
        if (
            value.get("token") not in {r["token"] for r in public["clips"]}
            or value.get("category") not in public["labels"]
        ):
            raise ValueError("Invalid blinded clip rating.")
    elif kind == "pair":
        if value.get("token") not in {r["pair_token"] for r in public["pairs"]} or value.get(
            "meaningfully_different"
        ) not in ["yes", "no", "unsure"]:
            raise ValueError("Invalid blinded contrast rating.")
    else:
        raise ValueError("Unknown rating type.")
    if not isinstance(value.get("timestamp"), str):
        raise ValueError("Rating timestamp required.")
    folder = root / "ratings" / listener
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (value["token"] + ".json")
    # No changing judgments after models are revealed.
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
    return {"saved": True}


def make_handler(root):
    manifest = json.loads((root / "protocol.json").read_text())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, value, status=200):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == "/api/plan":
                    return self.reply(
                        {
                            "protocol": manifest,
                            "recordings": list(planned(manifest).values()),
                            "completed": [r["recording_id"] for r in records(root)],
                        }
                    )
                if path == "/api/listener-plan":
                    pub, _ = listener_plan(root, manifest)
                    pub["completed"] = {
                        p.name: [f.stem for f in p.glob("*.json")]
                        for p in (root / "ratings").glob("L*")
                        if p.is_dir()
                    }
                    return self.reply(pub)
                if path.startswith("/audio/"):
                    _, private = listener_plan(root, manifest)
                    name = private["token_to_recording"].get(path.split("/")[-1])
                    if name is None:
                        raise ValueError("Unknown blind audio token.")
                    file = root / "records" / (name + ".wav")
                else:
                    allowed = {
                        "/": "index.html",
                        "/listen": "listener.html",
                        "/capture.js": "capture.js",
                        "/style.css": "style.css",
                    }
                    if path not in allowed:
                        return self.reply({"error": "Not found"}, 404)
                    file = WEB / allowed[path]
                data = file.read_bytes()
                mime = {
                    ".html": "text/html; charset=utf-8",
                    ".css": "text/css",
                    ".js": "text/javascript",
                    ".wav": "audio/wav",
                }[file.suffix]
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            except (ValueError, OSError) as e:
                self.reply({"error": str(e)}, 400)

        def do_POST(self):
            try:
                # Same-origin loopback UI only; never expose this collection endpoint publicly.
                origin = self.headers.get("Origin")
                if origin and origin != "http://" + self.headers.get("Host", ""):
                    raise ValueError("Cross-origin uploads are disabled.")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 12_000_000:
                    raise ValueError("Invalid upload size.")
                body = self.rfile.read(length)
                if self.path == "/api/record":
                    meta = json.loads(base64.b64decode(self.headers["X-Recording-Metadata"]))
                    value = store_recording(root, manifest, meta, body)
                    return self.reply({"saved": value["recording_id"], "sha256": value["waveform_sha256"]})
                if self.path == "/api/rating":
                    return self.reply(store_rating(root, manifest, json.loads(body)))
                self.reply({"error": "Not found"}, 404)
            except (ValueError, OSError, KeyError) as e:
                self.reply({"error": str(e)}, 400)

    return Handler


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", default="artifacts/submission/controlled_delivery")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    protocol = (WEB / "phrases.json").read_bytes()
    file = root / "protocol.json"
    if file.exists() and file.read_bytes() != protocol:
        raise ValueError("Recording protocol is immutable.")
    if not file.exists():
        file.write_bytes(protocol)
    print(
        f"Record: http://localhost:{args.port}/  |  Blind ratings: http://localhost:{args.port}/listen",
        flush=True,
    )
    print(f"Local recordings only: {root}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(root)).serve_forever()


if __name__ == "__main__":
    main()

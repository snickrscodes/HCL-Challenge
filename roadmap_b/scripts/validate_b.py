"""Run fresh-process synthetic/unit and pinned pretrained checks; never score MELD test."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--baseline-root", required=True)
parser.add_argument("--output-root", default="artifacts/roadmap_b")
args = parser.parse_args()
root = Path(args.output_root).resolve()
base = Path(args.baseline_root).resolve()
if root == base or root.is_relative_to(base):
    raise ValueError("Validation output must not modify Roadmap A")
root.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, RUN_PRETRAINED="1", B_BASELINE_ROOT=str(base), HF_HUB_OFFLINE="1")
commands = [
    [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts/validate_b.py"],
    [sys.executable, "-m", "ruff", "format", "--check", "src", "tests", "scripts/validate_b.py"],
    [sys.executable, "-m", "pytest", "-q", "--junitxml=" + str(root / "pytest.xml")],
]
runs = []
for command in commands:
    proc = subprocess.run(command, env=env, capture_output=True, text=True)
    runs.append({"command": command, "returncode": proc.returncode, "output": proc.stdout + proc.stderr})
    print(proc.stdout + proc.stderr, end="", flush=True)
    if proc.returncode:
        break
result = {
    "passed": len(runs) == len(commands) and all(r["returncode"] == 0 for r in runs),
    "scope": "Unit/synthetic integration plus official pretrained weights; no MELD test predictions",
    "runs": runs,
}
(root / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
raise SystemExit(0 if result["passed"] else 1)

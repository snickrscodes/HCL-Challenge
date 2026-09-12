#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  command -v curl >/dev/null 2>&1 || { echo "curl is required to install uv" >&2; exit 1; }
  curl -LsSf https://astral.sh/uv/0.12.13/install.sh | sh
fi
uv python install 3.12.14
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12.14 .venv
fi
uv pip sync --python .venv/bin/python requirements.lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
echo "Ready. Activate with: source .venv/bin/activate"

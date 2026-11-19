#!/usr/bin/env bash
# One-time setup on macOS / Linux. Installs into ./.venv, no admin rights needed.
set -euo pipefail
cd "$(dirname "$0")"
echo "HUVEC Damage Analyzer - one-time setup (downloads about 1 GB)"
if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Python installer from astral.sh)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
uv venv .venv --python 3.12 --allow-existing
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -c "import torch, streamlit; print('OK - PyTorch', torch.__version__, '/ Streamlit', streamlit.__version__)"
echo "Setup finished. Run ./start.sh to open the app."

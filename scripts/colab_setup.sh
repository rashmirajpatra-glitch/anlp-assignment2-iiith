#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --quiet uv
fi

uv python install 3.11
uv sync --python 3.11 --no-dev

uv run python - <<'PY'
import shutil
import subprocess
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Expected Python 3.11, found {sys.version.split()[0]}")
print(f"Python version: {sys.version.split()[0]}")

import torch
import huggingface_hub
import wandb

if not shutil.which("hf"):
    raise SystemExit("Hugging Face CLI (`hf`) is missing after dependency sync.")
if not shutil.which("wandb"):
    raise SystemExit("W&B CLI (`wandb`) is missing after dependency sync.")
print(f"PyTorch: {torch.__version__}; huggingface_hub and wandb import successfully.")
print("Hugging Face CLI: available; W&B CLI: available.")

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable. In Colab select Runtime > Change runtime type > T4 GPU, then rerun."
    )
properties = torch.cuda.get_device_properties(0)
print(f"CUDA device: {properties.name}")
print(f"Total GPU memory: {properties.total_memory / 1024**3:.1f} GiB")
print(f"GPU compute capability: {properties.major}.{properties.minor}")
print(f"CUDA BF16 supported: {torch.cuda.is_bf16_supported()}")

def authenticated(command):
    try:
        return subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0
    except OSError:
        return False

if authenticated(["hf", "auth", "whoami"]):
    print("Hugging Face authentication: available.")
else:
    print("Hugging Face is not authenticated yet. Authenticate when needed: hf auth login")

if authenticated(["wandb", "status"]):
    print("W&B authentication: available.")
else:
    print("W&B is not authenticated yet. Authenticate when needed: wandb login")
PY

echo "Setup complete. Keep the Colab runtime on GPU and run the smoke/training commands from the README."

#!/usr/bin/env bash
# PPAI v0.3.0 — GCloud L4 VM Setup Script
# Usage: bash ppai_v3/examples/qwen3b/setup.sh
#
# Creates a venv, installs dependencies, and verifies ppai_v3 imports.
# Run this AFTER uploading the ppai_v3 directory to the VM.
#
# Tested on: GCloud g2-standard-16 (L4 GPU, 24GB VRAM, 64GB RAM)
# OS image: Deep Learning VM with CUDA 12.4, Python 3.11

set -euo pipefail

echo "========================================"
echo "PPAI v0.3.0 — GCloud L4 Setup"
echo "========================================"

# ── Check GPU ──────────────────────────────────────
echo ""
echo "[1/6] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "  WARNING: nvidia-smi not found. GPU may not be available."
fi

# ── Create venv ────────────────────────────────────
echo ""
echo "[2/6] Creating virtual environment..."
VENV_DIR="${HOME}/ppai_v3_env"
if [ -d "$VENV_DIR" ]; then
    echo "  Existing venv found, reusing: $VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
    echo "  Created: $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "  Python: $(python3 --version)"
echo "  pip:    $(pip --version)"

# ── Install dependencies ──────────────────────────
echo ""
echo "[3/6] Installing PyTorch (CUDA 12.4)..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "[4/6] Installing transformers and utilities..."
pip install transformers accelerate numpy psutil

# ── Install ppai_v3 ───────────────────────────────
echo ""
echo "[5/6] Installing ppai_v3..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PPAI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
pip install -e "$PPAI_ROOT/ppai_v3"

# ── Verify ─────────────────────────────────────────
echo ""
echo "[6/6] Verifying installation..."
python3 -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

import transformers
print(f'  Transformers {transformers.__version__}')

import ppai_v3
print(f'  ppai_v3 {ppai_v3.__version__}')

from ppai_v3.layers.linear import PPAILinear
from ppai_v3.compress.convert import convert_model
from ppai_v3.trace.recorder import TraceRecorder
from ppai_v3.trace.verifier import verify_trace
print('  All ppai_v3 imports OK')
"

echo ""
echo "========================================"
echo "Setup complete!"
echo ""
echo "To run the pipeline:"
echo "  source $VENV_DIR/bin/activate"
echo "  python3 ppai_v3/examples/qwen3b/patch_qwen.py"
echo ""
echo "To auto-stop VM when done (cost savings):"
echo "  python3 ppai_v3/examples/qwen3b/patch_qwen.py --auto-stop"
echo "========================================"

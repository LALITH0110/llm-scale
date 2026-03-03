#!/usr/bin/env bash
# Download small models for M2 Mac dev (1B Q4_0 + Q8_0 only)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"
mkdir -p "$MODELS_DIR"

if ! command -v huggingface-cli &>/dev/null; then
  pip install huggingface_hub[cli] -q
fi

echo "=== Downloading local dev models (1B only) ==="

# Llama 3.2 1B Q4_0 (~700MB)
echo "Downloading Llama-3.2-1B Q4_0..."
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
  Llama-3.2-1B-Instruct-Q4_0.gguf \
  --local-dir "$MODELS_DIR" \
  --local-dir-use-symlinks False

mv "$MODELS_DIR/Llama-3.2-1B-Instruct-Q4_0.gguf" \
   "$MODELS_DIR/llama-3.2-1b-q4_0.gguf" 2>/dev/null || true

# Llama 3.2 1B Q8_0 (~1.3GB)
echo "Downloading Llama-3.2-1B Q8_0..."
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
  Llama-3.2-1B-Instruct-Q8_0.gguf \
  --local-dir "$MODELS_DIR" \
  --local-dir-use-symlinks False

mv "$MODELS_DIR/Llama-3.2-1B-Instruct-Q8_0.gguf" \
   "$MODELS_DIR/llama-3.2-1b-q8_0.gguf" 2>/dev/null || true

# Check free space before optional 3B
FREE_GB=$(df -g "$MODELS_DIR" | awk 'NR==2{print $4}')
if [ "${FREE_GB:-0}" -gt 5 ]; then
  echo "Sufficient space (${FREE_GB}GB free). Downloading Llama-3.2-3B Q4_0 (~2GB)..."
  huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
    Llama-3.2-3B-Instruct-Q4_0.gguf \
    --local-dir "$MODELS_DIR" \
    --local-dir-use-symlinks False

  mv "$MODELS_DIR/Llama-3.2-3B-Instruct-Q4_0.gguf" \
     "$MODELS_DIR/llama-3.2-3b-q4_0.gguf" 2>/dev/null || true
else
  echo "Low disk space (${FREE_GB}GB). Skipping 3B model."
fi

echo ""
echo "=== Downloaded models ==="
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "No .gguf files found"

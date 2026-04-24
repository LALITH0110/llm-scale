#!/usr/bin/env bash
# Download small models for M2 Mac dev (1B Q4_0 + Q8_0 only)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"
mkdir -p "$MODELS_DIR"

python3 -c "import huggingface_hub" 2>/dev/null || python3 -m pip install huggingface_hub -q

download_model() {
  local repo="$1" hf_filename="$2" local_name="$3"
  if [ -f "$MODELS_DIR/$local_name" ]; then
    echo "  Already exists: $local_name"; return
  fi
  echo "  Downloading $local_name..."
  python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(repo_id='$repo', filename='$hf_filename', local_dir='$MODELS_DIR')
dest = os.path.join('$MODELS_DIR', '$local_name')
if os.path.abspath(path) != os.path.abspath(dest):
    shutil.move(path, dest)
print('  -> ' + dest)
"
}

echo "=== Downloading local dev models (1B only) ==="

download_model bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_0.gguf llama-3.2-1b-q4_0.gguf
download_model bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q8_0.gguf llama-3.2-1b-q8_0.gguf

# Check free space before optional 3B
FREE_GB=$(df -BG "$MODELS_DIR" | awk 'NR==2{gsub("G","",$4); print $4}')
if [ "${FREE_GB:-0}" -gt 5 ]; then
  echo "Sufficient space (${FREE_GB}GB free). Downloading Llama-3.2-3B Q4_0 (~2GB)..."
  download_model bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_0.gguf llama-3.2-3b-q4_0.gguf
else
  echo "Low disk space (${FREE_GB}GB). Skipping 3B model."
fi

echo ""
echo "=== Downloaded models ==="
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "No .gguf files found"

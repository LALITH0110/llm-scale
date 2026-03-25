#!/usr/bin/env bash
# Download all models × all quant levels for Chameleon Cloud
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"
mkdir -p "$MODELS_DIR"

python3 -c "import huggingface_hub" 2>/dev/null || python3 -m pip install huggingface_hub -q

echo "=== Downloading full model suite for Chameleon ==="
echo ""

download_model() {
  local repo="$1"
  local hf_filename="$2"
  local local_name="$3"

  if [ -f "$MODELS_DIR/$local_name" ]; then
    echo "  Already exists: $local_name"
    return
  fi

  echo "  Downloading $local_name from $repo..."
  python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(repo_id='$repo', filename='$hf_filename')
dest = os.path.join('$MODELS_DIR', '$local_name')
shutil.copy2(path, dest)
print(f'  Saved: {os.path.getsize(dest) / 1e9:.1f} GB')
"
}

# Llama 3.2 1B
echo "--- Llama 3.2 1B ---"
download_model bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-f16.gguf llama-3.2-1b-f16.gguf
download_model bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q8_0.gguf llama-3.2-1b-q8_0.gguf
download_model bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_0.gguf llama-3.2-1b-q4_0.gguf
download_model bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q3_K_L.gguf llama-3.2-1b-q3_k_l.gguf

# Llama 3.2 3B
echo "--- Llama 3.2 3B ---"
download_model bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-f16.gguf llama-3.2-3b-f16.gguf
download_model bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q8_0.gguf llama-3.2-3b-q8_0.gguf
download_model bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_0.gguf llama-3.2-3b-q4_0.gguf
download_model bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q3_K_L.gguf llama-3.2-3b-q3_k_l.gguf

# DeepSeek 7B (TheBloke, no FP16 available)
echo "--- DeepSeek 7B ---"
download_model TheBloke/deepseek-llm-7B-chat-GGUF deepseek-llm-7b-chat.Q8_0.gguf deepseek-7b-q8_0.gguf
download_model TheBloke/deepseek-llm-7B-chat-GGUF deepseek-llm-7b-chat.Q4_0.gguf deepseek-7b-q4_0.gguf
download_model TheBloke/deepseek-llm-7B-chat-GGUF deepseek-llm-7b-chat.Q3_K_L.gguf deepseek-7b-q3_k_l.gguf
download_model TheBloke/deepseek-llm-7B-chat-GGUF deepseek-llm-7b-chat.Q2_K.gguf deepseek-7b-q2_k.gguf

echo ""
echo "=== Download complete ==="
echo "Models:"
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null | awk '{print $5, $9}'
echo ""
TOTAL=$(du -sh "$MODELS_DIR" | cut -f1)
echo "Total size: $TOTAL"

#!/usr/bin/env bash
# Download all models × all quant levels for Chameleon Cloud
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"
mkdir -p "$MODELS_DIR"

if ! command -v huggingface-cli &>/dev/null; then
  pip install huggingface_hub[cli] -q
fi

echo "=== Downloading full model suite for Chameleon ==="
echo "Warning: ~50-100GB of disk space required."
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
  huggingface-cli download "$repo" "$hf_filename" \
    --local-dir "$MODELS_DIR" \
    --local-dir-use-symlinks False

  # Rename to our convention
  if [ -f "$MODELS_DIR/$hf_filename" ] && [ "$hf_filename" != "$local_name" ]; then
    mv "$MODELS_DIR/$hf_filename" "$MODELS_DIR/$local_name"
  fi
}

# Llama 3.2 1B
echo "--- Llama 3.2 1B ---"
download_model bartowski/Llama-3.2-1B-Instruct-GGUF \
  Llama-3.2-1B-Instruct-f16.gguf llama-3.2-1b-f16.gguf
download_model bartowski/Llama-3.2-1B-Instruct-GGUF \
  Llama-3.2-1B-Instruct-Q8_0.gguf llama-3.2-1b-q8_0.gguf
download_model bartowski/Llama-3.2-1B-Instruct-GGUF \
  Llama-3.2-1B-Instruct-Q4_0.gguf llama-3.2-1b-q4_0.gguf
download_model bartowski/Llama-3.2-1B-Instruct-GGUF \
  Llama-3.2-1B-Instruct-Q2_K.gguf llama-3.2-1b-q2_k.gguf

# Llama 3.2 3B
echo "--- Llama 3.2 3B ---"
download_model bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-f16.gguf llama-3.2-3b-f16.gguf
download_model bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q8_0.gguf llama-3.2-3b-q8_0.gguf
download_model bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_0.gguf llama-3.2-3b-q4_0.gguf
download_model bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q2_K.gguf llama-3.2-3b-q2_k.gguf

# DeepSeek 7B
echo "--- DeepSeek 7B ---"
download_model bartowski/deepseek-llm-7b-chat-GGUF \
  deepseek-llm-7b-chat-f16.gguf deepseek-7b-f16.gguf
download_model bartowski/deepseek-llm-7b-chat-GGUF \
  deepseek-llm-7b-chat-Q8_0.gguf deepseek-7b-q8_0.gguf
download_model bartowski/deepseek-llm-7b-chat-GGUF \
  deepseek-llm-7b-chat-Q4_0.gguf deepseek-7b-q4_0.gguf
download_model bartowski/deepseek-llm-7b-chat-GGUF \
  deepseek-llm-7b-chat-Q2_K.gguf deepseek-7b-q2_k.gguf

echo ""
echo "=== Download complete ==="
echo "Models:"
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null | awk '{print $5, $9}'
echo ""
TOTAL=$(du -sh "$MODELS_DIR" | cut -f1)
echo "Total size: $TOTAL"

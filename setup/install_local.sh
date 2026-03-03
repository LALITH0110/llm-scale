#!/usr/bin/env bash
# Install for local M2 Mac dev with Metal acceleration
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== LLM-SCALE local (M2 Mac) setup ==="

# Homebrew deps
if ! command -v cmake &>/dev/null; then
  echo "Installing cmake..."
  brew install cmake
fi

if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 not found. Install via brew or pyenv."
  exit 1
fi

# llama-cpp-python with Metal
echo "Installing llama-cpp-python with Metal support..."
CMAKE_ARGS="-DLLAMA_METAL=on -DLLAMA_NATIVE=on" \
  pip install llama-cpp-python --force-reinstall --no-cache-dir

# Rest of requirements
echo "Installing requirements..."
pip install -r "$REPO_ROOT/requirements.txt"

# Generate gRPC stubs
echo "Generating gRPC stubs..."
python -m grpc_tools.protoc \
  -I "$REPO_ROOT/src/disaggregated/proto" \
  --python_out="$REPO_ROOT/src/disaggregated" \
  --grpc_python_out="$REPO_ROOT/src/disaggregated" \
  "$REPO_ROOT/src/disaggregated/proto/kvcache.proto"

# Fix relative imports in generated stubs (grpc_tools generates broken imports)
sed -i '' 's/^import kvcache_pb2/from . import kvcache_pb2/' \
  "$REPO_ROOT/src/disaggregated/kvcache_pb2_grpc.py" 2>/dev/null || true

echo ""
echo "=== Setup complete ==="
echo "Set LLMSCALE_ENV=local before running experiments:"
echo "  export LLMSCALE_ENV=local"
echo "  make download-local"
echo "  make exp1"

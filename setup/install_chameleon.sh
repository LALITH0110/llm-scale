#!/usr/bin/env bash
# Install for Chameleon Cloud (Ubuntu, CPU-only)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== LLM-SCALE Chameleon Cloud setup ==="

# System deps
echo "Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
  numactl \
  libnuma-dev \
  linux-tools-common \
  linux-tools-$(uname -r) \
  cmake \
  build-essential \
  python3-pip \
  python3-dev \
  wget \
  curl

# Prometheus node exporter
PROM_VERSION="1.8.2"
PROM_ARCH="linux-amd64"
if ! command -v node_exporter &>/dev/null; then
  echo "Installing Prometheus node exporter..."
  wget -q "https://github.com/prometheus/node_exporter/releases/download/v${PROM_VERSION}/node_exporter-${PROM_VERSION}.${PROM_ARCH}.tar.gz"
  tar xzf "node_exporter-${PROM_VERSION}.${PROM_ARCH}.tar.gz"
  sudo mv "node_exporter-${PROM_VERSION}.${PROM_ARCH}/node_exporter" /usr/local/bin/
  rm -rf "node_exporter-${PROM_VERSION}.${PROM_ARCH}"*
  echo "node_exporter installed. Start with: nohup node_exporter &"
fi

# Detect CPU features
AVX512_FLAG=""
if grep -q avx512f /proc/cpuinfo 2>/dev/null; then
  AVX512_FLAG="-DLLAMA_AVX512=on"
  echo "AVX512 detected: enabled"
fi

# llama-cpp-python CPU-optimized
echo "Installing llama-cpp-python (CPU, AVX2${AVX512_FLAG:+ + AVX512})..."
CMAKE_ARGS="-DLLAMA_NATIVE=on -DLLAMA_AVX2=on ${AVX512_FLAG}" \
  pip install llama-cpp-python --force-reinstall --no-cache-dir

# Python deps
echo "Installing Python requirements..."
pip install -r "$REPO_ROOT/requirements.txt"

# Generate gRPC stubs
echo "Generating gRPC stubs..."
python -m grpc_tools.protoc \
  -I "$REPO_ROOT/src/disaggregated/proto" \
  --python_out="$REPO_ROOT/src/disaggregated" \
  --grpc_python_out="$REPO_ROOT/src/disaggregated" \
  "$REPO_ROOT/src/disaggregated/proto/kvcache.proto"

# Fix relative imports
sed -i 's/^import kvcache_pb2/from . import kvcache_pb2/' \
  "$REPO_ROOT/src/disaggregated/kvcache_pb2_grpc.py" 2>/dev/null || true

echo ""
echo "=== Setup complete ==="
echo "Run on Chameleon:"
echo "  export LLMSCALE_ENV=chameleon"
echo "  make download-full  # downloads all models"
echo "  make exp1"
echo ""
echo "For multi-node disaggregated:"
echo "  PREFILL_HOST=<ip> DECODE_HOSTS=<ip1>,<ip2> make exp2"

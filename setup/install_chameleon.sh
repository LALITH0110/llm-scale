#!/usr/bin/env bash
# Install for Chameleon Cloud (Ubuntu).
# Detects GPU via nvidia-smi and installs CUDA llama-cpp-python + vLLM on GPU nodes.
# Falls back to CPU-only (AVX2/AVX512) build on CPU nodes.
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
  linux-tools-"$(uname -r)" 2>/dev/null || true
sudo apt-get install -y \
  cmake \
  build-essential \
  python3-pip \
  python3-dev \
  python3-venv \
  wget \
  curl

# Add local bin to PATH
export PATH="$HOME/.local/bin:$PATH"

# Upgrade pip and packaging to avoid scikit-build-core compat issues on Python 3.10
echo "Upgrading pip and packaging..."
python3 -m pip install --upgrade pip setuptools packaging wheel

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

# ---------------------------------------------------------------------------
# Detect GPU presence
# ---------------------------------------------------------------------------
HAS_GPU=false
if command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q .; then
  HAS_GPU=true
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  echo "GPU detected: ${GPU_NAME}"
else
  echo "No GPU detected (or nvidia-smi unavailable) — CPU-only path"
fi

if [ "$HAS_GPU" = "true" ]; then
  # -------------------------------------------------------------------------
  # GPU node: install llama-cpp-python with CUDA support
  # -------------------------------------------------------------------------
  # Chameleon CUDA images have nvcc at /usr/local/cuda/bin but not in PATH
  export PATH="/usr/local/cuda/bin:$PATH"
  export CUDACXX="/usr/local/cuda/bin/nvcc"
  echo "CUDA compiler: $(nvcc --version | head -1)"

  echo "Installing llama-cpp-python with CUDA (source build, GGML_CUDA=on)..."
  # native arch requires CMake 3.24+; Chameleon has 3.22 — use explicit sm_75 (RTX 6000 / Turing)
  CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75" \
    python3 -m pip install llama-cpp-python --force-reinstall --no-cache-dir

  # Verify CUDA offload works
  python3 -c "import llama_cpp; assert llama_cpp.llama_supports_gpu_offload(), 'CUDA offload not enabled!'; print('llama-cpp-python CUDA: OK')"

  # -------------------------------------------------------------------------
  # vLLM for Exp 8 (GPU batched baseline)
  # Chameleon CUDA driver varies (12.1-12.4); let pip auto-select the wheel.
  # -------------------------------------------------------------------------
  echo "Installing vLLM for GPU batched baseline (Exp 8)..."
  python3 -m pip install vllm

else
  # -------------------------------------------------------------------------
  # CPU node: AVX2 + AVX512 llama-cpp-python
  # -------------------------------------------------------------------------
  AVX512_FLAG=""
  if grep -q avx512f /proc/cpuinfo 2>/dev/null; then
    AVX512_FLAG="-DLLAMA_AVX512=on"
    echo "AVX512 detected: enabled"
  fi

  echo "Installing llama-cpp-python (CPU, AVX2${AVX512_FLAG:+ + AVX512})..."
  CMAKE_ARGS="-DLLAMA_NATIVE=on -DLLAMA_AVX2=on ${AVX512_FLAG}" \
    python3 -m pip install llama-cpp-python --force-reinstall --no-cache-dir
fi

# Python deps
echo "Installing Python requirements..."
python3 -m pip install -r "$REPO_ROOT/requirements.txt"

# Generate gRPC stubs
echo "Generating gRPC stubs..."
python3 -m grpc_tools.protoc \
  -I "$REPO_ROOT/src/disaggregated/proto" \
  --python_out="$REPO_ROOT/src/disaggregated" \
  --grpc_python_out="$REPO_ROOT/src/disaggregated" \
  "$REPO_ROOT/src/disaggregated/proto/kvcache.proto"

# Note: kvcache_pb2_grpc.py uses absolute imports (import kvcache_pb2)
# because decode_server.py adds the directory to sys.path directly.
# Do NOT convert to relative imports.

echo ""
echo "=== Setup complete ==="
echo "Run on Chameleon:"
echo "  export LLMSCALE_ENV=chameleon"
echo "  make download-full  # downloads all models"
echo "  make exp1"
echo ""
echo "For GPU experiments:"
echo "  make exp4  # GPU colocated baseline"
echo "  make exp8  # vLLM batched baseline"
echo ""
echo "For multi-node disaggregated:"
echo "  PREFILL_HOST=<ip> DECODE_HOSTS=<ip1>,<ip2> make exp2"
echo "For hybrid (GPU prefill + CPU decode):"
echo "  PREFILL_HOST=<gpu-ip> DECODE_HOSTS=<cpu-ip> make exp6"

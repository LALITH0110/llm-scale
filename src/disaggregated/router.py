"""
HTTP router for disaggregated prefill/decode serving.

POST /generate  → prefill (local or remote) → gRPC to decode node → return result + metrics
GET  /health    → check prefill + decode node health
GET  /metrics   → Prometheus metrics
"""
import os
import sys
import time
import uuid
import logging
import argparse
from pathlib import Path
from typing import List, Optional

import grpc
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "disaggregated"))

try:
    import kvcache_pb2
    import kvcache_pb2_grpc
except ImportError:
    print("ERROR: gRPC stubs not found. Run: make proto")
    sys.exit(1)

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [router] %(message)s")
log = logging.getLogger("router")

# Prometheus metrics
REQUEST_COUNTER = Counter("llmscale_requests_total", "Total generate requests", ["status"])
TTFT_HIST = Histogram("llmscale_ttft_ms", "Time to first token (ms)", buckets=[10, 50, 100, 250, 500, 1000, 2000, 5000])
TPOT_HIST = Histogram("llmscale_tpot_ms", "Time per output token (ms)", buckets=[1, 5, 10, 20, 50, 100, 250, 500])
KV_TRANSFER_HIST = Histogram("llmscale_kv_transfer_ms", "KV cache transfer latency (ms)", buckets=[1, 5, 10, 25, 50, 100, 250, 500])
THROUGHPUT_HIST = Histogram("llmscale_throughput_tps", "Decode throughput (tokens/sec)", buckets=[0.5, 1, 2, 5, 10, 20, 50, 100])


class GenerateRequest(BaseModel):
    prompt: str
    model_id: str = "llama-3.2-1b:q4_0"
    n_predict: int = 128
    request_id: Optional[str] = None


class GenerateResponse(BaseModel):
    request_id: str
    text: str
    ttft_ms: float
    tpot_ms: float
    tpot_std_ms: float
    throughput_tps: float
    tokens_generated: int
    kv_transfer_ms: float
    total_ms: float
    prefill_node: str
    decode_node: str
    error: Optional[str] = None


class RouterConfig:
    def __init__(
        self,
        prefill_host: str,
        prefill_port: int,
        decode_hosts: List[str],
        decode_port_base: int,
        model_path: Optional[str] = None,
        n_threads: int = 4,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        local_prefill: bool = False,
    ):
        self.prefill_host = prefill_host
        self.prefill_port = prefill_port
        self.decode_hosts = decode_hosts
        self.decode_port_base = decode_port_base
        self.model_path = model_path
        self.n_threads = n_threads
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.local_prefill = local_prefill

        # For local prefill (in-process), load model directly
        self._local_llm: Optional[Llama] = None
        if local_prefill and model_path and Llama:
            log.info(f"Loading model for local prefill: {model_path}")
            self._local_llm = Llama(
                model_path=model_path,
                n_threads=n_threads,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            log.info("Local prefill model loaded.")

        # gRPC channels to decode nodes (round-robin)
        self._decode_channels = []
        self._decode_stubs = []
        self._decode_idx = 0

        for i, host in enumerate(decode_hosts):
            port = decode_port_base + i
            addr = f"{host}:{port}"
            ch = grpc.insecure_channel(addr)
            stub = kvcache_pb2_grpc.KVCacheServiceStub(ch)
            self._decode_channels.append(ch)
            self._decode_stubs.append(stub)
            log.info(f"Decode node registered: {addr}")

    def next_decode_stub(self) -> tuple:
        """Round-robin decode node selection."""
        idx = self._decode_idx % len(self._decode_stubs)
        self._decode_idx += 1
        host = self.decode_hosts[idx]
        port = self.decode_port_base + idx
        return self._decode_stubs[idx], f"{host}:{self.decode_port_base + idx}"

    def run_prefill(self, prompt: str, request_id: str) -> tuple[bytes, int, float]:
        """Run prefill either locally or via gRPC to prefill node."""
        if self._local_llm:
            tokens = self._local_llm.tokenize(prompt.encode("utf-8"))
            t0 = time.perf_counter()
            self._local_llm.eval(tokens)
            ttft_ms = (time.perf_counter() - t0) * 1000.0
            kv_bytes = bytes(self._local_llm.save_state())
            n_past = len(tokens)
            return kv_bytes, n_past, ttft_ms
        else:
            # Remote prefill via gRPC (not yet implemented — use local for now)
            raise NotImplementedError("Remote prefill gRPC not yet wired; use --local-prefill")


# Global config (set at startup)
_router_config: Optional[RouterConfig] = None

app = FastAPI(title="LLM-SCALE Router")


@app.get("/health")
def health():
    return {"status": "ok", "decode_nodes": len(_router_config.decode_hosts) if _router_config else 0}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if _router_config is None:
        raise HTTPException(500, "Router not configured")

    request_id = req.request_id or str(uuid.uuid4())
    t_total_start = time.perf_counter()

    try:
        # --- Prefill ---
        t_prefill = time.perf_counter()
        kv_bytes, n_past, ttft_ms = _router_config.run_prefill(req.prompt, request_id)
        prefill_ms = (time.perf_counter() - t_prefill) * 1000.0

        TTFT_HIST.observe(ttft_ms)
        log.info(f"[{request_id}] prefill: ttft={ttft_ms:.1f}ms kv_size={len(kv_bytes)/1024:.1f}KB")

        # --- KV transfer to decode node ---
        decode_stub, decode_addr = _router_config.next_decode_stub()

        grpc_request = kvcache_pb2.GenerateRequest(
            kv_state=kv_bytes,
            n_past=n_past,
            request_id=request_id,
            model_id=req.model_id,
            n_predict=req.n_predict,
            prefill_ttft_ms=ttft_ms,
        )

        t_transfer = time.perf_counter()
        token_parts = []
        tpot_values = []
        total_decode_ms = 0.0
        total_tokens = 0
        kv_transfer_ms = 0.0
        first_response = True

        for token_resp in decode_stub.GenerateTokens(grpc_request, timeout=300):
            if first_response:
                kv_transfer_ms = (time.perf_counter() - t_transfer) * 1000.0
                KV_TRANSFER_HIST.observe(kv_transfer_ms)
                first_response = False

            if token_resp.token:
                token_parts.append(token_resp.token)
            if token_resp.tpot_ms > 0:
                tpot_values.append(token_resp.tpot_ms)
            if token_resp.is_last:
                total_decode_ms = token_resp.total_decode_ms
                total_tokens = token_resp.total_tokens
                break

        # Metrics
        import statistics
        tpot_mean = statistics.mean(tpot_values) if tpot_values else 0.0
        tpot_std = statistics.stdev(tpot_values) if len(tpot_values) > 1 else 0.0
        throughput = total_tokens / (total_decode_ms / 1000.0) if total_decode_ms > 0 else 0.0

        TPOT_HIST.observe(tpot_mean)
        THROUGHPUT_HIST.observe(throughput)
        REQUEST_COUNTER.labels(status="success").inc()

        total_ms = (time.perf_counter() - t_total_start) * 1000.0

        return GenerateResponse(
            request_id=request_id,
            text="".join(token_parts),
            ttft_ms=ttft_ms,
            tpot_ms=tpot_mean,
            tpot_std_ms=tpot_std,
            throughput_tps=throughput,
            tokens_generated=total_tokens,
            kv_transfer_ms=kv_transfer_ms,
            total_ms=total_ms,
            prefill_node=f"{_router_config.prefill_host}:{_router_config.prefill_port}" if not _router_config.local_prefill else "local",
            decode_node=decode_addr,
        )

    except Exception as e:
        REQUEST_COUNTER.labels(status="error").inc()
        log.error(f"[{request_id}] error: {e}")
        raise HTTPException(500, str(e))


def main(args):
    global _router_config

    env = os.environ.get("LLMSCALE_ENV", "chameleon")
    n_gpu_layers = -1 if env == "local" else 0

    decode_hosts = args.decode_hosts.split(",") if args.decode_hosts else ["localhost"]

    _router_config = RouterConfig(
        prefill_host=args.prefill_host,
        prefill_port=args.prefill_port,
        decode_hosts=decode_hosts,
        decode_port_base=args.decode_port_base,
        model_path=args.model_path,
        n_threads=args.n_threads,
        n_ctx=args.n_ctx,
        n_gpu_layers=n_gpu_layers,
        local_prefill=args.local_prefill,
    )

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-host", default="localhost")
    parser.add_argument("--prefill-port", type=int, default=50051)
    parser.add_argument("--decode-hosts", default="localhost")
    parser.add_argument("--decode-port-base", type=int, default=50052)
    parser.add_argument("--model-path", default=None, help="Model path for local prefill")
    parser.add_argument("--n-threads", type=int, default=os.cpu_count() // 2)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--local-prefill", action="store_true",
                        help="Run prefill in-process (no separate prefill server)")
    args = parser.parse_args()
    main(args)

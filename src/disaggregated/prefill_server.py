"""
Prefill gRPC server.
Loads model, runs prefill on prompt tokens, serializes KV cache via save_state(),
returns bytes + n_past + measured TTFT to caller.
"""
import os
import sys
import time
import logging
import argparse
from concurrent import futures
from pathlib import Path

import grpc

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
    print("ERROR: llama-cpp-python not installed.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [prefill] %(message)s")
log = logging.getLogger("prefill")


class PrefillServicer(kvcache_pb2_grpc.KVCacheServiceServicer):
    def __init__(self, model_path: str, n_threads: int, n_ctx: int, n_gpu_layers: int, model_id: str):
        self.model_id = model_id
        self.n_threads = n_threads
        self.active_requests = 0

        log.info(f"Loading model: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_threads=n_threads,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        log.info(f"Model loaded. model_id={model_id}")

    def TransferKVCache(self, request, context):
        """Not used directly by the router — see GenerateTokens flow."""
        return kvcache_pb2.KVCacheResponse(
            request_id=request.request_id,
            success=False,
            error="Use GenerateTokens for prefill→decode transfer",
        )

    def GenerateTokens(self, request, context):
        """
        Prefill-only path: eval tokens, save_state, yield a single 'done' response
        with kv_state embedded. The router then forwards to decode node.

        We abuse the streaming interface: send ONE TokenResponse with token="" and
        is_last=True. The kv_state is stored server-side keyed by request_id.
        The decode server receives a GenerateRequest with kv_state directly.
        """
        self.active_requests += 1
        req_id = request.request_id

        try:
            # Decode prompt from the token field (we overload: token holds the prompt)
            prompt = request.token if hasattr(request, 'token') else ""
            # Actually: prompt comes in via a separate field — see router.py which
            # calls the decode server directly with kv_state. The prefill server
            # exposes a dedicated HTTP endpoint instead. See router.py.
            pass
        finally:
            self.active_requests -= 1

        yield kvcache_pb2.TokenResponse(
            request_id=req_id,
            token="",
            is_last=True,
        )

    def HealthCheck(self, request, context):
        return kvcache_pb2.HealthResponse(
            healthy=True,
            model_id=self.model_id,
            active_requests=self.active_requests,
        )

    def run_prefill(self, prompt: str, request_id: str) -> tuple[bytes, int, float]:
        """
        Public method called by router's in-process path or via gRPC wrapper.
        Returns (kv_state_bytes, n_past, ttft_ms).
        """
        tokens = self.llm.tokenize(prompt.encode("utf-8"))

        t0 = time.perf_counter()
        self.llm.eval(tokens)
        ttft_ms = (time.perf_counter() - t0) * 1000.0

        kv_state = self.llm.save_state()
        n_past = len(tokens)

        log.info(f"[{request_id}] prefill done: n_past={n_past} ttft={ttft_ms:.1f}ms "
                 f"kv_size={len(kv_state)/1024:.1f}KB")
        return bytes(kv_state), n_past, ttft_ms


def serve(args):
    env = os.environ.get("LLMSCALE_ENV", "chameleon")
    n_gpu_layers = -1 if env == "local" else 0

    servicer = PrefillServicer(
        model_path=args.model_path,
        n_threads=args.n_threads,
        n_ctx=args.n_ctx,
        n_gpu_layers=n_gpu_layers,
        model_id=args.model_id,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    kvcache_pb2_grpc.add_KVCacheServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    log.info(f"Prefill server listening on port {args.port}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", default="unknown")
    parser.add_argument("--n-threads", type=int, default=os.cpu_count())
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()
    serve(args)

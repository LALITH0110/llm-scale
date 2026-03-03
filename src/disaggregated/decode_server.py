"""
Decode gRPC server.
Receives KV cache bytes (from prefill node), restores via load_state(),
generates tokens one-by-one, streams each as TokenResponse with per-token TPOT.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [decode] %(message)s")
log = logging.getLogger("decode")


class DecodeServicer(kvcache_pb2_grpc.KVCacheServiceServicer):
    def __init__(self, model_path: str, n_threads: int, n_ctx: int, n_gpu_layers: int, model_id: str):
        self.model_id = model_id
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

    def GenerateTokens(self, request: kvcache_pb2.GenerateRequest, context):
        """
        1. Restore KV state from request.kv_state via load_state()
        2. Generate up to n_predict tokens
        3. Stream each token as TokenResponse with tpot_ms
        """
        self.active_requests += 1
        req_id = request.request_id
        n_predict = request.n_predict or 128
        prefill_ttft_ms = request.prefill_ttft_ms

        log.info(f"[{req_id}] decode start: kv_size={len(request.kv_state)/1024:.1f}KB "
                 f"n_past={request.n_past} n_predict={n_predict}")

        try:
            # Restore KV cache from prefill node
            t_restore_start = time.perf_counter()
            self.llm.load_state(request.kv_state)
            restore_ms = (time.perf_counter() - t_restore_start) * 1000.0
            log.info(f"[{req_id}] KV restored in {restore_ms:.1f}ms")

            token_latencies = []
            total_tokens = 0
            t_decode_total_start = time.perf_counter()

            for i in range(n_predict):
                t_tok = time.perf_counter()
                token_id = self.llm.sample(
                    top_k=1,
                    top_p=1.0,
                    temp=0.0,
                    repeat_penalty=1.0,
                )
                tpot = (time.perf_counter() - t_tok) * 1000.0
                token_latencies.append(tpot)

                if token_id == self.llm.token_eos():
                    # Stream final summary
                    total_decode_ms = (time.perf_counter() - t_decode_total_start) * 1000.0
                    yield kvcache_pb2.TokenResponse(
                        request_id=req_id,
                        token="",
                        is_last=True,
                        tpot_ms=tpot,
                        token_index=i,
                        total_decode_ms=total_decode_ms,
                        total_tokens=total_tokens,
                    )
                    break

                # Detokenize
                token_bytes = self.llm.detokenize([token_id])
                token_str = token_bytes.decode("utf-8", errors="replace")
                total_tokens += 1

                is_last = (i == n_predict - 1)
                total_decode_ms = (time.perf_counter() - t_decode_total_start) * 1000.0 if is_last else 0.0

                yield kvcache_pb2.TokenResponse(
                    request_id=req_id,
                    token=token_str,
                    is_last=is_last,
                    tpot_ms=tpot,
                    token_index=i,
                    total_decode_ms=total_decode_ms if is_last else 0.0,
                    total_tokens=total_tokens if is_last else 0,
                )

                self.llm.eval([token_id])

            log.info(f"[{req_id}] decode complete: {total_tokens} tokens, "
                     f"avg_tpot={sum(token_latencies)/len(token_latencies):.2f}ms")

        except Exception as e:
            log.error(f"[{req_id}] decode error: {e}")
            yield kvcache_pb2.TokenResponse(
                request_id=req_id,
                token=f"ERROR: {e}",
                is_last=True,
            )
        finally:
            self.active_requests -= 1

    def TransferKVCache(self, request, context):
        return kvcache_pb2.KVCacheResponse(
            request_id=request.request_id,
            success=False,
            error="Use GenerateTokens on decode server",
        )

    def HealthCheck(self, request, context):
        return kvcache_pb2.HealthResponse(
            healthy=True,
            model_id=self.model_id,
            active_requests=self.active_requests,
        )


def serve(args):
    env = os.environ.get("LLMSCALE_ENV", "chameleon")
    n_gpu_layers = -1 if env == "local" else 0

    servicer = DecodeServicer(
        model_path=args.model_path,
        n_threads=args.n_threads,
        n_ctx=args.n_ctx,
        n_gpu_layers=n_gpu_layers,
        model_id=args.model_id,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    kvcache_pb2_grpc.add_KVCacheServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    log.info(f"Decode server listening on port {args.port}")

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
    parser.add_argument("--port", type=int, default=50052)
    args = parser.parse_args()
    serve(args)

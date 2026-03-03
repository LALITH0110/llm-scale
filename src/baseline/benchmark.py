"""Single-node colocated inference benchmark. Measures TTFT, TPOT, throughput."""
import os
import sys
import time
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional

import psutil

# llama-cpp-python import with graceful error
try:
    from llama_cpp import Llama
except ImportError:
    print("ERROR: llama-cpp-python not installed. Run: make setup-local or make setup-chameleon")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    model_path: str
    n_threads: int
    prompt_name: str
    prompt_len_tokens: int
    n_predict: int
    n_ctx: int

    # Timing (ms)
    ttft_ms: float = 0.0        # Time To First Token
    tpot_ms: float = 0.0        # Time Per Output Token (mean)
    tpot_std_ms: float = 0.0    # std dev of per-token latencies
    total_decode_ms: float = 0.0

    # Throughput
    tokens_generated: int = 0
    throughput_tps: float = 0.0  # tokens/sec (decode phase)

    # Memory / bandwidth
    rss_mb_before: float = 0.0
    rss_mb_peak: float = 0.0
    mem_bw_est_gbps: float = 0.0  # estimated from psutil IO counters (rough)

    # Run metadata
    n_gpu_layers: int = 0
    env: str = "unknown"
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _measure_peak_rss(proc: psutil.Process, interval: float, result: list, stop_event: threading.Event):
    """Background thread: sample RSS periodically, store peak."""
    peak = 0.0
    while not stop_event.is_set():
        try:
            rss = proc.memory_info().rss / 1024**2
            if rss > peak:
                peak = rss
        except psutil.NoSuchProcess:
            break
        time.sleep(interval)
    result.append(peak)


def benchmark_colocated(
    model_path: str,
    n_threads: int,
    prompt: str,
    prompt_name: str,
    n_predict: int = 128,
    n_ctx: int = 4096,
    n_gpu_layers: int = 0,
    verbose: bool = False,
) -> BenchmarkResult:
    """
    Run colocated (prefill+decode on same node) inference.

    Measures:
    - TTFT: time from start of eval() to first token available
    - TPOT: per-token decode latency distribution
    - throughput: tokens/sec during decode
    """
    env = os.environ.get("LLMSCALE_ENV", "unknown")

    result = BenchmarkResult(
        model_path=model_path,
        n_threads=n_threads,
        prompt_name=prompt_name,
        prompt_len_tokens=0,
        n_predict=n_predict,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        env=env,
    )

    proc = psutil.Process(os.getpid())
    result.rss_mb_before = proc.memory_info().rss / 1024**2

    try:
        # Load model (not timed — we time inference only)
        llm = Llama(
            model_path=model_path,
            n_threads=n_threads,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            logits_all=False,
        )

        # Tokenize prompt
        tokens = llm.tokenize(prompt.encode("utf-8"))
        result.prompt_len_tokens = len(tokens)

        # Start peak RSS monitor
        peak_rss_buf: list = []
        stop_evt = threading.Event()
        rss_thread = threading.Thread(
            target=_measure_peak_rss,
            args=(proc, 0.05, peak_rss_buf, stop_evt),
            daemon=True,
        )
        rss_thread.start()

        # --- TTFT: run prefill ---
        t_prefill_start = time.perf_counter()
        llm.eval(tokens)
        t_first_token = time.perf_counter()
        result.ttft_ms = (t_first_token - t_prefill_start) * 1000.0

        # --- Decode: generate tokens one at a time ---
        token_latencies_ms = []
        generated_tokens = []

        # Sampling params (greedy for reproducibility)
        for _ in range(n_predict):
            t_tok_start = time.perf_counter()
            token_id = llm.sample(
                top_k=1,
                top_p=1.0,
                temp=0.0,
                repeat_penalty=1.0,
            )
            t_tok_end = time.perf_counter()

            if token_id == llm.token_eos():
                break

            token_latencies_ms.append((t_tok_end - t_tok_start) * 1000.0)
            generated_tokens.append(token_id)
            llm.eval([token_id])

        stop_evt.set()
        rss_thread.join(timeout=1.0)

        # Metrics
        result.tokens_generated = len(generated_tokens)
        if token_latencies_ms:
            import statistics
            result.tpot_ms = statistics.mean(token_latencies_ms)
            result.tpot_std_ms = statistics.stdev(token_latencies_ms) if len(token_latencies_ms) > 1 else 0.0
            result.total_decode_ms = sum(token_latencies_ms)
            result.throughput_tps = (
                result.tokens_generated / (result.total_decode_ms / 1000.0)
                if result.total_decode_ms > 0 else 0.0
            )

        result.rss_mb_peak = peak_rss_buf[0] if peak_rss_buf else proc.memory_info().rss / 1024**2

        # Rough memory bandwidth estimate: bytes read ≈ model_size_on_disk × tokens_generated
        # (each token does ~1 pass over weights)
        model_size_bytes = os.path.getsize(model_path)
        if result.total_decode_ms > 0:
            result.mem_bw_est_gbps = (
                model_size_bytes * result.tokens_generated
                / (result.total_decode_ms / 1000.0)
                / 1e9
            )

    except Exception as e:
        result.error = str(e)

    return result

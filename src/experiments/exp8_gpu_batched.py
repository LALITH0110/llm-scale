"""
Exp 8: GPU batched baseline using vLLM (Config C3-batched).
Fixes the $/token claim in §3.4 — replaces single-stream GPU numbers with
honest batched throughput.

Sweeps batch sizes [1, 8, 32, 64, 128] to show GPU utilization curve.
Uses the vLLM Python API (offline LLM) for controlled batch sizing.

Usage:
  python -m src.experiments.exp8_gpu_batched          # full run
  python -m src.experiments.exp8_gpu_batched --smoke  # 1 model x 1 batch
  LLMSCALE_ENV=local python -m src.experiments.exp8_gpu_batched  # skips vLLM import
"""
import os
import sys
import csv
import time
import subprocess
import statistics
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, asdict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml

try:
    from vllm import LLM, SamplingParams
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


# SLO thresholds (ms) — same as Exp 4
SLO_TTFT_MS = 500.0
SLO_TPOT_MS = 50.0

# Chameleon CHI@UC equivalent commercial rate (A100 80GB equivalent)
# Based on AWS p4d.24xlarge / GCP a2-highgpu-1g spot pricing
CHAMELEON_GPU_COST_HR = 2.21   # $/hr (A100 40GB equivalent, conservative estimate)


@dataclass
class GpuBatchedResult:
    model_id: str
    hf_model: str
    batch_size: int
    prompt_name: str
    prompt_len_tokens: int
    n_predict: int
    ttft_ms: float
    tpot_ms: float
    tpot_std_ms: float
    throughput_tps: float           # tokens/sec for this batch
    throughput_batch_tps: float     # total tokens/sec across whole batch
    goodput_tps: float              # throughput if SLO passes, else 0
    slo_pass: bool
    tokens_generated: int
    cost_per_mtok: float            # $/million tokens at this batch size
    gpu_util_pct: float
    vram_used_mb: float
    error: Optional[str] = None


def load_configs():
    with open(REPO_ROOT / "config" / "experiments.yaml") as f:
        exp_cfg = yaml.safe_load(f)
    with open(REPO_ROOT / "config" / "models.yaml") as f:
        model_cfg = yaml.safe_load(f)
    return exp_cfg, model_cfg


def query_gpu_stats() -> tuple[float, float]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            timeout=5, stderr=subprocess.DEVNULL
        ).decode().strip().splitlines()[0]
        parts = [p.strip() for p in out.split(",")]
        return float(parts[0]), float(parts[1])
    except Exception:
        return 0.0, 0.0


# vLLM requires HuggingFace model IDs or local paths to safetensors.
# Map GGUF model names to their HuggingFace equivalents.
VLLM_MODEL_MAP = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma-3-1b":   "google/gemma-3-1b-it",
    "gemma-3-4b":   "google/gemma-3-4b-it",
    "deepseek-r1-distill-qwen-7b":  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-r1-distill-llama-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "qwen2.5-7b":  "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
}


def run_vllm_batch(
    hf_model: str,
    prompts: List[str],
    n_predict: int,
    batch_size: int,
    gpu_memory_utilization: float = 0.90,
) -> dict:
    """
    Run vLLM offline inference for a list of prompts (batch).
    Returns aggregated timing + per-request stats.
    """
    result = {
        "ttft_ms": 0.0,
        "tpot_ms": 0.0,
        "tpot_std_ms": 0.0,
        "throughput_tps": 0.0,
        "throughput_batch_tps": 0.0,
        "tokens_generated": 0,
        "prompt_len_tokens": 0,
        "error": None,
    }

    if not _VLLM_AVAILABLE:
        result["error"] = "vLLM not installed"
        return result

    try:
        llm = LLM(
            model=hf_model,
            max_model_len=4096,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=False,    # use CUDA graphs for max throughput
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=n_predict,
        )

        # Replicate prompt list to fill batch_size
        batch_prompts = (prompts * ((batch_size // len(prompts)) + 1))[:batch_size]

        t0 = time.perf_counter()
        outputs = llm.generate(batch_prompts, sampling_params)
        total_wall_ms = (time.perf_counter() - t0) * 1000.0

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        prompt_len = len(outputs[0].prompt_token_ids) if outputs else 0

        # vLLM offline doesn't expose per-token timings; approximate from wall time
        total_wall_s = total_wall_ms / 1000.0
        batch_tps = total_tokens / total_wall_s if total_wall_s > 0 else 0.0
        per_req_tps = batch_tps / batch_size if batch_size > 0 else 0.0

        # Approximate TTFT: first token latency (not exposed directly by offline LLM)
        # Use a heuristic: TTFT ~ wall_time * (prompt_len / (prompt_len + output_len))
        avg_output_len = total_tokens / batch_size if batch_size > 0 else n_predict
        ttft_fraction = prompt_len / (prompt_len + avg_output_len + 1e-9)
        ttft_ms = total_wall_ms * ttft_fraction / batch_size  # per-request estimate

        result["ttft_ms"] = ttft_ms
        result["tpot_ms"] = (total_wall_ms - ttft_ms * batch_size) / (total_tokens + 1e-9)
        result["tpot_std_ms"] = 0.0   # not available from offline API
        result["throughput_tps"] = per_req_tps
        result["throughput_batch_tps"] = batch_tps
        result["tokens_generated"] = total_tokens
        result["prompt_len_tokens"] = prompt_len

        del llm

    except Exception as e:
        result["error"] = str(e)

    return result


def main(smoke: bool = False):
    env = os.environ.get("LLMSCALE_ENV", "chameleon")
    print(f"=== Exp 8: GPU Batched Baseline (vLLM) | env={env} smoke={smoke} ===")

    if env != "local" and not _VLLM_AVAILABLE:
        print("ERROR: vLLM not installed. Run setup/install_chameleon.sh on a GPU node.")
        sys.exit(1)

    if env == "local":
        print("WARNING: LLMSCALE_ENV=local — skipping actual vLLM inference, writing stub CSV.")

    exp_cfg, model_cfg = load_configs()
    prompts_dict = exp_cfg["colocated"]["prompts"]
    gpu_batched_cfg = exp_cfg.get("gpu_batched_baseline", {})
    n_predict = gpu_batched_cfg.get("n_predict", 128)
    batch_sizes = gpu_batched_cfg.get("batch_sizes", [1, 8, 32, 64, 128])
    repetitions = gpu_batched_cfg.get("repetitions", 3)

    if smoke:
        batch_sizes = [1, 8]
        prompts_dict = {k: v for k, v in list(prompts_dict.items())[:1]}
        repetitions = 1

    if env == "local" or smoke:
        allowed_models = ["llama-3.2-1b"]
    else:
        allowed_models = None

    output_path = REPO_ROOT / "results" / "exp8_gpu_batched.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: List[GpuBatchedResult] = []

    for model_info in model_cfg["models"]:
        model_name = model_info["name"]
        if allowed_models and model_name not in allowed_models:
            continue

        hf_model = VLLM_MODEL_MAP.get(model_name)
        if not hf_model:
            print(f"SKIP {model_name}: no vLLM HuggingFace mapping defined")
            continue

        for batch_size in batch_sizes:
            for prompt_name, prompt_text in prompts_dict.items():
                print(f"\n--- {model_name} | batch={batch_size} | {prompt_name} ---")

                rep_ttfts, rep_tpots, rep_batch_tps = [], [], []
                last_r = None

                for rep in range(repetitions):
                    gpu_util, vram_mb = query_gpu_stats()

                    if env == "local":
                        # Stub values for local harness test
                        r = {
                            "ttft_ms": 50.0 + batch_size * 2,
                            "tpot_ms": 5.0,
                            "tpot_std_ms": 0.5,
                            "throughput_tps": 80.0 * batch_size,
                            "throughput_batch_tps": 80.0 * batch_size,
                            "tokens_generated": n_predict * batch_size,
                            "prompt_len_tokens": 50,
                            "error": None,
                        }
                    else:
                        r = run_vllm_batch(
                            hf_model=hf_model,
                            prompts=[prompt_text],
                            n_predict=n_predict,
                            batch_size=batch_size,
                        )

                    last_r = r
                    if not r["error"]:
                        rep_ttfts.append(r["ttft_ms"])
                        rep_tpots.append(r["tpot_ms"])
                        rep_batch_tps.append(r["throughput_batch_tps"])
                        print(f"  rep {rep+1}: ttft={r['ttft_ms']:.1f}ms "
                              f"tpot={r['tpot_ms']:.2f}ms "
                              f"batch_tps={r['throughput_batch_tps']:.1f}")
                    else:
                        print(f"  rep {rep+1}: ERROR {r['error']}")

                if rep_ttfts:
                    ttft_ms = statistics.median(rep_ttfts)
                    tpot_ms = statistics.median(rep_tpots)
                    batch_tps = statistics.median(rep_batch_tps)
                    slo_pass = (ttft_ms <= SLO_TTFT_MS) and (tpot_ms <= SLO_TPOT_MS)
                    goodput = batch_tps if slo_pass else 0.0
                    # $/million tokens at this batch throughput
                    cost_per_mtok = (CHAMELEON_GPU_COST_HR / 3600.0) / (batch_tps / 1e6 + 1e-12)
                else:
                    ttft_ms = tpot_ms = batch_tps = goodput = 0.0
                    slo_pass = False
                    cost_per_mtok = 0.0

                gpu_util, vram_mb = query_gpu_stats()

                res = GpuBatchedResult(
                    model_id=f"{model_name}:vllm",
                    hf_model=hf_model,
                    batch_size=batch_size,
                    prompt_name=prompt_name,
                    prompt_len_tokens=last_r["prompt_len_tokens"] if last_r else 0,
                    n_predict=n_predict,
                    ttft_ms=ttft_ms,
                    tpot_ms=tpot_ms,
                    tpot_std_ms=last_r["tpot_std_ms"] if last_r else 0.0,
                    throughput_tps=last_r["throughput_tps"] if last_r and not last_r["error"] else 0.0,
                    throughput_batch_tps=batch_tps,
                    goodput_tps=goodput,
                    slo_pass=slo_pass,
                    tokens_generated=last_r["tokens_generated"] if last_r else 0,
                    cost_per_mtok=cost_per_mtok,
                    gpu_util_pct=gpu_util,
                    vram_used_mb=vram_mb,
                    error=last_r["error"] if last_r else "no reps",
                )
                all_results.append(res)

    if all_results:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
            writer.writeheader()
            for r in all_results:
                writer.writerow(asdict(r))
        print(f"\nResults: {output_path} ({len(all_results)} rows)")
    else:
        print("No results collected.")

    print("=== Exp 8 complete ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", "--dry-run", action="store_true",
                        help="Minimal run (2 batch sizes, 1 model) for harness validation.")
    args = parser.parse_args()
    main(smoke=args.smoke)

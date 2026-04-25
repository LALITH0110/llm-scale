"""
Exp 4: GPU colocated baseline (Config C3).
Single GPU node, llama-cpp-python CUDA backend (n_gpu_layers=-1).
Sweeps batch sizes [1, 4, 16, 32, 64] — each batch_size runs that many
sequential inferences per rep to show aggregate throughput vs GPU single-stream.
Metrics: TTFT, TPOT, throughput, goodput (SLO: TTFT<500ms, TPOT<50ms), GPU util, VRAM.

Usage:
  python -m src.experiments.exp4_gpu_colocated          # full Chameleon run
  python -m src.experiments.exp4_gpu_colocated --smoke  # 1 model x 1 quant x 1 batch
  LLMSCALE_ENV=local python -m src.experiments.exp4_gpu_colocated
"""
import gc
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
from tqdm import tqdm

try:
    from llama_cpp import Llama
    _LLAMA_AVAILABLE = True
except ImportError:
    _LLAMA_AVAILABLE = False


SLO_TTFT_MS = 500.0
SLO_TPOT_MS = 50.0


@dataclass
class GpuColocResult:
    model_id: str
    quant: str
    batch_size: int
    prompt_name: str
    prompt_len_tokens: int
    n_predict: int
    ttft_ms: float
    tpot_ms: float
    tpot_std_ms: float
    throughput_tps: float
    goodput_tps: float          # throughput * slo_pass (0 if SLO violated)
    slo_pass: bool
    tokens_generated: int
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
    """Return (utilization_pct, vram_used_mb). Returns (0, 0) if unavailable."""
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


def run_one_inference(llm: "Llama", prompt: str, n_predict: int) -> dict:
    """
    One inference pass on an already-loaded model.
    Resets model state before running so it can be called in a batch loop.
    """
    result = {
        "ttft_ms": 0.0,
        "tpot_ms": 0.0,
        "tpot_std_ms": 0.0,
        "throughput_tps": 0.0,
        "tokens_generated": 0,
        "prompt_len_tokens": 0,
        "error": None,
    }
    try:
        llm.reset()
        tokens = llm.tokenize(prompt.encode("utf-8"))
        result["prompt_len_tokens"] = len(tokens)

        t0 = time.perf_counter()
        llm.eval(tokens)
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        result["ttft_ms"] = ttft_ms

        token_latencies: List[float] = []
        tokens_generated = 0
        for _ in range(n_predict):
            t_tok = time.perf_counter()
            token_id = llm.sample(top_k=1, top_p=1.0, temp=0.0, repeat_penalty=1.0)
            tpot = (time.perf_counter() - t_tok) * 1000.0
            token_latencies.append(tpot)
            if token_id == llm.token_eos():
                break
            tokens_generated += 1
            llm.eval([token_id])

        result["tokens_generated"] = tokens_generated
        if token_latencies:
            result["tpot_ms"] = statistics.mean(token_latencies)
            result["tpot_std_ms"] = statistics.stdev(token_latencies) if len(token_latencies) > 1 else 0.0
            total_ms = sum(token_latencies)
            result["throughput_tps"] = tokens_generated / (total_ms / 1000.0) if total_ms > 0 else 0.0

    except Exception as e:
        result["error"] = str(e)
    return result


def run_batch(llm: "Llama", prompt: str, n_predict: int, batch_size: int) -> dict:
    """
    Run batch_size sequential inferences on the same loaded model.
    Reports: mean per-request TTFT/TPOT, aggregate throughput (total_tokens / total_time).
    This simulates sequential request processing — honest baseline for llama.cpp GPU.
    """
    ttfts, tpots, tpot_stds, total_tokens = [], [], [], 0
    t_batch_start = time.perf_counter()
    first_error = None

    for _ in range(batch_size):
        r = run_one_inference(llm, prompt, n_predict)
        if r["error"]:
            first_error = r["error"]
            break
        ttfts.append(r["ttft_ms"])
        tpots.append(r["tpot_ms"])
        tpot_stds.append(r["tpot_std_ms"])
        total_tokens += r["tokens_generated"]

    total_batch_ms = (time.perf_counter() - t_batch_start) * 1000.0

    if first_error or not ttfts:
        return {
            "ttft_ms": 0.0, "tpot_ms": 0.0, "tpot_std_ms": 0.0,
            "throughput_tps": 0.0, "tokens_generated": 0,
            "prompt_len_tokens": 0, "error": first_error or "no results",
        }

    return {
        "ttft_ms": statistics.mean(ttfts),
        "tpot_ms": statistics.mean(tpots),
        "tpot_std_ms": statistics.mean(tpot_stds),
        # Aggregate tps: all tokens across the full batch wall time
        "throughput_tps": total_tokens / (total_batch_ms / 1000.0) if total_batch_ms > 0 else 0.0,
        "tokens_generated": total_tokens,
        "prompt_len_tokens": 0,   # set by caller
        "error": None,
    }


def main(smoke: bool = False):
    env = os.environ.get("LLMSCALE_ENV", "chameleon")
    print(f"=== Exp 4: GPU Colocated Baseline | env={env} smoke={smoke} ===")

    if not _LLAMA_AVAILABLE:
        print("ERROR: llama-cpp-python not installed. Run setup/install_chameleon.sh on a GPU node.")
        sys.exit(1)

    exp_cfg, model_cfg = load_configs()
    gpu_cfg = exp_cfg.get("gpu_colocated", {})
    prompts = exp_cfg["colocated"]["prompts"]
    n_predict = gpu_cfg.get("n_predict", 128)
    n_ctx = gpu_cfg.get("n_ctx", 4096)
    batch_sizes = gpu_cfg.get("batch_sizes", [1, 4, 16, 32, 64])
    repetitions = gpu_cfg.get("repetitions", 3)

    if smoke:
        batch_sizes = [1]
        repetitions = 1
        prompts = {k: v for k, v in list(prompts.items())[:1]}

    n_gpu_layers = -1 if env != "local" else exp_cfg["local_overrides"].get("n_gpu_layers", -1)

    if env == "local":
        allowed_models = exp_cfg["local_overrides"]["models"]
        allowed_quants = exp_cfg["local_overrides"]["quants"]
    elif smoke:
        allowed_models = ["llama-3.2-1b"]
        allowed_quants = ["q4_0"]
    else:
        allowed_models = None
        allowed_quants = None

    output_path = REPO_ROOT / "results" / "exp4_gpu_colocated.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: List[GpuColocResult] = []

    model_list = model_cfg["models"]
    if smoke:
        model_list = model_list[:1]

    for model_info in model_list:
        model_name = model_info["name"]
        if allowed_models and model_name not in allowed_models:
            continue

        for quant, rel_path in model_info["variants"].items():
            if allowed_quants and quant not in allowed_quants:
                continue

            model_path = str(REPO_ROOT / rel_path)
            if not Path(model_path).exists():
                print(f"SKIP (missing): {model_path}")
                continue

            model_id = f"{model_name}:{quant}"
            print(f"\n=== Loading {model_id} ===")

            # Load model once; reuse across all batch sizes + prompts + reps
            llm = None
            load_error = None
            try:
                llm = Llama(
                    model_path=model_path,
                    n_threads=4,
                    n_ctx=n_ctx,      # fixed — no batch_size multiplier
                    n_gpu_layers=n_gpu_layers,
                    n_batch=512,
                    verbose=False,
                )
                print(f"  Loaded OK — n_ctx={n_ctx} n_gpu_layers={n_gpu_layers}")
            except Exception as e:
                load_error = str(e)
                print(f"  LOAD ERROR: {load_error}")

            if load_error:
                # Record load error for all configs, skip inference
                for batch_size in batch_sizes:
                    for prompt_name in prompts:
                        res = GpuColocResult(
                            model_id=model_id, quant=quant, batch_size=batch_size,
                            prompt_name=prompt_name, prompt_len_tokens=0,
                            n_predict=n_predict, ttft_ms=0.0, tpot_ms=0.0,
                            tpot_std_ms=0.0, throughput_tps=0.0, goodput_tps=0.0,
                            slo_pass=False, tokens_generated=0,
                            gpu_util_pct=0.0, vram_used_mb=0.0, error=load_error,
                        )
                        all_results.append(res)
                continue

            for batch_size in batch_sizes:
                for prompt_name, prompt_text in prompts.items():
                    print(f"\n--- {model_id} | batch={batch_size} | {prompt_name} ---")

                    rep_ttfts, rep_tpots, rep_tps = [], [], []
                    last_r = None

                    for rep in range(repetitions):
                        r = run_batch(llm, prompt_text, n_predict, batch_size)
                        gpu_util_after, vram_after = query_gpu_stats()
                        last_r = r

                        if not r["error"]:
                            rep_ttfts.append(r["ttft_ms"])
                            rep_tpots.append(r["tpot_ms"])
                            rep_tps.append(r["throughput_tps"])
                            print(f"  rep {rep+1}: ttft={r['ttft_ms']:.1f}ms "
                                  f"tpot={r['tpot_ms']:.2f}ms "
                                  f"tps={r['throughput_tps']:.1f} "
                                  f"gpu={gpu_util_after:.0f}% vram={vram_after:.0f}MB")
                        else:
                            print(f"  rep {rep+1}: ERROR {r['error']}")

                    if rep_ttfts:
                        ttft_ms = statistics.median(rep_ttfts)
                        tpot_ms = statistics.median(rep_tpots)
                        throughput_tps = statistics.median(rep_tps)
                        slo_pass = (ttft_ms <= SLO_TTFT_MS) and (tpot_ms <= SLO_TPOT_MS)
                        goodput = throughput_tps if slo_pass else 0.0
                    else:
                        ttft_ms = tpot_ms = throughput_tps = goodput = 0.0
                        slo_pass = False

                    gpu_util_pct, vram_used_mb = query_gpu_stats()

                    res = GpuColocResult(
                        model_id=model_id,
                        quant=quant,
                        batch_size=batch_size,
                        prompt_name=prompt_name,
                        prompt_len_tokens=last_r["prompt_len_tokens"] if last_r else 0,
                        n_predict=n_predict,
                        ttft_ms=ttft_ms,
                        tpot_ms=tpot_ms,
                        tpot_std_ms=last_r["tpot_std_ms"] if last_r and not last_r["error"] else 0.0,
                        throughput_tps=throughput_tps,
                        goodput_tps=goodput,
                        slo_pass=slo_pass,
                        tokens_generated=last_r["tokens_generated"] if last_r else 0,
                        gpu_util_pct=gpu_util_pct,
                        vram_used_mb=vram_used_mb,
                        error=last_r["error"] if last_r else "no reps",
                    )
                    all_results.append(res)

            # Unload model before loading next
            del llm
            gc.collect()

    if all_results:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
            writer.writeheader()
            for r in all_results:
                writer.writerow(asdict(r))
        print(f"\nResults: {output_path} ({len(all_results)} rows)")
    else:
        print("No results collected.")

    print("=== Exp 4 complete ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", "--dry-run", action="store_true",
                        help="Minimal run (1 model x 1 batch x 1 rep) for harness validation.")
    args = parser.parse_args()
    main(smoke=args.smoke)

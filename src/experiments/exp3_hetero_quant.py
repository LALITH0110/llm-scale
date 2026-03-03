"""
Exp 3: Heterogeneous quantization — FP16 prefill × Q4 decode (and other combos).
Tests whether load_state() works cross-quantization.
Each combo: run prefill with quant_A, save_state, load in quant_B model, decode.
"""
import os
import sys
import csv
import time
import signal
import subprocess
import httpx
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, asdict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml

try:
    from llama_cpp import Llama
except ImportError:
    print("ERROR: llama-cpp-python not installed.")
    sys.exit(1)


@dataclass
class HeteroResult:
    model_name: str
    prefill_quant: str
    decode_quant: str
    prompt_name: str
    prompt_len_tokens: int
    n_predict: int
    ttft_ms: float
    kv_transfer_ms: float
    tpot_ms: float
    tpot_std_ms: float
    throughput_tps: float
    tokens_generated: int
    kv_size_kb: float
    cross_quant_success: bool
    error: Optional[str] = None


def load_configs():
    with open(REPO_ROOT / "config" / "experiments.yaml") as f:
        exp_cfg = yaml.safe_load(f)
    with open(REPO_ROOT / "config" / "models.yaml") as f:
        model_cfg = yaml.safe_load(f)
    return exp_cfg, model_cfg


def run_hetero_inference(
    prefill_model_path: str,
    decode_model_path: str,
    prompt: str,
    n_predict: int,
    n_ctx: int,
    n_threads: int,
    n_gpu_layers: int,
) -> dict:
    """
    In-process hetero quant test:
    1. Load prefill model, run prefill, save_state
    2. Load decode model (different quant), load_state, decode
    """
    result = {
        "ttft_ms": 0.0,
        "kv_transfer_ms": 0.0,
        "kv_size_kb": 0.0,
        "tpot_ms": 0.0,
        "tpot_std_ms": 0.0,
        "throughput_tps": 0.0,
        "tokens_generated": 0,
        "prompt_len_tokens": 0,
        "cross_quant_success": False,
        "error": None,
    }

    try:
        # Load prefill model
        prefill_llm = Llama(
            model_path=prefill_model_path,
            n_threads=n_threads,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        tokens = prefill_llm.tokenize(prompt.encode("utf-8"))
        result["prompt_len_tokens"] = len(tokens)

        # Prefill
        t0 = time.perf_counter()
        prefill_llm.eval(tokens)
        result["ttft_ms"] = (time.perf_counter() - t0) * 1000.0

        # Save KV state
        t_save = time.perf_counter()
        kv_bytes = bytes(prefill_llm.save_state())
        result["kv_size_kb"] = len(kv_bytes) / 1024.0

        # Free prefill model
        del prefill_llm

        # Load decode model (different quant)
        decode_llm = Llama(
            model_path=decode_model_path,
            n_threads=n_threads,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        # Try to load state from different quant
        t_load = time.perf_counter()
        decode_llm.load_state(kv_bytes)
        result["kv_transfer_ms"] = (time.perf_counter() - t_load) * 1000.0
        result["cross_quant_success"] = True

        # Decode
        import statistics
        token_latencies = []
        tokens_generated = 0

        for i in range(n_predict):
            t_tok = time.perf_counter()
            token_id = decode_llm.sample(top_k=1, top_p=1.0, temp=0.0, repeat_penalty=1.0)
            tpot = (time.perf_counter() - t_tok) * 1000.0
            token_latencies.append(tpot)

            if token_id == decode_llm.token_eos():
                break
            tokens_generated += 1
            decode_llm.eval([token_id])

        result["tokens_generated"] = tokens_generated
        if token_latencies:
            result["tpot_ms"] = statistics.mean(token_latencies)
            result["tpot_std_ms"] = statistics.stdev(token_latencies) if len(token_latencies) > 1 else 0.0
            total_decode_ms = sum(token_latencies)
            result["throughput_tps"] = tokens_generated / (total_decode_ms / 1000.0) if total_decode_ms > 0 else 0.0

        del decode_llm

    except Exception as e:
        result["error"] = str(e)
        result["cross_quant_success"] = False

    return result


def main():
    env = os.environ.get("LLMSCALE_ENV", "chameleon")
    print(f"=== Exp 3: Heterogeneous Quantization | env={env} ===")

    exp_cfg, model_cfg = load_configs()
    prompts = exp_cfg["colocated"]["prompts"]
    n_predict = exp_cfg["colocated"]["n_predict"]
    n_ctx = exp_cfg["colocated"]["n_ctx"]
    combos = exp_cfg["hetero_quant"]["combos"]
    n_gpu_layers = exp_cfg["local_overrides"]["n_gpu_layers"] if env == "local" else 0
    n_threads = os.cpu_count() or 8

    output_path = REPO_ROOT / "results" / "exp3_hetero_quant.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: List[HeteroResult] = []

    for model_info in model_cfg["models"]:
        model_name = model_info["name"]

        # Local: only 1B
        if env == "local" and model_name not in exp_cfg["local_overrides"]["models"]:
            continue

        for combo in combos:
            pq = combo["prefill_quant"]
            dq = combo["decode_quant"]

            prefill_path = REPO_ROOT / model_info["variants"].get(pq, "")
            decode_path = REPO_ROOT / model_info["variants"].get(dq, "")

            if not prefill_path.exists():
                print(f"SKIP: {model_name} {pq} (missing)")
                continue
            if not decode_path.exists():
                print(f"SKIP: {model_name} {dq} (missing)")
                continue

            print(f"\n--- {model_name} | prefill={pq} decode={dq} ---")

            for prompt_name, prompt_text in prompts.items():
                print(f"  Prompt: {prompt_name}")

                r = run_hetero_inference(
                    prefill_model_path=str(prefill_path),
                    decode_model_path=str(decode_path),
                    prompt=prompt_text,
                    n_predict=n_predict,
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    n_gpu_layers=n_gpu_layers,
                )

                res = HeteroResult(
                    model_name=model_name,
                    prefill_quant=pq,
                    decode_quant=dq,
                    prompt_name=prompt_name,
                    prompt_len_tokens=r["prompt_len_tokens"],
                    n_predict=n_predict,
                    ttft_ms=r["ttft_ms"],
                    kv_transfer_ms=r["kv_transfer_ms"],
                    tpot_ms=r["tpot_ms"],
                    tpot_std_ms=r["tpot_std_ms"],
                    throughput_tps=r["throughput_tps"],
                    tokens_generated=r["tokens_generated"],
                    kv_size_kb=r["kv_size_kb"],
                    cross_quant_success=r["cross_quant_success"],
                    error=r["error"],
                )
                all_results.append(res)

                status = "OK" if res.cross_quant_success else f"FAIL({res.error})"
                print(f"    cross_quant={status} ttft={res.ttft_ms:.1f}ms "
                      f"tpot={res.tpot_ms:.2f}ms kv={res.kv_size_kb:.0f}KB")

    if all_results:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
            writer.writeheader()
            for r in all_results:
                writer.writerow(asdict(r))
        print(f"\nResults: {output_path}")

        # Summary of cross-quant compatibility
        successes = sum(1 for r in all_results if r.cross_quant_success)
        print(f"\nCross-quant load_state compatibility: {successes}/{len(all_results)} success")
    else:
        print("No results (check model paths).")

    print("=== Exp 3 complete ===")


if __name__ == "__main__":
    main()

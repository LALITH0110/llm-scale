"""
Outer sweep loop for colocated baseline.
Reads config/experiments.yaml + config/models.yaml, runs benchmark_colocated
for each (model, quant, core_count, prompt) combo, writes results/exp1_colocated.csv.
"""
import os
import sys
import csv
import time
from pathlib import Path
from typing import List

import yaml
from tqdm import tqdm

# Allow imports from project root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.baseline.benchmark import benchmark_colocated, BenchmarkResult


def load_configs():
    with open(REPO_ROOT / "config" / "experiments.yaml") as f:
        exp_cfg = yaml.safe_load(f)
    with open(REPO_ROOT / "config" / "models.yaml") as f:
        model_cfg = yaml.safe_load(f)
    return exp_cfg, model_cfg


def build_sweep_configs(exp_cfg, model_cfg, env: str):
    """Build list of (model_name, quant, model_path, n_threads, prompt_name, prompt) dicts."""
    configs = []

    if env == "local":
        core_counts = exp_cfg["local_overrides"]["core_counts"]
        allowed_models = exp_cfg["local_overrides"]["models"]
        allowed_quants = exp_cfg["local_overrides"]["quants"]
        n_gpu_layers = exp_cfg["local_overrides"]["n_gpu_layers"]
    else:
        core_counts = exp_cfg["colocated"]["core_counts"]
        allowed_models = None  # all
        allowed_quants = None  # all
        n_gpu_layers = 0

    prompts = exp_cfg["colocated"]["prompts"]
    n_predict = exp_cfg["colocated"]["n_predict"]
    n_ctx = exp_cfg["colocated"]["n_ctx"]
    repetitions = exp_cfg["colocated"].get("repetitions", 1)

    for model_info in model_cfg["models"]:
        model_name = model_info["name"]
        if allowed_models and model_name not in allowed_models:
            continue

        for quant, rel_path in model_info["variants"].items():
            if allowed_quants and quant not in allowed_quants:
                continue

            model_path = REPO_ROOT / rel_path
            if not model_path.exists():
                print(f"  SKIP (missing): {model_path}")
                continue

            for n_threads in core_counts:
                for prompt_name, prompt_text in prompts.items():
                    for rep in range(repetitions):
                        configs.append({
                            "model_name": model_name,
                            "quant": quant,
                            "model_path": str(model_path),
                            "n_threads": n_threads,
                            "prompt_name": prompt_name,
                            "prompt": prompt_text,
                            "n_predict": n_predict,
                            "n_ctx": n_ctx,
                            "n_gpu_layers": n_gpu_layers,
                            "repetition": rep,
                        })
    return configs


def run_sweep(output_path: Path = None) -> List[BenchmarkResult]:
    env = os.environ.get("LLMSCALE_ENV", "chameleon")
    exp_cfg, model_cfg = load_configs()

    if output_path is None:
        output_path = REPO_ROOT / "results" / "exp1_colocated.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sweep_configs = build_sweep_configs(exp_cfg, model_cfg, env)
    if not sweep_configs:
        print("No valid sweep configs found. Check model paths and config/models.yaml.")
        return []

    print(f"Sweep: {len(sweep_configs)} configs | env={env}")

    results: List[BenchmarkResult] = []
    fieldnames = None

    with open(output_path, "w", newline="") as csvf:
        writer = None

        for cfg in tqdm(sweep_configs, desc="Sweep"):
            print(f"\n  model={cfg['model_name']} quant={cfg['quant']} "
                  f"threads={cfg['n_threads']} prompt={cfg['prompt_name']} rep={cfg['repetition']}")

            result = benchmark_colocated(
                model_path=cfg["model_path"],
                n_threads=cfg["n_threads"],
                prompt=cfg["prompt"],
                prompt_name=cfg["prompt_name"],
                n_predict=cfg["n_predict"],
                n_ctx=cfg["n_ctx"],
                n_gpu_layers=cfg["n_gpu_layers"],
            )

            row = result.to_dict()
            row["model_name"] = cfg["model_name"]
            row["quant"] = cfg["quant"]
            row["repetition"] = cfg["repetition"]

            if writer is None:
                fieldnames = list(row.keys())
                writer = csv.DictWriter(csvf, fieldnames=fieldnames)
                writer.writeheader()

            writer.writerow(row)
            csvf.flush()
            results.append(result)

            if result.error:
                print(f"  ERROR: {result.error}")
            else:
                print(f"  TTFT={result.ttft_ms:.1f}ms TPOT={result.tpot_ms:.2f}ms "
                      f"tokens={result.tokens_generated} tps={result.throughput_tps:.1f}")

    print(f"\nResults written to: {output_path}")
    return results


if __name__ == "__main__":
    run_sweep()

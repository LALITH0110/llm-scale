"""
Exp 7: CPU prefill + GPU decode (Config C6) — sanity check.
Expected: worst of both worlds (slow CPU prefill, GPU underutilized at decode).
Small run: 1 model x 1 quant only, confirming hypothesis.

Environment variables:
  PREFILL_HOST    - CPU node IP (default: localhost)
  DECODE_HOSTS    - GPU node IP (default: localhost)
  LLMSCALE_ENV    - local | chameleon

Usage:
  PREFILL_HOST=<cpu-ip> DECODE_HOSTS=<gpu-ip> python -m src.experiments.exp7_reverse_hybrid
  python -m src.experiments.exp7_reverse_hybrid --smoke
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


@dataclass
class ReverseHybridResult:
    model_id: str
    prefill_quant: str
    decode_quant: str
    output_length: int
    prompt_name: str
    prompt_len_tokens: int
    ttft_ms: float              # CPU prefill time (expected: slow)
    kv_transfer_ms: float
    tpot_ms: float              # GPU decode TPOT
    tpot_std_ms: float
    throughput_tps: float
    tokens_generated: int
    total_ms: float
    kv_size_kb: float = 0.0
    prefill_backend: str = "cpu"
    decode_backend: str = "cuda"
    error: Optional[str] = None


def load_configs():
    with open(REPO_ROOT / "config" / "experiments.yaml") as f:
        exp_cfg = yaml.safe_load(f)
    with open(REPO_ROOT / "config" / "models.yaml") as f:
        model_cfg = yaml.safe_load(f)
    return exp_cfg, model_cfg


def wait_for_server(url: str, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.5)
    return False


def launch_decode_server(model_path: str, port: int, n_threads: int,
                         model_id: str, env: str, backend: str) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "src" / "disaggregated" / "decode_server.py"),
        "--model-path", model_path,
        "--model-id", model_id,
        "--n-threads", str(n_threads),
        "--port", str(port),
        "--backend", backend,
    ]
    proc_env = os.environ.copy()
    proc_env["LLMSCALE_ENV"] = env
    return subprocess.Popen(cmd, env=proc_env)


def launch_router(model_path: str, decode_hosts: str, decode_port_base: int,
                  router_port: int, n_threads: int, env: str,
                  prefill_backend: str) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "src" / "disaggregated" / "router.py"),
        "--model-path", model_path,
        "--decode-hosts", decode_hosts,
        "--decode-port-base", str(decode_port_base),
        "--port", str(router_port),
        "--n-threads", str(n_threads),
        "--local-prefill",
        "--backend", prefill_backend,
    ]
    proc_env = os.environ.copy()
    proc_env["LLMSCALE_ENV"] = env
    return subprocess.Popen(cmd, env=proc_env)


def run_reverse_experiment(
    model_path: str,
    model_id: str,
    prefill_quant: str,
    decode_quant: str,
    output_length: int,
    prompts: dict,
    decode_port: int,
    router_port: int,
    n_threads: int,
    env: str,
) -> List[ReverseHybridResult]:
    results = []
    processes = []

    try:
        decode_host = os.environ.get("DECODE_HOSTS", "localhost").split(",")[0]
        print(f"  Launching decode server (cuda) on {decode_host}:{decode_port}")
        decode_proc = launch_decode_server(
            model_path=model_path,
            port=decode_port,
            n_threads=n_threads,
            model_id=model_id,
            env=env,
            backend="cuda",
        )
        processes.append(decode_proc)

        print(f"  Launching router (prefill=cpu) on port {router_port}")
        router_proc = launch_router(
            model_path=model_path,
            decode_hosts=decode_host,
            decode_port_base=decode_port,
            router_port=router_port,
            n_threads=n_threads,
            env=env,
            prefill_backend="cpu",
        )
        processes.append(router_proc)

        router_url = f"http://localhost:{router_port}"
        if not wait_for_server(f"{router_url}/health", timeout=150):
            raise RuntimeError("Router failed to start within 150s")

        print(f"  Servers ready. output_length={output_length}")

        for prompt_name, prompt_text in prompts.items():
            try:
                r = httpx.post(
                    f"{router_url}/generate",
                    json={
                        "prompt": prompt_text,
                        "model_id": model_id,
                        "n_predict": output_length,
                    },
                    timeout=600.0,
                )
                r.raise_for_status()
                data = r.json()
                results.append(ReverseHybridResult(
                    model_id=model_id,
                    prefill_quant=prefill_quant,
                    decode_quant=decode_quant,
                    output_length=output_length,
                    prompt_name=prompt_name,
                    prompt_len_tokens=0,
                    ttft_ms=data["ttft_ms"],
                    kv_transfer_ms=data["kv_transfer_ms"],
                    tpot_ms=data["tpot_ms"],
                    tpot_std_ms=data["tpot_std_ms"],
                    throughput_tps=data["throughput_tps"],
                    tokens_generated=data["tokens_generated"],
                    total_ms=data["total_ms"],
                ))
                print(f"    {prompt_name}: ttft={data['ttft_ms']:.1f}ms "
                      f"tpot={data['tpot_ms']:.2f}ms")
            except Exception as e:
                results.append(ReverseHybridResult(
                    model_id=model_id,
                    prefill_quant=prefill_quant,
                    decode_quant=decode_quant,
                    output_length=output_length,
                    prompt_name=prompt_name,
                    prompt_len_tokens=0,
                    ttft_ms=0, kv_transfer_ms=0, tpot_ms=0, tpot_std_ms=0,
                    throughput_tps=0, tokens_generated=0, total_ms=0,
                    error=str(e),
                ))
                print(f"    ERROR {prompt_name}: {e}")
    finally:
        for proc in processes:
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

    return results


def main(smoke: bool = False):
    env = os.environ.get("LLMSCALE_ENV", "chameleon")
    print(f"=== Exp 7: Reverse Hybrid CPU-Prefill + GPU-Decode (C6) | env={env} smoke={smoke} ===")

    exp_cfg, model_cfg = load_configs()
    prompts = exp_cfg["colocated"]["prompts"]
    rev_cfg = exp_cfg.get("reverse_hybrid_cpu_prefill_gpu_decode", {})
    output_lengths = rev_cfg.get("output_lengths", [128])
    n_ctx = rev_cfg.get("n_ctx", 4096)
    decode_port_base = rev_cfg.get("decode_port_base", 50052)
    router_port = rev_cfg.get("router_port", 8003)
    prefill_quant = rev_cfg.get("prefill_quant", "q4_0")
    decode_quant = rev_cfg.get("decode_quant", "fp16")

    # Restrict to small model list by default (sanity check, not full sweep)
    allowed_models = rev_cfg.get("models", ["llama-3.2-3b"])
    allowed_quants = rev_cfg.get("quants", ["q4_0"])

    if smoke:
        allowed_models = ["llama-3.2-1b"]
        prompts = {k: v for k, v in list(prompts.items())[:1]}

    total_cores = os.cpu_count() or 8
    n_threads = max(1, total_cores // 2)

    output_path = REPO_ROOT / "results" / "exp7_reverse_hybrid.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: List[ReverseHybridResult] = []

    for model_info in model_cfg["models"]:
        model_name = model_info["name"]
        if model_name not in allowed_models:
            continue

        # Use same model file for both prefill and decode (different backend only)
        model_rel = model_info["variants"].get(prefill_quant)
        if not model_rel:
            print(f"SKIP {model_name}: missing quant {prefill_quant}")
            continue

        model_path = str(REPO_ROOT / model_rel)
        if not Path(model_path).exists():
            print(f"SKIP (missing): {model_path}")
            continue

        model_id = f"{model_name}:{prefill_quant}"

        for output_length in output_lengths:
            print(f"\n--- {model_id} | output_length={output_length} (CPU->GPU) ---")

            results = run_reverse_experiment(
                model_path=model_path,
                model_id=model_id,
                prefill_quant=prefill_quant,
                decode_quant=decode_quant,
                output_length=output_length,
                prompts=prompts,
                decode_port=decode_port_base,
                router_port=router_port,
                n_threads=n_threads,
                env=env,
            )
            all_results.extend(results)

    if all_results:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
            writer.writeheader()
            for r in all_results:
                writer.writerow(asdict(r))
        print(f"\nResults: {output_path} ({len(all_results)} rows)")
    else:
        print("No results collected.")

    print("=== Exp 7 complete ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", "--dry-run", action="store_true",
                        help="Minimal run for harness validation.")
    args = parser.parse_args()
    main(smoke=args.smoke)

"""
Exp 5: Full GPU disaggregation (Config C4) — DistServe replica on Chameleon 10GbE.
Both prefill + decode nodes are GPU. Validates whether DistServe's KV-transfer
speedup survives on 10GbE (vs NVLink).

Environment variables:
  PREFILL_HOST    - IP/hostname of GPU prefill node (default: localhost)
  DECODE_HOSTS    - comma-separated GPU decode hosts (default: localhost)
  MODEL_PATH      - path to model file (optional override)
  LLMSCALE_ENV    - local | chameleon

Usage:
  PREFILL_HOST=10.52.0.1 DECODE_HOSTS=10.52.0.2 python -m src.experiments.exp5_gpu_disagg
  python -m src.experiments.exp5_gpu_disagg --smoke
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
from tqdm import tqdm


@dataclass
class GpuDisaggResult:
    model_id: str
    quant: str
    pd_ratio: str
    n_prefill: int
    n_decode: int
    prompt_name: str
    prompt_len_tokens: int
    n_predict: int
    ttft_ms: float
    tpot_ms: float
    tpot_std_ms: float
    throughput_tps: float
    tokens_generated: int
    kv_transfer_ms: float
    total_ms: float
    kv_size_kb: float = 0.0
    prefill_backend: str = "cuda"
    decode_backend: str = "cuda"
    error: Optional[str] = None


def load_configs():
    with open(REPO_ROOT / "config" / "experiments.yaml") as f:
        exp_cfg = yaml.safe_load(f)
    with open(REPO_ROOT / "config" / "models.yaml") as f:
        model_cfg = yaml.safe_load(f)
    return exp_cfg, model_cfg


def wait_for_server(url: str, timeout: float = 90.0) -> bool:
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


def run_experiment(
    model_path: str,
    model_id: str,
    quant: str,
    n_prefill: int,
    decode_hosts: List[str],
    decode_port_base: int,
    router_port: int,
    n_threads_prefill: int,
    n_threads_decode: int,
    prompts: dict,
    n_predict: int,
    env: str,
    pd_ratio_str: str,
    prefill_backend: str,
    decode_backend: str,
) -> List[GpuDisaggResult]:
    results = []
    processes = []

    try:
        for i, host in enumerate(decode_hosts):
            port = decode_port_base + i
            print(f"  Launching decode server ({decode_backend}) on {host}:{port}")
            proc = launch_decode_server(model_path, port, n_threads_decode, model_id, env, decode_backend)
            processes.append(proc)

        print(f"  Launching router (prefill={prefill_backend}) on port {router_port}")
        router_proc = launch_router(
            model_path=model_path,
            decode_hosts=",".join(decode_hosts),
            decode_port_base=decode_port_base,
            router_port=router_port,
            n_threads=n_threads_prefill,
            env=env,
            prefill_backend=prefill_backend,
        )
        processes.append(router_proc)

        router_url = f"http://localhost:{router_port}"
        if not wait_for_server(f"{router_url}/health", timeout=120):
            raise RuntimeError("Router failed to start within 120s")

        print(f"  Router ready. Running {len(prompts)} prompts...")

        for prompt_name, prompt_text in prompts.items():
            try:
                r = httpx.post(
                    f"{router_url}/generate",
                    json={
                        "prompt": prompt_text,
                        "model_id": model_id,
                        "n_predict": n_predict,
                    },
                    timeout=300.0,
                )
                r.raise_for_status()
                data = r.json()
                results.append(GpuDisaggResult(
                    model_id=model_id,
                    quant=quant,
                    pd_ratio=pd_ratio_str,
                    n_prefill=n_prefill,
                    n_decode=len(decode_hosts),
                    prompt_name=prompt_name,
                    prompt_len_tokens=0,
                    n_predict=n_predict,
                    ttft_ms=data["ttft_ms"],
                    tpot_ms=data["tpot_ms"],
                    tpot_std_ms=data["tpot_std_ms"],
                    throughput_tps=data["throughput_tps"],
                    tokens_generated=data["tokens_generated"],
                    kv_transfer_ms=data["kv_transfer_ms"],
                    total_ms=data["total_ms"],
                    prefill_backend=prefill_backend,
                    decode_backend=decode_backend,
                ))
                print(f"    {prompt_name}: ttft={data['ttft_ms']:.1f}ms "
                      f"tpot={data['tpot_ms']:.2f}ms "
                      f"kv_xfer={data['kv_transfer_ms']:.1f}ms")
            except Exception as e:
                results.append(GpuDisaggResult(
                    model_id=model_id,
                    quant=quant,
                    pd_ratio=pd_ratio_str,
                    n_prefill=n_prefill,
                    n_decode=len(decode_hosts),
                    prompt_name=prompt_name,
                    prompt_len_tokens=0,
                    n_predict=n_predict,
                    ttft_ms=0, tpot_ms=0, tpot_std_ms=0,
                    throughput_tps=0, tokens_generated=0,
                    kv_transfer_ms=0, total_ms=0,
                    prefill_backend=prefill_backend,
                    decode_backend=decode_backend,
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
    print(f"=== Exp 5: GPU Disaggregated Serving (C4) | env={env} smoke={smoke} ===")

    exp_cfg, model_cfg = load_configs()
    prompts = exp_cfg["colocated"]["prompts"]
    gpu_disagg_cfg = exp_cfg.get("gpu_disaggregated", {})
    n_predict = gpu_disagg_cfg.get("n_predict", 128)
    n_ctx = exp_cfg["colocated"]["n_ctx"]
    pd_ratios = gpu_disagg_cfg.get("pd_ratios", [{"prefill": 1, "decode": 1}])
    decode_port_base = gpu_disagg_cfg.get("decode_port_base", 50052)
    router_port = gpu_disagg_cfg.get("router_port", 8001)

    if smoke:
        prompts = {k: v for k, v in list(prompts.items())[:1]}
        pd_ratios = [pd_ratios[0]]

    if env == "local":
        allowed_models = exp_cfg["local_overrides"]["models"]
        allowed_quants = exp_cfg["local_overrides"]["quants"]
    elif smoke:
        allowed_models = ["llama-3.2-1b"]
        allowed_quants = ["q4_0"]
    else:
        allowed_models = None
        allowed_quants = None

    total_cores = os.cpu_count() or 8

    output_path = REPO_ROOT / "results" / "exp5_gpu_disagg.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: List[GpuDisaggResult] = []

    for model_info in model_cfg["models"]:
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

            for ratio in pd_ratios:
                n_prefill = ratio["prefill"]
                n_decode = ratio["decode"]
                pd_str = f"{n_prefill}:{n_decode}"

                print(f"\n--- {model_id} | P:D={pd_str} (GPU:GPU) ---")

                threads_per_node = max(1, total_cores // (n_prefill + n_decode))
                n_threads_prefill = threads_per_node * n_prefill
                n_threads_decode = threads_per_node

                decode_hosts_env = os.environ.get("DECODE_HOSTS", "")
                decode_hosts = decode_hosts_env.split(",")[:n_decode] if decode_hosts_env else ["localhost"] * n_decode

                results = run_experiment(
                    model_path=model_path,
                    model_id=model_id,
                    quant=quant,
                    n_prefill=n_prefill,
                    decode_hosts=decode_hosts,
                    decode_port_base=decode_port_base,
                    router_port=router_port,
                    n_threads_prefill=n_threads_prefill,
                    n_threads_decode=n_threads_decode,
                    prompts=prompts,
                    n_predict=n_predict,
                    env=env,
                    pd_ratio_str=pd_str,
                    prefill_backend="cuda",
                    decode_backend="cuda",
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

    print("=== Exp 5 complete ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", "--dry-run", action="store_true",
                        help="Minimal run for harness validation.")
    args = parser.parse_args()
    main(smoke=args.smoke)

"""
Exp 6: GPU prefill + CPU decode hybrid (Config C5) — main new contribution.

Phase A: FP16 both sides (concept proof, no quant mismatch).
Phase B: FP16 prefill + Q4 decode with cache-type-k f16 (real hybrid).

Sweeps output lengths [128, 512, 2048] to find the TTFT-saved vs KV-transfer-cost
break-even point.

Environment variables:
  PREFILL_HOST    - GPU node IP (default: localhost)
  DECODE_HOSTS    - CPU node IP (default: localhost)
  LLMSCALE_ENV    - local | chameleon

Usage:
  PREFILL_HOST=<gpu-ip> DECODE_HOSTS=<cpu-ip> python -m src.experiments.exp6_hybrid
  python -m src.experiments.exp6_hybrid --phase a --smoke
  python -m src.experiments.exp6_hybrid --phase b
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
class HybridResult:
    model_id: str
    prefill_quant: str
    decode_quant: str
    phase: str                  # "a" or "b"
    output_length: int
    prompt_name: str
    prompt_len_tokens: int
    ttft_ms: float              # GPU prefill time
    kv_transfer_ms: float       # gRPC KV transfer
    tpot_ms: float              # CPU decode TPOT
    tpot_std_ms: float
    throughput_tps: float
    tokens_generated: int
    total_ms: float
    kv_size_kb: float = 0.0
    prefill_backend: str = "cuda"
    decode_backend: str = "cpu"
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
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


def launch_decode_server(
    model_path: str, port: int, n_threads: int, model_id: str, env: str,
    backend: str, cache_type_k: str, cache_type_v: str,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "src" / "disaggregated" / "decode_server.py"),
        "--model-path", model_path,
        "--model-id", model_id,
        "--n-threads", str(n_threads),
        "--port", str(port),
        "--backend", backend,
        "--cache-type-k", cache_type_k,
        "--cache-type-v", cache_type_v,
    ]
    proc_env = os.environ.copy()
    proc_env["LLMSCALE_ENV"] = env
    return subprocess.Popen(cmd, env=proc_env)


def launch_router(
    model_path: str, decode_hosts: str, decode_port_base: int,
    router_port: int, n_threads: int, env: str, prefill_backend: str,
) -> subprocess.Popen:
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


def run_hybrid_experiment(
    prefill_model_path: str,
    decode_model_path: str,
    model_id: str,
    prefill_quant: str,
    decode_quant: str,
    phase: str,
    output_length: int,
    prompts: dict,
    decode_port: int,
    router_port: int,
    n_threads: int,
    env: str,
    prefill_backend: str,
    decode_backend: str,
    cache_type_k: str,
    cache_type_v: str,
) -> List[HybridResult]:
    results = []
    processes = []

    try:
        print(f"  Launching decode server ({decode_backend}) on port {decode_port} "
              f"cache_k={cache_type_k}")
        decode_proc = launch_decode_server(
            model_path=decode_model_path,
            port=decode_port,
            n_threads=n_threads,
            model_id=model_id,
            env=env,
            backend=decode_backend,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
        )
        processes.append(decode_proc)

        decode_host = os.environ.get("DECODE_HOSTS", "localhost").split(",")[0]
        print(f"  Launching router (prefill={prefill_backend}) on port {router_port}")
        router_proc = launch_router(
            model_path=prefill_model_path,
            decode_hosts=decode_host,
            decode_port_base=decode_port,
            router_port=router_port,
            n_threads=n_threads,
            env=env,
            prefill_backend=prefill_backend,
        )
        processes.append(router_proc)

        router_url = f"http://localhost:{router_port}"
        if not wait_for_server(f"{router_url}/health", timeout=150):
            raise RuntimeError("Router/decode failed to start within 150s")

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
                results.append(HybridResult(
                    model_id=model_id,
                    prefill_quant=prefill_quant,
                    decode_quant=decode_quant,
                    phase=phase,
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
                    prefill_backend=prefill_backend,
                    decode_backend=decode_backend,
                    cache_type_k=cache_type_k,
                    cache_type_v=cache_type_v,
                ))
                print(f"    {prompt_name}: ttft={data['ttft_ms']:.1f}ms "
                      f"kv_xfer={data['kv_transfer_ms']:.1f}ms "
                      f"tpot={data['tpot_ms']:.2f}ms")
            except Exception as e:
                results.append(HybridResult(
                    model_id=model_id,
                    prefill_quant=prefill_quant,
                    decode_quant=decode_quant,
                    phase=phase,
                    output_length=output_length,
                    prompt_name=prompt_name,
                    prompt_len_tokens=0,
                    ttft_ms=0, kv_transfer_ms=0, tpot_ms=0, tpot_std_ms=0,
                    throughput_tps=0, tokens_generated=0, total_ms=0,
                    prefill_backend=prefill_backend,
                    decode_backend=decode_backend,
                    cache_type_k=cache_type_k,
                    cache_type_v=cache_type_v,
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


def main(phase: str = "a", smoke: bool = False):
    env = os.environ.get("LLMSCALE_ENV", "chameleon")
    print(f"=== Exp 6: Hybrid GPU-Prefill + CPU-Decode (C5) | phase={phase} env={env} smoke={smoke} ===")

    exp_cfg, model_cfg = load_configs()
    prompts = exp_cfg["colocated"]["prompts"]
    hybrid_cfg = exp_cfg.get("hybrid_gpu_prefill_cpu_decode", {})
    output_lengths = hybrid_cfg.get("output_lengths", [128, 512, 2048])
    n_ctx = hybrid_cfg.get("n_ctx", 4096)
    decode_port_base = hybrid_cfg.get("decode_port_base", 50052)
    router_port = hybrid_cfg.get("router_port", 8002)
    repetitions = hybrid_cfg.get("repetitions", 3)

    if smoke:
        output_lengths = [128]
        prompts = {k: v for k, v in list(prompts.items())[:1]}
        repetitions = 1

    # Phase A: FP16 both sides
    if phase == "a":
        phase_cfg = hybrid_cfg.get("phase_a", {})
        prefill_quant = phase_cfg.get("prefill_quant", "fp16")
        decode_quant = phase_cfg.get("decode_quant", "fp16")
        cache_type_k = "f16"
        cache_type_v = "f16"
    # Phase B: FP16 prefill -> Q4 decode with f16 KV cache
    else:
        phase_cfg = hybrid_cfg.get("phase_b", {})
        prefill_quant = phase_cfg.get("prefill_quant", "fp16")
        decode_quant = phase_cfg.get("decode_quant", "q4_0")
        cache_type_k = phase_cfg.get("cache_type_k", "f16")
        cache_type_v = phase_cfg.get("cache_type_v", "f16")

    if env == "local" or smoke:
        allowed_models = ["llama-3.2-1b"]
    else:
        allowed_models = None

    total_cores = os.cpu_count() or 8
    n_threads = max(1, total_cores // 2)

    output_path = REPO_ROOT / "results" / "exp6_hybrid.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Append mode so phase A and phase B can be combined in one file
    file_exists = output_path.exists()
    all_results: List[HybridResult] = []

    for model_info in model_cfg["models"]:
        model_name = model_info["name"]
        if allowed_models and model_name not in allowed_models:
            continue

        prefill_rel = model_info["variants"].get(prefill_quant)
        decode_rel = model_info["variants"].get(decode_quant)
        if not prefill_rel or not decode_rel:
            print(f"SKIP {model_name}: missing quant {prefill_quant} or {decode_quant}")
            continue

        prefill_path = str(REPO_ROOT / prefill_rel)
        decode_path = str(REPO_ROOT / decode_rel)

        for path, label in [(prefill_path, "prefill"), (decode_path, "decode")]:
            if not Path(path).exists():
                print(f"SKIP (missing {label}): {path}")
                break
        else:
            model_id = f"{model_name}:{prefill_quant}x{decode_quant}"

            for output_length in output_lengths:
                print(f"\n--- {model_id} | phase={phase} | output_length={output_length} ---")

                results = run_hybrid_experiment(
                    prefill_model_path=prefill_path,
                    decode_model_path=decode_path,
                    model_id=model_id,
                    prefill_quant=prefill_quant,
                    decode_quant=decode_quant,
                    phase=phase,
                    output_length=output_length,
                    prompts=prompts,
                    decode_port=decode_port_base,
                    router_port=router_port,
                    n_threads=n_threads,
                    env=env,
                    prefill_backend="cuda",
                    decode_backend="cpu",
                    cache_type_k=cache_type_k,
                    cache_type_v=cache_type_v,
                )
                all_results.extend(results)

    if all_results:
        write_header = not file_exists
        with open(output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
            if write_header:
                writer.writeheader()
            for r in all_results:
                writer.writerow(asdict(r))
        print(f"\nResults appended: {output_path} ({len(all_results)} rows)")
    else:
        print("No results collected.")

    print(f"=== Exp 6 phase {phase} complete ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["a", "b"], default="a",
                        help="Phase a: FP16 both sides. Phase b: FP16 prefill + Q4 decode.")
    parser.add_argument("--smoke", "--dry-run", action="store_true",
                        help="Minimal run (1 model x output=128 x 1 rep).")
    args = parser.parse_args()
    main(phase=args.phase, smoke=args.smoke)

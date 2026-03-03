"""
Exp 1: Colocated baseline sweep.
Drives sweep.py over all (cores × models × quant × prompts).
On Chameleon: wraps each run in numactl for CPU affinity.
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def has_numactl() -> bool:
    try:
        subprocess.run(["numactl", "--hardware"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_numa_nodes() -> list[int]:
    """Return list of NUMA node IDs available on this machine."""
    try:
        result = subprocess.run(
            ["numactl", "--hardware"],
            capture_output=True, text=True, check=True
        )
        nodes = []
        for line in result.stdout.splitlines():
            if line.startswith("available:"):
                # "available: 2 nodes (0-1)"
                parts = line.split()
                n = int(parts[1])
                nodes = list(range(n))
        return nodes
    except Exception:
        return [0]


def main():
    env = os.environ.get("LLMSCALE_ENV", "chameleon")
    print(f"=== Exp 1: Colocated Baseline | env={env} ===")

    use_numa = has_numactl() and env == "chameleon"
    if use_numa:
        numa_nodes = get_numa_nodes()
        print(f"NUMA available: {len(numa_nodes)} nodes")
    else:
        numa_nodes = [None]
        if env == "chameleon":
            print("WARNING: numactl not found — running without NUMA binding")

    # Import sweep (runs in-process)
    from src.baseline.sweep import run_sweep, load_configs, build_sweep_configs
    import yaml

    exp_cfg, model_cfg = load_configs()

    if use_numa:
        # For each NUMA node, run a separate sweep bound to that node
        for node in numa_nodes:
            output_path = REPO_ROOT / "results" / f"exp1_colocated_numa{node}.csv"
            print(f"\n--- NUMA node {node} ---")

            # Set env var so benchmark.py can annotate results
            os.environ["LLMSCALE_NUMA_NODE"] = str(node)

            # We run via subprocess with numactl for proper CPU binding
            cmd = [
                "numactl",
                f"--cpunodebind={node}",
                f"--membind={node}",
                sys.executable,
                "-c",
                f"""
import sys
sys.path.insert(0, '{REPO_ROOT}')
import os
os.environ['LLMSCALE_ENV'] = '{env}'
os.environ['LLMSCALE_NUMA_NODE'] = '{node}'
from src.baseline.sweep import run_sweep
from pathlib import Path
run_sweep(Path('{output_path}'))
""",
            ]
            print(f"Running: {' '.join(cmd[:4])} ...")
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(f"WARNING: NUMA node {node} sweep exited with code {result.returncode}")
    else:
        # Single sweep (local or non-NUMA Chameleon)
        output_path = REPO_ROOT / "results" / "exp1_colocated.csv"
        run_sweep(output_path)

    print("\n=== Exp 1 complete ===")
    print("Next: make analyze  (or make exp2 for disaggregated)")


if __name__ == "__main__":
    main()

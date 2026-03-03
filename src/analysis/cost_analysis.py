"""
Cost analysis: throughput-per-dollar vs published H100/A100 baselines.
Chameleon CPU-only instances vs GPU cloud.

Uses throughput from exp1/exp2 results + known Chameleon/cloud pricing.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np

FIGURES_DIR = REPO_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "monospace",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# ---------------------------------------------------------------------------
# Published GPU baseline throughputs (tokens/sec on single GPU, 7B-class model)
# Sources: llama.cpp benchmarks, vLLM benchmarks (approximate)
# ---------------------------------------------------------------------------
GPU_BASELINES = {
    "H100 SXM (80GB)": {"tps": 120.0, "cost_hr": 3.50, "notes": "vLLM Llama-7B ~120t/s"},
    "A100 SXM (80GB)": {"tps": 80.0,  "cost_hr": 2.50, "notes": "vLLM Llama-7B ~80t/s"},
    "RTX 4090 (24GB)": {"tps": 60.0,  "cost_hr": 0.74, "notes": "llama.cpp Q4_0 ~60t/s"},
    "RTX 3090 (24GB)": {"tps": 35.0,  "cost_hr": 0.35, "notes": "llama.cpp Q4_0 ~35t/s"},
}

# Chameleon Cloud pricing (approximate, check current rates)
# Chameleon bare-metal nodes: ~$0/hr for research allocations, but include compute-cost equivalent
CHAMELEON_COST_HR = 0.50  # conservative estimate (bare-metal ~128 core node)


def load_best_colocated() -> pd.DataFrame:
    paths = list((REPO_ROOT / "results").glob("exp1_colocated*.csv"))
    if not paths:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["model_quant"] = df["model_name"] + ":" + df["quant"]
    # Best throughput per model_quant + prompt
    best = df.loc[df.groupby(["model_quant", "prompt_name"])["throughput_tps"].idxmax()]
    return best


def compute_cost_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Add tokens_per_dollar column."""
    df = df.copy()
    df["cost_hr"] = CHAMELEON_COST_HR
    df["tps_per_dollar"] = df["throughput_tps"] / df["cost_hr"]  # tokens/sec per $/hr
    # Normalize: tokens/million-tokens-cost
    df["tokens_per_dollar"] = df["throughput_tps"] * 3600 / df["cost_hr"]  # tokens/$
    return df


def plot_throughput_vs_cost(df: pd.DataFrame):
    """Scatter: throughput (t/s) vs cost ($/hr). Pareto frontier."""
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("tab10")

    # Plot CPU results
    if not df.empty:
        for i, mq in enumerate(df["model_quant"].unique()):
            sub = df[df["model_quant"] == mq]
            best_tps = sub["throughput_tps"].max()
            ax.scatter(CHAMELEON_COST_HR, best_tps, s=120, marker="o",
                       color=palette[i % len(palette)], label=f"CPU: {mq}", zorder=5)

    # Plot GPU baselines
    gpu_colors = ["#e74c3c", "#e67e22", "#8e44ad", "#2980b9"]
    for i, (name, info) in enumerate(GPU_BASELINES.items()):
        ax.scatter(info["cost_hr"], info["tps"], s=150, marker="*",
                   color=gpu_colors[i % len(gpu_colors)],
                   label=f"GPU: {name}", zorder=5, edgecolors="black", linewidths=0.5)
        ax.annotate(name, (info["cost_hr"], info["tps"]),
                    textcoords="offset points", xytext=(8, 3), fontsize=7)

    ax.set_xlabel("Cost ($/hr)")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Throughput vs Cost: CPU Disaggregated vs GPU Cloud", fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    out = FIGURES_DIR / "cost_throughput_scatter.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_tokens_per_dollar(df: pd.DataFrame):
    """Bar chart: tokens-per-dollar for CPU configs vs GPU baselines."""
    fig, ax = plt.subplots(figsize=(12, 6))

    bars_data = {}

    # CPU results
    if not df.empty:
        for mq in df["model_quant"].unique():
            sub = df[df["model_quant"] == mq]
            best_tps = sub["throughput_tps"].max()
            tokens_per_dollar = best_tps * 3600 / CHAMELEON_COST_HR
            bars_data[f"CPU\n{mq}"] = tokens_per_dollar

    # GPU baselines
    for name, info in GPU_BASELINES.items():
        tokens_per_dollar = info["tps"] * 3600 / info["cost_hr"]
        bars_data[f"GPU\n{name}"] = tokens_per_dollar

    labels = list(bars_data.keys())
    values = list(bars_data.values())
    colors = (
        [sns.color_palette("tab10")[i] for i in range(len(df["model_quant"].unique()) if not df.empty else 0)]
        + ["#e74c3c"] * len(GPU_BASELINES)
    )

    bars = ax.bar(range(len(labels)), values, color=colors[:len(labels)], edgecolor="none", alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Tokens per dollar (tokens/$)")
    ax.set_title("Cost Efficiency: Tokens Generated per Dollar", fontweight="bold")

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val/1e6:.1f}M", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = FIGURES_DIR / "tokens_per_dollar.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def print_summary_table(df: pd.DataFrame):
    print("\n=== Cost Efficiency Summary ===")
    print(f"{'System':<35} {'TPS':>8} {'$/hr':>8} {'Tok/$':>12} {'Tok/$/hr':>12}")
    print("-" * 77)

    if not df.empty:
        for mq in df["model_quant"].unique():
            sub = df[df["model_quant"] == mq]
            best_tps = sub["throughput_tps"].max()
            tpd = best_tps * 3600 / CHAMELEON_COST_HR
            print(f"  CPU {mq:<30} {best_tps:>8.1f} {CHAMELEON_COST_HR:>8.2f} {tpd:>12,.0f} {tpd:>12,.0f}")

    for name, info in GPU_BASELINES.items():
        tpd = info["tps"] * 3600 / info["cost_hr"]
        print(f"  GPU {name:<30} {info['tps']:>8.1f} {info['cost_hr']:>8.2f} {tpd:>12,.0f} {tpd:>12,.0f}")


def main():
    df = load_best_colocated()
    if df.empty:
        print("No exp1 results. Running cost analysis with GPU baselines only.")
    else:
        df = compute_cost_efficiency(df)

    plot_throughput_vs_cost(df)
    plot_tokens_per_dollar(df)
    print_summary_table(df)
    print("\nCost analysis complete.")


if __name__ == "__main__":
    main()

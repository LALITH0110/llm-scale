"""
Cost analysis — Round 2.
Produces four figures comparing all six configs (C1-C6):
  pareto_cost_latency.png          — Pareto frontier: SLO compliance vs $/Mtok
  goodput_vs_concurrency.png       — goodput (SLO-weighted TPS) vs concurrency
  hybrid_breakeven_output_length.png — hybrid break-even vs output length
  config_comparison_all_six.png    — C1-C6 bar chart: throughput + cost

Chameleon CHI@UC cost model (equivalent commercial rates, not actual allocation cost):
  CPU node (Cascade Lake 96c): $0.50/hr
  GPU node (A100 40GB equiv): $2.21/hr
  GPU node (RTX 6000 equiv):  $1.10/hr
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
# Chameleon CHI@UC cost model (equivalent commercial rates)
# ---------------------------------------------------------------------------
COST = {
    "cpu":         0.50,   # $/hr — 96c Cascade Lake bare-metal (AWS m5.24xlarge equiv)
    "gpu_a100":    2.21,   # $/hr — A100 40GB (GCP a2-highgpu-1g on-demand)
    "gpu_rtx6000": 1.10,   # $/hr — RTX 6000 (Lambda Labs A6000 equiv)
}

# Config labels
CONFIG_LABELS = {
    "C1": "C1: CPU colocated",
    "C2": "C2: CPU disagg",
    "C3": "C3: GPU colocated",
    "C4": "C4: GPU disagg",
    "C5": "C5: GPU-pf + CPU-dec\n(hybrid)",
    "C6": "C6: CPU-pf + GPU-dec\n(reverse)",
}

# Config node cost (per hour, sum of all nodes involved)
CONFIG_COST_HR = {
    "C1": COST["cpu"],
    "C2": COST["cpu"] * 2,
    "C3": COST["gpu_a100"],
    "C4": COST["gpu_a100"] * 2,
    "C5": COST["gpu_a100"] + COST["cpu"],
    "C6": COST["cpu"] + COST["gpu_a100"],
}

# SLO thresholds
SLO_TTFT_MS = 500.0
SLO_TPOT_MS = 50.0

# ---------------------------------------------------------------------------
# Published GPU baselines (for backward-compat with original plot)
# ---------------------------------------------------------------------------
GPU_BASELINES = {
    "H100 SXM (80GB)": {"tps": 120.0, "cost_hr": 3.50},
    "A100 SXM (80GB)": {"tps": 80.0,  "cost_hr": 2.50},
    "RTX 4090 (24GB)": {"tps": 60.0,  "cost_hr": 0.74},
    "RTX 3090 (24GB)": {"tps": 35.0,  "cost_hr": 0.35},
}

CHAMELEON_COST_HR = COST["cpu"]  # kept for backward compat


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_exp1() -> pd.DataFrame:
    paths = list((REPO_ROOT / "results").glob("exp1_colocated*.csv"))
    if not paths:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["config"] = "C1"
    df["cost_hr"] = CONFIG_COST_HR["C1"]
    return df


def load_exp2() -> pd.DataFrame:
    p = REPO_ROOT / "results" / "exp2_disaggregated.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["config"] = "C2"
    df["cost_hr"] = CONFIG_COST_HR["C2"]
    return df


def load_exp4() -> pd.DataFrame:
    p = REPO_ROOT / "results" / "exp4_gpu_colocated.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["config"] = "C3"
    df["cost_hr"] = CONFIG_COST_HR["C3"]
    return df


def load_exp5() -> pd.DataFrame:
    p = REPO_ROOT / "results" / "exp5_gpu_disagg.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["config"] = "C4"
    df["cost_hr"] = CONFIG_COST_HR["C4"]
    return df


def load_exp6() -> pd.DataFrame:
    p = REPO_ROOT / "results" / "exp6_hybrid.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["config"] = "C5"
    df["cost_hr"] = CONFIG_COST_HR["C5"]
    return df


def load_exp7() -> pd.DataFrame:
    p = REPO_ROOT / "results" / "exp7_reverse_hybrid.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["config"] = "C6"
    df["cost_hr"] = CONFIG_COST_HR["C6"]
    return df


def load_exp8() -> pd.DataFrame:
    p = REPO_ROOT / "results" / "exp8_gpu_batched.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["config"] = "C3-batched"
    df["cost_hr"] = CONFIG_COST_HR["C3"]
    return df


def _tps_col(df: pd.DataFrame) -> pd.Series:
    """Resolve throughput column name across different CSV schemas."""
    for col in ["throughput_tps", "throughput_batch_tps"]:
        if col in df.columns:
            return df[col]
    return pd.Series([0.0] * len(df))


def _ttft_col(df: pd.DataFrame) -> pd.Series:
    if "ttft_ms" in df.columns:
        return df["ttft_ms"]
    return pd.Series([0.0] * len(df))


def _tpot_col(df: pd.DataFrame) -> pd.Series:
    if "tpot_ms" in df.columns:
        return df["tpot_ms"]
    return pd.Series([0.0] * len(df))


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add cost and SLO columns."""
    if df.empty:
        return df
    df = df.copy()
    df["throughput_tps_"] = _tps_col(df)
    df["ttft_ms_"] = _ttft_col(df)
    df["tpot_ms_"] = _tpot_col(df)
    df["slo_pass"] = (df["ttft_ms_"] <= SLO_TTFT_MS) & (df["tpot_ms_"] <= SLO_TPOT_MS)
    df["cost_per_mtok"] = (df["cost_hr"] / 3600.0) / (df["throughput_tps_"] / 1e6 + 1e-12)
    df["goodput_tps"] = df["throughput_tps_"] * df["slo_pass"].astype(float)
    return df


# ---------------------------------------------------------------------------
# Fig 1: Pareto frontier — SLO compliance vs $/Mtok
# ---------------------------------------------------------------------------

def plot_pareto_cost_latency(frames: dict[str, pd.DataFrame]):
    """
    x-axis: latency SLO compliance rate (fraction of requests passing TTFT + TPOT SLOs)
    y-axis: $/million tokens (log scale)
    Each config is a point; Pareto-optimal configs are connected.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = sns.color_palette("tab10", n_colors=8)

    points = []
    for i, (config_id, df) in enumerate(frames.items()):
        if df.empty or "slo_pass" not in df.columns:
            continue
        slo_rate = df["slo_pass"].mean()
        cost_mtok = df["cost_per_mtok"].median()
        if cost_mtok <= 0 or not np.isfinite(cost_mtok):
            continue
        label = CONFIG_LABELS.get(config_id, config_id)
        ax.scatter(slo_rate, cost_mtok, s=180, marker="o",
                   color=palette[i % len(palette)], label=label, zorder=5)
        ax.annotate(config_id, (slo_rate, cost_mtok),
                    textcoords="offset points", xytext=(6, 4), fontsize=9, fontweight="bold")
        points.append((slo_rate, cost_mtok))

    if not points:
        # Placeholder with synthetic data if no results yet
        _plot_placeholder(ax, "No experiment results yet\n(run exp4-exp8 on Chameleon)")

    ax.set_xlabel("SLO Compliance Rate (TTFT<500ms and TPOT<50ms)", fontsize=11)
    ax.set_ylabel("Cost ($/million tokens)", fontsize=11)
    ax.set_yscale("log")
    ax.set_xlim(-0.05, 1.1)
    ax.set_title("Pareto Frontier: Latency-SLO Compliance vs Cost Efficiency", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    out = FIGURES_DIR / "pareto_cost_latency.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 2: Goodput vs concurrency (batch size as proxy for concurrency)
# ---------------------------------------------------------------------------

def plot_goodput_vs_concurrency(exp4_df: pd.DataFrame, exp8_df: pd.DataFrame):
    """
    x-axis: batch size (proxy for concurrency)
    y-axis: goodput (SLO-gated throughput, tokens/sec)
    Lines: GPU colocated (Exp4) and vLLM batched (Exp8)
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for df, label, color in [
        (exp4_df, "GPU colocated (llama.cpp CUDA)", "#2980b9"),
        (exp8_df, "GPU batched (vLLM)", "#e74c3c"),
    ]:
        if df.empty or "batch_size" not in df.columns:
            continue
        grp = df.groupby("batch_size")["goodput_tps"].median().reset_index()
        ax.plot(grp["batch_size"], grp["goodput_tps"], marker="o",
                color=color, label=label, linewidth=2)

    ax.set_xlabel("Batch Size (concurrency proxy)", fontsize=11)
    ax.set_ylabel("Goodput (tokens/sec, SLO-gated)", fontsize=11)
    ax.set_title("GPU Goodput vs Concurrency\n(SLO: TTFT<500ms, TPOT<50ms)", fontweight="bold")
    ax.legend(fontsize=9)

    if exp4_df.empty and exp8_df.empty:
        _plot_placeholder(ax, "No experiment results yet\n(run exp4 and exp8 on Chameleon)")

    plt.tight_layout()
    out = FIGURES_DIR / "goodput_vs_concurrency.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 3: Hybrid break-even vs output length
# ---------------------------------------------------------------------------

def plot_hybrid_breakeven(hybrid_df: pd.DataFrame, cpu_df: pd.DataFrame):
    """
    x-axis: output length (tokens)
    y-axis: total latency (ms) or throughput (tps)
    Compare C5 hybrid vs C1 CPU baseline vs C3 GPU colocated.
    Break-even: output length where hybrid TTFT saving > KV transfer cost.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: total latency
    ax = axes[0]
    for df, label, color in [
        (hybrid_df, "C5: GPU-pf + CPU-dec (hybrid)", "#27ae60"),
        (cpu_df, "C1: CPU colocated", "#2980b9"),
    ]:
        if df.empty or "output_length" not in df.columns and "n_predict" not in df.columns:
            continue
        len_col = "output_length" if "output_length" in df.columns else "n_predict"
        if "total_ms" not in df.columns:
            continue
        grp = df.groupby(len_col)["total_ms"].median().reset_index()
        ax.plot(grp[len_col], grp["total_ms"], marker="o",
                color=color, label=label, linewidth=2)

    ax.set_xlabel("Output Length (tokens)", fontsize=10)
    ax.set_ylabel("Total Latency (ms)", fontsize=10)
    ax.set_title("Hybrid Break-even: Total Latency", fontweight="bold")
    ax.legend(fontsize=8)

    # Right: throughput
    ax = axes[1]
    for df, label, color in [
        (hybrid_df, "C5: hybrid", "#27ae60"),
        (cpu_df, "C1: CPU colocated", "#2980b9"),
    ]:
        if df.empty:
            continue
        len_col = "output_length" if "output_length" in df.columns else "n_predict"
        if len_col not in df.columns:
            continue
        grp = df.groupby(len_col)["throughput_tps_"].median().reset_index() if "throughput_tps_" in df.columns else pd.DataFrame()
        if grp.empty:
            continue
        ax.plot(grp[len_col], grp["throughput_tps_"], marker="s",
                color=color, label=label, linewidth=2)

    ax.set_xlabel("Output Length (tokens)", fontsize=10)
    ax.set_ylabel("Throughput (tokens/sec)", fontsize=10)
    ax.set_title("Hybrid Break-even: Throughput", fontweight="bold")
    ax.legend(fontsize=8)

    if hybrid_df.empty:
        for ax in axes:
            _plot_placeholder(ax, "No hybrid results yet\n(run exp6 on Chameleon)")

    plt.tight_layout()
    out = FIGURES_DIR / "hybrid_breakeven_output_length.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 4: Config comparison C1-C6 bar chart
# ---------------------------------------------------------------------------

def plot_config_comparison_all_six(frames: dict[str, pd.DataFrame]):
    """
    Grouped bar chart: throughput (primary y) + $/Mtok (secondary y)
    One group per config, bars = model families.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    palette = sns.color_palette("tab10", n_colors=len(frames))

    configs = list(CONFIG_LABELS.keys())
    x = np.arange(len(configs))
    bar_w = 0.55

    tps_vals = []
    cost_vals = []

    for cfg in configs:
        df = frames.get(cfg, pd.DataFrame())
        if df.empty or "throughput_tps_" not in df.columns:
            tps_vals.append(0.0)
            cost_vals.append(0.0)
        else:
            tps_vals.append(df["throughput_tps_"].median())
            cost_vals.append(df["cost_per_mtok"].replace([np.inf, -np.inf], np.nan).median())

    colors = [palette[i] for i in range(len(configs))]

    bars1 = ax1.bar(x, tps_vals, width=bar_w, color=colors, edgecolor="none", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([CONFIG_LABELS[c].replace("\n", "\n") for c in configs],
                        fontsize=8, rotation=20, ha="right")
    ax1.set_ylabel("Median Throughput (tokens/sec)", fontsize=10)
    ax1.set_title("Throughput by Config (C1-C6)", fontweight="bold")
    for bar, val in zip(bars1, tps_vals):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                     f"{val:.0f}", ha="center", va="bottom", fontsize=8)

    bars2 = ax2.bar(x, cost_vals, width=bar_w, color=colors, edgecolor="none", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([CONFIG_LABELS[c].replace("\n", "\n") for c in configs],
                        fontsize=8, rotation=20, ha="right")
    ax2.set_ylabel("Median Cost ($/million tokens)", fontsize=10)
    ax2.set_title("Cost Efficiency by Config (C1-C6)", fontweight="bold")
    for bar, val in zip(bars2, cost_vals):
        if val and val > 0 and np.isfinite(val):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                     f"${val:.3f}", ha="center", va="bottom", fontsize=8)

    if all(v == 0 for v in tps_vals):
        _plot_placeholder(ax1, "No results yet")
        _plot_placeholder(ax2, "Run all experiments first")

    plt.tight_layout()
    out = FIGURES_DIR / "config_comparison_all_six.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Legacy plots (kept for backward compat)
# ---------------------------------------------------------------------------

def load_best_colocated() -> pd.DataFrame:
    df = load_exp1()
    if df.empty:
        return df
    df["model_quant"] = df.get("model_name", df.get("model_id", "?")).astype(str) + ":" + df.get("quant", "").astype(str)
    best = df.loc[df.groupby(["model_quant", "prompt_name"])["throughput_tps"].idxmax()]
    return best


def plot_throughput_vs_cost(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("tab10")

    if not df.empty and "model_quant" in df.columns:
        for i, mq in enumerate(df["model_quant"].unique()):
            sub = df[df["model_quant"] == mq]
            best_tps = sub["throughput_tps"].max()
            ax.scatter(CHAMELEON_COST_HR, best_tps, s=120, marker="o",
                       color=palette[i % len(palette)], label=f"CPU: {mq}", zorder=5)

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
    fig, ax = plt.subplots(figsize=(12, 6))
    bars_data = {}

    if not df.empty and "model_quant" in df.columns:
        for mq in df["model_quant"].unique():
            sub = df[df["model_quant"] == mq]
            best_tps = sub["throughput_tps"].max()
            bars_data[f"CPU\n{mq}"] = best_tps * 3600 / CHAMELEON_COST_HR

    for name, info in GPU_BASELINES.items():
        bars_data[f"GPU\n{name}"] = info["tps"] * 3600 / info["cost_hr"]

    labels = list(bars_data.keys())
    values = list(bars_data.values())
    n_cpu = len(df["model_quant"].unique()) if not df.empty and "model_quant" in df.columns else 0
    colors = (
        [sns.color_palette("tab20")[i] for i in range(n_cpu)]
        + ["#e74c3c"] * len(GPU_BASELINES)
    )

    bars = ax.bar(range(len(labels)), values, color=colors[:len(labels)], edgecolor="none", alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Tokens per dollar (tokens/$)")
    ax.set_title("Cost Efficiency: Tokens Generated per Dollar", fontweight="bold")
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
    print(f"{'System':<35} {'TPS':>8} {'$/hr':>8} {'Tok/$':>12}")
    print("-" * 67)
    if not df.empty and "model_quant" in df.columns:
        for mq in df["model_quant"].unique():
            sub = df[df["model_quant"] == mq]
            best_tps = sub["throughput_tps"].max()
            tpd = best_tps * 3600 / CHAMELEON_COST_HR
            print(f"  CPU {mq:<30} {best_tps:>8.1f} {CHAMELEON_COST_HR:>8.2f} {tpd:>12,.0f}")
    for name, info in GPU_BASELINES.items():
        tpd = info["tps"] * 3600 / info["cost_hr"]
        print(f"  GPU {name:<30} {info['tps']:>8.1f} {info['cost_hr']:>8.2f} {tpd:>12,.0f}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plot_placeholder(ax, msg: str):
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, fontsize=11, color="gray",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8", edgecolor="lightgray"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Cost Analysis (Round 2) ===")

    # Load all experiment results
    e1 = enrich(load_exp1())
    e2 = enrich(load_exp2())
    e4 = enrich(load_exp4())
    e5 = enrich(load_exp5())
    e6 = enrich(load_exp6())
    e7 = enrich(load_exp7())
    e8 = enrich(load_exp8())

    frames = {
        "C1": e1, "C2": e2, "C3": e4,
        "C4": e5, "C5": e6, "C6": e7,
    }

    # New Round 2 figures
    plot_pareto_cost_latency(frames)
    plot_goodput_vs_concurrency(e4, e8)
    plot_hybrid_breakeven(e6, e1)
    plot_config_comparison_all_six(frames)

    # Legacy figures (kept)
    legacy_df = load_best_colocated()
    if not legacy_df.empty:
        plot_throughput_vs_cost(legacy_df)
        plot_tokens_per_dollar(legacy_df)
        print_summary_table(legacy_df)

    print("\nCost analysis complete.")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

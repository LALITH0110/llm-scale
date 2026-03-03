"""
Plot TTFT and TPOT vs core count for colocated baseline (Exp 1).
Generates: results/figures/scaling_ttft.png, scaling_tpot.png, scaling_throughput.png
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

PALETTE = sns.color_palette("tab10")

# Style: clean, no chart junk
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "monospace",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def load_data() -> pd.DataFrame:
    paths = list((REPO_ROOT / "results").glob("exp1_colocated*.csv"))
    if not paths:
        raise FileNotFoundError("No exp1_colocated*.csv found in results/. Run: make exp1")
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    df["model_quant"] = df["model_name"] + ":" + df["quant"]
    return df


def plot_metric_vs_cores(df: pd.DataFrame, metric: str, ylabel: str, title: str, fname: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    prompts = df["prompt_name"].unique()

    for ax, prompt in zip(axes, prompts):
        sub = df[df["prompt_name"] == prompt]
        for i, mq in enumerate(sub["model_quant"].unique()):
            grp = sub[sub["model_quant"] == mq].groupby("n_threads")[metric]
            means = grp.mean()
            stds = grp.std().fillna(0)

            ax.plot(means.index, means.values, marker="o", label=mq,
                    color=PALETTE[i % len(PALETTE)], linewidth=1.5)
            ax.fill_between(means.index,
                            means.values - stds.values,
                            means.values + stds.values,
                            alpha=0.15, color=PALETTE[i % len(PALETTE)])

        ax.set_title(prompt.replace("_", " "), fontsize=10)
        ax.set_xlabel("CPU threads")
        ax.set_ylabel(ylabel if ax == axes[0] else "")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=7, framealpha=0.5)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / fname
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_scaling_efficiency(df: pd.DataFrame):
    """Throughput scaling efficiency: actual / ideal (linear)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, mq in enumerate(df["model_quant"].unique()):
        sub = df[(df["model_quant"] == mq) & (df["prompt_name"] == "short_128")]
        grp = sub.groupby("n_threads")["throughput_tps"].mean()
        if grp.empty:
            continue

        baseline_threads = grp.index.min()
        baseline_tps = grp[baseline_threads]
        efficiency = grp / (baseline_tps * grp.index / baseline_threads)

        ax.plot(grp.index, efficiency.values, marker="s", label=mq,
                color=PALETTE[i % len(PALETTE)], linewidth=1.5)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="ideal linear")
    ax.set_xlabel("CPU threads")
    ax.set_ylabel("Scaling efficiency (actual / ideal)")
    ax.set_title("Throughput Scaling Efficiency (short prompts)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.5)

    out = FIGURES_DIR / "scaling_efficiency.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    try:
        df = load_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    print(f"Loaded {len(df)} rows, {df['model_quant'].nunique()} model/quant combos")

    plot_metric_vs_cores(df, "ttft_ms", "TTFT (ms)", "Time to First Token vs CPU Threads", "scaling_ttft.png")
    plot_metric_vs_cores(df, "tpot_ms", "TPOT (ms)", "Time Per Output Token vs CPU Threads", "scaling_tpot.png")
    plot_metric_vs_cores(df, "throughput_tps", "Throughput (tokens/s)", "Decode Throughput vs CPU Threads", "scaling_throughput.png")
    plot_metric_vs_cores(df, "mem_bw_est_gbps", "Est. Mem BW (GB/s)", "Estimated Memory Bandwidth vs CPU Threads", "scaling_membw.png")
    plot_scaling_efficiency(df)

    print("Analysis complete. Figures in results/figures/")


if __name__ == "__main__":
    main()

"""
Compare colocated vs disaggregated: bar charts for TTFT, TPOT, throughput.
Also plots P:D ratio comparison for Exp 2.
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


def load_colocated() -> pd.DataFrame:
    paths = list((REPO_ROOT / "results").glob("exp1_colocated*.csv"))
    if not paths:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["model_quant"] = df["model_name"] + ":" + df["quant"]
    df["mode"] = "colocated"
    return df


def load_disaggregated() -> pd.DataFrame:
    path = REPO_ROOT / "results" / "exp2_disaggregated.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["mode"] = "disaggregated"
    return df


def plot_colocated_vs_disagg(col: pd.DataFrame, dis: pd.DataFrame):
    """Side-by-side TTFT comparison."""
    if col.empty or dis.empty:
        print("Need both exp1 and exp2 results for comparison plot.")
        return

    # Use best (max threads) colocated config
    best_col = col.loc[col.groupby(["model_quant", "prompt_name"])["n_threads"].idxmax()]
    best_col = best_col[["model_quant", "prompt_name", "ttft_ms", "tpot_ms", "throughput_tps"]]
    best_col["pd_ratio"] = "colocated"

    dis_sub = dis[["model_id", "prompt_name", "ttft_ms", "tpot_ms", "throughput_tps", "pd_ratio"]].copy()
    dis_sub = dis_sub.rename(columns={"model_id": "model_quant"})

    combined = pd.concat([
        best_col.assign(pd_ratio="colocated"),
        dis_sub
    ], ignore_index=True)

    model_quants = combined["model_quant"].unique()

    fig, axes = plt.subplots(len(model_quants), 3, figsize=(15, 5 * len(model_quants)))
    if len(model_quants) == 1:
        axes = axes.reshape(1, -1)

    metrics = [("ttft_ms", "TTFT (ms)"), ("tpot_ms", "TPOT (ms)"), ("throughput_tps", "Throughput (t/s)")]

    for row, mq in enumerate(model_quants):
        sub = combined[combined["model_quant"] == mq]
        for col_idx, (metric, label) in enumerate(metrics):
            ax = axes[row, col_idx]
            pivot = sub.pivot_table(index="prompt_name", columns="pd_ratio", values=metric, aggfunc="mean")
            pivot.plot(kind="bar", ax=ax, colormap="tab10", width=0.7, edgecolor="none")
            ax.set_title(f"{mq}\n{label}", fontsize=9)
            ax.set_xlabel("")
            ax.set_ylabel(label if col_idx == 0 else "")
            ax.tick_params(axis="x", rotation=30)
            ax.legend(title="P:D", fontsize=7)

    fig.suptitle("Colocated vs Disaggregated: TTFT / TPOT / Throughput", fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "comparison_colocated_vs_disagg.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_pd_ratio_impact(dis: pd.DataFrame):
    """TTFT / TPOT vs P:D ratio."""
    if dis.empty:
        print("No exp2 data for P:D ratio plot.")
        return

    model_ids = dis["model_id"].unique()
    fig, axes = plt.subplots(len(model_ids), 2, figsize=(12, 5 * len(model_ids)))
    if len(model_ids) == 1:
        axes = axes.reshape(1, -1)

    for row, mid in enumerate(model_ids):
        sub = dis[dis["model_id"] == mid]
        for col_idx, (metric, label) in enumerate([("ttft_ms", "TTFT (ms)"), ("tpot_ms", "TPOT (ms)")]):
            ax = axes[row, col_idx]
            pivot = sub.groupby(["pd_ratio", "prompt_name"])[metric].mean().unstack()
            pivot.plot(kind="bar", ax=ax, width=0.7, edgecolor="none")
            ax.set_title(f"{mid}\n{label}", fontsize=9)
            ax.set_xlabel("P:D ratio")
            ax.set_ylabel(label if col_idx == 0 else "")
            ax.tick_params(axis="x", rotation=0)
            ax.legend(title="prompt", fontsize=7)

    fig.suptitle("Impact of P:D Ratio on Latency", fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "pd_ratio_impact.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    col = load_colocated()
    dis = load_disaggregated()

    if col.empty:
        print("No exp1 data. Run: make exp1")
    if dis.empty:
        print("No exp2 data. Run: make exp2")

    plot_colocated_vs_disagg(col, dis)
    plot_pd_ratio_impact(dis)
    print("Comparison analysis complete.")


if __name__ == "__main__":
    main()

"""
KV cache transfer overhead analysis.
- KV transfer latency vs model size / quant level
- KV state size vs context length
- Transfer overhead as % of total latency
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
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

# Approximate model sizes in GB (on disk) for annotation
MODEL_SIZES_GB = {
    "llama-3.2-1b:fp16": 2.5,
    "llama-3.2-1b:q8_0": 1.3,
    "llama-3.2-1b:q4_0": 0.7,
    "llama-3.2-1b:q2_k": 0.4,
    "llama-3.2-3b:fp16": 6.4,
    "llama-3.2-3b:q8_0": 3.4,
    "llama-3.2-3b:q4_0": 1.9,
    "llama-3.2-3b:q2_k": 1.1,
    "deepseek-7b:fp16": 14.0,
    "deepseek-7b:q8_0": 7.7,
    "deepseek-7b:q4_0": 4.1,
    "deepseek-7b:q2_k": 2.5,
}


def load_exp2() -> pd.DataFrame:
    p = REPO_ROOT / "results" / "exp2_disaggregated.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_exp3() -> pd.DataFrame:
    p = REPO_ROOT / "results" / "exp3_hetero_quant.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def plot_kv_transfer_vs_model(df: pd.DataFrame):
    """KV transfer latency vs model_id (bar chart)."""
    if df.empty:
        return

    grp = df.groupby("model_id")["kv_transfer_ms"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(grp))
    bars = ax.bar(x, grp["mean"], yerr=grp["std"].fillna(0), capsize=4,
                  color=sns.color_palette("tab10", len(grp)), edgecolor="none", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(grp["model_id"].tolist(), rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("KV transfer latency (ms)")
    ax.set_title("KV Cache Transfer Latency by Model", fontweight="bold")

    # Annotate with model size
    for bar, model_id in zip(bars, grp["model_id"]):
        size = MODEL_SIZES_GB.get(model_id, "?")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{size}GB", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out = FIGURES_DIR / "kv_transfer_latency.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_kv_overhead_fraction(df: pd.DataFrame):
    """KV transfer as % of total end-to-end latency."""
    if df.empty or "kv_transfer_ms" not in df.columns or "total_ms" not in df.columns:
        return

    df = df.copy()
    df["kv_frac"] = df["kv_transfer_ms"] / df["total_ms"].clip(lower=1e-9)

    fig, ax = plt.subplots(figsize=(10, 5))
    order = df.groupby("model_id")["kv_frac"].mean().sort_values(ascending=False).index

    sns.boxplot(data=df, x="model_id", y="kv_frac", order=order, ax=ax,
                palette="tab10", width=0.5)
    ax.axhline(0.1, color="red", linestyle="--", linewidth=1, label="10% threshold")
    ax.set_ylabel("KV transfer / total latency")
    ax.set_xlabel("")
    ax.set_title("KV Transfer Overhead Fraction", fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.legend()

    plt.tight_layout()
    out = FIGURES_DIR / "kv_overhead_fraction.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_hetero_quant_summary(df3: pd.DataFrame):
    """Cross-quant success heatmap from Exp 3."""
    if df3.empty:
        return

    # Pivot: prefill_quant × decode_quant → success rate
    pivot = df3.groupby(["prefill_quant", "decode_quant"])["cross_quant_success"].mean().unstack()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".0%", cmap="RdYlGn", ax=ax,
                vmin=0, vmax=1, linewidths=0.5, cbar_kws={"label": "Success rate"})
    ax.set_title("load_state() Cross-Quantization Compatibility\n(1.0 = works, 0.0 = fails)",
                 fontweight="bold")
    ax.set_xlabel("Decode quant")
    ax.set_ylabel("Prefill quant")

    plt.tight_layout()
    out = FIGURES_DIR / "hetero_quant_compat.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # TPOT comparison: same quant vs cross quant
    if "tpot_ms" in df3.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        df3["combo"] = df3["prefill_quant"] + " → " + df3["decode_quant"]
        order = df3[df3["cross_quant_success"]].groupby("combo")["tpot_ms"].mean().sort_values().index
        sns.barplot(data=df3[df3["cross_quant_success"]], x="combo", y="tpot_ms",
                    order=order, ax=ax, palette="tab10", errorbar="sd")
        ax.set_xlabel("prefill_quant → decode_quant")
        ax.set_ylabel("TPOT (ms)")
        ax.set_title("TPOT by Hetero Quant Combo", fontweight="bold")
        ax.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        out = FIGURES_DIR / "hetero_quant_tpot.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


def main():
    df2 = load_exp2()
    df3 = load_exp3()

    if df2.empty:
        print("No exp2 data. Run: make exp2")
    if df3.empty:
        print("No exp3 data. Run: make exp3")

    plot_kv_transfer_vs_model(df2)
    plot_kv_overhead_fraction(df2)
    plot_hetero_quant_summary(df3)
    print("KV overhead analysis complete.")


if __name__ == "__main__":
    main()

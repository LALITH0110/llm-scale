"""
Hybrid-specific plots for Exp 6 (GPU prefill + CPU decode).

Figures produced:
  hybrid_ttft_breakdown.png        — stacked bar: GPU prefill / KV transfer / CPU decode
  hybrid_breakeven_output_length.png (also in cost_analysis.py; here with more detail)
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

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


def load_hybrid() -> pd.DataFrame:
    p = REPO_ROOT / "results" / "exp6_hybrid.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_cpu_baseline() -> pd.DataFrame:
    """Load exp1 (CPU colocated) for comparison."""
    paths = list((REPO_ROOT / "results").glob("exp1_colocated*.csv"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def load_gpu_colocated() -> pd.DataFrame:
    p = REPO_ROOT / "results" / "exp4_gpu_colocated.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _placeholder(ax, msg: str):
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, fontsize=11, color="gray",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8", edgecolor="lightgray"))


# ---------------------------------------------------------------------------
# Fig A: Stacked bar — TTFT breakdown
# ---------------------------------------------------------------------------

def plot_ttft_breakdown(hybrid_df: pd.DataFrame):
    """
    Stacked horizontal bar per model:
      segment 1 (GPU prefill)  = ttft_ms
      segment 2 (KV transfer)  = kv_transfer_ms
      segment 3 (CPU decode)   = tpot_ms * tokens_generated  (total decode time)
    Grouped by output_length on x-axis.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    if hybrid_df.empty:
        _placeholder(ax, "No hybrid results yet\n(run exp6 on Chameleon)")
        plt.tight_layout()
        out = FIGURES_DIR / "hybrid_ttft_breakdown.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"Saved (placeholder): {out}")
        return

    required = {"ttft_ms", "kv_transfer_ms", "tpot_ms", "tokens_generated", "output_length", "model_id"}
    if not required.issubset(hybrid_df.columns):
        _placeholder(ax, f"Missing columns: {required - set(hybrid_df.columns)}")
        plt.tight_layout()
        out = FIGURES_DIR / "hybrid_ttft_breakdown.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"Saved (placeholder, missing cols): {out}")
        return

    # Compute decode total ms
    df = hybrid_df.copy()
    df["decode_total_ms"] = df["tpot_ms"] * df["tokens_generated"]

    # Group by output_length + phase; take median across models/prompts
    grp = df.groupby(["output_length", "phase"]).agg(
        ttft_ms=("ttft_ms", "median"),
        kv_transfer_ms=("kv_transfer_ms", "median"),
        decode_total_ms=("decode_total_ms", "median"),
    ).reset_index()

    output_lengths = sorted(grp["output_length"].unique())
    phases = sorted(grp["phase"].unique())
    n_groups = len(output_lengths)
    n_phases = len(phases)
    x = np.arange(n_groups)
    bar_w = 0.35
    phase_colors = {"a": "#3498db", "b": "#e74c3c"}

    for pi, phase in enumerate(phases):
        sub = grp[grp["phase"] == phase].set_index("output_length").reindex(output_lengths)
        offset = (pi - (n_phases - 1) / 2) * bar_w

        bottoms = np.zeros(n_groups)
        for label, col, color in [
            ("GPU prefill", "ttft_ms", "#f39c12"),
            ("KV transfer", "kv_transfer_ms", "#e74c3c"),
            ("CPU decode", "decode_total_ms", "#27ae60"),
        ]:
            vals = sub[col].fillna(0).values
            ax.bar(x + offset, vals, bar_w, bottom=bottoms,
                   label=f"{label} (phase {phase})" if pi == 0 else "_nolegend_",
                   color=color, alpha=0.75 if pi == 0 else 0.45,
                   edgecolor="none")
            bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"output={ol}" for ol in output_lengths], fontsize=9)
    ax.set_ylabel("Time (ms)", fontsize=11)
    ax.set_title("Hybrid Latency Breakdown: GPU Prefill / KV Transfer / CPU Decode",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    out = FIGURES_DIR / "hybrid_ttft_breakdown.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig B: Break-even output length — hybrid vs CPU vs GPU
# ---------------------------------------------------------------------------

def plot_breakeven_detail(hybrid_df: pd.DataFrame, cpu_df: pd.DataFrame, gpu_df: pd.DataFrame):
    """
    Three lines on one plot:
      - C1 CPU colocated: total latency vs output length
      - C3 GPU colocated: total latency vs output length
      - C5 Hybrid: total latency vs output length
    Intersection = break-even output length.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    series_list = [
        (hybrid_df, "C5: GPU-pf + CPU-dec (hybrid)", "#27ae60", "output_length"),
        (cpu_df,    "C1: CPU colocated",              "#2980b9", "n_predict"),
        (gpu_df,    "C3: GPU colocated",              "#e74c3c", "n_predict"),
    ]

    for ax_idx, metric in [(0, "total_ms"), (1, "throughput_tps")]:
        ax = axes[ax_idx]
        has_data = False

        for df, label, color, len_col in series_list:
            if df.empty or len_col not in df.columns or metric not in df.columns:
                continue
            grp = df.groupby(len_col)[metric].median().reset_index()
            ax.plot(grp[len_col], grp[metric], marker="o", color=color,
                    label=label, linewidth=2)
            has_data = True

        ax.set_xlabel("Output Length (tokens)", fontsize=10)
        ax.set_ylabel("Total Latency (ms)" if ax_idx == 0 else "Throughput (tokens/sec)",
                      fontsize=10)
        title = ("Break-even: Total Latency" if ax_idx == 0
                 else "Break-even: Throughput")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)

        if not has_data:
            _placeholder(ax, "No results yet\n(run exp1, exp4, exp6)")

    plt.tight_layout()
    out = FIGURES_DIR / "hybrid_breakeven_output_length.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Hybrid Plots ===")

    hybrid_df = load_hybrid()
    cpu_df = load_cpu_baseline()
    gpu_df = load_gpu_colocated()

    plot_ttft_breakdown(hybrid_df)
    plot_breakeven_detail(hybrid_df, cpu_df, gpu_df)

    print("Hybrid plots complete.")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Final

# ---------- Data ----------
df: pd.DataFrame = pd.read_csv("results/benchmark_real.csv")
ok_df: pd.DataFrame = df[df["status"] == "ok"].copy()

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12})

PALETTE: Final[dict[str, str]] = {
    "BS": "#1f77b4",
    "Greedy": "#ff7f0e",
}

OUTPUT_DIR: Final[str] = os.path.join(os.getcwd(), "Final_benchmark_graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Plot Function ----------
def generate_plot(
    metric: str,
    y_label: str,
    title: str,
    filename: str,
    log_y: bool = False,
    log_x: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        data=ok_df,
        x="n",
        y=metric,
        hue="algo",
        style="k",
        alpha=0.7,
        s=50,
        palette=PALETTE,
        ax=ax,
    )

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("Number of Nodes (n)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(title="Algorithm / k", bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ---------- Generate Figures ----------
generate_plot(
    "sparseness",
    "Sparseness (|E_H| / n)",
    "Spanner Sparseness vs. Graph Size",
    "fig_sparseness.png",
    log_y=True,
)

generate_plot(
    "lightness",
    "Lightness (w(E_H) / w(MST))",
    "Spanner Lightness vs. Graph Size",
    "fig_lightness.png",
    log_y=True,
)

generate_plot(
    "eff_stretch",
    "Effective Stretch (max d_H / d_G)",
    "Effective Stretch vs. Graph Size",
    "fig_stretch.png",
)

generate_plot(
    "runtime_s",
    "Runtime (seconds)",
    "Execution Time vs. Graph Size",
    "fig_runtime.png",
    log_y=True,
)

# ---------- NEW: Peak Memory Plot ----------
generate_plot(
    "peak_mem_kb",
    "Peak Memory (KB)",
    "Peak Memory vs. Graph Size",
    "fig_peak_memory.png",
    log_y=True,
)
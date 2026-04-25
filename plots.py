from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Final

# ---------- Data ----------
df: pd.DataFrame = pd.read_csv("results/benchmark_real.csv")
ok_df: pd.DataFrame = df[df["status"] == "ok"].copy()

# ════════════════════════════════════════════════════════════
# RESULTS TALLY
# ════════════════════════════════════════════════════════════

# ── 1. Total unique graph instances ──────────────────────────────────────────
unique_instances = df[["family", "instance"]].drop_duplicates()
total_unique = len(unique_instances)

print("=" * 60)
print("UNIQUE GRAPH INSTANCES TALLY")
print("=" * 60)
print(f"\nTotal unique instances (across all families): {total_unique}")

# ── 2. Unique instances by family ─────────────────────────────────────────────
by_family = (
    df[["family", "instance"]]
    .drop_duplicates()
    .groupby("family")
    .size()
    .rename("unique_instances")
    .sort_values(ascending=False)
)
print("\nUnique instances per family:")
print(by_family.to_string())

# ── 3. Execution status breakdown (rows, not instances) ──────────────────────
status_counts = df["status"].value_counts()
print("\n" + "=" * 60)
print("EXECUTION STATUS BREAKDOWN  (row-level counts)")
print("=" * 60)
print(status_counts.to_string())

# Unique instances that saw at least one skipped row
skipped_instances = (
    df[df["status"] == "skipped_greedy_n_gt_500"][["family", "instance"]]
    .drop_duplicates()
)
print(f"\nUnique instances with at least one skipped row: {len(skipped_instances)}")

# ── 4. Algorithm-level unique instance counts ─────────────────────────────────
bs_ok = (
    df[(df["algo"] == "BS") & (df["status"] == "ok")][["family", "instance"]]
    .drop_duplicates()
)
greedy_ok = (
    df[(df["algo"] == "Greedy") & (df["status"] == "ok")][["family", "instance"]]
    .drop_duplicates()
)

print("\n" + "=" * 60)
print("ALGORITHM-LEVEL UNIQUE INSTANCE COUNTS  (status == ok)")
print("=" * 60)
print(f"  Baswana-Sen (BS)  successfully evaluated: {len(bs_ok):>5}")
print(f"  Greedy            successfully evaluated: {len(greedy_ok):>5}")

both_ok = pd.merge(bs_ok, greedy_ok, on=["family", "instance"])
print(f"  Both algorithms   successfully evaluated: {len(both_ok):>5}")

# ── 5. Final verified count ───────────────────────────────────────────────────
if "verified" in df.columns:
    verified_instances = (
        df[df["verified"] == True][["family", "instance"]]
        .drop_duplicates()
    )
    final_valid = len(verified_instances)
else:
    # Fall back: instances where every row is ok
    all_ok = df.groupby(["family", "instance"])["status"].apply(
        lambda s: (s == "ok").all()
    )
    final_valid = all_ok.sum()

print("\n" + "=" * 60)
print(f"FINAL VERIFIED COUNT OF VALID INSTANCES: {final_valid}")
print("=" * 60)

# ════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════

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
    fig, ax = plt.subplots(figsize=(8, 6))  # pyright: ignore[reportUnknownMemberType]

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
        ax.set_xscale("log")  # pyright: ignore[reportUnknownMemberType]
    if log_y:
        ax.set_yscale("log")  # pyright: ignore[reportUnknownMemberType]

    ax.set_xlabel("Number of Nodes (n)")  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel(y_label)  # pyright: ignore[reportUnknownMemberType]
    ax.set_title(title)  # pyright: ignore[reportUnknownMemberType]
    ax.legend(title="Algorithm / k", bbox_to_anchor=(1.05, 1), loc="upper left")  # pyright: ignore[reportUnknownMemberType]

    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
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

generate_plot(
    "peak_mem_kb",
    "Peak Memory (KB)",
    "Peak Memory vs. Graph Size",
    "fig_peak_memory.png",
    log_y=True,
)
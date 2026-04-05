"""
Greedy (2k-1)-Spanner — implementation and benchmark.
Matches the experimental setup of Chimani & Stutzenstein (ESA 2022).

This script has two goals:
1) Build graph spanners using the classic greedy algorithm.
2) Benchmark quality and performance of those spanners on random weighted graphs.

Important concept:
A (2k-1)-spanner of a weighted graph G is a sparse subgraph H where distances in H
approximate distances in G. Formally, for every node pair (u, v),
distance_H(u, v) <= (2k-1) * distance_G(u, v).

Intuition:
- We try to keep far fewer edges than the original graph.
- We still want shortest-path distances to stay close to original.
- The factor (2k-1) is the allowed multiplicative distortion (also called stretch bound).
"""

# pyright: reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false

import time
import tracemalloc
import random
import math
import csv
import os
from typing import Any, Literal, TypedDict, cast

import networkx as nx
import numpy as np
import matplotlib
# Use a non-interactive backend so plots can be saved to files in scripts/servers.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ResultRow(TypedDict):
    """Schema for one raw experiment result row (one graph instance, one k value)."""

    # Number of nodes in the original graph.
    n: int
    # Number of edges in the original graph.
    m: int
    # Erdos-Renyi edge probability used to generate the graph.
    density: float
    # Repetition index (different random seed for same n and density).
    rep: int
    # User-selected parameter controlling the spanner stretch factor.
    k: int
    # Theoretical stretch bound alpha = 2k - 1.
    alpha: int
    # Number of edges kept in the spanner.
    n_spanner_edges: int
    # Edge sparsity metric |E_H| / n.
    sparseness: float
    # Total spanner weight divided by MST weight.
    lightness: float
    # Max observed ratio d_H(u,v) / d_G(u,v).
    eff_stretch: float
    # Runtime in seconds for constructing the spanner.
    runtime_s: float
    # Peak memory usage during construction (KB), measured by tracemalloc.
    peak_mem_kb: float


class AggRow(TypedDict):
    """Schema for aggregated statistics (mean values across repetitions)."""

    # Number of nodes.
    n: int
    # Erdos-Renyi density.
    density: float
    # Spanner parameter.
    k: int
    # Corresponding stretch bound alpha.
    alpha: int
    # Mean sparseness over repetitions.
    sparseness: float
    # Mean lightness over repetitions.
    lightness: float
    # Mean effective stretch over repetitions.
    eff_stretch: float
    # Mean runtime over repetitions.
    runtime_s: float
    # Mean peak memory usage over repetitions.
    peak_mem_kb: float


MetricField = Literal[
    # Restrict valid metric names to keys that are numeric metrics.
    "sparseness",
    "lightness",
    "eff_stretch",
    "runtime_s",
    "peak_mem_kb",
]

# ── Greedy spanner ────────────────────────────────────────────────────────────

def greedy_spanner(G: nx.Graph, k: int) -> nx.Graph:
    """
    Construct a weighted greedy (2k-1)-spanner of an undirected graph.

    Purpose:
    Build a sparse subgraph H from input graph G while preserving approximate
    shortest-path distances. This follows the standard greedy spanner strategy.

    Parameters:
    G:
        Input undirected weighted graph (NetworkX graph). Every edge must have
        a numeric weight stored under key "weight".
    k:
        Positive integer controlling allowed distortion. The theoretical bound
        used by this algorithm is alpha = 2k - 1.

    Return value:
    nx.Graph:
        A new graph H with the same nodes as G and a selected subset of edges,
        such that each inserted edge is necessary under the greedy criterion.

    High-level algorithm idea:
    1) Sort all edges by increasing weight.
    2) Process edges from lightest to heaviest.
    3) For candidate edge (u, v, w), compute current shortest-path distance
       d_H(u, v) in the partially built spanner H.
    4) Add the edge only if d_H(u, v) > alpha * w.

    Why this works intuitively:
    - Light edges are considered first, so cheap connectivity is established early.
    - If current H already connects u and v with a path not too much longer than
      w (within factor alpha), we skip the edge to keep H sparse.
    - Otherwise, edge is needed to avoid violating stretch bound locally.
    """
    # The allowed multiplicative stretch factor for distances in the spanner.
    alpha = 2 * k - 1
    # Start from an empty graph that will become the spanner.
    H = nx.Graph()
    # Spanner must keep the exact same node set as original graph.
    H.add_nodes_from(G.nodes())

    # Sort edges by weight ascending
    # Why sort? The greedy proof/intuition relies on considering cheaper edges first,
    # so expensive edges are added only if no sufficiently short alternative exists.
    edges_sorted = sorted(G.edges(data="weight"), key=lambda e: e[2])

    # Process each edge candidate in nondecreasing weight order.
    for u, v, w in edges_sorted:
        # Shortest path in current spanner H
        # We repeatedly run Dijkstra because H changes after each accepted edge,
        # and we need up-to-date shortest-path distances.
        # Dijkstra is suitable because all generated edge weights are positive.
        try:
            # Distance currently achievable between u and v using already accepted edges.
            d = nx.dijkstra_path_length(H, u, v, weight="weight")
        except nx.NetworkXNoPath:
            # If no path exists yet, treat distance as infinity to force insertion.
            d = math.inf

        # Greedy acceptance test:
        # add edge if existing path is too long relative to edge weight.
        # Mathematical check: d_H(u,v) > alpha * w.
        if d > alpha * w:
            # Keep this edge in the spanner because it improves distance guarantees.
            H.add_edge(u, v, weight=w)

    # Return constructed spanner subgraph.
    return H


# ── Metrics ───────────────────────────────────────────────────────────────────

def sparseness(G: nx.Graph, H: nx.Graph) -> float:
    """
    Compute spanner sparseness as |E_H| / n.

    Purpose:
    Measure how many edges the spanner keeps per node, which indicates how sparse
    (compact) the resulting subgraph is.

    Parameters:
    G:
        Original graph (used only for node count n).
    H:
        Spanner graph whose edge count is measured.

    Return value:
    float:
        Ratio |E_H| / n, or 0.0 if n == 0.

    High-level idea:
    Divide the number of spanner edges by number of nodes to normalize graph size.
    """
    # Number of vertices in the original problem instance.
    n = G.number_of_nodes()
    # Protect against division by zero for empty graphs.
    # Formula used: sparseness = |E_H| / n.
    return H.number_of_edges() / n if n > 0 else 0.0


def lightness(G: nx.Graph, H: nx.Graph) -> float:
    """
    Compute spanner lightness as total_weight(H) / total_weight(MST(G)).

    Purpose:
    Compare weight of spanner to the minimum possible connected backbone.
    A lightness close to 1 means the spanner is close in weight to an MST.

    Parameters:
    G:
        Original weighted graph.
    H:
        Spanner graph.

    Return value:
    float:
        Weight ratio w(H) / w(MST(G)), or 0.0 if MST weight is zero.

    High-level idea:
    - MST is a baseline for minimum total weight while preserving connectivity.
    - Spanner usually has extra edges for distance quality.
    - Ratio tells how costly that extra structure is.
    """
    # Sum weights of MST edges of original graph.
    # NetworkX helper returns edges of one minimum spanning tree.
    mst_weight = sum(d["weight"] for _, _, d in nx.minimum_spanning_edges(G, data=True))
    # Sum weights of all edges currently present in the spanner.
    h_weight   = sum(d["weight"] for _, _, d in H.edges(data=True))
    # Protect against zero-division; otherwise return lightness ratio.
    return h_weight / mst_weight if mst_weight > 0 else 0.0


def effective_stretch(
    G: nx.Graph,
    H: nx.Graph,
    n_samples: int = 500,
    exact_threshold: int = 150,
) -> float:
    """
    Estimate worst-case observed stretch between graph and spanner distances.

    Purpose:
    Quantify how much shortest-path distances increase when using H instead of G.

    Parameters:
    G:
        Original weighted graph.
    H:
        Spanner graph built from G.
    n_samples:
        Number of random node pairs to test when graph is large.
    exact_threshold:
        If n <= this threshold, evaluate all node pairs exactly.

    Return value:
    float:
        Maximum observed ratio d_H(u,v) / d_G(u,v) among tested pairs.

    High-level algorithm idea:
    - For small graphs: check every node pair (exact but potentially expensive).
    - For large graphs: sample random pairs to control runtime.
    - For each pair: compute shortest paths in G and H via Dijkstra, then ratio.
    - Track the maximum observed ratio as "effective stretch".

    Why sampling is used for large n:
    Number of all pairs is n(n-1)/2, which grows quadratically and can become
    expensive. Sampling gives a practical approximation for benchmarking.
    """
    # Materialize node list once for pair generation.
    nodes = cast(list[int], list(G.nodes()))
    # Number of nodes controls whether to use exact or sampled evaluation.
    n = len(nodes)

    # For small graphs, evaluate all unordered pairs for exact max stretch.
    if n <= exact_threshold:
        # Generate all pairs (i < j) to avoid duplicates and self-pairs.
        pairs: list[tuple[int, int]] = [
            (nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n)
        ]
    else:
        # Fixed seed makes random sampling deterministic/reproducible.
        random.seed(42)
        # Randomly sample node pairs for scalable approximation.
        pairs = [
            (sample[0], sample[1]) for sample in (random.sample(nodes, 2) for _ in range(n_samples))
        ]

    # Track worst observed distance inflation from G to H.
    max_stretch = 0.0
    # Evaluate each selected pair independently.
    for u, v in pairs:
        try:
            # True shortest-path distance in original graph.
            dG = nx.dijkstra_path_length(G, u, v, weight="weight")
            # Corresponding shortest-path distance in spanner.
            dH = nx.dijkstra_path_length(H, u, v, weight="weight")
            # Ignore degenerate zero-distance case to avoid divide-by-zero.
            if dG > 0:
                # Stretch formula for this pair: d_H(u,v) / d_G(u,v).
                max_stretch = max(max_stretch, dH / dG)
        except nx.NetworkXNoPath:
            # If either graph has no path for this pair, skip pair for observed metric.
            pass

    # Return maximum observed stretch across tested pairs.
    return max_stretch


# ── Graph generation ──────────────────────────────────────────────────────────

def make_random_graph(n: int, density: float, seed: int) -> nx.Graph:
    """
    Generate a connected random weighted graph using Erdos-Renyi edge sampling.

    Purpose:
    Create synthetic benchmark instances with controllable size and density.

    Parameters:
    n:
        Number of nodes.
    density:
        Probability p of including each possible undirected edge (i, j).
        Must be in (0, 1].
    seed:
        Random seed for reproducible graph generation.

    Return value:
    nx.Graph:
        Connected undirected graph with integer edge weights in [1, n].

    High-level algorithm idea:
    1) Add n nodes.
    2) For each pair of nodes, add edge with probability = density.
    3) Assign random integer weight to each accepted edge.
    4) If graph is disconnected, connect components by adding bridging edges.

    Why enforce connectivity:
    Spanner metrics and shortest-path comparisons are most meaningful when graph
    is connected. This avoids many unreachable-pair corner cases.
    """
    # Local RNG instance keeps reproducibility isolated from global random state.
    rng = random.Random(seed)
    # Start with empty undirected graph.
    G = nx.Graph()
    # Add nodes 0..n-1.
    G.add_nodes_from(range(n))
    # Iterate all unordered node pairs.
    for i in range(n):
        for j in range(i+1, n):
            # Include edge with Erdos-Renyi probability p = density.
            if rng.random() < density:
                # Assign positive integer weight; positive weights fit Dijkstra assumptions.
                w = rng.randint(1, n)
                # Insert weighted undirected edge.
                G.add_edge(i, j, weight=w)
    # Ensure connectivity
    # If disconnected, connect neighboring components with one random bridge each.
    if not nx.is_connected(G):
        # Connected components are disjoint node sets.
        comps = list(nx.connected_components(G))
        # Chain components together: comp0-comp1, comp1-comp2, ...
        for idx in range(len(comps)-1):
            # Pick one representative node from each neighboring component.
            u = next(iter(comps[idx]))
            v = next(iter(comps[idx+1]))
            # Random bridge weight, same scale as other edges.
            w = rng.randint(1, n)
            # Add bridge so total graph becomes connected after processing all pairs.
            G.add_edge(u, v, weight=w)
    # Return generated benchmark graph.
    return G


# ── Experiment ────────────────────────────────────────────────────────────────

# Subset of paper's setup (8 sizes × 3 densities × 3 graphs = 72 instances)
# Node counts to test; larger n means larger input graphs.
SIZES     = [10, 20, 50, 100, 200]
# Edge probability settings for sparse-to-dense random graphs.
DENSITIES = [0.1, 0.3, 0.5]
# Number of random repetitions per (n, density) setting.
REPS      = 3
# Spanner k values; alpha bound will be 2k-1.
K_VALUES  = [2, 3]

# Collect one result dictionary per run.
results: list[ResultRow] = []

# Total number of benchmark runs = all parameter combinations.
total = len(SIZES) * len(DENSITIES) * REPS * len(K_VALUES)
# Progress counter.
done  = 0

# Inform user that full benchmark sweep has started.
print(f"Running {total} experiments...")

# Outer loop over graph sizes.
for n in SIZES:
    # Loop over density settings for each size.
    for density in DENSITIES:
        # Repeat each configuration with different seeds for robustness.
        for rep in range(REPS):
            # Deterministic seed composition for reproducible runs.
            seed = n * 1000 + int(density * 10) * 10 + rep
            # Generate one connected random weighted graph instance.
            G = make_random_graph(n, density, seed)

            # For the same graph instance, build/evaluate each spanner parameter k.
            for k in K_VALUES:
                # Start memory tracking only around the construction step.
                # tracemalloc measures Python-level allocations and peak memory.
                tracemalloc.start()
                # High-resolution start time for runtime measurement.
                t0 = time.perf_counter()
                # Build greedy spanner for this graph and k.
                H  = greedy_spanner(G, k)
                # End time immediately after construction.
                t1 = time.perf_counter()
                # Read current and peak traced memory; keep peak for reporting.
                _, peak_mem = tracemalloc.get_traced_memory()
                # Stop tracing to avoid overhead accumulation across iterations.
                tracemalloc.stop()

                # Compute structural sparsity metric.
                sp  = sparseness(G, H)
                # Compute weight efficiency metric relative to MST.
                li  = lightness(G, H)
                # Compute observed distance distortion metric.
                es  = effective_stretch(G, H)
                # Runtime in seconds for constructing H.
                rt  = t1 - t0
                # Convert bytes to kilobytes for easier reading.
                pm  = peak_mem / 1024  # KB

                # Store full raw result row for later CSV export and aggregation.
                results.append({
                    "n":        n,
                    "m":        G.number_of_edges(),
                    "density":  density,
                    "rep":      rep,
                    "k":        k,
                    "alpha":    2*k - 1,
                    "n_spanner_edges": H.number_of_edges(),
                    "sparseness":  round(sp,  4),
                    "lightness":   round(li,  4),
                    "eff_stretch": round(es,  4),
                    "runtime_s":   round(rt,  4),
                    "peak_mem_kb": round(pm,  2),
                })

                # Update progress counter after each completed (graph, k) run.
                done += 1
                # Print periodic progress updates to keep long runs transparent.
                if done % 20 == 0:
                    print(f"  {done}/{total} done")

# All benchmark runs finished.
print(f"All {total} experiments complete.")

# ── Save CSV ──────────────────────────────────────────────────────────────────

def resolve_output_dir() -> str:
    """
    Resolve output directory where CSV and plots are written.

    Purpose:
    Keep all generated artifacts in the current working directory so they appear
    directly in the project folder from which the script is run.

    Parameters:
    None.

    Return value:
    str:
        Path to directory used for output files.

    High-level idea:
    Use os.getcwd() as a simple, predictable output target.
    """
    # Save outputs in the current working directory.
    return os.getcwd()


# Resolve output folder once and reuse for all files.
fig_dir = resolve_output_dir()
# Build CSV file path in that output directory.
csv_path = os.path.join(fig_dir, "results.csv")

# Write all raw result rows to CSV.
with open(csv_path, "w", newline="") as f:
    # Use keys from first result row as stable CSV header order.
    fieldnames: list[str] = list(results[0].keys())
    # DictWriter maps each dictionary row to CSV columns by header names.
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    # Emit header row first.
    writer.writeheader()
    # Emit all benchmark rows.
    writer.writerows(results)

# Inform user where CSV was saved.
print(f"Results saved to {csv_path}")

# ── Aggregate for reporting ───────────────────────────────────────────────────

import collections

# Mean per (n, density, k) across reps
# Group rows by experimental condition so repetitions can be averaged.
agg: collections.defaultdict[tuple[int, float, int], list[ResultRow]] = collections.defaultdict(list)
# Insert each raw row into its condition bucket.
for r in results:
    # Grouping key identifies one condition independent of repetition.
    key = (r["n"], r["density"], r["k"])
    agg[key].append(r)

def mean_field(rows: list[ResultRow], field: MetricField) -> float:
    """
    Compute mean of one numeric metric over a list of result rows.

    Purpose:
    Support aggregation step by averaging repeated runs per condition.

    Parameters:
    rows:
        Raw result rows belonging to same (n, density, k) condition.
    field:
        Metric key to average (restricted by MetricField type).

    Return value:
    float:
        Rounded mean value with 4 decimal places.

    High-level idea:
    Extract selected metric from each row, average with numpy, round for display.
    """
    # Convert values to float explicitly for numeric consistency.
    return round(float(np.mean([float(r[field]) for r in rows])), 4)

# Final list of aggregated rows (one per unique condition).
agg_rows: list[AggRow] = []
# Sort keys for deterministic output ordering in table/plots.
for (n, d, k), rows in sorted(agg.items()):
    # Build one summarized row with means across repetitions.
    agg_rows.append({
        "n": n, "density": d, "k": k, "alpha": 2*k-1,
        "sparseness":  mean_field(rows, "sparseness"),
        "lightness":   mean_field(rows, "lightness"),
        "eff_stretch": mean_field(rows, "eff_stretch"),
        "runtime_s":   mean_field(rows, "runtime_s"),
        "peak_mem_kb": mean_field(rows, "peak_mem_kb"),
    })

# Print summary table
# Print table header with aligned fixed-width columns.
print(f"\n{'n':>6} {'dens':>5} {'k':>2} {'alpha':>5} {'sparse':>8} "
      f"{'light':>7} {'stretch':>8} {'time(s)':>9} {'mem(KB)':>9}")
# Visual separator line.
print("-" * 70)
# Print one formatted line per aggregated condition.
for r in agg_rows:
    print(f"{r['n']:>6} {r['density']:>5.1f} {r['k']:>2} {r['alpha']:>5} "
          f"{r['sparseness']:>8.4f} {r['lightness']:>7.4f} "
          f"{r['eff_stretch']:>8.4f} {r['runtime_s']:>9.4f} "
          f"{r['peak_mem_kb']:>9.2f}")

# ── Plots ─────────────────────────────────────────────────────────────────────

# Keep a relaxed type alias for matplotlib object access in strict typing mode.
plt_any: Any = plt

# 1. Sparseness vs n for each density × k
# Create a 1x2 subplot figure: one panel for k=2 and one for k=3.
fig, axes = plt_any.subplots(1, 2, figsize=(10, 4), sharey=False)
# Color palette by density for consistent visual mapping across plots.
colors = {0.1: "#2166ac", 0.3: "#f4a582", 0.5: "#d6604d"}
# Marker style by density to make lines distinguishable in grayscale/print.
markers = {0.1: "o", 0.3: "s", 0.5: "^"}

# Iterate subplot index and k value together.
for ax_idx, k in enumerate([2, 3]):
    # Select subplot axis for this k.
    ax = axes[ax_idx]
    # Draw one line per density on the current axis.
    for d in DENSITIES:
        # Filter aggregated rows for this (k, density) slice.
        rows = [r for r in agg_rows if r["k"] == k and r["density"] == d]
        # X values: node sizes n.
        ns   = [r["n"] for r in rows]
        # Y values: average sparseness.
        sp   = [r["sparseness"] for r in rows]
        # Plot line with consistent color/marker for this density.
        ax.plot(ns, sp, marker=markers[d], color=colors[d],
                label=f"density={d}", linewidth=1.5)
    # Log x-scale spreads small and large n values for clearer trend comparison.
    ax.set_xscale("log")
    # Label x-axis with math notation.
    ax.set_xlabel("$n$ (log scale)")
    # Label y-axis with sparseness formula.
    ax.set_ylabel("Sparseness $|E_H|/n$")
    # Plot title includes corresponding alpha bound for this k.
    ax.set_title(f"Greedy spanner, $\\alpha={2*k-1}$ ($k={k}$)")
    # Show legend identifying density curves.
    ax.legend(fontsize=8)
    # Add light grid for readability.
    ax.grid(True, alpha=0.3)

# Improve spacing to avoid label overlap.
plt_any.tight_layout()
# Save vector version suitable for publication-quality scaling.
plt_any.savefig(f"{fig_dir}/fig1_sparseness.pdf", bbox_inches="tight")
# Save raster version for easy sharing/viewing.
plt_any.savefig(f"{fig_dir}/fig1_sparseness.png", dpi=150, bbox_inches="tight")
# Close figure to release memory.
plt_any.close()

# 2. Runtime vs n (k=2, all densities)
# Single-axis figure focused on runtime scaling behavior.
fig, ax = plt_any.subplots(figsize=(5.5, 4))
# Plot runtime lines for each density at fixed k=2.
for d in DENSITIES:
    # Select rows for this density and k=2.
    rows = [r for r in agg_rows if r["k"] == 2 and r["density"] == d]
    # X values: graph size.
    ns   = [r["n"] for r in rows]
    # Y values: average runtime seconds.
    rt   = [r["runtime_s"] for r in rows]
    # Plot runtime trend for this density.
    ax.plot(ns, rt, marker=markers[d], color=colors[d],
            label=f"density={d}", linewidth=1.5)
# Log-log scaling helps reveal growth rates (power-law-like trends).
ax.set_xscale("log")
ax.set_yscale("log")
# Axis labels and title explain what is being benchmarked.
ax.set_xlabel("$n$ (log scale)")
ax.set_ylabel("Runtime (s, log scale)")
ax.set_title("Greedy runtime ($\\alpha=3$, $k=2$)")
# Legend and grid improve interpretability.
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
# Layout and file export steps.
plt_any.tight_layout()
plt_any.savefig(f"{fig_dir}/fig2_runtime.pdf", bbox_inches="tight")
plt_any.savefig(f"{fig_dir}/fig2_runtime.png", dpi=150, bbox_inches="tight")
plt_any.close()

# 3. Effective stretch vs n (k=2, density=0.3)
# Single-axis comparison of observed stretch quality for two alpha values.
fig, ax = plt_any.subplots(figsize=(5.5, 4))
# Compare k=2 and k=3 at fixed density=0.3.
for k in [2, 3]:
    # Select rows for this k at chosen density.
    rows = [r for r in agg_rows if r["k"] == k and r["density"] == 0.3]
    # X values: graph size.
    ns   = [r["n"] for r in rows]
    # Y values: observed effective stretch.
    es   = [r["eff_stretch"] for r in rows]
    # Plot observed stretch line with marker distinguishing k.
    ax.plot(ns, es, marker="o" if k==2 else "s",
            label=f"$\\alpha={2*k-1}$", linewidth=1.5)
# Horizontal reference lines show theoretical bounds for each alpha.
ax.axhline(y=3, color="#2166ac", linestyle="--", alpha=0.5, label="theoretical bound $\\alpha=3$")
ax.axhline(y=5, color="#d6604d", linestyle="--", alpha=0.5, label="theoretical bound $\\alpha=5$")
# Log x-scale again for consistent treatment of n across wide range.
ax.set_xscale("log")
# Labels/title/legend/grid for readability.
ax.set_xlabel("$n$ (log scale)")
ax.set_ylabel("Effective stretch")
ax.set_title("Effective stretch vs $n$ (density = 0.3)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
# Final export of third figure.
plt_any.tight_layout()
plt_any.savefig(f"{fig_dir}/fig3_stretch.pdf", bbox_inches="tight")
plt_any.savefig(f"{fig_dir}/fig3_stretch.png", dpi=150, bbox_inches="tight")
plt_any.close()

# Final completion message for plotting stage.
print("\nPlots saved.")
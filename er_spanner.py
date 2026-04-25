"""
Greedy (2k-1)-Spanner — implementation and benchmark.
Matches the experimental setup of Chimani & Stutzenstein (ESA 2022).
"""

# pyright: reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false

import time
import tracemalloc
import random
import math
import csv
import os
import collections
from typing import Any, Literal, TypedDict, cast

import networkx as nx
import numpy as np
from final_spanner import baswana_sen_spanner
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ResultRow(TypedDict):
    algo: str
    n: int
    m: int
    density: float
    rep: int
    k: int
    alpha: int
    n_spanner_edges: int
    sparseness: float
    lightness: float
    eff_stretch: float
    runtime_s: float
    peak_mem_kb: float


class AggRow(TypedDict):
    algo: str
    n: int
    density: float
    k: int
    alpha: int
    sparseness: float
    lightness: float
    eff_stretch: float
    runtime_s: float
    peak_mem_kb: float


MetricField = Literal[
    "sparseness",
    "lightness",
    "eff_stretch",
    "runtime_s",
    "peak_mem_kb",
]


def greedy_spanner(G: nx.Graph, k: int) -> nx.Graph:
    """
    Build a greedy (2k-1)-spanner of G.

    Sorts edges by weight and adds (u, v, w) only if d_H(u, v) > (2k-1)*w.
    """
    alpha = 2 * k - 1
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    edges_sorted = sorted(G.edges(data="weight"), key=lambda e: e[2])
    for u, v, w in edges_sorted:
        try:
            d = nx.dijkstra_path_length(H, u, v, weight="weight")
        except nx.NetworkXNoPath:
            d = math.inf
        if d > alpha * w:
            H.add_edge(u, v, weight=w)
    return H



def sparseness(G: nx.Graph, H: nx.Graph) -> float:
    """Return |E_H| / n."""
    n = G.number_of_nodes()
    return H.number_of_edges() / n if n > 0 else 0.0


def lightness(G: nx.Graph, H: nx.Graph) -> float:
    """Return w(H) / w(MST(G))."""
    mst_weight = sum(d["weight"] for _, _, d in nx.minimum_spanning_edges(G, data=True))
    h_weight = sum(d["weight"] for _, _, d in H.edges(data=True))
    return h_weight / mst_weight if mst_weight > 0 else 0.0


def effective_stretch(
    G: nx.Graph,
    H: nx.Graph,
    n_samples: int = 500,
    exact_threshold: int = 150,
) -> float:
    """Return the maximum d_H(u,v)/d_G(u,v) over tested node pairs."""
    nodes = cast(list[int], list(G.nodes()))
    n = len(nodes)

    if n <= exact_threshold:
        pairs: list[tuple[int, int]] = [
            (nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n)
        ]
    else:
        random.seed(42)
        pairs = [
            (sample[0], sample[1]) for sample in (random.sample(nodes, 2) for _ in range(n_samples))
        ]

    max_stretch = 0.0
    for u, v in pairs:
        try:
            dG = nx.dijkstra_path_length(G, u, v, weight="weight")
            dH = nx.dijkstra_path_length(H, u, v, weight="weight")
            if dG > 0:
                max_stretch = max(max_stretch, dH / dG)
        except nx.NetworkXNoPath:
            pass

    return max_stretch



def make_random_graph(n: int, density: float, seed: int) -> nx.Graph:
    """Generate a connected random weighted graph with integer weights in [1, n]."""
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < density:
                w = rng.randint(1, n)
                G.add_edge(i, j, weight=w)
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for idx in range(len(comps)-1):
            u = next(iter(comps[idx]))
            v = next(iter(comps[idx+1]))
            w = rng.randint(1, n)
            G.add_edge(u, v, weight=w)
    return G



# Subset of paper's setup (8 sizes × 3 densities × 3 graphs = 72 instances)
SIZES = [10, 20, 50, 100, 200]
DENSITIES = [0.1, 0.3, 0.5]
REPS = 3
K_VALUES = [2, 3]
ALGORITHMS = ["Greedy", "BS"]

RESULT_FIELDNAMES = [
    "algo",
    "n",
    "m",
    "density",
    "rep",
    "k",
    "alpha",
    "n_spanner_edges",
    "sparseness",
    "lightness",
    "eff_stretch",
    "runtime_s",
    "peak_mem_kb",
]

# Store ER benchmark outputs in a dedicated folder.
fig_dir = os.path.join(os.getcwd(), "ER_graphs")
os.makedirs(fig_dir, exist_ok=True)
csv_path = os.path.join(fig_dir, "results.csv")

# Collect one result dictionary per run.
results: list[ResultRow] = []

if os.path.exists(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            algo_raw = (raw.get("algo") or "Greedy").strip() or "Greedy"
            row: ResultRow = {
                "algo": algo_raw,
                "n": int(raw["n"]),
                "m": int(raw["m"]),
                "density": float(raw["density"]),
                "rep": int(raw["rep"]),
                "k": int(raw["k"]),
                "alpha": int(raw["alpha"]),
                "n_spanner_edges": int(raw["n_spanner_edges"]),
                "sparseness": float(raw["sparseness"]),
                "lightness": float(raw["lightness"]),
                "eff_stretch": float(raw["eff_stretch"]),
                "runtime_s": float(raw["runtime_s"]),
                "peak_mem_kb": float(raw["peak_mem_kb"]),
            }
            results.append(row)
    print(f"Loaded {len(results)} existing rows from {csv_path}")

existing_run_keys = {
    (r["algo"], r["n"], r["density"], r["rep"], r["k"]) for r in results
}

pending_run_keys = [
    (algo, n, density, rep, k)
    for n in SIZES
    for density in DENSITIES
    for rep in range(REPS)
    for k in K_VALUES
    for algo in ALGORITHMS
    if (algo, n, density, rep, k) not in existing_run_keys
]

total = len(pending_run_keys)
done = 0

if total == 0:
    print("No pending experiments. Reusing existing CSV rows.")
else:
    print(f"Running {total} pending experiments...")

for n in SIZES:
    for density in DENSITIES:
        for rep in range(REPS):
            if not any(
                (algo, n, density, rep, k) not in existing_run_keys
                for k in K_VALUES
                for algo in ALGORITHMS
            ):
                continue

            seed = n * 1000 + int(density * 10) * 10 + rep
            G = make_random_graph(n, density, seed)

            for k in K_VALUES:
                for algo in ALGORITHMS:
                    run_key = (algo, n, density, rep, k)
                    if run_key in existing_run_keys:
                        continue

                    tracemalloc.start()
                    t0 = time.perf_counter()
                    if algo == "Greedy":
                        h_spanner = greedy_spanner(G, k)
                    else:
                        h_spanner = baswana_sen_spanner(G, k, seed=seed)
                    t1 = time.perf_counter()
                    _, peak_mem = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    sp = sparseness(G, h_spanner)
                    li = lightness(G, h_spanner)
                    es = effective_stretch(G, h_spanner)
                    rt = t1 - t0
                    pm = peak_mem / 1024

                    results.append({
                        "algo": algo,
                        "n": n,
                        "m": G.number_of_edges(),
                        "density": density,
                        "rep": rep,
                        "k": k,
                        "alpha": 2*k - 1,
                        "n_spanner_edges": h_spanner.number_of_edges(),
                        "sparseness": round(sp, 4),
                        "lightness": round(li, 4),
                        "eff_stretch": round(es, 4),
                        "runtime_s": round(rt, 4),
                        "peak_mem_kb": round(pm, 2),
                    })

                    existing_run_keys.add(run_key)

                    done += 1
                    if done % 20 == 0:
                        print(f"  {done}/{total} done")

print(f"Completed {done} new experiments.")
print(f"Total rows available for analysis: {len(results)}")

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
    writer.writeheader()
    writer.writerows(results)

print(f"Results saved to {csv_path}")


agg: collections.defaultdict[tuple[str, int, float, int], list[ResultRow]] = collections.defaultdict(list)
# Insert each raw row into its condition bucket.
for r in results:
    key = (r["algo"], r["n"], r["density"], r["k"])
    agg[key].append(r)

def mean_field(rows: list[ResultRow], field: MetricField) -> float:
    """Return the mean of field across rows, rounded to 4 dp."""
    return round(float(np.mean([float(r[field]) for r in rows])), 4)

agg_rows: list[AggRow] = []
for (algo, n, d, k), rows in sorted(agg.items()):
    agg_rows.append({
        "algo": algo,
        "n": n, "density": d, "k": k, "alpha": 2*k-1,
        "sparseness":  mean_field(rows, "sparseness"),
        "lightness":   mean_field(rows, "lightness"),
        "eff_stretch": mean_field(rows, "eff_stretch"),
        "runtime_s":   mean_field(rows, "runtime_s"),
        "peak_mem_kb": mean_field(rows, "peak_mem_kb"),
    })

print(f"\n{'algo':>8} {'n':>6} {'dens':>5} {'k':>2} {'alpha':>5} {'sparse':>8} "
      f"{'light':>7} {'stretch':>8} {'time(s)':>9} {'mem(KB)':>9}")
print("-" * 80)
for r in agg_rows:
    print(f"{r['algo']:>8} {r['n']:>6} {r['density']:>5.1f} {r['k']:>2} {r['alpha']:>5} "
          f"{r['sparseness']:>8.4f} {r['lightness']:>7.4f} "
          f"{r['eff_stretch']:>8.4f} {r['runtime_s']:>9.4f} "
          f"{r['peak_mem_kb']:>9.2f}")


plt_any: Any = plt

# 1. Sparseness vs n: one panel for k=2 and one for k=3.
fig, axes = plt_any.subplots(1, 2, figsize=(10, 4), sharey=False)
colors = {0.1: "#2166ac", 0.3: "#f4a582", 0.5: "#d6604d"}
markers = {0.1: "o", 0.3: "s", 0.5: "^"}

for ax_idx, k in enumerate([2, 3]):
    ax = axes[ax_idx]
    for d in DENSITIES:
        rows = [
            r for r in agg_rows
            if r["algo"] == "Greedy" and r["k"] == k and r["density"] == d
        ]
        ns = [r["n"] for r in rows]
        sp = [r["sparseness"] for r in rows]
        ax.plot(ns, sp, marker=markers[d], color=colors[d],
                label=f"density={d}", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("$n$ (log scale)")
    ax.set_ylabel("Sparseness $|E_H|/n$")
    ax.set_title(f"Greedy spanner, $\\alpha={2*k-1}$ ($k={k}$)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt_any.tight_layout()
plt_any.savefig(f"{fig_dir}/fig1_sparseness.pdf", bbox_inches="tight")
plt_any.savefig(f"{fig_dir}/fig1_sparseness.png", dpi=150, bbox_inches="tight")
plt_any.close()

# 2. Runtime vs n (k=2)
fig, ax = plt_any.subplots(figsize=(5.5, 4))
for d in DENSITIES:
    rows = [
        r for r in agg_rows
        if r["algo"] == "Greedy" and r["k"] == 2 and r["density"] == d
    ]
    ns = [r["n"] for r in rows]
    rt = [r["runtime_s"] for r in rows]
    ax.plot(ns, rt, marker=markers[d], color=colors[d],
            label=f"density={d}", linewidth=1.5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$n$ (log scale)")
ax.set_ylabel("Runtime (s, log scale)")
ax.set_title("Greedy runtime ($\\alpha=3$, $k=2$)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt_any.tight_layout()
plt_any.savefig(f"{fig_dir}/fig2_runtime.pdf", bbox_inches="tight")
plt_any.savefig(f"{fig_dir}/fig2_runtime.png", dpi=150, bbox_inches="tight")
plt_any.close()

# 3. Effective stretch vs n (density=0.3)
fig, ax = plt_any.subplots(figsize=(5.5, 4))
for k in [2, 3]:
    rows = [
        r for r in agg_rows
        if r["algo"] == "Greedy" and r["k"] == k and r["density"] == 0.3
    ]
    ns = [r["n"] for r in rows]
    es = [r["eff_stretch"] for r in rows]
    ax.plot(ns, es, marker="o" if k==2 else "s",
            label=f"$\\alpha={2*k-1}$", linewidth=1.5)
ax.axhline(y=3, color="#2166ac", linestyle="--", alpha=0.5, label="theoretical bound $\\alpha=3$")
ax.axhline(y=5, color="#d6604d", linestyle="--", alpha=0.5, label="theoretical bound $\\alpha=5$")
ax.set_xscale("log")
ax.set_xlabel("$n$ (log scale)")
ax.set_ylabel("Effective stretch")
ax.set_title("Effective stretch vs $n$ (density = 0.3)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt_any.tight_layout()
plt_any.savefig(f"{fig_dir}/fig3_stretch.pdf", bbox_inches="tight")
plt_any.savefig(f"{fig_dir}/fig3_stretch.png", dpi=150, bbox_inches="tight")
plt_any.close()

print("\nPlots saved.")
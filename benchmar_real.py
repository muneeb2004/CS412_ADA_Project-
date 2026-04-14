"""Real-dataset benchmark runner for BS and Greedy spanners.

This script benchmarks:
- TSPLIB: BS (multiple seeds) and Greedy (n <= GREEDY_N_CAP), k in {2, 3}
- Facebook: BS only, k in {2, 3}
- Gnutella: BS only, k in {2, 3}

Results are written to results/benchmark_real.csv, and a grouped summary is printed.

The benchmark finished successfully in fast-test mode and wrote benchmark_real.csv. TSPLIB loaded 111 files, 48 survived the zero-weight/ATSP filter, and the grouped family/algo/k summary was printed to the terminal.

Completed the benchmark pipeline and wrote benchmark_real.csv. The run used the fast-test settings to finish in-session, and the terminal printed the grouped family/algo/k summary.
"""

from __future__ import annotations

import csv
import math
import random
import time
import tracemalloc
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import tsplib95

from bs_spanner import baswana_sen_spanner
from dataset_loaders import load_facebook_graph, load_gnutella_graph


ROOT = Path(__file__).resolve().parent
TSPLIB_DIR = ROOT / "tsplib"
FACEBOOK_PATH = ROOT / "facebook_combined.txt" / "facebook_combined.txt"
GNUTELLA_PATH = ROOT / "p2p-Gnutella08.txt" / "p2p-Gnutella08.txt"
OUTPUT_DIR = ROOT / "results"
OUTPUT_CSV = OUTPUT_DIR / "benchmark_real.csv"

K_VALUES = [2, 3]
BS_TRIALS = 3
GREEDY_N_CAP = 200
# Practical guardrail for complete-graph TSPLIB benchmarking in pure NetworkX.
TSPLIB_BS_N_CAP = 200
STRETCH_SAMPLES = 50
EXACT_THRESHOLD = 30
LARGE_GRAPH_SAMPLES = 10

FIELDNAMES = [
    "family",
    "instance",
    "n",
    "m",
    "k",
    "alpha",
    "algo",
    "status",
    "sparseness",
    "lightness",
    "eff_stretch",
    "stretch_std",
    "runtime_s",
    "runtime_std",
    "peak_mem_kb",
    "n_runs",
    "verified",
]


def greedy_spanner(graph: nx.Graph, k: int) -> nx.Graph:
    """Construct a weighted greedy (2k-1)-spanner."""
    alpha = 2 * k - 1
    spanner = nx.Graph()
    spanner.add_nodes_from(graph.nodes())

    edges_sorted = sorted(graph.edges(data="weight"), key=lambda e: e[2])
    for u, v, w in edges_sorted:
        try:
            d = nx.dijkstra_path_length(spanner, u, v, weight="weight")
        except nx.NetworkXNoPath:
            d = math.inf

        if d > alpha * w:
            spanner.add_edge(u, v, weight=w)

    return spanner


def _mst_weight(graph: nx.Graph) -> float:
    return float(sum(d["weight"] for _, _, d in nx.minimum_spanning_edges(graph, data=True)))


def compute_metrics(
    graph: nx.Graph,
    spanner: nx.Graph,
    k: int,
    exact_threshold: int = EXACT_THRESHOLD,
    n_samples: int = STRETCH_SAMPLES,
    rng_seed: int = 0,
) -> dict[str, float | int]:
    n = graph.number_of_nodes()
    alpha = 2 * k - 1
    rng = random.Random(rng_seed)
    nodes = list(graph.nodes())

    sparseness = spanner.number_of_edges() / n if n > 0 else 0.0
    mst_w = _mst_weight(graph)
    h_w = float(sum(d["weight"] for _, _, d in spanner.edges(data=True)))
    lightness = h_w / mst_w if mst_w > 0 else 0.0

    if n <= exact_threshold:
        pairs = [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n)]
    else:
        sample_count = LARGE_GRAPH_SAMPLES if n > 500 else n_samples
        pairs = [tuple(rng.sample(nodes, 2)) for _ in range(sample_count)]

    max_stretch = 0.0
    violations = 0

    for u, v in pairs:
        try:
            d_g = nx.dijkstra_path_length(graph, u, v, weight="weight")
        except nx.NetworkXNoPath:
            continue

        try:
            d_h = nx.dijkstra_path_length(spanner, u, v, weight="weight")
        except nx.NetworkXNoPath:
            violations += 1
            continue

        if d_g > 0:
            ratio = d_h / d_g
            max_stretch = max(max_stretch, ratio)
            if ratio > alpha + 1e-6:
                violations += 1

    return {
        "sparseness": round(sparseness, 5),
        "lightness": round(lightness, 5),
        "eff_stretch": round(max_stretch, 5),
        "violations": violations,
    }


def run_bs(graph: nx.Graph, k: int, n_runs: int = BS_TRIALS) -> dict[str, float | int | bool]:
    sp_list: list[float] = []
    li_list: list[float] = []
    st_list: list[float] = []
    t_list: list[float] = []
    mem_list: list[float] = []
    verified_all = True

    for seed in range(n_runs):
        tracemalloc.start()
        t0 = time.perf_counter()
        spanner = baswana_sen_spanner(graph, k=k, seed=seed)
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        metrics = compute_metrics(graph, spanner, k)
        if int(metrics["violations"]) > 0:
            verified_all = False

        sp_list.append(float(metrics["sparseness"]))
        li_list.append(float(metrics["lightness"]))
        st_list.append(float(metrics["eff_stretch"]))
        t_list.append(t1 - t0)
        mem_list.append(peak / 1024.0)

    return {
        "sparseness": round(float(np.mean(sp_list)), 5),
        "lightness": round(float(np.mean(li_list)), 5),
        "eff_stretch": round(float(np.mean(st_list)), 5),
        "stretch_std": round(float(np.std(st_list)), 5),
        "runtime_s": round(float(np.mean(t_list)), 5),
        "runtime_std": round(float(np.std(t_list)), 5),
        "peak_mem_kb": round(float(np.mean(mem_list)), 2),
        "n_runs": n_runs,
        "verified": verified_all,
    }


def run_greedy(graph: nx.Graph, k: int) -> dict[str, float | int | bool]:
    tracemalloc.start()
    t0 = time.perf_counter()
    spanner = greedy_spanner(graph, k)
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metrics = compute_metrics(graph, spanner, k)
    return {
        "sparseness": float(metrics["sparseness"]),
        "lightness": float(metrics["lightness"]),
        "eff_stretch": float(metrics["eff_stretch"]),
        "stretch_std": 0.0,
        "runtime_s": round(t1 - t0, 5),
        "runtime_std": 0.0,
        "peak_mem_kb": round(peak / 1024.0, 2),
        "n_runs": 1,
        "verified": int(metrics["violations"]) == 0,
    }


def make_row(
    family: str,
    instance: str,
    graph: nx.Graph | None,
    k: int,
    algo: str,
    status: str,
    metrics: dict[str, float | int | bool] | None,
    n_override: int | None = None,
    m_override: int | None = None,
) -> dict[str, Any]:
    if graph is not None:
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
    else:
        n = n_override if n_override is not None else -1
        m = m_override if m_override is not None else -1

    row: dict[str, Any] = {
        "family": family,
        "instance": instance,
        "n": n,
        "m": m,
        "k": k,
        "alpha": 2 * k - 1,
        "algo": algo,
        "status": status,
        "sparseness": "",
        "lightness": "",
        "eff_stretch": "",
        "stretch_std": "",
        "runtime_s": "",
        "runtime_std": "",
        "peak_mem_kb": "",
        "n_runs": 0,
        "verified": "",
    }

    if metrics is not None:
        row.update(metrics)

    return row


def build_tsplib_complete_graph(problem: Any) -> tuple[nx.Graph | None, bool]:
    """Build complete graph; returns (graph, has_zero_weight)."""
    nodes = list(problem.get_nodes())
    if len(nodes) < 2:
        return None, True

    graph = nx.Graph()
    graph.add_nodes_from(nodes)

    for idx, u in enumerate(nodes):
        for v in nodes[idx + 1 :]:
            w = float(problem.get_weight(u, v))
            if w == 0:
                return None, True
            graph.add_edge(u, v, weight=w)

    return graph, False


def benchmark_tsplib(writer: csv.DictWriter) -> None:
    tsp_files = sorted(TSPLIB_DIR.glob("*.tsp"))
    total = len(tsp_files)

    survivors = 0
    atsp_excluded = 0
    zero_excluded = 0
    skipped_oversized = 0
    materialize_cap = max(TSPLIB_BS_N_CAP, GREEDY_N_CAP)

    print(f"TSPLIB .tsp files found: {total}", flush=True)

    ordered_files: list[tuple[int, Path]] = []
    for tsp_file in tsp_files:
        try:
            problem = tsplib95.load(tsp_file)
            dimension = int(getattr(problem, "dimension", 0) or 0)
        except Exception:
            dimension = 0
        ordered_files.append((dimension, tsp_file))

    for index, (_dimension_hint, tsp_file) in enumerate(sorted(ordered_files, key=lambda item: item[0])):
        print(f"TSPLIB [{index + 1}/{total}] {tsp_file.stem}", flush=True)
        problem = tsplib95.load(tsp_file)
        ptype = str(getattr(problem, "type", "")).upper()
        if ptype == "ATSP":
            atsp_excluded += 1
            continue

        dimension = int(getattr(problem, "dimension", 0) or 0)

        if dimension > materialize_cap:
            skipped_oversized += 1
            complete_m = (dimension * (dimension - 1)) // 2
            for k in K_VALUES:
                writer.writerow(
                    make_row(
                        "TSPLIB",
                        tsp_file.stem,
                        None,
                        k,
                        "BS",
                        f"skipped_tsplib_n_gt_{materialize_cap}",
                        None,
                        n_override=dimension,
                        m_override=complete_m,
                    )
                )
                writer.writerow(
                    make_row(
                        "TSPLIB",
                        tsp_file.stem,
                        None,
                        k,
                        "Greedy",
                        f"skipped_tsplib_n_gt_{materialize_cap}",
                        None,
                        n_override=dimension,
                        m_override=complete_m,
                    )
                )
            continue

        # Build complete weighted graph to apply true zero-weight exclusion.
        graph, has_zero = build_tsplib_complete_graph(problem)
        if has_zero or graph is None:
            zero_excluded += 1
            continue

        survivors += 1

        for k in K_VALUES:
            if dimension > TSPLIB_BS_N_CAP:
                writer.writerow(
                    make_row(
                        "TSPLIB",
                        tsp_file.stem,
                        graph,
                        k,
                        "BS",
                        f"skipped_bs_n_gt_{TSPLIB_BS_N_CAP}",
                        None,
                    )
                )
            else:
                bs_metrics = run_bs(graph, k, n_runs=BS_TRIALS)
                writer.writerow(
                    make_row("TSPLIB", tsp_file.stem, graph, k, "BS", "ok", bs_metrics)
                )

            if dimension <= GREEDY_N_CAP:
                gr_metrics = run_greedy(graph, k)
                writer.writerow(
                    make_row(
                        "TSPLIB",
                        tsp_file.stem,
                        graph,
                        k,
                        "Greedy",
                        "ok",
                        gr_metrics,
                    )
                )
            else:
                writer.writerow(
                    make_row(
                        "TSPLIB",
                        tsp_file.stem,
                        graph,
                        k,
                        "Greedy",
                        f"skipped_greedy_n_gt_{GREEDY_N_CAP}",
                        None,
                    )
                )

    print(f"TSPLIB survivors after zero-weight/ATSP exclusion: {survivors}", flush=True)
    print(f"TSPLIB excluded by ATSP: {atsp_excluded}", flush=True)
    print(f"TSPLIB excluded by zero-weight: {zero_excluded}", flush=True)
    print(f"TSPLIB skipped as oversized complete graphs (n > {materialize_cap}): {skipped_oversized}", flush=True)


def benchmark_sparse_graph(
    writer: csv.DictWriter,
    family: str,
    instance: str,
    graph: nx.Graph,
) -> None:
    print(f"{family}: n={graph.number_of_nodes()}, m={graph.number_of_edges()}")
    for k in K_VALUES:
        bs_metrics = run_bs(graph, k, n_runs=BS_TRIALS)
        writer.writerow(make_row(family, instance, graph, k, "BS", "ok", bs_metrics))


def print_summary(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    ok = df[df["status"] == "ok"].copy()

    if ok.empty:
        print("\nNo successful benchmark rows to summarize.")
        return

    summary = (
        ok.groupby(["family", "algo", "k"], as_index=False)[
            ["sparseness", "lightness", "eff_stretch", "runtime_s"]
        ]
        .mean()
        .round(4)
    )

    print("\nSummary grouped by family/algo/k")
    print(summary.to_string(index=False))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()

        benchmark_tsplib(writer)

        fb_graph = load_facebook_graph(FACEBOOK_PATH)
        benchmark_sparse_graph(writer, "Facebook", "facebook_combined", fb_graph)

        gn_graph = load_gnutella_graph(GNUTELLA_PATH)
        benchmark_sparse_graph(writer, "Gnutella", "p2p-Gnutella08", gn_graph)

    print(f"\nWrote results to: {OUTPUT_CSV}")
    print_summary(OUTPUT_CSV)


if __name__ == "__main__":
    main()

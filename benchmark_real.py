"""Real-dataset benchmark runner for BS and Greedy spanners.

This script benchmarks:
- TSPLIB: BS (multiple seeds) and Greedy (n <= GREEDY_N_CAP), k in {2, 3}
- Facebook: BS only, k in {2, 3}
- Gnutella: BS only, k in {2, 3}
- Steinlib: BS and Greedy (n <= GREEDY_N_CAP), k in {2, 3}

Results are written to results/benchmark_real.csv, and a grouped summary is printed.
"""

from __future__ import annotations

import csv
import math
import random
import time
import tracemalloc
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, TypeAlias, cast

import networkx as nx
import numpy as np
import pandas as pd
import tsplib95  # pyright: ignore[reportMissingTypeStubs]
import dataset_loaders

from bs_spanner import baswana_sen_spanner  # pyright: ignore[reportUnknownVariableType]


NodeId = int
if TYPE_CHECKING:
    GraphT: TypeAlias = nx.Graph[NodeId]
else:
    GraphT: TypeAlias = nx.Graph


class TsplibProblem(Protocol):
    type: str
    dimension: int

    def get_nodes(self) -> list[NodeId]: ...

    def get_weight(self, start: NodeId, end: NodeId) -> float | int: ...


class BSBuilder(Protocol):
    def __call__(self, G: GraphT, k: int, seed: int | None = None) -> GraphT: ...


class GraphLoader(Protocol):
    def __call__(self, path: str | Path, seed: int = 42) -> GraphT: ...


class GraphsLoader(Protocol):
    def __call__(self, path: str | Path) -> dict[str, GraphT]: ...


BS_SPANNER = cast(BSBuilder, baswana_sen_spanner)
LOAD_FACEBOOK_GRAPH = cast(GraphLoader, dataset_loaders.load_facebook_graph)  # pyright: ignore[reportUnknownMemberType]
LOAD_GNUTELLA_GRAPH = cast(GraphLoader, dataset_loaders.load_gnutella_graph)  # pyright: ignore[reportUnknownMemberType]
LOAD_STEINLIB_GRAPHS = cast(GraphsLoader, dataset_loaders.load_steinlib_graphs)  # pyright: ignore[reportUnknownMemberType]
TSPLIB_LOAD = cast(Callable[[Path], Any], tsplib95.load)  # pyright: ignore[reportUnknownMemberType]


ROOT = Path(__file__).resolve().parent
TSPLIB_DIR = ROOT / "tsplib"
FACEBOOK_PATH = ROOT / "facebook_combined.txt" / "facebook_combined.txt"
GNUTELLA_PATH = ROOT / "p2p-Gnutella08.txt" / "p2p-Gnutella08.txt"
STEINLIB_DIR = ROOT / "steinlib"
OUTPUT_DIR = ROOT / "results"
OUTPUT_CSV = OUTPUT_DIR / "benchmark_real.csv"

# Fast test-run config
K_VALUES = [2, 3]
BS_TRIALS = 3
GREEDY_N_CAP = 200
# Practical guardrail for complete-graph TSPLIB benchmarking in pure NetworkX.
TSPLIB_BS_N_CAP = 200
STRETCH_SAMPLES = 50
EXACT_THRESHOLD = 30
LARGE_GRAPH_SAMPLES = 10

# Production config
# K_VALUES = [2, 3]
# BS_TRIALS = 10
# GREEDY_N_CAP = 1200
# TSPLIB_BS_N_CAP = 1200
# STRETCH_SAMPLES = 1000
# EXACT_THRESHOLD = 30
# LARGE_GRAPH_SAMPLES = 1000

FIELDNAMES = [
    "family"
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


def load_tsplib_problem(path: Path) -> TsplibProblem:
    return cast(TsplibProblem, TSPLIB_LOAD(path))


def greedy_spanner(graph: GraphT, k: int) -> GraphT:
    """Construct a weighted greedy (2k-1)-spanner."""
    alpha = 2 * k - 1
    spanner: GraphT = cast(GraphT, nx.Graph())
    spanner.add_nodes_from(graph.nodes())

    edges_sorted = sorted(
        graph.edges(data=True),
        key=lambda e: float(e[2].get("weight", 1.0)),
    )
    for u, v, data in edges_sorted:
        w = float(data.get("weight", 1.0))
        try:
            d = nx.dijkstra_path_length(spanner, u, v, weight="weight")
        except nx.NetworkXNoPath:
            d = math.inf

        if d > alpha * w:
            spanner.add_edge(u, v, weight=w)

    return spanner


def _mst_weight(graph: GraphT) -> float:
    mst_edges = cast(
        list[tuple[NodeId, NodeId, dict[str, Any]]],
        list(nx.minimum_spanning_edges(graph, data=True)),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    )
    total = 0.0
    for _, _, data in mst_edges:
        total += float(data.get("weight", 0.0))
    return total


def compute_metrics(
    graph: GraphT,
    spanner: GraphT,
    k: int,
    exact_threshold: int = EXACT_THRESHOLD,
    n_samples: int = STRETCH_SAMPLES,
    rng_seed: int = 0,
) -> dict[str, float | int]:
    n = graph.number_of_nodes()
    alpha = 2 * k - 1
    rng = random.Random(rng_seed)
    nodes: list[NodeId] = list(graph.nodes())
    pairs: list[tuple[NodeId, NodeId]]

    sparseness = spanner.number_of_edges() / n if n > 0 else 0.0
    mst_w = _mst_weight(graph)
    h_w = float(sum(d["weight"] for _, _, d in spanner.edges(data=True)))
    lightness = h_w / mst_w if mst_w > 0 else 0.0

    if n <= exact_threshold:
        pairs = [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n)]
    else:
        sample_count = LARGE_GRAPH_SAMPLES if n > 500 else n_samples
        pairs = [cast(tuple[NodeId, NodeId], tuple(rng.sample(nodes, 2))) for _ in range(sample_count)]

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


def run_bs(graph: GraphT, k: int, n_runs: int = BS_TRIALS) -> dict[str, float | int | bool]:
    sp_list: list[float] = []
    li_list: list[float] = []
    st_list: list[float] = []
    t_list: list[float] = []
    mem_list: list[float] = []
    verified_all = True

    for seed in range(n_runs):
        tracemalloc.start()
        t0 = time.perf_counter()
        spanner = BS_SPANNER(graph, k=k, seed=seed)
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


def run_greedy(graph: GraphT, k: int) -> dict[str, float | int | bool]:
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
    graph: GraphT | None,
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


def build_tsplib_complete_graph(problem: TsplibProblem) -> tuple[GraphT | None, bool]:
    """Build complete graph; returns (graph, has_zero_weight)."""
    nodes = list(problem.get_nodes())
    if len(nodes) < 2:
        return None, True

    graph: GraphT = cast(GraphT, nx.Graph())
    graph.add_nodes_from(nodes)

    for idx, u in enumerate(nodes):
        for v in nodes[idx + 1 :]:
            w = float(problem.get_weight(u, v))
            if w == 0:
                return None, True
            graph.add_edge(u, v, weight=w)

    return graph, False


def benchmark_tsplib(writer: csv.DictWriter[str]) -> None:
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
            problem = load_tsplib_problem(tsp_file)
            dimension = int(getattr(problem, "dimension", 0) or 0)
        except Exception:
            dimension = 0
        ordered_files.append((dimension, tsp_file))

    for index, (_dimension_hint, tsp_file) in enumerate(sorted(ordered_files, key=lambda item: item[0])):
        print(f"TSPLIB [{index + 1}/{total}] {tsp_file.stem}", flush=True)
        problem = load_tsplib_problem(tsp_file)
        ptype = str(getattr(problem, "type", "")).upper()
        if ptype == "ATSP":
            atsp_excluded += 1
            continue

        dimension = int(getattr(problem, "dimension", 0) or 0)

        # Build complete weighted graph to apply true zero-weight exclusion.
        graph, has_zero = build_tsplib_complete_graph(problem)
        if has_zero or graph is None:
            zero_excluded += 1
            continue

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
    writer: csv.DictWriter[str],
    family: str,
    instance: str,
    graph: GraphT,
    include_greedy: bool = False,
) -> None:
    print(f"{family}: n={graph.number_of_nodes()}, m={graph.number_of_edges()}")
    for k in K_VALUES:
        bs_metrics = run_bs(graph, k, n_runs=BS_TRIALS)
        writer.writerow(make_row(family, instance, graph, k, "BS", "ok", bs_metrics))
        if include_greedy:
            if graph.number_of_nodes() <= GREEDY_N_CAP:
                gr_metrics = run_greedy(graph, k)
                writer.writerow(make_row(family, instance, graph, k, "Greedy", "ok", gr_metrics))
            else:
                writer.writerow(
                    make_row(
                        family,
                        instance,
                        graph,
                        k,
                        "Greedy",
                        f"skipped_greedy_n_gt_{GREEDY_N_CAP}",
                        None,
                    )
                )


def benchmark_sparse_family(
    writer: csv.DictWriter[str],
    family: str,
    instances: list[tuple[str, GraphT]],
    include_greedy: bool = False,
) -> None:
    total = len(instances)
    print(f"{family} graphs loaded: {total}", flush=True)

    for idx, (instance, graph) in enumerate(instances, start=1):
        print(
            f"{family} [{idx}/{total}] {instance} n={graph.number_of_nodes()} m={graph.number_of_edges()}",
            flush=True,
        )
        benchmark_sparse_graph(
            writer,
            family,
            instance,
            graph,
            include_greedy=include_greedy,
        )

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
        writer: csv.DictWriter[str] = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()

        benchmark_tsplib(writer)

        fb_graph = LOAD_FACEBOOK_GRAPH(FACEBOOK_PATH)
        benchmark_sparse_family(
            writer,
            "Facebook",
            [("facebook_combined", fb_graph)],
            include_greedy=False,
        )

        gn_graph = LOAD_GNUTELLA_GRAPH(GNUTELLA_PATH)
        benchmark_sparse_family(
            writer,
            "Gnutella",
            [("p2p-Gnutella08", gn_graph)],
            include_greedy=False,
        )

        steinlib_graphs = LOAD_STEINLIB_GRAPHS(STEINLIB_DIR)
        benchmark_sparse_family(
            writer,
            "Steinlib",
            sorted(steinlib_graphs.items()),
            include_greedy=True,
        )
        

    print(f"\nWrote results to: {OUTPUT_CSV}")
    print_summary(OUTPUT_CSV)

    


if __name__ == "__main__":
    main()

"""Benchmark runner for BS and Greedy spanners on real datasets.

Benchmarks:
- TSPLIB: BS (multiple seeds) and Greedy (n <= GREEDY_N_CAP), k in {2, 3}
- Facebook: BS only, k in {2, 3}
- Gnutella: BS only, k in {2, 3}
- Steinlib: BS and Greedy (n <= GREEDY_N_CAP), k in {2, 3}

Results are written to results/benchmark_real.csv, and a grouped summary is printed.
"""

from __future__ import annotations

import csv
import math
import os
import random
import time
import tracemalloc
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, TextIO, TypeAlias, cast

import networkx as nx
import numpy as np
import pandas as pd
import tsplib95  # pyright: ignore[reportMissingTypeStubs]
import dataset_loaders

from final_spanner import baswana_sen_spanner  # pyright: ignore[reportUnknownVariableType]


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
LOAD_STEINLIB_GRAPH = cast(Callable[[str | Path], GraphT | None], dataset_loaders.load_steinlib_graph)  # pyright: ignore[reportUnknownMemberType]
LOAD_STEINLIB_GRAPHS = cast(GraphsLoader, dataset_loaders.load_steinlib_graphs)  # pyright: ignore[reportUnknownMemberType]
TSPLIB_LOAD = cast(Callable[[Path], Any], tsplib95.load)  # pyright: ignore[reportUnknownMemberType]


ROOT = Path(__file__).resolve().parent
TSPLIB_DIR = ROOT / "tsplib"
FACEBOOK_PATH = ROOT / "facebook_combined.txt" / "facebook_combined.txt"
GNUTELLA_PATH = ROOT / "p2p-Gnutella08.txt" / "p2p-Gnutella08.txt"
STEINLIB_DIR = ROOT / "steinlib"
OUTPUT_DIR = ROOT / "results"
OUTPUT_CSV = OUTPUT_DIR / "benchmark_real.csv"

# # Fast test-run config
# K_VALUES = [2, 3]
# BS_TRIALS = 3
# GREEDY_N_CAP = 200
# Practical guardrail for complete-graph TSPLIB benchmarking in pure NetworkX.
# TSPLIB_BS_N_CAP = 200
# STRETCH_SAMPLES = 50
# EXACT_THRESHOLD = 30
# LARGE_GRAPH_SAMPLES = 10

# Production config
K_VALUES = [2, 3]
BS_TRIALS = 10
GREEDY_N_CAP = 500 # Greedy is very slow on complete graphs (tried running it for 10 hours but only 77 graphs were completed rendering the experiment impractical), so we set a lower cap than BS. This means some large TSPLIB instances will skip Greedy benchmarks, but it keeps runtimes reasonable.
TSPLIB_BS_N_CAP = 1200
STRETCH_SAMPLES = 1000
EXACT_THRESHOLD = 30
LARGE_GRAPH_SAMPLES = 1000

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

RowKey: TypeAlias = tuple[str, str, int, str]


def make_row_key(family: str, instance: str, k: int, algo: str) -> RowKey:
    return (family, instance, k, algo)


def row_key_from_row(row: dict[str, Any]) -> RowKey | None:
    family = str(row.get("family", "")).strip()
    instance = str(row.get("instance", "")).strip()
    algo = str(row.get("algo", "")).strip()
    raw_k = row.get("k", "")
    try:
        k = int(raw_k)
    except (TypeError, ValueError):
        try:
            k = int(float(raw_k))
        except (TypeError, ValueError):
            return None
    if not family or not instance or not algo:
        return None
    return make_row_key(family, instance, k, algo)


def load_completed_row_keys(csv_path: Path) -> set[RowKey]:
    completed: set[RowKey] = set()
    if (not csv_path.exists()) or csv_path.stat().st_size == 0:
        return completed

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row_key_from_row(row)
            if key is not None:
                completed.add(key)
    return completed


def open_results_writer(csv_path: Path) -> tuple[TextIO, csv.DictWriter[str], set[RowKey], bool]:
    completed = load_completed_row_keys(csv_path)
    resume_mode = csv_path.exists() and csv_path.stat().st_size > 0
    mode = "a" if resume_mode else "w"
    handle = csv_path.open(mode, newline="", encoding="utf-8")
    writer: csv.DictWriter[str] = csv.DictWriter(handle, fieldnames=FIELDNAMES)

    if not resume_mode:
        writer.writeheader()
        handle.flush()
        os.fsync(handle.fileno())

    return handle, writer, completed, resume_mode


def emit_row_with_checkpoint(
    writer: csv.DictWriter[str],
    handle: TextIO,
    completed: set[RowKey],
    row: dict[str, Any],
) -> None:
    writer.writerow(row)
    handle.flush()
    os.fsync(handle.fileno())
    key = row_key_from_row(row)
    if key is not None:
        completed.add(key)


def is_row_done(completed: set[RowKey], family: str, instance: str, k: int, algo: str) -> bool:
    return make_row_key(family, instance, k, algo) in completed


def load_tsplib_problem(path: Path) -> TsplibProblem:
    return cast(TsplibProblem, TSPLIB_LOAD(path))


def safe_load_tsplib_problem(path: Path) -> TsplibProblem | None:
    try:
        return load_tsplib_problem(path)
    except Exception:
        return None


def preflight_checks() -> None:
    # Validate dataset paths/types and output writability before long runs.
    required_paths: list[tuple[Path, str]] = [
        (TSPLIB_DIR, "dir"),
        (FACEBOOK_PATH, "file"),
        (GNUTELLA_PATH, "file"),
        (STEINLIB_DIR, "dir"),
    ]
    missing: list[str] = []
    for path, expected_kind in required_paths:
        if expected_kind == "dir" and not path.is_dir():
            missing.append(f"Missing directory: {path}")
        if expected_kind == "file" and not path.is_file():
            missing.append(f"Missing file: {path}")

    if missing:
        message = "\n".join(missing)
        raise FileNotFoundError(f"Preflight failed:\n{message}")

    tsp_files = sorted(TSPLIB_DIR.glob("*.tsp"))
    if not tsp_files:
        raise RuntimeError(f"Preflight failed: no TSPLIB .tsp files found in {TSPLIB_DIR}")

    stp_files = sorted(STEINLIB_DIR.rglob("*.stp"))
    if not stp_files:
        raise RuntimeError(f"Preflight failed: no Steinlib .stp files found in {STEINLIB_DIR}")

    for value_name, value in [
        ("BS_TRIALS", BS_TRIALS),
        ("GREEDY_N_CAP", GREEDY_N_CAP),
        ("TSPLIB_BS_N_CAP", TSPLIB_BS_N_CAP),
        ("STRETCH_SAMPLES", STRETCH_SAMPLES),
        ("EXACT_THRESHOLD", EXACT_THRESHOLD),
        ("LARGE_GRAPH_SAMPLES", LARGE_GRAPH_SAMPLES),
    ]:
        if value <= 0:
            raise ValueError(f"Preflight failed: {value_name} must be > 0, got {value}")

    if not K_VALUES or any(k < 1 for k in K_VALUES):
        raise ValueError(f"Preflight failed: K_VALUES must be non-empty and >= 1, got {K_VALUES}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_file = OUTPUT_DIR / ".benchmark_write_test"
    test_file.write_text("ok", encoding="utf-8")
    test_file.unlink(missing_ok=True)

    fb = LOAD_FACEBOOK_GRAPH(FACEBOOK_PATH, seed=42)
    if fb.number_of_nodes() == 0 or fb.number_of_edges() == 0:
        raise RuntimeError("Preflight failed: Facebook graph loaded empty")

    gn = LOAD_GNUTELLA_GRAPH(GNUTELLA_PATH, seed=42)
    if gn.number_of_nodes() == 0 or gn.number_of_edges() == 0:
        raise RuntimeError("Preflight failed: Gnutella graph loaded empty")

    parsed_steinlib = False
    for stp_file in stp_files[:20]:
        graph = LOAD_STEINLIB_GRAPH(stp_file)
        if graph is not None and graph.number_of_edges() > 0:
            parsed_steinlib = True
            break
    if not parsed_steinlib:
        raise RuntimeError("Preflight failed: could not parse any of the first 20 Steinlib files")

    tsplib_meta_ok = False
    for tsp_file in tsp_files[:20]:
        problem = safe_load_tsplib_problem(tsp_file)
        if problem is None:
            continue
        _ = str(getattr(problem, "type", "")).upper()
        _ = int(getattr(problem, "dimension", 0) or 0)
        tsplib_meta_ok = True
        break
    if not tsplib_meta_ok:
        raise RuntimeError("Preflight failed: could not parse any of the first 20 TSPLIB files")

    print(
        "Preflight checks passed: paths, dataset loaders, graph types, and output directory are valid.",
        flush=True,
    )


def greedy_spanner(graph: GraphT, k: int) -> GraphT:
    """Build a greedy (2k-1)-spanner."""
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
        print(
            f"  BS seed progress: k={k}, seed {seed + 1}/{n_runs}, n={graph.number_of_nodes()}, m={graph.number_of_edges()}",
            flush=True,
        )
        seed_start = time.perf_counter()
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

        seed_elapsed = time.perf_counter() - seed_start
        print(
            f"  BS seed done: k={k}, seed {seed + 1}/{n_runs}, elapsed={seed_elapsed:.2f}s",
            flush=True,
        )

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
    """Build complete graph; returns (graph, has_invalid_weight_or_parse_error)."""
    try:
        nodes = list(problem.get_nodes())
    except Exception:
        return None, True
    if len(nodes) < 2:
        return None, True

    graph: GraphT = cast(GraphT, nx.Graph())
    graph.add_nodes_from(nodes)

    try:
        for idx, u in enumerate(nodes):
            for v in nodes[idx + 1 :]:
                w = float(problem.get_weight(u, v))
                if (not math.isfinite(w)) or w <= 0:
                    return None, True
                graph.add_edge(u, v, weight=w)
    except Exception:
        return None, True

    return graph, False


def benchmark_tsplib(
    emit_row: Callable[[dict[str, Any]], None],
    completed: set[RowKey],
) -> None:
    tsp_files = sorted(TSPLIB_DIR.glob("*.tsp"))
    total = len(tsp_files)

    survivors = 0
    atsp_excluded = 0
    zero_excluded = 0
    parse_failed = 0
    skipped_oversized = 0
    materialize_cap = max(TSPLIB_BS_N_CAP, GREEDY_N_CAP)

    print(f"TSPLIB .tsp files found: {total}", flush=True)

    ordered_files: list[tuple[int, Path]] = []
    for tsp_file in tsp_files:
        problem = safe_load_tsplib_problem(tsp_file)
        if problem is not None:
            dimension = int(getattr(problem, "dimension", 0) or 0)
        else:
            dimension = 0
        ordered_files.append((dimension, tsp_file))

    for index, (_dimension_hint, tsp_file) in enumerate(sorted(ordered_files, key=lambda item: item[0])):
        print(f"TSPLIB [{index + 1}/{total}] {tsp_file.stem}", flush=True)
        instance = tsp_file.stem
        expected_keys = [
            make_row_key("TSPLIB", instance, k, algo)
            for k in K_VALUES
            for algo in ("BS", "Greedy")
        ]
        if expected_keys and all(key in completed for key in expected_keys):
            print(
                f"TSPLIB [{index + 1}/{total}] {instance} already checkpointed, skipping",
                flush=True,
            )
            continue

        problem = safe_load_tsplib_problem(tsp_file)
        if problem is None:
            parse_failed += 1
            for k in K_VALUES:
                if not is_row_done(completed, "TSPLIB", instance, k, "BS"):
                    emit_row(
                        make_row(
                            "TSPLIB",
                            instance,
                            None,
                            k,
                            "BS",
                            "skipped_parse_error",
                            None,
                        )
                    )
                if not is_row_done(completed, "TSPLIB", instance, k, "Greedy"):
                    emit_row(
                        make_row(
                            "TSPLIB",
                            instance,
                            None,
                            k,
                            "Greedy",
                            "skipped_parse_error",
                            None,
                        )
                    )
            continue

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
                if not is_row_done(completed, "TSPLIB", instance, k, "BS"):
                    emit_row(
                        make_row(
                            "TSPLIB",
                            instance,
                            None,
                            k,
                            "BS",
                            f"skipped_tsplib_n_gt_{materialize_cap}",
                            None,
                            n_override=dimension,
                            m_override=complete_m,
                        )
                    )
                if not is_row_done(completed, "TSPLIB", instance, k, "Greedy"):
                    emit_row(
                        make_row(
                            "TSPLIB",
                            instance,
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
            if not is_row_done(completed, "TSPLIB", instance, k, "BS"):
                if dimension > TSPLIB_BS_N_CAP:
                    emit_row(
                        make_row(
                            "TSPLIB",
                            instance,
                            graph,
                            k,
                            "BS",
                            f"skipped_bs_n_gt_{TSPLIB_BS_N_CAP}",
                            None,
                        )
                    )
                else:
                    bs_metrics = run_bs(graph, k, n_runs=BS_TRIALS)
                    emit_row(
                        make_row("TSPLIB", instance, graph, k, "BS", "ok", bs_metrics)
                    )

            if not is_row_done(completed, "TSPLIB", instance, k, "Greedy"):
                if dimension <= GREEDY_N_CAP:
                    gr_metrics = run_greedy(graph, k)
                    emit_row(
                        make_row(
                            "TSPLIB",
                            instance,
                            graph,
                            k,
                            "Greedy",
                            "ok",
                            gr_metrics,
                        )
                    )
                else:
                    emit_row(
                        make_row(
                            "TSPLIB",
                            instance,
                            graph,
                            k,
                            "Greedy",
                            f"skipped_greedy_n_gt_{GREEDY_N_CAP}",
                            None,
                        )
                    )

    print(f"TSPLIB survivors after zero-weight/ATSP exclusion: {survivors}", flush=True)
    print(f"TSPLIB excluded by ATSP: {atsp_excluded}", flush=True)
    print(f"TSPLIB excluded by invalid/zero/non-finite weight: {zero_excluded}", flush=True)
    print(f"TSPLIB parse failures: {parse_failed}", flush=True)
    print(f"TSPLIB skipped as oversized complete graphs (n > {materialize_cap}): {skipped_oversized}", flush=True)


def benchmark_sparse_graph(
    emit_row: Callable[[dict[str, Any]], None],
    completed: set[RowKey],
    family: str,
    instance: str,
    graph: GraphT,
    include_greedy: bool = False,
) -> None:
    print(f"{family}: n={graph.number_of_nodes()}, m={graph.number_of_edges()}")
    for k in K_VALUES:
        if not is_row_done(completed, family, instance, k, "BS"):
            bs_metrics = run_bs(graph, k, n_runs=BS_TRIALS)
            emit_row(make_row(family, instance, graph, k, "BS", "ok", bs_metrics))
        if include_greedy:
            if not is_row_done(completed, family, instance, k, "Greedy"):
                if graph.number_of_nodes() <= GREEDY_N_CAP:
                    gr_metrics = run_greedy(graph, k)
                    emit_row(make_row(family, instance, graph, k, "Greedy", "ok", gr_metrics))
                else:
                    emit_row(
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
    emit_row: Callable[[dict[str, Any]], None],
    completed: set[RowKey],
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
            emit_row,
            completed,
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


def format_elapsed(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    


def main() -> None:
    run_started = time.perf_counter()
    preflight_checks()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    handle, writer, completed, resume_mode = open_results_writer(OUTPUT_CSV)
    try:
        if resume_mode:
            print(
                f"Resume mode: found {len(completed)} checkpointed rows in {OUTPUT_CSV}",
                flush=True,
            )
        else:
            print(f"Starting new benchmark file: {OUTPUT_CSV}", flush=True)
        def emit_row(row: dict[str, Any]) -> None:
            emit_row_with_checkpoint(writer, handle, completed, row)
        benchmark_tsplib(emit_row, completed)
        fb_graph = LOAD_FACEBOOK_GRAPH(FACEBOOK_PATH)
        benchmark_sparse_family(
            emit_row,
            completed,
            "Facebook",
            [("facebook_combined", fb_graph)],
            include_greedy=False,
        )
        gn_graph = LOAD_GNUTELLA_GRAPH(GNUTELLA_PATH)
        benchmark_sparse_family(
            emit_row,
            completed,
            "Gnutella",
            [("p2p-Gnutella08", gn_graph)],
            include_greedy=False,
        )
        steinlib_graphs = LOAD_STEINLIB_GRAPHS(STEINLIB_DIR)
        benchmark_sparse_family(
            emit_row,
            completed,
            "Steinlib",
            sorted(steinlib_graphs.items()),
            include_greedy=True,
        )
    finally:
        handle.close()
    print(f"\nWrote results to: {OUTPUT_CSV}")
    print_summary(OUTPUT_CSV)
    elapsed = time.perf_counter() - run_started
    print(f"Elapsed wall time: {format_elapsed(elapsed)} ({elapsed:.2f} s)")

if __name__ == "__main__":
    main()

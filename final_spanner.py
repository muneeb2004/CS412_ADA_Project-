"""
Baswana-Sen (2k-1)-spanner: implementation, verification, and benchmark.

Reference:
    Baswana & Sen, "A Simple and Linear Time Randomized Algorithm for Computing
    Sparse Spanners in Weighted Graphs", Random Structures & Algorithms
    30(4):532-563, 2007.
"""

# pyright: reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false

import math
import random
import time
import csv
import os
from typing import TypedDict

import networkx as nx
import numpy as np


class VerifyResult(TypedDict):
    passed: bool
    violations: list[str]
    max_stretch: float



def baswana_sen_spanner(G: nx.Graph, k: int, seed: int | None = None) -> nx.Graph:
    """
    Construct a (2k-1)-spanner of G using the Baswana-Sen Las Vegas algorithm.

    Returns a sparse subgraph H with the same node set as G such that for every
    pair (u, v): d_H(u, v) <= (2k-1) * d_G(u, v).

    Expected edge count: O(k * n^(1+1/k)). Expected runtime: O(km).
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    rng = random.Random(seed)
    n = G.number_of_nodes()
    if n == 0 or k == 1:
        return G.copy()

    residual = nx.Graph()
    residual.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        residual.add_edge(u, v, weight=w)

    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    def add_to_spanner(u: int, v: int) -> None:
        w = G[u][v].get("weight", 1.0)
        if not H.has_edge(u, v) or H[u][v]["weight"] > w:
            H.add_edge(u, v, weight=w)

    cluster_of = {v: v for v in G.nodes()}

    p = n ** (-1.0 / k)
    # Reject rounds that produce too many edges (Las Vegas sparsity guard).
    size_limit = 2 * (n ** (1.0 + 1.0 / k))
    for _round in range(k - 1):
        while True:
            current_centers = set(cluster_of.values())
            sampled = {c for c in current_centers if rng.random() < p}

            # Batch all decisions before applying so the round uses a consistent snapshot.
            edges_to_add = set()
            edges_to_remove = set()
            new_cluster_of = {}

            for v in list(residual.nodes()):
                if cluster_of[v] in sampled:
                    continue

                lightest_nbr = {}
                lightest_weight = {}

                for u in residual.adj[v]:
                    c_u = cluster_of[u]
                    w = residual[v][u]["weight"]
                    if c_u not in lightest_weight or w < lightest_weight[c_u]:
                        lightest_nbr[c_u] = u
                        lightest_weight[c_u] = w

                nbr_sampled = set(lightest_weight.keys()) & sampled

                if not nbr_sampled:
                    # Disband: secure local connectivity then exit the residual.
                    for u in lightest_nbr.values():
                        edges_to_add.add((v, u))
                    for u in list(residual.adj[v]):
                        edges_to_remove.add((v, u))

                else:
                    closest = min(nbr_sampled, key=lambda c: lightest_weight[c])
                    w_closest = lightest_weight[closest]
                    u_closest = lightest_nbr[closest]

                    edges_to_add.add((v, u_closest))
                    new_cluster_of[v] = closest

                    # Add edges to cheaper neighboring clusters (required for stretch proof).
                    for c, w_c in lightest_weight.items():
                        if w_c < w_closest:
                            edges_to_add.add((v, lightest_nbr[c]))

                    for u in list(residual.adj[v]):
                        c_u = cluster_of[u]
                        w_u = lightest_weight.get(c_u, math.inf)
                        if c_u == closest or w_u < w_closest:
                            edges_to_remove.add((v, u))

            if len(edges_to_add) > size_limit:
                continue
            break

        for u, v in edges_to_add:
            add_to_spanner(u, v)

        residual.remove_edges_from(edges_to_remove)

        for v, c in cluster_of.items():
            if c in sampled and v not in new_cluster_of:
                new_cluster_of[v] = c
        cluster_of = new_cluster_of

        intra = [(u, v) for u, v in residual.edges()
                 if cluster_of.get(u) == cluster_of.get(v) and
                    cluster_of.get(u) is not None]
        residual.remove_edges_from(intra)

        disbanded = [v for v in list(residual.nodes()) if v not in cluster_of]
        residual.remove_nodes_from(disbanded)

    # Phase 2: each remaining vertex adds one lightest edge per neighboring cluster.
    for v in list(residual.nodes()):
        lightest_nbr = {}
        lightest_weight = {}
        for u in residual.adj[v]:
            c_u = cluster_of.get(u)
            if c_u is None:
                continue
            w = residual[v][u]["weight"]
            if c_u not in lightest_weight or w < lightest_weight[c_u]:
                lightest_nbr[c_u] = u
                lightest_weight[c_u] = w
        for u in lightest_nbr.values():
            add_to_spanner(v, u)

    return H


def verify_spanner(
    G: nx.Graph,
    H: nx.Graph,
    k: int,
    exact_threshold: int = 300,
    n_samples: int = 1000,
    rng_seed: int = 0,
) -> VerifyResult:
    """
    Check whether H is a valid (2k-1)-spanner of G.

    For n <= exact_threshold tests all pairs; otherwise samples n_samples random
    pairs. Returns pass/fail status, violation messages, and max observed stretch.
    """
    alpha = 2 * k - 1
    n = G.number_of_nodes()
    rng = random.Random(rng_seed)
    nodes = list(G.nodes())
    violations = []

    for u, v, d in H.edges(data=True):
        if not G.has_edge(u, v):
            violations.append(f"Edge ({u},{v}) in H but not in G")
        else:
            if abs(d.get("weight",1.0) - G[u][v].get("weight",1.0)) > 1e-9:
                violations.append(f"Edge ({u},{v}) weight mismatch")

    if nx.is_connected(G) and not nx.is_connected(H):
        violations.append("G connected but H is not")

    if n <= exact_threshold:
        pairs = [(nodes[i], nodes[j]) for i in range(n) for j in range(i+1, n)]
    else:
        pairs = [tuple(rng.sample(nodes, 2)) for _ in range(n_samples)]

    max_ratio = 0.0
    for u, v in pairs:
        try:
            d_G = nx.dijkstra_path_length(G, u, v, weight="weight")
        except nx.NetworkXNoPath:
            continue
        try:
            d_H = nx.dijkstra_path_length(H, u, v, weight="weight")
        except nx.NetworkXNoPath:
            violations.append(f"No H-path ({u},{v}), G-dist={d_G:.3f}")
            continue
        if d_G > 0:
            ratio = d_H / d_G
            max_ratio = max(max_ratio, ratio)
            if ratio > alpha + 1e-6:
                violations.append(
                    f"Stretch ({u},{v}): dH={d_H:.4f} dG={d_G:.4f} "
                    f"ratio={ratio:.4f} > alpha={alpha}"
                )

    return {"passed": len(violations)==0, "violations": violations, "max_stretch": max_ratio}


def make_graph(n: int, density: float, seed: int) -> nx.Graph:
    """Generate a connected random weighted graph with integer weights in [1, n]."""
    rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < density:
                G.add_edge(i, j, weight=rng.randint(1, n))
    if n > 0 and not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for idx in range(len(comps)-1):
            u = next(iter(comps[idx]))
            v = next(iter(comps[idx+1]))
            G.add_edge(u, v, weight=rng.randint(1, n))
    return G


def run_tests() -> bool:
    print("=" * 60)
    print("VERIFICATION SUITE")
    print("=" * 60)
    all_ok = True

    # k=1 is the identity case.
    g = make_graph(10, 0.5, 0)
    h = baswana_sen_spanner(g, k=1, seed=0)
    assert set(h.edges()) == set(g.edges()), "k=1 must return G"
    print("[PASS] k=1 returns G unchanged")

    h = baswana_sen_spanner(nx.Graph(), k=2, seed=0)
    assert h.number_of_nodes() == 0
    print("[PASS] Empty graph")

    # Path graphs are sparse and expose disband/join errors.
    Gp = nx.path_graph(20)
    for u, v in Gp.edges():
        Gp[u][v]["weight"] = 1
    for k in [2, 3]:
        for seed in range(20):
            h = baswana_sen_spanner(Gp, k=k, seed=seed)
            r = verify_spanner(Gp, h, k)
            if not r["passed"]:
                print(f"  [FAIL] Path k={k} seed={seed}: {r['violations'][:2]}")
                all_ok = False
    print(f"[{'PASS' if all_ok else 'FAIL'}] Path graph k=2,3 (20 seeds each)")

    # Complete graph: every vertex sees many neighboring clusters.
    Gc = nx.complete_graph(20)
    rng_ref = random.Random(7)
    for e in Gc.edges():
        Gc[e[0]][e[1]]["weight"] = rng_ref.randint(1, 50)
    k4_ok = True
    for k in [2, 3, 4]:
        for seed in range(10):
            h = baswana_sen_spanner(Gc, k=k, seed=seed)
            r = verify_spanner(Gc, h, k)
            if not r["passed"]:
                print(f"  [FAIL] K_20 k={k} seed={seed}: {r['violations'][:2]}")
                k4_ok = False; all_ok = False
    print(f"[{'PASS' if k4_ok else 'FAIL'}] Complete K_20 k=2,3,4 (10 seeds each)")

    SIZES = [8, 12, 15, 20, 30, 50]
    DENSITIES = [0.2, 0.4, 0.7]
    K_VALUES = [2, 3, 4]
    TRIALS = 5
    fails = total = 0
    for n in SIZES:
        for d in DENSITIES:
            g = make_graph(n, d, seed=n*100+int(d*10))
            for k in K_VALUES:
                for trial in range(TRIALS):
                    h = baswana_sen_spanner(g, k=k, seed=trial)
                    r = verify_spanner(g, h, k, exact_threshold=n+1)
                    if not r["passed"]:
                        fails += 1; all_ok = False
                        print(f"  [FAIL] n={n} d={d} k={k} seed={trial}: "
                              f"{r['violations'][:2]}")
                    total += 1
    print(f"[{'PASS' if fails==0 else 'FAIL'}] "
          f"Exhaustive small-graph suite: {total} runs, {fails} failures")

    G_big = make_graph(60, 0.4, 999)
    sub_ok = True
    for k in [2, 3]:
        for trial in range(15):
            h = baswana_sen_spanner(G_big, k=k, seed=trial)
            for u, v in h.edges():
                if not G_big.has_edge(u, v):
                    print(f"  [FAIL] H edge ({u},{v}) not in G")
                    sub_ok = False; all_ok = False
    print(f"[{'PASS' if sub_ok else 'FAIL'}] H ⊆ G always (n=60, 30 runs)")

    cmp_ok = True
    for n, d, k, seed in [(20,0.5,2,0),(30,0.4,3,1),(50,0.3,2,5)]:
        g = make_graph(n, d, seed*100+n)
        H_mine = baswana_sen_spanner(g, k=k, seed=seed)
        H_nx   = nx.spanner(g, stretch=2*k-1, weight="weight", seed=seed)
        r_mine = verify_spanner(g, H_mine, k)
        r_nx   = verify_spanner(g, H_nx,   k)
        if not r_mine["passed"]:
            print(f"  [FAIL] Mine invalid n={n} d={d} k={k}")
            cmp_ok = False; all_ok = False
        if not r_nx["passed"]:
            print(f"  [WARN] NX invalid n={n} d={d} k={k}")
    print(f"[{'PASS' if cmp_ok else 'FAIL'}] Cross-check vs NetworkX nx.spanner()")

    print("=" * 60)
    print(f"SUITE {'PASSED' if all_ok else 'FAILED'}")
    print("=" * 60)
    return all_ok


def greedy_spanner(G: nx.Graph, k: int) -> nx.Graph:
    """
    Build a greedy (2k-1)-spanner of G for comparison against Baswana-Sen.

    Processes edges in non-decreasing weight order and adds (u, v, w) only if
    d_H(u, v) > (2k-1) * w in the current partial spanner.
    """
    alpha = 2*k - 1
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, w in sorted(G.edges(data="weight"), key=lambda e: e[2]):
        try:
            d = nx.dijkstra_path_length(H, u, v, weight="weight")
        except nx.NetworkXNoPath:
            d = math.inf
        if d > alpha * w:
            H.add_edge(u, v, weight=w)
    return H


def benchmark() -> None:
    print()
    print("=" * 80)
    print("BENCHMARK: Baswana-Sen vs Greedy")
    print(f"{'n':>5} {'d':>4} {'k':>2}  "
          f"{'BS_sp':>7} {'BS_li':>7} {'BS_str':>7} {'BS_t':>8}  "
          f"{'GR_sp':>7} {'GR_li':>7} {'GR_str':>7} {'GR_t':>8}  "
          f"{'ok':>4}")
    print("-" * 80)

    SIZES = [10, 20, 50, 100, 200]
    DENSITIES = [0.1, 0.3, 0.5]
    K_VALUES = [2, 3]
    BS_TRIALS = 10
    GR_REPS = 3

    rows = []
    for n in SIZES:
        for density in DENSITIES:
            G = make_graph(n, density, seed=n*1000 + int(density*10)*10)
            mst_w = sum(d["weight"] for _, _, d in
                        nx.minimum_spanning_edges(G, data=True))
            for k in K_VALUES:
                alpha = 2*k-1

                t0 = time.perf_counter()
                H_gr: nx.Graph | None = None
                for _ in range(GR_REPS):
                    H_gr = greedy_spanner(G, k)
                assert H_gr is not None
                t_gr = (time.perf_counter()-t0) / GR_REPS
                gr_sp = H_gr.number_of_edges() / n
                gr_li = sum(d["weight"] for _, _, d in H_gr.edges(data=True)) / mst_w
                gr_res = verify_spanner(G, H_gr, k, exact_threshold=n+1)

                bs_sp_list, bs_li_list, bs_st_list, bs_t_list = [], [], [], []
                bs_ok = True
                for trial in range(BS_TRIALS):
                    t0 = time.perf_counter()
                    H_bs = baswana_sen_spanner(G, k=k, seed=trial)
                    t_bs = time.perf_counter() - t0
                    r = verify_spanner(G, H_bs, k, exact_threshold=n+1)
                    if not r["passed"]:
                        bs_ok = False
                        print(f"  !! VERIFY FAIL n={n} d={density} k={k} seed={trial}: "
                              f"{r['violations'][:1]}")
                    bs_sp_list.append(H_bs.number_of_edges() / n)
                    bs_li_list.append(
                        sum(d["weight"] for _, _, d in H_bs.edges(data=True)) / mst_w)
                    bs_st_list.append(r["max_stretch"])
                    bs_t_list.append(t_bs)

                row = dict(
                    n=n, density=density, k=k, alpha=alpha,
                    bs_sp=round(np.mean(bs_sp_list),4),
                    bs_li=round(np.mean(bs_li_list),4),
                    bs_str=round(np.mean(bs_st_list),4),
                    bs_t=round(np.mean(bs_t_list),5),
                    gr_sp=round(gr_sp,4), gr_li=round(gr_li,4),
                    gr_str=round(gr_res["max_stretch"],4), gr_t=round(t_gr,4),
                    bs_ok=bs_ok,
                )
                rows.append(row)
                print(f"{n:>5} {density:>4.1f} {k:>2}  "
                      f"{row['bs_sp']:>7.4f} {row['bs_li']:>7.4f} "
                      f"{row['bs_str']:>7.4f} {row['bs_t']:>8.5f}  "
                      f"{row['gr_sp']:>7.4f} {row['gr_li']:>7.4f} "
                      f"{row['gr_str']:>7.4f} {row['gr_t']:>8.4f}  "
                      f"{'ok' if bs_ok else 'FAIL':>4}")

    os.makedirs("/mnt/user-data/outputs/spanner_results", exist_ok=True)
    path = "/mnt/user-data/outputs/spanner_results/bs_vs_greedy.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print()
    all_ok = all(r["bs_ok"] for r in rows)
    print(f"All BS instances verified: {all_ok}")
    print(f"Avg BS  sparseness: {np.mean([r['bs_sp'] for r in rows]):.4f}")
    print(f"Avg GR  sparseness: {np.mean([r['gr_sp'] for r in rows]):.4f}")
    print(f"Avg BS  runtime:    {np.mean([r['bs_t']  for r in rows]):.5f}s")
    print(f"Avg GR  runtime:    {np.mean([r['gr_t']  for r in rows]):.4f}s")
    speedup = np.mean([r['gr_t'] for r in rows]) / max(np.mean([r['bs_t'] for r in rows]),1e-9)
    print(f"Speedup (GR/BS):    {speedup:.1f}x")
    print(f"Results saved -> {path}")


if __name__ == "__main__":
    ok = run_tests()
    if ok:
        benchmark()
    else:
        print("Verification failed — not proceeding to benchmark.")
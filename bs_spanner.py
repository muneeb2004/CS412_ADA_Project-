"""
Baswana-Sen (2k-1)-Spanner implementation with verification and benchmarking.

A weighted graph spanner is a sparse subgraph H of a graph G such that shortest
path distances in H approximate those in G within a multiplicative stretch
factor. Here we target stretch (2k-1), so for every pair (u, v):
distance_H(u, v) <= (2k-1) * distance_G(u, v).

The Baswana-Sen algorithm is a randomized construction that, in expectation,
produces O(k * n^(1+1/k)) edges in O(km) time while guaranteeing stretch.
This file implements the same operational structure used in NetworkX's
authoritative reference implementation.

Two design choices are central to correctness and sparsity:
1) Residual graph processing: once an edge is "covered" by the algorithmic
     progress of a round, it is removed from consideration in later steps.
2) Las Vegas retry: if a sampled round would add too many edges, the round is
     resampled. This preserves the guarantee while controlling output size.

Reference:
    Baswana & Sen, "A Simple and Linear Time Randomized Algorithm for Computing
    Sparse Spanners in Weighted Graphs", Random Structures & Algorithms
    30(4):532-563, 2007.

Algorithm (Phase 1, rounds i = 1 to k-1):
    1. Sample R_i ⊆ cluster_centers with prob n^{-1/k}.
    2. For each vertex v whose cluster center is NOT in R_i:
         Let lightest(C) = min-weight edge from v to cluster C in residual graph.
         (a) If some neighboring cluster center is in R_i:
                     - Let C* = the closest (lightest edge) sampled cluster.
                     - Add edge to C* to spanner; v joins C*.
                     - Also add edges to any cluster C with weight(C) < weight(C*).
                     - Remove from residual: edges to C* and to any cheaper C.
         (b) If NO neighboring cluster center is in R_i:
                     - Add ONE edge to EVERY neighboring cluster (lightest edge each).
                     - Remove ALL incident edges from v in residual graph.
                     - v is disbanded (removed from residual graph).
    3. Remove intra-cluster edges from residual graph.
    4. Remove disbanded vertices from residual graph.

Phase 2:
    For each vertex v in the residual graph:
        Add one edge (lightest) to each neighboring cluster.
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


# ============================================================
# Core algorithm
# ============================================================

def baswana_sen_spanner(G: nx.Graph, k: int, seed: int | None = None) -> nx.Graph:
    """
    Construct a weighted (2k-1)-spanner of an undirected graph.

    Purpose:
    Build a sparse subgraph H that preserves all-pairs distances up to stretch
    factor (2k-1), using the Baswana-Sen Las Vegas randomized strategy.

    Inputs:
    G:
        Undirected weighted NetworkX graph. Missing edge weights default to 1.0.
    k:
        Integer >= 1 controlling the target stretch bound (2k-1).
    seed:
        Optional PRNG seed for reproducible random sampling.

    Output:
    nx.Graph:
        Spanner H with the same node set as G and a subset of weighted edges.

    Complexity:
    - Expected runtime: O(km)
    - Expected edge count: O(k * n^(1+1/k))

    Guarantees and invariants:
    - Las Vegas property: returned H always satisfies the stretch guarantee.
    - H remains a weighted subgraph of G.
    - The residual graph monotonically loses edges/nodes as they become covered.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    rng = random.Random(seed)
    n = G.number_of_nodes()
    if n == 0 or k == 1:
        return G.copy()

    # ── Residual graph: starts as G; edges removed as they get covered ────────
    # The residual graph is the algorithm's "work frontier". If an edge has
    # already been accounted for by a round decision, it is removed so future
    # rounds focus only on unresolved connectivity obligations.
    residual = nx.Graph()
    residual.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        residual.add_edge(u, v, weight=w)

    # ── Spanner (edges accumulated) ───────────────────────────────────────────
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    def add_to_spanner(u: int, v: int) -> None:
        w = G[u][v].get("weight", 1.0)
        if not H.has_edge(u, v) or H[u][v]["weight"] > w:
            H.add_edge(u, v, weight=w)

    # ── Clustering: cluster_of[v] = cluster center ────────────────────────────
    # Initially each node is its own singleton cluster center.
    cluster_of = {v: v for v in G.nodes()}

    # Baswana-Sen sampling probability n^(-1/k): larger k means slower cluster
    # shrinking and lower stretch pressure per round.
    p           = n ** (-1.0 / k)
    # Las Vegas edge cap for one round. If a sampled round exceeds this, we
    # discard the sample and retry to preserve the expected sparsity bound.
    size_limit  = 2 * (n ** (1.0 + 1.0 / k))

    # ── Phase 1: k-1 rounds ───────────────────────────────────────────────────
    for _round in range(k - 1):
        while True:   # Las Vegas: retry if too many edges generated
            # Step 1: sample cluster centers
            current_centers = set(cluster_of.values())
            sampled = {c for c in current_centers if rng.random() < p}

            # Compute changes (don't apply until end of round)
            # Batching ensures all decisions in this round are based on the same
            # clustering snapshot; eager updates would introduce order bias.
            edges_to_add    = set()
            edges_to_remove = set()
            new_cluster_of  = {}

            for v in list(residual.nodes()):
                if cluster_of[v] in sampled:
                    continue    # v's cluster survived; nothing to do

                # Find lightest edge from v to each neighboring cluster
                # in the residual graph (using CURRENT round's clustering snapshot)
                lightest_nbr    = {}   # cluster_center -> neighbor node
                lightest_weight = {}   # cluster_center -> edge weight

                for u in residual.adj[v]:
                    c_u = cluster_of[u]
                    w   = residual[v][u]["weight"]
                    if c_u not in lightest_weight or w < lightest_weight[c_u]:
                        lightest_nbr[c_u]    = u
                        lightest_weight[c_u] = w

                # Which of those neighboring clusters are sampled?
                nbr_sampled = set(lightest_weight.keys()) & sampled

                if not nbr_sampled:
                    # Disband: add one edge to EACH neighboring cluster,
                    # remove ALL incident edges from residual.
                    # Intuition: without any sampled adjacent cluster to join,
                    # the node secures local connectivity guarantees now and is
                    # then removed from active residual processing.
                    for u in lightest_nbr.values():
                        edges_to_add.add((v, u))
                    for u in list(residual.adj[v]):
                        edges_to_remove.add((v, u))
                    # v leaves clustering (will be removed from residual)

                else:
                    # Join: find closest sampled cluster.
                    # This becomes the node's new cluster center for next round.
                    closest = min(nbr_sampled, key=lambda c: lightest_weight[c])
                    w_closest = lightest_weight[closest]
                    u_closest = lightest_nbr[closest]

                    edges_to_add.add((v, u_closest))
                    new_cluster_of[v] = closest

                    # Also add edges to any cheaper neighboring cluster.
                    # These edges are required by the analysis to preserve
                    # stretch while transitioning cluster memberships.
                    for c, w_c in lightest_weight.items():
                        if w_c < w_closest:
                            edges_to_add.add((v, lightest_nbr[c]))

                    # Remove edges to closest cluster and all cheaper clusters.
                    # They are now "covered" by this round's spanner additions.
                    for u in list(residual.adj[v]):
                        c_u = cluster_of[u]
                        w_u = lightest_weight.get(c_u, math.inf)
                        if c_u == closest or w_u < w_closest:
                            edges_to_remove.add((v, u))

            # Las Vegas check: reject pathological samples that would violate the
            # expected sparsity target, then resample this round.
            if len(edges_to_add) > size_limit:
                continue    # retry this round with a new sample
            break           # round succeeded

        # Apply changes to spanner
        for u, v in edges_to_add:
            add_to_spanner(u, v)

        # Remove covered edges from residual graph so they are never reconsidered.
        residual.remove_edges_from(edges_to_remove)

        # Propagate old clustering for surviving cluster members.
        # This is done after round decisions so membership updates are atomic at
        # round granularity, matching the algorithmic model.
        for v, c in cluster_of.items():
            if c in sampled and v not in new_cluster_of:
                new_cluster_of[v] = c
        cluster_of = new_cluster_of

        # Step 4: remove intra-cluster edges from residual graph
        # Intra-cluster edges are irrelevant for future inter-cluster decisions,
        # so dropping them reduces residual density and speeds later rounds.
        intra = [(u, v) for u, v in residual.edges()
                 if cluster_of.get(u) == cluster_of.get(v) and
                    cluster_of.get(u) is not None]
        residual.remove_edges_from(intra)

        # Remove disbanded vertices from residual graph
        # Disbanded nodes have already connected to all neighboring clusters by
        # lightest edges; they no longer need participation in subsequent rounds.
        disbanded = [v for v in list(residual.nodes()) if v not in cluster_of]
        residual.remove_nodes_from(disbanded)

    # ── Phase 2: connect remaining vertices to neighboring clusters ───────────
    # Final cleanup pass: each surviving residual vertex adds one lightest edge
    # per neighboring cluster, completing coverage obligations.
    for v in list(residual.nodes()):
        lightest_nbr    = {}
        lightest_weight = {}
        for u in residual.adj[v]:
            c_u = cluster_of.get(u)
            if c_u is None:
                continue
            w = residual[v][u]["weight"]
            if c_u not in lightest_weight or w < lightest_weight[c_u]:
                lightest_nbr[c_u]    = u
                lightest_weight[c_u] = w
        for u in lightest_nbr.values():
            add_to_spanner(v, u)

    return H


# ============================================================
# Verification
# ============================================================

def verify_spanner(
    G: nx.Graph,
    H: nx.Graph,
    k: int,
    exact_threshold: int = 300,
    n_samples: int = 1000,
    rng_seed: int = 0,
) -> VerifyResult:
    """
    Verify whether H is a valid weighted (2k-1)-spanner of G.

    Purpose:
    Provide a practical correctness oracle used by the test suite and benchmark.
    The checker validates both structural consistency (H subset of G) and
    distance-stretch constraints.

    Inputs:
    G:
        Original weighted graph.
    H:
        Candidate spanner graph.
    k:
        Stretch parameter, so alpha = 2k-1.
    exact_threshold:
        If n <= exact_threshold, test all node pairs exactly.
    n_samples:
        Number of random pairs for approximate checking on larger graphs.
    rng_seed:
        PRNG seed for deterministic sampled verification.

    Output:
    VerifyResult with fields:
        passed: True if no violations were detected.
        violations: human-readable descriptions of all failures.
        max_stretch: maximum observed d_H / d_G among checked pairs.

    Complexity:
    - Structural checks are linear in edge count.
    - Distance checking dominates and runs one Dijkstra in G and H per tested
      pair (all pairs for small graphs, sampled pairs for larger graphs).
    """
    alpha      = 2 * k - 1
    n          = G.number_of_nodes()
    rng        = random.Random(rng_seed)
    nodes      = list(G.nodes())
    violations = []

    # Check H ⊆ G with matching weights to ensure this is a true subgraph
    # spanner, not an arbitrary auxiliary graph.
    for u, v, d in H.edges(data=True):
        if not G.has_edge(u, v):
            violations.append(f"Edge ({u},{v}) in H but not in G")
        else:
            if abs(d.get("weight",1.0) - G[u][v].get("weight",1.0)) > 1e-9:
                violations.append(f"Edge ({u},{v}) weight mismatch")

    # Connectivity sanity check: if G is connected, H must remain connected,
    # otherwise stretch is undefined for unreachable pairs.
    if nx.is_connected(G) and not nx.is_connected(H):
        violations.append("G connected but H is not")

    # Stretch evaluation strategy:
    # - Exact mode for small graphs gives a complete guarantee.
    # - Sampling mode for larger graphs provides a scalable estimate.
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
        # Ignore degenerate zero-distance pairs (same node would be excluded,
        # but this guard keeps ratio computation numerically safe).
        if d_G > 0:
            ratio = d_H / d_G
            max_ratio = max(max_ratio, ratio)
            # Core stretch criterion for a (2k-1)-spanner.
            if ratio > alpha + 1e-6:
                violations.append(
                    f"Stretch ({u},{v}): dH={d_H:.4f} dG={d_G:.4f} "
                    f"ratio={ratio:.4f} > alpha={alpha}"
                )

    return {"passed": len(violations)==0, "violations": violations, "max_stretch": max_ratio}


# ============================================================
# Test suite
# ============================================================

def make_graph(n: int, density: float, seed: int) -> nx.Graph:
    """
    Generate a connected random undirected weighted graph.

    Purpose:
    Create reproducible test/benchmark instances using an Erdos-Renyi style
    edge sampling process, then enforce connectivity if needed.

    Inputs:
    n:
        Number of vertices.
    density:
        Independent probability for each potential edge.
    seed:
        PRNG seed for deterministic generation.

    Output:
    nx.Graph with integer edge weights in [1, n].

    Complexity:
    O(n^2) edge sampling in the dense upper bound.

    Invariant:
    For n > 0, the returned graph is connected.
    """
    rng = random.Random(seed)
    G   = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < density:
                G.add_edge(i, j, weight=rng.randint(1, n))
    if n > 0 and not nx.is_connected(G):
        # Connect components by adding one bridge between consecutive components.
        # This preserves randomness while guaranteeing connectivity for spanner
        # verification and stretch measurements.
        comps = list(nx.connected_components(G))
        for idx in range(len(comps)-1):
            u = next(iter(comps[idx]))
            v = next(iter(comps[idx+1]))
            G.add_edge(u, v, weight=rng.randint(1, n))
    return G


def run_tests() -> bool:
    """
    Execute a focused correctness suite for the Baswana-Sen implementation.

    Purpose:
    Exercise edge cases, structured graphs, randomized instances, and a
    cross-check against NetworkX's built-in implementation.

    Output:
    bool indicating whether all test families passed.

    Notes:
    - Tests emphasize correctness guarantees (subgraph property, connectivity,
      and stretch bound), not micro-performance.
    - The suite intentionally mixes deterministic and randomized scenarios.
    """
    print("=" * 60)
    print("VERIFICATION SUITE")
    print("=" * 60)
    all_ok = True

    # T1: k=1 is the identity case. Stretch=1 implies the full graph is needed.
    g = make_graph(10, 0.5, 0)
    h = baswana_sen_spanner(g, k=1, seed=0)
    assert set(h.edges()) == set(g.edges()), "k=1 must return G"
    print("[PASS] k=1 returns G unchanged")

    # T2: empty input should be handled without special-case failures.
    h = baswana_sen_spanner(nx.Graph(), k=2, seed=0)
    assert h.number_of_nodes() == 0
    print("[PASS] Empty graph")

    # T3: path graph regression test. Paths are sparse and sensitive to errors
    # in disband/join handling, so this catches subtle logic mistakes.
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

    # T4: dense graph stress test on a complete graph with random weights.
    # This validates behavior when every vertex sees many neighboring clusters.
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

    # T5: broad randomized coverage on small/medium sizes using exact pair
    # checking, giving high confidence in stretch correctness.
    SIZES     = [8, 12, 15, 20, 30, 50]
    DENSITIES = [0.2, 0.4, 0.7]
    K_VALUES  = [2, 3, 4]
    TRIALS    = 5
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

    # T6: structural invariant check: every spanner edge must come from G.
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

    # T7: external cross-check with NetworkX's Baswana-Sen implementation.
    # Both should satisfy the same formal stretch guarantee.
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
            print(f"  [WARN] NX invalid n={n} d={d} k={k}")   # shouldn't happen
    print(f"[{'PASS' if cmp_ok else 'FAIL'}] Cross-check vs NetworkX nx.spanner()")

    print("=" * 60)
    print(f"SUITE {'PASSED' if all_ok else 'FAILED'}")
    print("=" * 60)
    return all_ok


# ============================================================
# Benchmark: BS vs Greedy
# ============================================================

def greedy_spanner(G: nx.Graph, k: int) -> nx.Graph:
    """
    Build a weighted greedy (2k-1)-spanner for comparison purposes.

    Purpose:
    Provide a baseline spanner constructor to compare against Baswana-Sen in
    edge count, lightness, stretch, and runtime.

    Inputs:
    G:
        Undirected weighted graph.
    k:
        Stretch parameter, giving threshold alpha = 2k-1.

    Output:
    nx.Graph containing a greedy-selected subset of edges.

    Complexity:
    Edges are processed in nondecreasing weight order; each step runs a
    shortest-path query on the current spanner, which is typically expensive
    but serves as a useful quality baseline.
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

    SIZES     = [10, 20, 50, 100, 200]
    DENSITIES = [0.1, 0.3, 0.5]
    K_VALUES  = [2, 3]
    BS_TRIALS = 10
    GR_REPS   = 3

    rows = []
    for n in SIZES:
        for density in DENSITIES:
            G = make_graph(n, density, seed=n*1000 + int(density*10)*10)
            mst_w = sum(d["weight"] for _,_,d in
                        nx.minimum_spanning_edges(G, data=True))
            for k in K_VALUES:
                alpha = 2*k-1

                # Greedy
                t0 = time.perf_counter()
                H_gr: nx.Graph | None = None
                for _ in range(GR_REPS):
                    H_gr = greedy_spanner(G, k)
                assert H_gr is not None
                t_gr = (time.perf_counter()-t0) / GR_REPS
                gr_sp = H_gr.number_of_edges() / n
                gr_li = sum(d["weight"] for _,_,d in H_gr.edges(data=True)) / mst_w
                gr_res = verify_spanner(G, H_gr, k, exact_threshold=n+1)

                # BS (multiple seeds)
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
                        sum(d["weight"] for _,_,d in H_bs.edges(data=True)) / mst_w)
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
                      f"{'✓' if bs_ok else '✗':>4}")

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
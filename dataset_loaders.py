"""Dataset loader utilities for TSPLIB, Facebook, and Gnutella graphs.

The loaders normalize each source into weighted undirected NetworkX graphs
suitable for spanner experiments.
"""

from __future__ import annotations

from pathlib import Path
import random

import networkx as nx
import tsplib95


def load_tsplib_complete_graphs(tsplib_dir: str | Path) -> dict[str, nx.Graph]:
    """Load TSPLIB instances as complete weighted graphs.

    Exclusion criteria (matching Chimani & Stutzenstein ESA 2022):
      - Directed instances (type == 'ATSP')
      - Any instance where any pairwise weight equals 0

    NOTE: is_depictable() is NOT used as a filter — it returns False for
    EXPLICIT matrix instances which are valid and used in the paper.

    Returns:
        Mapping of instance stem (e.g., 'berlin52') to a weighted graph.
    """
    directory = Path(tsplib_dir)
    graphs: dict[str, nx.Graph] = {}

    for tsp_file in sorted(directory.glob("*.tsp")):
        try:
            problem = tsplib95.load(tsp_file)
        except Exception:
            continue

        # Skip directed instances
        problem_type = getattr(problem, "type", "TSP")
        if isinstance(problem_type, str) and problem_type.upper() == "ATSP":
            continue

        try:
            nodes = list(problem.get_nodes())
        except Exception:
            continue

        if len(nodes) < 2:
            continue

        graph = nx.Graph()
        graph.add_nodes_from(nodes)

        has_zero = False
        try:
            for idx, u in enumerate(nodes):
                for v in nodes[idx + 1:]:
                    w = float(problem.get_weight(u, v))
                    if w == 0:
                        has_zero = True
                        break
                    graph.add_edge(u, v, weight=w)
                if has_zero:
                    break
        except Exception:
            continue

        if has_zero:
            continue

        graphs[tsp_file.stem] = graph

    return graphs


def load_facebook_graph(path: str | Path, seed: int = 42) -> nx.Graph:
    """Load Facebook edge list and assign deterministic random weights.

    Expects facebook_combined.txt (or .gz) format.
    Already undirected and connected — no filtering needed.
    Edge weights: random integers in [1, n], fixed seed.
    """
    graph = nx.read_edgelist(
        str(path), nodetype=int, data=False, create_using=nx.Graph
    )
    _assign_random_integer_weights(graph, seed=seed)
    return graph


def load_gnutella_graph(path: str | Path, seed: int = 42) -> nx.Graph:
    """Load Gnutella P2P graph, extract largest CC, assign weights.

    Expects p2p-Gnutella08.txt (or .gz) format.
    Steps: read directed -> undirected -> largest CC -> assign weights.
    Edge weights: random integers in [1, n], fixed seed.
    """
    directed = nx.read_edgelist(
        str(path),
        nodetype=int,
        data=False,
        create_using=nx.DiGraph,
        comments="#",
    )
    undirected = directed.to_undirected()
    largest_cc = max(nx.connected_components(undirected), key=len)
    graph = undirected.subgraph(largest_cc).copy()
    _assign_random_integer_weights(graph, seed=seed)
    return graph


def _assign_random_integer_weights(graph: nx.Graph, seed: int) -> None:
    """Assign random integer edge weights in [1, n] in-place."""
    rng = random.Random(seed)
    n = graph.number_of_nodes()
    for u, v in graph.edges():
        graph[u][v]["weight"] = rng.randint(1, n)
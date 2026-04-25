"""Dataset loader utilities for TSPLIB, Facebook, Gnutella, and Steinlib graphs.

The loaders normalize each source into weighted undirected NetworkX graphs
suitable for spanner experiments.
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import TYPE_CHECKING, Any, Callable, Protocol, TypeAlias, cast

import networkx as nx
import tsplib95  # pyright: ignore[reportMissingTypeStubs]


NodeId = int
if TYPE_CHECKING:
    GraphT: TypeAlias = nx.Graph[NodeId]
else:
    GraphT: TypeAlias = nx.Graph


class TsplibProblem(Protocol):
    type: str

    def get_nodes(self) -> list[NodeId]: ...

    def get_weight(self, start: NodeId, end: NodeId) -> float | int: ...


TSPLIB_LOAD = cast(Callable[[Path], Any], tsplib95.load)  # pyright: ignore[reportUnknownMemberType]


def load_tsplib_complete_graphs(tsplib_dir: str | Path) -> dict[str, GraphT]:
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
    graphs: dict[str, GraphT] = {}

    for tsp_file in sorted(directory.glob("*.tsp")):
        try:
            problem = cast(TsplibProblem, TSPLIB_LOAD(tsp_file))
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

        graph: GraphT = cast(GraphT, nx.Graph())
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


def load_facebook_graph(path: str | Path, seed: int = 42) -> GraphT:
    """Load Facebook edge list and assign deterministic random weights.

    Expects facebook_combined.txt (or .gz) format.
    Already undirected and connected — no filtering needed.
    Edge weights: random integers in [1, n], fixed seed.
    """
    graph = cast(
        GraphT,
        nx.read_edgelist(  # pyright: ignore[reportUnknownMemberType]
            str(path), nodetype=int, data=False, create_using=nx.Graph
        ),
    )
    _assign_random_integer_weights(graph, seed=seed)
    return graph


def load_gnutella_graph(path: str | Path, seed: int = 42) -> GraphT:
    """Load Gnutella P2P graph, extract largest CC, assign weights.

    Expects p2p-Gnutella08.txt (or .gz) format.
    Steps: read directed -> undirected -> largest CC -> assign weights.
    Edge weights: random integers in [1, n], fixed seed.
    """
    directed = cast(
        "nx.DiGraph[NodeId]",
        nx.read_edgelist(  # pyright: ignore[reportUnknownMemberType]
            str(path),
            nodetype=int,
            data=False,
            create_using=nx.DiGraph,
            comments="#",
        ),
    )
    undirected: GraphT = directed.to_undirected()
    largest_cc: set[NodeId] = max(nx.connected_components(undirected), key=len)  # pyright: ignore[reportUnknownArgumentType]
    graph: GraphT = undirected.subgraph(largest_cc).copy()
    _assign_random_integer_weights(graph, seed=seed)
    return graph


def _assign_random_integer_weights(graph: GraphT, seed: int) -> None:
    """Assign random integer edge weights in [1, n] in-place."""
    rng = random.Random(seed)
    n = graph.number_of_nodes()
    for u, v in graph.edges():
        graph[u][v]["weight"] = rng.randint(1, n)


def load_steinlib_graph(path: str | Path) -> GraphT | None:
    """Parse a Steinlib .stp file and return a weighted undirected graph.

    STP format (relevant sections):
        SECTION Graph
        Nodes <n>
        Edges <m>
        E <u> <v> <w>
        ...
        END

    Returns None if the file is malformed or has no edges.
    Node IDs are 1-indexed in STP; preserved as-is.
    """
    graph: GraphT = cast(GraphT, nx.Graph())
    in_graph_section = False

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                upper = line.upper()

                if upper.startswith("SECTION GRAPH"):
                    in_graph_section = True
                    continue

                if in_graph_section:
                    if upper == "END":
                        break
                    if upper.startswith("NODES"):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                n = int(parts[1])
                                graph.add_nodes_from(range(1, n + 1))
                            except ValueError:
                                pass
                    elif upper.startswith("E "):
                        parts = line.split()
                        if len(parts) >= 4:
                            try:
                                u, v, w = int(parts[1]), int(parts[2]), float(parts[3])
                                if w > 0:
                                    graph.add_edge(u, v, weight=w)
                            except ValueError:
                                pass
    except Exception:
        return None

    if graph.number_of_edges() == 0:
        return None

    return graph

def load_steinlib_graphs(steinlib_dir: str | Path) -> dict[str, GraphT]:
    """Load all valid .stp files from a Steinlib directory tree.

    Walks subdirectories recursively (Steinlib is organized into 44 subsets).
    Exclusion: files with zero edges or parse failures.

    Returns:
        Mapping of relative stem path (e.g., 'cc/cc3-10') to graph.
    """
    root = Path(steinlib_dir)
    graphs: dict[str, GraphT] = {}

    for stp_file in sorted(root.rglob("*.stp")):
        graph = load_steinlib_graph(stp_file)
        if graph is None:
            continue
        # Use path relative to root as key to preserve subset identity
        try:
            key = str(stp_file.relative_to(root).with_suffix(""))
        except ValueError:
            key = stp_file.stem
        graphs[key] = graph

    return graphs



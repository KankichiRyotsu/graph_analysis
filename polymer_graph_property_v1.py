from __future__ import annotations
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Callable

# ------------------------------------------------------------
# Helpers & data structures
# ------------------------------------------------------------

@dataclass
class ReduceStats:
    n_bridge_center: int = 0
    n_dangling_front: int = 0
    n_ineffective_loop: int = 0
    rounds: int = 0

def _ensure_edge_len(G: nx.Graph, default: int = 1) -> None:
    """Ensure every edge has integer 'len' attribute (edge length / weight)."""
    for u, v in G.edges():
        if "len" not in G[u][v]:
            G[u][v]["len"] = default

def _is_junction(G: nx.Graph, n: int, thr: int = 3) -> bool:
    """Node is a junction if degree >= thr."""
    return G.degree(n) >= thr

# ------------------------------------------------------------
# (1) Bridge center elimination
#   Interpret as: suppress degree-2 nodes by contracting paths
#   and accumulating edge 'len'. This is a standard line-graph
#   simplification for polymer backbones.
# ------------------------------------------------------------

def eliminate_bridge_centers(
    G: nx.Graph,
    degree2_ok_on_cycles: bool = False,
) -> Tuple[nx.Graph, int]:
    """
    Remove degree-2 intermediates by contracting u-j-v into u-v and summing lengths.
    If an u-v edge already exists, accumulate 'len'.
    degree2_ok_on_cycles: if False, skips degree-2 nodes that lie on cycles
                          (only contract degree-2 on bridges).
    Returns new graph and #removed nodes.
    """
    H = G.copy()
    _ensure_edge_len(H, 1)

    removed = 0
    # Collect degree-2 candidates first to avoid mutation during iteration
    candidates = [n for n in H.nodes() if H.degree(n) == 2]
    if not candidates:
        return H, 0

    # Optionally restrict to nodes not on cycles
    if not degree2_ok_on_cycles:
        # node on a cycle iff it belongs to some cycle basis
        on_cycle: Set[int] = set().union(*nx.cycle_basis(H)) if H.number_of_edges() else set()
        candidates = [n for n in candidates if n not in on_cycle]

    for j in candidates:
        if j not in H or H.degree(j) != 2:
            continue
        nbrs = list(H.neighbors(j))
        if len(nbrs) != 2:
            continue
        a, b = nbrs
        # Accumulate edge length: len(a-b) = len(a-j) + len(j-b)
        len_aj = H[a][j].get("len", 1)
        len_jb = H[j][b].get("len", 1)
        new_len = len_aj + len_jb

        # Remove j and connect a-b
        H.remove_node(j)
        removed += 1

        if H.has_edge(a, b):
            H[a][b]["len"] = H[a][b].get("len", 1) + new_len
        else:
            H.add_edge(a, b, len=new_len)

    return H, removed

# ------------------------------------------------------------
# (2) Dangling front elimination
#   Peel a leaf chain until a junction (deg>=3) or stub end.
#   Accumulate total length into the junction via callback.
# ------------------------------------------------------------

def remove_leaves_once(
    G: nx.Graph,
    *,
    on_leaf: Optional[Callable[[nx.Graph, int, int, int], None]] = None,
    # callback(H_before, leaf, neighbor, edge_len)
    remove_isolates: bool = False,  # optional: also drop degree-0 nodes after removal
) -> Tuple[nx.Graph, int, List[int]]:
    """
    Delete ALL degree-1 nodes present in this round (strict reading of panel d).
    Returns:
        H: graph after this round
        n_removed: number of leaves removed
        leaves: list of removed leaf node ids
    """
    H = G.copy()

    # collect current leaves
    leaves = [n for n, d in H.degree() if d == 1]
    if not leaves:
        return H, 0, []

    # optional: notify before removal (for A-updates etc.)
    if on_leaf is not None:
        for u in leaves:
            nbrs = list(H.neighbors(u))
            if nbrs:
                v = nbrs[0]                 # leaf has exactly one neighbor
                w = H[u][v].get("len", 1)   # bond length/weight; default=1
                on_leaf(H, u, v, w)

    H.remove_nodes_from(leaves)

    if remove_isolates:
        isolates = [n for n, d in H.degree() if d == 0]
        if isolates:
            H.remove_nodes_from(isolates)

    return H, len(leaves), leaves

# ------------------------------------------------------------
# (3) Ineffective loop elimination
#   Remove a cycle (biconnected component that is a pure ring) that attaches
#   to the rest of the graph through a single articulation node.
# ------------------------------------------------------------

def eliminate_one_selfloop_edge(G: nx.Graph, select="first"):
    H = G.copy()
    try:
        loops = list(H.selfloop_edges(keys=True))  # MultiGraph
        if not loops: return H, None
        if select == "random":
            import random; u, _, k = random.choice(loops)
        elif select == "maxdeg":
            u, _, k = max(loops, key=lambda t: H.degree(t[0]))
        else:
            u, _, k = loops[0]
        H.remove_edge(u, u, key=k)
        return H, u
    except TypeError:
        loops = list(H.selfloop_edges())          # simple Graph
        if not loops: return H, None
        u, _ = loops[0]
        H.remove_edge(u, u)
        return H, u

# ------------------------------------------------------------
# Full pipeline (Fig. a)
# ------------------------------------------------------------

def reduce_effective_graph(
    G: nx.Graph,
    *,
    junction_degree: int = 3,
    max_rounds: int = 10_000,
    # Optional callbacks to implement your Fig.(c)(d)(e) A-updates:
    on_bridge_contracted: Optional[Callable[[nx.Graph, int], None]] = None,
    on_leaf: Optional[Callable[[nx.Graph, int, int, int], None]] = None,
    on_loop_eliminated: Optional[Callable[[nx.Graph, int, int], None]] = None,
    contract_degree2_on_cycles: bool = True,
) -> Tuple[nx.Graph, ReduceStats]:
    """
    Implements the flow in panel (a):
      - Bridge center detection & elimination
      - Dangling front detection & elimination
      - Ineffective loop detection & elimination
    Repeats until none is detected, then returns the effective graph.

    Callbacks:
      on_bridge_contracted(H, count_removed_degree2_nodes)
      on_dangling_peeled(H, junction_node, peeled_length)
      on_loop_eliminated(H, articulation_node, loop_length)

    Notes:
      - Edge attribute 'len' is maintained/accumulated (default 1 per bond).
      - You can store/update node attributes (e.g., 'A') in callbacks to match (c)(d)(e).
    """
    H = G.copy()
    _ensure_edge_len(H, 1)
    stats = ReduceStats()

    for r in range(1, max_rounds + 1):
        changed = False

        # (c) Bridge center elimination (degree-2 contraction across the graph)
        H2, removed_bridge = eliminate_bridge_centers(
            H, degree2_ok_on_cycles=contract_degree2_on_cycles
        )
        if removed_bridge > 0:
            if on_bridge_contracted:
                on_bridge_contracted(H2, removed_bridge)
            H = H2
            stats.n_bridge_center += removed_bridge
            changed = True

        # (d) Dangling front elimination (one per iteration)
        H2, n_leaves, _ = remove_leaves_once(H, on_leaf=on_leaf, remove_isolates=False)
        if n_leaves > 0:
            H = H2
            stats.n_dangling_front += 1  # one detection/elimination event in this round
            changed = True

        # (e) Ineffective loop elimination (one per iteration)
        H2, removed_loop = eliminate_one_selfloop_edge(
            H,
            select="first",
        )
        if removed_loop > 0:
            H = H2
            stats.n_ineffective_loop += 1
            changed = True

        stats.rounds += 1
        if not changed:
            break

    return H, stats

# ------------------------------------------------------------
# Example callbacks for A-updates (you may replace by the paper's formulae)
# ------------------------------------------------------------

def make_A_dict(G: nx.Graph) -> Dict[int, int]:
    """Initialize A_i = 0 for all nodes."""
    return {n: 0 for n in G.nodes()}

def cb_dangling_increment_A(A: Dict[int, int]) -> Callable[[nx.Graph, int, int], None]:
    """
    Return a callback that adds peeled length to the terminal junction's A.
    Matches an intuitive 'arm length accumulation' for (d).
    """
    def _cb(H: nx.Graph, junction: int, peeled_len: int) -> None:
        A[junction] = A.get(junction, 0) + peeled_len
    return _cb

def cb_loop_decrement_A_by_two(A: Dict[int, int]) -> Callable[[nx.Graph, int, int], None]:
    """
    Example mapping for (e): decrement A at the articulation by 2 for each loop.
    Adjust if your paper defines a different rule.
    """
    def _cb(H: nx.Graph, articulation: int, loop_len: int) -> None:
        A[articulation] = A.get(articulation, 0) - 2
    return _cb


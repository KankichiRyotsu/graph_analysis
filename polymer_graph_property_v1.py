from __future__ import annotations
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Callable

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path


def draw_multiedge_graph(
    G: nx.MultiGraph,
    *,
    title: str = "",
    node_size: int = 300,
    out_png: Optional[str | Path] = None,
):
    """
    A-B 間の多重辺を左右対称に曲げて描画する可視化関数。
    2本なら +rad, -rad で左右対称。
    """
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(5, 5))
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=node_size, edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=10)

    ax = plt.gca()

    # 各ペアごとにエッジを集める
    edge_groups = {}
    for u, v, k in G.edges(keys=True):
        if u == v:
            continue  # 自己ループは省略
        pair = tuple(sorted((u, v)))
        edge_groups.setdefault(pair, []).append((u, v, k))

    for pair, edges in edge_groups.items():
        u, v = pair
        n = len(edges)
        # rad 値を左右対称に並べる
        # 例えば2本なら [-0.2, +0.2]、3本なら [-0.3, 0, +0.3]
        if n % 2 == 1:
            rads = [(i - n//2) * 0.3 for i in range(n)]
        else:
            rads = [((i - n/2) + 0.5) * 0.3 for i in range(n)]
        for rad in rads:
            patch = FancyArrowPatch(
                pos[u], pos[v],
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-",
                linewidth=2,
                color="black",
                alpha=0.8,
            )
            ax.add_patch(patch)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if out_png:
        out_png = Path(out_png)
        plt.savefig(out_png, dpi=150)
        print(f"[draw_multiedge_graph] saved to {out_png.resolve()}")
        plt.close()
    else:
        plt.show()



# ------------------------------------------------------------
# Helpers & data structures
# ------------------------------------------------------------

@dataclass
class ReduceStats:
    n_bridge_center: int = 0
    n_dangling_front: int = 0
    n_ineffective_loop: int = 0
    rounds: int = 0


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

    j = candidates[0]
    nbrs = list(H.neighbors(j))
    if len(nbrs) == 2:
        a, b = nbrs
        # Remove j and connect a-b
        H.remove_node(j)
        removed += 1
        H.add_edge(a, b)
        return H, removed
    if len(nbrs) == 1:
        a = nbrs[0]
        if a == j:
            H.remove_node(j)
            return H, 0
        H.remove_node(j)
        removed += 1
        H.add_edge(a, a)
        return H, removed

# ------------------------------------------------------------
# (2) Dangling front elimination
#   Peel a leaf chain until a junction (deg>=3) or stub end.
#   Accumulate total length into the junction via callback.
# ------------------------------------------------------------

def remove_leaves_once(
    G: nx.Graph,
    *,
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
        return H, 0

    H.remove_node(leaves[0])

    if remove_isolates:
        isolates = [n for n, d in H.degree() if d == 0]
        if isolates:
            H.remove_nodes_from(isolates)

    return H, 1

# ------------------------------------------------------------
# (3) Ineffective loop elimination
#   Remove a cycle (biconnected component that is a pure ring) that attaches
#   to the rest of the graph through a single articulation node.
# ------------------------------------------------------------

def eliminate_one_selfloop_edge(G: nx.Graph):
    H = G.copy()
    loops =  [(u, v, k) for u, v, k in H.edges(keys=True) if u == v]
    if not loops:
        return H, 0
    u, _, k = loops[0]
    H.remove_edge(u, u, key=k)
    return H, 1


# ------------------------------------------------------------
# Full pipeline (Fig. a)
# ------------------------------------------------------------

def reduce_effective_graph(
    G: nx.Graph,
    *,
    max_rounds: int = 10_000,
    # Optional callbacks to implement your Fig.(c)(d)(e) A-updates:
    contract_degree2_on_cycles: bool = True,
) -> Tuple[nx.Graph, ReduceStats]:
    """
    Implements the flow in panel (a):
      - Bridge center detection & elimination
      - Dangling front detection & elimination
      - Ineffective loop detection & elimination
    Repeats until none is detected, then returns the effective graph.
    """
    H = G.copy()
    stats = ReduceStats()

    for r in range(1, max_rounds + 1):
        changed = False

        # (c) Bridge center elimination (degree-2 contraction across the graph)
        H2, removed_bridge = eliminate_bridge_centers(
            H, degree2_ok_on_cycles=contract_degree2_on_cycles
        )
        if removed_bridge > 0:
            H = H2
            stats.n_bridge_center += 1
            changed = True
            stats.rounds += 1
            continue

        # (d) Dangling front elimination (one per iteration)
        H2, n_leaves = remove_leaves_once(H, remove_isolates=False)
        if n_leaves > 0:
            H = H2
            stats.n_dangling_front += 1  # one detection/elimination event in this round
            changed = True
            stats.rounds += 1
            continue

        # (e) Ineffective loop elimination (one per iteration)
        H2, removed_loop = eliminate_one_selfloop_edge(H)
        if removed_loop > 0:
            H = H2
            stats.n_ineffective_loop += 1
            changed = True
            stats.rounds += 1
            continue

        if not changed:
            break

    return H, stats


if __name__ == "__main__":
    G = nx.MultiGraph()
    G.add_edges_from([(1,2),(2,3),(2,3)])
    draw_multiedge_graph(G)
    H, stats = reduce_effective_graph(G)
    print(stats)

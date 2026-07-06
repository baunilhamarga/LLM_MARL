"""
Compression utility for gym-dragon graphs
-----------------------------------------

>>> from gym_dragon.core.graph import Graph as DragonGraph
>>> from utils.graph_compression import CompressedGraph
>>> CG = CompressedGraph(env_graph)      # env_graph is a DragonGraph
>>> print(CG.region_adjlist_str("A"))
23 : B
34 : 38 41 B
...
>>> CG.entry_point(23, "B")              # → concrete node (e.g. 47)
"""

import math, random
from collections import defaultdict
from typing import Union, Callable, Optional, Hashable, Dict, List, Set, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from networkx.algorithms import community
import collections

import networkx as nx

# ----------------------------------------------------------------------
# 1) partitioning helpers (unchanged)
# ----------------------------------------------------------------------

def partition_graph(G: nx.Graph, method: str = "kl", k: int = 2, seed=None):
    """Return list[set] of node IDs (one set per region)."""
    method = method.lower()

    if method in {"kl", "kernighan-lin"} and k == 2:
        A, B = nx.algorithms.community.kernighan_lin_bisection(G, seed=seed)
        return [set(A), set(B)]

    if method in {"louvain", "leiden"}:
        try:
            import community as community_louvain
        except ImportError as e:
            raise ImportError("pip install python-louvain") from e
        part = community_louvain.best_partition(G, random_state=seed)
        buckets = defaultdict(set)
        for n, cid in part.items():
            buckets[cid].add(n)
        return list(buckets.values())

    if method == "spectral":
        from networkx.algorithms.community import spectral_modularity_maximization
        return list(spectral_modularity_maximization(G))

    # ---  simple round-robin ------------------------------------------
    if method == "simple":
        nodes = list(G.nodes())
        random.Random(seed).shuffle(nodes)
        size   = math.ceil(len(nodes) / k)
        parts  = [set(nodes[i*size:(i+1)*size]) for i in range(k)]
        return [p for p in parts if p]          # guaranteed non-empty

    # ---  spectral recursive bisection --------------------------------
    if method in {"spectral_recursive", "spectral_k"}:
        parts = [set(G.nodes())]
        while len(parts) < k:
            # always bisect the largest current part
            largest = max(parts, key=len)
            parts.remove(largest)
            a, b = community.kernighan_lin_bisection(G.subgraph(largest), seed=seed)
            # kernighan_lin always returns two non-empty sets
            parts.extend([set(a), set(b)])
        return parts   # exactly k parts

    if method.lower() == "metis":
        try:
            import nxmetis
        except ImportError as e:
            raise ImportError("pip install nxmetis") from e

        # ------------------------------------------------------------------
        # FIX: give NetworkX 3.x graphs the .node attribute expected by nxmetis
        # ------------------------------------------------------------------
        if not hasattr(G, "node"):
            G.node = G._node          # ← single-line shim

        _, parts = nxmetis.partition(G, k)
        return [set(p) for p in parts]

    # ------------------------------------------------------------------
    # balanced BFS region growing (always k, connected, near-equal) ----
    # ------------------------------------------------------------------
    if method.lower() in {"bfs_balance", "balanced_bfs"}:
        if k > G.number_of_nodes():
            raise ValueError("k cannot exceed number of nodes")

        rng   = random.Random(seed)
        # -- 1) pick k seeds by farthest-point -------------------------
        seeds = []
        first = rng.choice(list(G.nodes()))
        seeds.append(first)
        for _ in range(1, k):
            # distance to nearest chosen seed
            dists = nx.multi_source_dijkstra_path_length(G, seeds)
            farthest = max(dists, key=dists.get)
            seeds.append(farthest)

        # region id by node; −1 = unassigned
        region_of = {n: -1 for n in G}
        frontiers = [collections.deque([s]) for s in seeds]
        for rid, s in enumerate(seeds):
            region_of[s] = rid

        # -- 2) round-robin BFS growth --------------------------------
        target_size = math.ceil(G.number_of_nodes() / k)
        sizes = [1] * k
        while any(frontiers):
            # always grow the smallest current region
            rid = min((i for i,f in enumerate(frontiers) if f),
                    key=lambda i: sizes[i])
            v = frontiers[rid].popleft()
            for w in G[v]:
                if region_of[w] == -1:
                    region_of[w] = rid
                    sizes[rid] += 1
                    frontiers[rid].append(w)

        # produce list[set]
        parts = [set() for _ in range(k)]
        for n, rid in region_of.items():
            parts[rid].add(n)
        return parts

    raise ValueError(f"Unknown partitioning method '{method}'")

# ----------------------------------------------------------------------
# 2) CompressedGraph (Dragon edition)
# ----------------------------------------------------------------------

class CompressedGraph:
    """
    Stores a compressed multi-region view of a `gym_dragon.core.graph.Graph`.
    """

    # ───────────────────── constructor ──────────────────────────────────
    def __init__(
        self,
        env_graph,                                   # gym_dragon.core.graph.Graph
        method: str = "kl",
        k: int = 2,
        entry_policy: Union[str, Callable] = "random",
        seed: Optional[int] = None,
        labels: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ):
        self._rng = random.Random(seed)
        self.seed = seed
        self._labels = labels
        self.entry_policy = entry_policy
        self._env_graph = env_graph
        self._rng = random.Random(seed)

        # 2.1 build an undirected nx.Graph for partitioning ----------------
        nx_G = nx.Graph()
        for n in env_graph.nodes.keys():
            nx_G.add_node(n)
        for u, v in env_graph.edges:
            nx_G.add_edge(u, v)       # treat edges as bidirectional

        # 2.2 run chosen partition algorithm ------------------------------
        self.regions: List[Set[Hashable]] = partition_graph(nx_G, method, k, seed)
        if len(self.regions) > len(labels):
            raise ValueError("Not enough region labels for the partition result")

        # 2.3 helper look-up tables ---------------------------------------
        self.node2reg: Dict[Hashable, str] = {
            node: lbl for lbl, reg in zip(labels, self.regions) for node in reg
        }

        # 2.4 create undirected adjacency once (faster than env_graph.neighbors)
        self._adj: Dict[Hashable, Set[Hashable]] = defaultdict(set)
        for u, v in env_graph.edges:
            self._adj[u].add(v)
            self._adj[v].add(u)

        # 2.5 compress -----------------------------------------------------
        self.views, self.gateways = self._compress()

    # ───────────────────── public API ────────────────────────────────────
    # 1. concrete entry node
    def entry_point(self, node: Hashable, region_label: str) -> Hashable:
        return self.gateways[(node, region_label)]

    # 2. node → region
    def region_of(self, node: Hashable) -> str:
        return self.node2reg[node]

    # 3. textual view of one region
    def region_adjlist_str(self, region: str) -> str:
        """
        Return the compact adjacency-list view for *region*.

        • For every node in *region* show its original compressed neighbours
        (local room numbers + any region letters it can directly reach).
        • After the nodes, add one extra line per reachable *foreign* region
        listing only the gateway nodes that connect to it, e.g.

            B: 34 53 57

        No other region labels are injected and the current region label itself
        never appears.
        """
        view = self.views[region]                # {node: [nbrs…]}
        lines = []
        gateways = defaultdict(list)             # foreign_region -> [nodes…]

        # build node lines and collect gateway info
        for v in sorted(view):
            nbrs = []
            for w in view[v]:
                nbrs.append(str(w))
                if isinstance(w, str) and w != region:
                    gateways[w].append(v)        # remember v as a gateway to w
            lines.append(f"{v} : {' '.join(nbrs)}" if nbrs else f"{v} :")

        # add one line per foreign region, ordered alphabetically
        for foreign_lbl in sorted(gateways):
            nodes = " ".join(map(str, sorted(set(gateways[foreign_lbl]))))
            lines.append(f"{foreign_lbl} : {nodes}")

        return "  \n".join(lines)

    # 4. helpers
    def all_region_labels(self) -> List[str]:
        return list(self.views.keys())
    
    def nodes_per_region(self) -> Dict[str, List[Hashable]]:
        """
        Return a dict mapping region labels to lists of node IDs in each region.
        """
        return {lbl: list(reg) for lbl, reg in zip(self._labels, self.regions) if lbl in self.views}
    
    def nodes_per_region_str(self) -> str:
        """
        Return a string listing nodes per region, e.g.:
        Region A : nodeA1 nodeA2 ...
        """
        lines = []
        for lbl in self.all_region_labels():
            nodes = sorted(self._region_nodes(lbl))
            node_str = " ".join(map(str, nodes))
            lines.append(f"Nodes in region {lbl} : {node_str}")
        return "  \n".join(lines)

    def neighbors(self, x) -> List[Hashable]:
        """
        If x is a node ID (hashable) -> compressed neighbour list for that node.
        If x is a region label      -> members of that region.
        """
        if isinstance(x, str) and len(x) == 1 and x in self.views:
            return list(self._region_nodes(x))
        return self.views[self.region_of(x)][x]

    # ───────────────────── internal helpers ──────────────────────────────
    def _compress(self):
        views: Dict[str, Dict[Hashable, List[Hashable]]] = {}
        gateways: Dict[Tuple[Hashable, str], Hashable] = {}

        for lbl, reg in zip(self._labels, self.regions):
            view = {}
            for v in sorted(reg):
                local, foreign = [], defaultdict(list)

                # split neighbours into local / foreign
                for w in self._adj[v]:
                    if self.node2reg[w] == lbl:
                        local.append(w)
                    else:
                        foreign[self.node2reg[w]].append(w)

                # choose one border node per foreign region
                for tgt_lbl, border_nodes in foreign.items():
                    chosen = (
                        border_nodes[0]
                        if self.entry_policy == "first"
                        else self._rng.choice(border_nodes)
                        if self.entry_policy == "random"
                        else self.entry_policy(border_nodes, tgt_lbl)
                    )
                    local.append(tgt_lbl)
                    gateways[(v, tgt_lbl)] = chosen

                view[v] = local
            if view:
                views[lbl] = view

        return views, gateways

    def _region_nodes(self, region_label: str) -> Set[Hashable]:
        idx = self._labels.index(region_label)
        return self.regions[idx]

    # ───────────────────── dunder helpers ────────────────────────────────
    def __repr__(self):
        sizes = ", ".join(
            f"{lbl}:{len(r)}" for lbl, r in zip(self._labels, self.regions)
        )
        return f"<CompressedGraph [{sizes}]>"

    def to_multiregion_string(self) -> str:
        return "\n\n".join(
            f"--- Region {lbl} ---\n{self.region_adjlist_str(lbl)}"
            for lbl in self.views
        )
    # ───────────────────────────── drawing helper ──────────────────────────
    def draw(
        self,
        ax=None,
        *,
        with_labels=True,
        node_size=400,
        path: Optional[str] = None,
        margin: float = 0.05,
        **nx_kwargs,
    ):
        """
        Visualise the full map.

        • Nodes use their original centroids.
        • Region membership is colour-coded; legend sits *outside* the plot.
        • Whitespace around the graph is trimmed to `margin` (fraction of range).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Draw on this Axes (created if None).
        with_labels : bool, default=True
            Draw node-ID labels.
        node_size : int, default=400
            Size passed to `networkx.draw`.
        path : str, optional
            If given, save the figure to this file (PNG, PDF, etc.).
        margin : float, default=0.05
            Extra whitespace (as a fraction of coordinate span) around the tight
            bounding box of the nodes.
        **nx_kwargs :
            Extra keyword args forwarded to `networkx.draw`.
        """
        # 1) undirected NetworkX view ----------------------------------------
        G = nx.Graph()
        G.add_nodes_from(self.node2reg)
        for u, nbrs in self._adj.items():
            G.add_edges_from((u, v) for v in nbrs if u < v)

        # 2) positions -------------------------------------------------------
        pos = {n: self._env_graph.nodes[n].centroid for n in G.nodes}
        # Swap x and y coordinates for each node to accomodate (x, z) format
        pos = {n: (y, x) for n, (x, y) in ((n, self._env_graph.nodes[n].centroid) for n in G.nodes)}

        # 3) colour map ------------------------------------------------------
        cmap           = plt.colormaps.get_cmap("tab10")
        region_labels  = self.all_region_labels()
        colour_of      = {lbl: cmap(i % cmap.N) for i, lbl in enumerate(region_labels)}
        node_colours   = [colour_of[self.node2reg[n]] for n in G.nodes]

        # 4) create figure/axes ---------------------------------------------
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure

        # 5) draw graph ------------------------------------------------------
        nx.draw(
            G,
            pos=pos,
            ax=ax,
            node_color=node_colours,
            node_size=node_size,
            with_labels=with_labels,
            **nx_kwargs,
        )

        # 6) tight limits with a small margin --------------------------------
        xs, ys = zip(*pos.values())
        dx     = max(xs) - min(xs)
        dy     = max(ys) - min(ys)
        pad_x  = dx * margin
        pad_y  = dy * margin
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
        ax.set_aspect("equal")
        ax.set_axis_off()

        # 7) legend outside the plot area ------------------------------------
        patches = [mpatches.Patch(color=colour_of[lbl], label=lbl) for lbl in region_labels]
        ax.legend(
            handles=patches,
            title="Regions",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )

        # 8) save or return --------------------------------------------------
        fig.tight_layout(pad=0)  # remove white frame
        if path is not None:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.02)

        return ax


if __name__ == "__main__":
    from gym_dragon.core.graph import Graph as DragonGraph
    import os
    # toy data ------------------------------------------------------
    centroids = {
        23: (0, 0), 34: (1, 0), 38: (2, 0), 41: (3, 0),
        47: (0, 1), 53: (1, 1), 57: (2, 1), 73: (3, 1)
    }
    edges = [
        (23, 47), (23, 73),
        (34, 38), (34, 41), (34, 57),
        (38, 53),
        (41, 47),
        (47, 57), (47, 73),
        (53, 73)
    ]
    dummy_agents = []           # agent list not needed for the compression
    env_graph = DragonGraph(centroids, edges, agents=dummy_agents)

    CG = CompressedGraph(env_graph, method="kl", k=2, seed=1)

    print(CG)                               # <CompressedGraph [A:4, B:4]>
    for region in CG.all_region_labels():
        print(f"View from region {region}:\n{CG.region_adjlist_str(region)}\n")
    print("Gateway 34→B:", CG.entry_point(34, "B"))
    
    CG3 = CompressedGraph(env_graph, method="balanced_bfs", k=3, seed=2)
    print(CG3)                               # <CompressedGraph [A:3, B:3, C:2]>
    for region in CG3.all_region_labels():
        print(f"View from region {region}:\n{CG3.region_adjlist_str(region)}\n")
    print("Gateway 47→C:", CG3.entry_point(47, "C"))
    print("Gateway 34→C:", CG3.entry_point(34, "C"))
    
    CG.draw()
    os.makedirs("tmp", exist_ok=True)
    plt.savefig("tmp/compressed_graph1.png")
    CG3.draw()
    plt.savefig("tmp/compressed_graph2.png")

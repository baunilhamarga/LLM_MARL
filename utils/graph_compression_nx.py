from __future__ import annotations
import random
import networkx as nx
from collections import defaultdict
from itertools import count


class CompressedGraph:
    """
    A light wrapper that keeps a *compressed* multi-region view of a NetworkX
    graph together with the hidden “gateway” table that maps abstract
    inter-region moves back to concrete border nodes.

    Parameters
    ----------
    G : networkx.Graph
    method : {"kl","louvain","spectral","metis"}, optional
        Partitioning algorithm (default `"kl"` → Kernighan-Lin bisection).
    k : int, optional
        Desired number of regions (ignored by KL unless k==2).
    entry_policy : {"first","random",callable}, optional
        Strategy to pick one border node per foreign region.
        * `"first"`  - smallest-ID border node
        * `"random"` - uniform random (seed-reproducible)
        * callable   - `f(border_nodes:list[int], tgt_label:str)->int`
    seed : int or None, optional
        RNG seed for reproducibility.
    labels : str, optional
        String providing region labels in order; extend if k>26.

    Notes
    -----
    * Nodes must be hashable (integers in your maps).
    * Edge direction is ignored - we compress on the undirected view.
    """

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        G: nx.Graph,
        method: str = "kl",
        k: int = 2,
        entry_policy: str | callable = "random",
        seed: int | None = None,
        labels: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ):
        self._rng = random.Random(seed)
        self._labels = labels
        self.seed = seed

        # 1) partition -------------------------------------------------------
        self.regions: list[set] = self._partition(G, method, k)
        if len(self.regions) > len(labels):
            raise ValueError(
                f"Need {len(self.regions)} distinct labels but only "
                f"{len(labels)} provided."
            )

        self.node2reg: dict[int, str] = {
            n: lbl for lbl, reg in zip(labels, self.regions) for n in reg
        }

        # 2) compress --------------------------------------------------------
        self.views, self.gateways = self._compress(
            G, entry_policy=entry_policy
        )

    # ---------------------------------------------------------------- utils

    # outer API -------------------------------------------------------------
    # 1. entry point --------------------------------------------------------

    def entry_point(self, node: int, target_region: str) -> int:
        """Return chosen border node when leaving *node* → *target_region*."""
        try:
            return self.gateways[(node, target_region)]
        except KeyError:
            raise ValueError(
                f"No gateway recorded from node {node} towards region "
                f"{target_region!r}"
            )

    # 2. node → region ------------------------------------------------------

    def region_of(self, node: int) -> str:
        return self.node2reg[node]

    # 3. region adjacency view ---------------------------------------------

    def region_adjlist_str(self, region: str) -> str:
        """Return the compact adjacency list string for *region*."""
        view = self.views[region]
        lines = []
        for v in sorted(view):
            nbrs = " ".join(map(str, view[v]))
            lines.append(f"{v} : {nbrs}" if nbrs else f"{v} :")
        return "  \n".join(lines)

    # 4. small helpers ------------------------------------------------------

    def all_region_labels(self) -> list[str]:
        return list(self.views.keys())

    def neighbors(self, node_or_region) -> list[int | str]:
        """
        * If int  -> return compressed neighbours of that node.
        * If str  -> return list of nodes inside that region.
        """
        if isinstance(node_or_region, int):
            region = self.region_of(node_or_region)
            return self.views[region][node_or_region]
        else:  # assume region label
            return list(self._region_nodes(node_or_region))

    # ------------------------------------------------------------------

    # --------------------- internal implementation --------------------

    # partition -------------------------------------------------------------

    @staticmethod
    def _partition(G: nx.Graph, method: str, k: int):
        m = method.lower()
        if m in {"kl", "kernighan-lin"} and k == 2:
            a, b = nx.algorithms.community.kernighan_lin_bisection(G)
            return [set(a), set(b)]

        if m in {"louvain", "leiden"}:
            try:
                import community as community_louvain
            except ImportError:
                raise ImportError("pip install python-louvain")
            part_dict = community_louvain.best_partition(G)
            buckets = defaultdict(set)
            for n, cid in part_dict.items():
                buckets[cid].add(n)
            return list(buckets.values())

        if m == "spectral":
            from networkx.algorithms.community import spectral_modularity_maximization
            return list(spectral_modularity_maximization(G))

        if m == "metis":
            try:
                import nxmetis
            except ImportError:
                raise ImportError("pip install nxmetis")
            _, parts = nxmetis.partition(G, k)
            return [set(p) for p in parts]

        raise ValueError(f"Unknown partitioning method: {method}")

    # compress --------------------------------------------------------------

    def _compress(self, G: nx.Graph, entry_policy="random"):
        views = {}
        gateways = {}
        policy = entry_policy
        rng = self._rng

        for lbl, reg in zip(self._labels, self.regions):
            view = {}
            for v in sorted(reg):
                local, foreign = [], defaultdict(list)
                for w in G[v]:
                    if self.node2reg[w] == lbl:
                        local.append(w)
                    else:
                        foreign[self.node2reg[w]].append(w)

                for tgt_lbl, border in foreign.items():
                    chosen = (
                        border[0]  # first
                        if policy == "first"
                        else rng.choice(border)  # random
                        if policy == "random"
                        else policy(border, tgt_lbl)  # callable
                    )
                    gateways[(v, tgt_lbl)] = chosen
                    local.append(tgt_lbl)

                view[v] = local
            if view:  # ignore empty regions
                views[lbl] = view
        return views, gateways

    # internal helper -------------------------------------------------------

    def _region_nodes(self, region: str):
        idx = self._labels.index(region)
        return self.regions[idx]

    # ---------------------------------------------------------------- repr

    def __repr__(self) -> str:  # short summary
        sizes = ", ".join(
            f"{lbl}:{len(r)}" for lbl, r in zip(self._labels, self.regions)
        )
        return f"<CompressedGraph [{sizes}]>"

    # pretty print all regions ---------------------------------------------

    def to_multiregion_string(self) -> str:
        blocks = [
            f"--- Region {lbl} ---\n{self.region_adjlist_str(lbl)}"
            for lbl in self.views
        ]
        return "\n\n".join(blocks)


# --------------------------------------------------------------------------
# Example usage ------------------------------------------------------------
if __name__ == "__main__":
    edges = [
        (23, 47),
        (23, 73),
        (34, 38),
        (34, 41),
        (34, 57),
        (38, 53),
        (41, 47),
        (47, 57),
        (47, 73),
        (53, 73),
    ]
    G = nx.Graph(edges)

    CG = CompressedGraph(G, method="kl", k=2, seed=1)

    print(CG)                                   # <CompressedGraph [A:4, B:4]>
    print("\nView of A:")
    print(CG.region_adjlist_str("A"))
    print("\nView of B:")
    print(CG.region_adjlist_str("B"))
    node = 47
    print(
        f"\nFrom {node} moving to region B enters via",
        CG.entry_point(node, "B"),
    )

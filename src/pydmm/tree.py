"""
Trees over features for the Dirichlet-tree multinomial mixture.

A ``Tree`` has the S features as leaves (ids 0..S-1) and n_nodes internal
nodes. It is stored the way the C core expects it: internal node u has
children ``child[child_start[u]:child_start[u + 1]]``, where a child id below
S is a leaf and S + u is internal node u. Internal nodes are in post-order
(children before parents), so the root is the last node. Internal nodes with
a single child are removed when a tree is built, since they add nothing.
"""

from typing import Dict, Hashable, List, Optional, Sequence

import numpy as np
import pandas as pd


class Tree:
    """
    Rooted tree with the features as leaves.

    Build one with :meth:`star`, :meth:`from_parents`, :meth:`from_linkage`,
    :meth:`from_taxonomy` or :meth:`from_correlation`.

    Attributes:
        n_leaves (int): Number of features
        n_nodes (int): Number of internal nodes
        n_edges (int): Number of edges, i.e. alpha parameters per component
        child_start (np.ndarray): int32, shape (n_nodes + 1,)
        child (np.ndarray): int32, shape (n_edges,)
        leaf_names (list or None): Feature names, if known
        node_names (list): One label per internal node
    """

    def __init__(self, child_start: Sequence[int], child: Sequence[int],
                 n_leaves: int, leaf_names: Optional[Sequence] = None,
                 node_names: Optional[Sequence[str]] = None):
        self.child_start = np.ascontiguousarray(child_start, dtype=np.int32)
        self.child = np.ascontiguousarray(child, dtype=np.int32)
        self.n_leaves = int(n_leaves)
        self.n_nodes = len(self.child_start) - 1
        self.n_edges = len(self.child)
        self.leaf_names = list(leaf_names) if leaf_names is not None else None
        self.node_names = (list(node_names) if node_names is not None
                           else [f"node{u}" for u in range(self.n_nodes)])
        self._validate()

    # ------------------------------------------------------------ building

    @classmethod
    def _from_children(cls, n_leaves: int, children: Dict[Hashable, List],
                       root: Hashable, leaf_names=None, labels=None) -> "Tree":
        """
        Build from a mapping internal key -> child keys (leaves are ints
        0..n_leaves-1, other keys are internal nodes). Unary nodes are removed.
        """
        labels = labels or {}
        child_start, child, node_names = [0], [], []
        seen = set()

        # iterative post-order walk; returns the id that replaces each subtree
        result: Dict[Hashable, int] = {}
        stack = [(root, False)]
        while stack:
            key, expanded = stack.pop()
            is_leaf = isinstance(key, (int, np.integer)) and key not in children
            if is_leaf:
                if not 0 <= key < n_leaves:
                    raise ValueError(f"leaf id {key} out of range")
                if key in seen:
                    raise ValueError(f"leaf {key} appears more than once")
                seen.add(key)
                result[key] = int(key)
                continue
            if not expanded:
                if key in result:
                    raise ValueError(f"node {key!r} appears more than once")
                kids = children.get(key, [])
                if not kids:
                    raise ValueError(f"internal node {key!r} has no children")
                stack.append((key, True))
                stack.extend((c, False) for c in reversed(kids))
                continue
            ids = [result[c] for c in children[key]]
            if len(ids) == 1:
                result[key] = ids[0]
                continue
            child.extend(ids)
            child_start.append(len(child))
            node_names.append(str(labels.get(key, f"node{len(node_names)}")))
            result[key] = n_leaves + len(node_names) - 1
        if len(seen) != n_leaves:
            missing = sorted(set(range(n_leaves)) - seen)
            raise ValueError(f"features not in the tree: {missing[:10]}")
        if not node_names:
            raise ValueError("a tree needs at least one internal node with 2+ children")
        return cls(child_start, child, n_leaves, leaf_names, node_names)

    @classmethod
    def star(cls, n_leaves: int, leaf_names: Optional[Sequence] = None) -> "Tree":
        """Root with every feature as a child; the DTMM then equals the DMM."""
        return cls([0, n_leaves], np.arange(n_leaves), n_leaves, leaf_names, ["root"])

    @classmethod
    def from_parents(cls, parents: Sequence[int], n_leaves: int,
                     leaf_names: Optional[Sequence] = None) -> "Tree":
        """
        Build from a parent array.

        Parameters:
            parents: parent id of every node; nodes 0..n_leaves-1 are the
                features, the rest are internal nodes. The root has parent -1.
            n_leaves: number of features
        """
        parents = np.asarray(parents, dtype=int)
        roots = np.flatnonzero(parents < 0)
        if len(roots) != 1:
            raise ValueError("exactly one node must have parent -1")
        if roots[0] < n_leaves:
            raise ValueError("the root cannot be a feature")
        children: Dict[Hashable, List] = {}
        for node, parent in enumerate(parents):
            if parent >= 0:
                if not n_leaves <= parent < len(parents):
                    raise ValueError(f"parent {parent} of node {node} is not an internal node")
                children.setdefault(("n", int(parent)), []).append(
                    node if node < n_leaves else ("n", node))
        return cls._from_children(n_leaves, children, ("n", int(roots[0])), leaf_names)

    @classmethod
    def from_linkage(cls, Z: np.ndarray, leaf_names: Optional[Sequence] = None) -> "Tree":
        """Binary tree from a SciPy linkage matrix (``scipy.cluster.hierarchy.linkage``)."""
        Z = np.asarray(Z)
        n_leaves = Z.shape[0] + 1
        node = lambda i: int(i) if i < n_leaves else ("n", int(i))  # noqa: E731
        children = {("n", n_leaves + r): [node(Z[r, 0]), node(Z[r, 1])] for r in range(len(Z))}
        return cls._from_children(n_leaves, children, ("n", 2 * n_leaves - 2), leaf_names)

    @classmethod
    def from_taxonomy(cls, taxonomy: pd.DataFrame) -> "Tree":
        """
        Tree from a taxonomy table.

        Parameters:
            taxonomy: one row per feature (index = feature names), one column
                per rank from the top (e.g. Phylum) down (e.g. Genus). Missing
                or empty values end a feature's path at the last known rank.
                Taxa are identified by their full path, so equal names under
                different parents stay separate.
        """
        if not isinstance(taxonomy, pd.DataFrame):
            raise ValueError("taxonomy must be a pandas DataFrame (features x ranks)")
        leaf_names = taxonomy.index.tolist()
        root = ("root",)
        children: Dict[Hashable, List] = {root: []}
        labels = {root: "root"}
        for leaf, (_, row) in enumerate(taxonomy.iterrows()):
            parent = root
            for rank, value in zip(taxonomy.columns, row.values):
                if value is None or (isinstance(value, float) and np.isnan(value)) \
                        or str(value).strip() == "":
                    break
                key = parent + (str(value),)
                if key not in children:
                    children[key] = []
                    children[parent].append(key)
                    labels[key] = f"{rank}:{value}"
                parent = key
            children[parent].append(leaf)
        return cls._from_children(len(leaf_names), children, root, leaf_names, labels)

    @classmethod
    def from_correlation(cls, X: np.ndarray, method: str = "average",
                         leaf_names: Optional[Sequence] = None,
                         n_clades: Optional[int] = None) -> "Tree":
        """
        Data-driven tree: hierarchical clustering of the features on
        1 - correlation of their centred log-ratios (pseudocount 0.5).
        Features that rise and fall together end up in the same clade.

        Parameters:
            X: counts (n_samples, n_features)
            method: SciPy linkage method
            leaf_names: optional feature names
            n_clades: None (default) keeps the full binary dendrogram. An
                integer cuts the dendrogram into at most that many clades and
                returns a two-level tree, root -> clades -> features, like a
                taxonomy with one rank (features in a clade of their own hang
                from the root). Far fewer nodes than the binary tree; on real
                microbiome data this gave much more reproducible clusters.
        """
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] < 2:
            raise ValueError("a tree needs at least 2 features")
        L = np.log(X + 0.5)
        C = L - L.mean(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            R = np.nan_to_num(np.corrcoef(C, rowvar=False))
        R = np.atleast_2d(R)
        D = np.clip(1.0 - R, 0.0, 2.0)
        np.fill_diagonal(D, 0.0)
        D = 0.5 * (D + D.T)
        Z = linkage(squareform(D, checks=False), method=method)
        if n_clades is None:
            return cls.from_linkage(Z, leaf_names)
        if int(n_clades) < 1:
            raise ValueError("n_clades must be a positive integer")
        return cls.from_clusters(fcluster(Z, int(n_clades), criterion="maxclust"), leaf_names)

    @classmethod
    def from_clusters(cls, labels: Sequence, leaf_names: Optional[Sequence] = None) -> "Tree":
        """
        Two-level tree from a clustering of the features: root -> one node per
        cluster with at least two features -> its features. Features alone in
        their cluster are children of the root.
        """
        labels = np.asarray(labels)
        S = len(labels)
        if S < 2:
            raise ValueError("a tree needs at least 2 features")
        groups = [np.flatnonzero(labels == c) for c in pd.unique(labels)]
        clades = [g for g in groups if len(g) > 1]
        root = S + len(clades)
        parents = np.full(root + 1, root)
        parents[root] = -1
        for u, g in enumerate(clades):
            parents[g] = S + u
        return cls.from_parents(parents, S, leaf_names)

    # ------------------------------------------------------------ queries

    def _validate(self) -> None:
        S, n = self.n_leaves, self.n_nodes
        if n < 1 or self.child_start[0] != 0 or self.child_start[-1] != self.n_edges:
            raise ValueError("invalid child_start")
        if np.any(np.diff(self.child_start) < 2):
            raise ValueError("every internal node needs at least 2 children")
        seen = np.zeros(S + n, dtype=bool)
        for u in range(n):
            for c in self.child[self.child_start[u]:self.child_start[u + 1]]:
                if c < 0 or c >= S + u or seen[c]:
                    raise ValueError("children must be leaves or earlier nodes, each used once")
                seen[c] = True
        if not seen[:S + n - 1].all():
            raise ValueError("every leaf and non-root node must have a parent")
        if self.leaf_names is not None and len(self.leaf_names) != S:
            raise ValueError("leaf_names must have one name per feature")
        if len(self.node_names) != n:
            raise ValueError("node_names must have one name per internal node")

    def children(self, u: int) -> np.ndarray:
        """Child ids of internal node u (ids >= n_leaves are internal nodes)."""
        return self.child[self.child_start[u]:self.child_start[u + 1]]

    def leaves(self, u: int) -> List[int]:
        """Feature indices below internal node u."""
        out, stack = [], [self.n_leaves + u]
        while stack:
            c = stack.pop()
            if c < self.n_leaves:
                out.append(int(c))
            else:
                stack.extend(self.children(c - self.n_leaves))
        return sorted(out)

    def label(self, node_id: int) -> str:
        """Name of a leaf (id < n_leaves) or internal node (id >= n_leaves)."""
        if node_id < self.n_leaves:
            return str(self.leaf_names[node_id]) if self.leaf_names is not None else f"feature{node_id}"
        return self.node_names[node_id - self.n_leaves]

    def edge_index(self) -> pd.MultiIndex:
        """(parent, child) labels of every edge, in parameter order."""
        parents = np.repeat(np.arange(self.n_nodes), np.diff(self.child_start))
        return pd.MultiIndex.from_arrays(
            [[self.node_names[p] for p in parents], [self.label(c) for c in self.child]],
            names=["parent", "child"])

    def node_counts(self, X: np.ndarray) -> List[np.ndarray]:
        """Counts of each internal node's children, one (N, n_children) array per node."""
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_leaves:
            raise ValueError(f"X has {X.shape[1]} features, the tree {self.n_leaves} leaves")
        totals = np.empty((X.shape[0], self.n_nodes))
        out = []
        for u in range(self.n_nodes):
            ids = self.children(u)
            C = np.column_stack([X[:, c] if c < self.n_leaves else totals[:, c - self.n_leaves]
                                 for c in ids])
            totals[:, u] = C.sum(axis=1)
            out.append(C)
        return out

    def reorder(self, feature_names: Sequence) -> "Tree":
        """Same tree with the leaves renumbered to follow feature_names."""
        if self.leaf_names is None:
            raise ValueError("the tree has no leaf names to match against")
        feature_names = list(feature_names)
        if sorted(map(str, feature_names)) != sorted(map(str, self.leaf_names)) \
                or len(feature_names) != self.n_leaves:
            raise ValueError("the tree's leaf names do not match the data's features")
        position = {str(name): i for i, name in enumerate(feature_names)}
        new_id = np.array([position[str(name)] for name in self.leaf_names])
        child = np.where(self.child < self.n_leaves, new_id[np.minimum(self.child, self.n_leaves - 1)],
                         self.child)
        return Tree(self.child_start, child, self.n_leaves, feature_names, self.node_names)

    def __repr__(self) -> str:
        return f"Tree(n_leaves={self.n_leaves}, n_nodes={self.n_nodes}, n_edges={self.n_edges})"

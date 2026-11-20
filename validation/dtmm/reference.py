"""
Reference (pure NumPy/SciPy) Dirichlet-tree multinomial mixture, fitted by EM.

Used to evaluate the model before writing the C implementation, and later as
an oracle for its tests. Mirrors the conventions of pyDMM's C core:
Gamma(0.1, 0.1) prior on every alpha, optimisation over lambda = log(alpha),
negative-log-likelihood scale for the criteria.

Model
-----
A rooted tree has the S features as leaves. Every internal node v splits the
count arriving at v over its children c(v) by a Dirichlet-multinomial with its
own parameter vector alpha_kv (per component k):

    log p(x_i | k) = sum_v log DM(n_i,c(v) | alpha_kv)    (+ multinomial coef.)

A star tree (root with all leaves as children) is exactly the DMM. Deeper
trees let each clade have its own total (positive correlation within a clade)
and each node its own precision (node-specific dispersion).
"""

from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gammaln, logsumexp

GAMMA_ITA = 0.1
GAMMA_NU = 0.1


# ----------------------------------------------------------------------- trees

class Tree:
    """
    Rooted tree over S leaves (features 0..S-1).

    children maps each internal node id (>= S) to its child ids. Every node
    except the root has exactly one parent.
    """

    def __init__(self, n_leaves: int, children: Dict[int, List[int]]):
        self.n_leaves = n_leaves
        self.children = {int(v): [int(c) for c in cs] for v, cs in children.items()}
        parents = {}
        for v, cs in self.children.items():
            if len(cs) < 2:
                raise ValueError(f"internal node {v} has fewer than 2 children")
            for c in cs:
                if c in parents:
                    raise ValueError(f"node {c} has more than one parent")
                parents[c] = v
        roots = [v for v in self.children if v not in parents]
        if len(roots) != 1:
            raise ValueError("tree must have exactly one root")
        missing = set(range(n_leaves)) - set(parents)
        if missing and not (n_leaves == 1):
            raise ValueError(f"leaves without a parent: {sorted(missing)[:5]}")
        self.root = roots[0]
        # internal nodes in post-order (children before parents)
        order, stack = [], [(self.root, False)]
        while stack:
            v, done = stack.pop()
            if v < n_leaves:
                continue
            if done:
                order.append(v)
            else:
                stack.append((v, True))
                stack.extend((c, False) for c in self.children[v])
        self.internal = order

    @property
    def n_params(self) -> int:
        """Number of alpha parameters per component (= number of edges)."""
        return sum(len(cs) for cs in self.children.values())

    def node_counts(self, X: np.ndarray) -> List[np.ndarray]:
        """Per internal node (in self.internal order): counts of its children, (N, c_v)."""
        totals = {j: X[:, j].astype(np.float64) for j in range(self.n_leaves)}
        out = []
        for v in self.internal:
            C = np.column_stack([totals[c] for c in self.children[v]])
            totals[v] = C.sum(axis=1)
            out.append(C)
        return out


def star_tree(n_leaves: int) -> Tree:
    """Root with every feature as a child: the DMM."""
    return Tree(n_leaves, {n_leaves: list(range(n_leaves))})


def grouped_tree(n_leaves: int, groups: List[List[int]]) -> Tree:
    """Root -> one node per group (of size >= 2) -> leaves. Singletons hang off the root."""
    children, root_children, nxt = {}, [], n_leaves
    for g in groups:
        if len(g) == 1:
            root_children.append(g[0])
        else:
            children[nxt] = list(g)
            root_children.append(nxt)
            nxt += 1
    children[nxt] = root_children
    return Tree(n_leaves, children)


def linkage_tree(Z: np.ndarray, n_leaves: int) -> Tree:
    """Binary tree from a SciPy linkage matrix (node S + i merges Z[i, 0] and Z[i, 1])."""
    return Tree(n_leaves, {n_leaves + i: [int(Z[i, 0]), int(Z[i, 1])] for i in range(len(Z))})


def correlation_tree(X: np.ndarray, method: str = "average") -> Tree:
    """Data-driven binary tree: hierarchical clustering on 1 - corr(CLR(X + 0.5))."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    L = np.log(X + 0.5)
    C = L - L.mean(axis=1, keepdims=True)
    R = np.nan_to_num(np.corrcoef(C, rowvar=False))
    D = np.clip(1.0 - R, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    return linkage_tree(linkage(squareform(D, checks=False), method=method), X.shape[1])


# ------------------------------------------------------------- node DM pieces

def dm_loglik(C: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """log DM(C_i | alpha) without the multinomial coefficient, shape (N,)."""
    a0 = alpha.sum()
    return (gammaln(a0) - gammaln(a0 + C.sum(axis=1))
            + (gammaln(C + alpha) - gammaln(alpha)).sum(axis=1))


def _neg_objective(lam, C, z):
    """Weighted negative log evidence of one node + Gamma prior, and its gradient."""
    alpha = np.exp(lam)
    a0 = alpha.sum()
    n = C.sum(axis=1)
    W = z.sum()
    f = -(z @ (gammaln(C + alpha) - gammaln(alpha)).sum(axis=1)
          + W * gammaln(a0) - z @ gammaln(a0 + n))
    f += GAMMA_NU * a0 - GAMMA_ITA * lam.sum()
    g_alpha = -(z @ (digamma(C + alpha) - digamma(alpha))
                + W * digamma(a0) - z @ digamma(a0 + n))
    g = alpha * (g_alpha + GAMMA_NU) - GAMMA_ITA
    return f, g


def fit_node(C: np.ndarray, z: np.ndarray, lam0: np.ndarray) -> np.ndarray:
    res = minimize(_neg_objective, lam0, args=(C, z), jac=True, method="L-BFGS-B",
                   options={"gtol": 1e-6, "maxiter": 1000})
    return res.x


# ------------------------------------------------------------------- mixture

def _init_assignments(X, K, seed):
    from sklearn.cluster import KMeans
    P = X / np.maximum(X.sum(axis=1, keepdims=True), 1)
    labels = KMeans(n_clusters=K, n_init=5, random_state=seed).fit_predict(P)
    z = np.zeros((X.shape[0], K))
    z[np.arange(X.shape[0]), labels] = 1.0
    return z


def fit_dtmm(X: np.ndarray, tree: Tree, K: int, seed: int = 0,
             max_iter: int = 200, tol: float = 1e-6,
             z_init: Optional[np.ndarray] = None,
             shared: Optional[np.ndarray] = None) -> dict:
    """
    EM for a Dirichlet-tree multinomial mixture.

    shared: optional boolean per internal node (tree.internal order). A shared
    node has one alpha for all components; its likelihood is the same for
    every component, so it does not influence the assignments.

    Returns dict with z (N, K), weights (K,), lam (list per k of list per
    node), loglik (N, K) per-component log-likelihoods, NLE (negative log
    likelihood incl. prior and multinomial coefficient, as in pyDMM),
    BIC, AIC, n_iter.
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    nodes = tree.node_counts(X)
    z = _init_assignments(X, K, seed) if z_init is None else z_init.copy()
    shared = np.zeros(len(nodes), dtype=bool) if shared is None else np.asarray(shared, bool)
    lam = [[np.zeros(C.shape[1]) for C in nodes] for _ in range(K)]
    ones = np.ones(N)
    for v, C in enumerate(nodes):
        if shared[v]:
            lam_v = fit_node(C, ones, np.zeros(C.shape[1]))
            for k in range(K):
                lam[k][v] = lam_v
    log_multinom = gammaln(X.sum(axis=1) + 1) - gammaln(X + 1).sum(axis=1)

    def m_step(z):
        for k in range(K):
            for v, C in enumerate(nodes):
                if not shared[v]:
                    lam[k][v] = fit_node(C, z[:, k], lam[k][v])

    def component_loglik():
        ll = np.zeros((N, K))
        for k in range(K):
            for v, C in enumerate(nodes):
                ll[:, k] += dm_loglik(C, np.exp(lam[k][v]))
        return ll

    m_step(z)
    prev = np.inf
    for it in range(max_iter):
        w = np.maximum(z.mean(axis=0), 1e-300)
        ll = component_loglik()
        a = ll + np.log(w)
        z = np.exp(a - logsumexp(a, axis=1, keepdims=True))
        m_step(z)
        w = np.maximum(z.mean(axis=0), 1e-300)
        ll = component_loglik()
        loglik = float(np.sum(logsumexp(ll + np.log(w), axis=1)) + log_multinom.sum())
        all_lam = np.concatenate([lam[k][v] for k in range(K) for v in range(len(nodes))
                                  if k == 0 or not shared[v]])
        prior = (len(all_lam) * (GAMMA_ITA * np.log(GAMMA_NU) - gammaln(GAMMA_ITA))
                 - GAMMA_NU * np.exp(all_lam).sum() + GAMMA_ITA * all_lam.sum())
        nle = -(loglik + prior)
        if abs(prev - nle) < tol:
            break
        prev = nle
    a = ll + np.log(w)
    z = np.exp(a - logsumexp(a, axis=1, keepdims=True))
    sizes = np.array([C.shape[1] for C in nodes])
    p = int(K * sizes[~shared].sum() + sizes[shared].sum() + K - 1)
    return {"z": z, "weights": w, "lam": lam, "loglik": ll, "NLE": nle,
            "BIC": nle + 0.5 * np.log(N) * p, "AIC": nle + p, "n_params": p,
            "n_iter": it + 1, "shared": shared, "nodes": nodes}


def node_gain(fit: dict) -> np.ndarray:
    """
    Per internal node: expected complete-data log-likelihood gain of
    component-specific alphas over one shared alpha, given the assignments.
    """
    z, lam, nodes = fit["z"], fit["lam"], fit["nodes"]
    K, N = z.shape[1], z.shape[0]
    gains = np.zeros(len(nodes))
    for v, C in enumerate(nodes):
        lam_sh = fit_node(C, np.ones(N), lam[0][v])
        specific = sum(z[:, k] @ dm_loglik(C, np.exp(fit_node(C, z[:, k], lam[k][v]))) for k in range(K))
        gains[v] = specific - dm_loglik(C, np.exp(lam_sh)).sum()
    return gains


def fit_dtmm_select(X: np.ndarray, tree: Tree, K: int, seed: int = 0,
                    max_rounds: int = 5, **kw) -> dict:
    """
    DTMM with node selection: a node is component-specific only when its
    likelihood gain beats a BIC penalty of 0.5 * log(N) * (K - 1) * c_v.

    Starts with all nodes specific, then alternates EM fits and per-node
    decisions until the set of shared nodes stops changing.
    """
    N = X.shape[0]
    fit = fit_dtmm(X, tree, K, seed=seed, **kw)
    kw.pop("z_init", None)
    for _ in range(max_rounds):
        sizes = np.array([C.shape[1] for C in fit["nodes"]])
        penalty = 0.5 * np.log(N) * (K - 1) * sizes
        shared = node_gain(fit) < penalty
        if shared.all():  # keep the most informative node so the mixture can still separate
            shared[np.argmax(node_gain(fit) - penalty)] = False
        if np.array_equal(shared, fit["shared"]):
            break
        fit = fit_dtmm(X, tree, K, seed=seed, z_init=fit["z"], shared=shared, **kw)
    return fit

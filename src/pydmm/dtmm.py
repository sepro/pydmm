"""
Dirichlet-tree multinomial mixture (DTMM).

Each component is a Dirichlet-tree multinomial: every internal node of a tree
over the features splits its count over its children with a
Dirichlet-multinomial. With node selection (the default) a node is
component-specific only when that is worth its BIC penalty; otherwise one
parameter vector is shared by all components, and the node does not influence
the cluster assignments.

References:
    Dennis, S.Y. (1991). On the hyper-Dirichlet type 1 and hyper-Liouville
    distributions. Communications in Statistics - Theory and Methods 20(12).
    Wang, T. & Zhao, H. (2017). A Dirichlet-tree multinomial regression model
    for associating dietary nutrients with gut microorganisms. Biometrics 73(3).
    Mao, J. & Ma, L. (2022). Dirichlet-tree multinomial mixtures for clustering
    microbiome compositions. Annals of Applied Statistics 16(3).
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.special import gammaln

from .calibration import (resolve_temperature, tempered_responsibilities,
                          validate_temperature)
from .core import DirichletMixture, DirichletMixtureResult
from .tree import Tree

try:
    from . import pydmm_core
except ImportError:  # pragma: no cover
    raise ImportError(  # pragma: no cover
        "pydmm_core module not available. Please install the package with "
        "'pip install -e .' to compile the C extension."
    )


def tree_component_log_likelihood(X: np.ndarray, tree: Tree,
                                  alpha: np.ndarray) -> np.ndarray:
    """
    Dirichlet-tree multinomial log-likelihood of every sample under every
    component, without the multinomial coefficient.

    Parameters:
        X: counts, shape (N, S), columns in the tree's leaf order
        tree: the tree
        alpha: Dirichlet parameters per edge, shape (n_edges, K)

    Returns:
        np.ndarray of shape (N, K)
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    out = np.zeros((X.shape[0], alpha.shape[1]))
    for u, C in enumerate(tree.node_counts(X)):
        a = alpha[tree.child_start[u]:tree.child_start[u + 1], :]  # (c, K)
        a0 = a.sum(axis=0)
        n = C.sum(axis=1, keepdims=True)
        out += gammaln(a0)[None, :] - gammaln(a0[None, :] + n)
        out += (gammaln(C[:, :, None] + a[None, :, :]) - gammaln(a)[None, :, :]).sum(axis=1)
    return out


class DirichletTreeMixtureResult(DirichletMixtureResult):
    """
    Results from Dirichlet-tree mixture fitting.

    As :class:`DirichletMixtureResult`, except that ``parameter_estimates``
    holds one row per tree edge (shape (n_edges, n_components)), plus:

    Attributes:
        tree (Tree): The tree used, leaves in the data's feature order
        node_shared (np.ndarray): bool per internal node, True if all
            components share the node's parameters
        n_select_rounds (int): Node selection rounds run (0 when off)
    """

    def __init__(self, result_dict: Dict[str, Any], n_components: int,
                 n_samples: int, n_features: int, tree: Tree,
                 sample_names: Optional[list] = None,
                 feature_names: Optional[list] = None):
        super().__init__(result_dict, n_components, n_samples, n_features,
                         sample_names, feature_names)
        self.tree = tree
        self.node_shared = np.asarray(result_dict["NodeShared"]).astype(bool)
        self.n_select_rounds = int(result_dict["SelectRounds"])

    def get_parameter_estimates_df(self) -> Dict[str, pd.DataFrame]:
        """Parameter estimates as DataFrames indexed by (parent, child) edge."""
        index = self.tree.edge_index()
        columns = [f"Component_{i}" for i in range(self.n_components)]
        return {key: pd.DataFrame(self.parameter_estimates[key], index=index, columns=columns)
                for key in ("Lower", "Estimate", "Upper")}

    def differential_nodes(self) -> pd.DataFrame:
        """
        Internal nodes whose parameters differ between components, i.e. the
        clades whose composition separates the clusters.

        Returns:
            pd.DataFrame with columns node, n_features, features
        """
        rows = []
        for u in np.flatnonzero(~self.node_shared):
            leaves = self.tree.leaves(int(u))
            rows.append({"node": self.tree.node_names[u], "n_features": len(leaves),
                         "features": [self.tree.label(j) for j in leaves]})
        return pd.DataFrame(rows, columns=["node", "n_features", "features"])

    def summary(self) -> Dict[str, Any]:
        out = super().summary()
        out["n_nodes"] = self.tree.n_nodes
        out["n_differential_nodes"] = int((~self.node_shared).sum())
        return out


class DirichletTreeMixture(DirichletMixture):
    """
    Dirichlet-tree multinomial mixture for compositional count data.

    Like :class:`DirichletMixture`, but each component models the counts
    along a tree over the features: a taxonomy, or a tree built from the data.
    Taxa in one clade can then rise and fall together, and every split has its
    own dispersion. With a star tree (``Tree.star``) and ``node_selection=False``
    the model is the DMM.

    Parameters:
        n_components (int): Number of mixture components (default: 2)
        tree (Tree or None): Tree over the features (default: None, build one
            from the data with ``Tree.from_correlation``). If the tree has leaf
            names and X is a DataFrame, leaves are matched to columns by name.
        node_selection (bool): Share a node's parameters across components
            unless component-specific parameters improve the likelihood by
            more than a BIC penalty (default: True). Nodes that do not differ
            between clusters then do not add noise to the assignments.
        verbose (bool): Whether to print fitting progress (default: False)
        random_state (int): Random seed for reproducibility (default: 42)
        temperature (None, float, "auto" or "one-step"): Temper the assignment
            probabilities, as in ``DirichletMixture`` (default: None, off).
            On real data "auto" can make a DTMM almost uniform (BUGS.md,
            bug 5); "one-step" does not have this problem.

    Attributes:
        result_ (DirichletTreeMixtureResult): Fitting results (after fit)
        tree_ (Tree): The tree used, leaves in the order of the data's features
        classes_ (np.ndarray): Cluster labels (after fit)
        feature_names_in_ (np.ndarray): Feature names (after fit with a DataFrame)
        temperature_ (float or None): Tempering factor used (after fit)
    """

    def __init__(self, n_components: int = 2, tree: Optional[Tree] = None,
                 node_selection: bool = True, verbose: bool = False,
                 random_state: int = 42, temperature=None):
        self.n_components = n_components
        self.tree = tree
        self.node_selection = node_selection
        self.verbose = verbose
        self.random_state = random_state
        self.temperature = temperature
        self.is_fitted = False
        self.result_ = None

    def _resolve_tree(self, X_array: np.ndarray, feature_names: Optional[list]) -> Tree:
        n_features = X_array.shape[1]
        if self.tree is None:
            return Tree.from_correlation(X_array, leaf_names=feature_names)
        if not isinstance(self.tree, Tree):
            raise ValueError("tree must be a pydmm.Tree or None")
        if feature_names is not None and self.tree.leaf_names is not None:
            return self.tree.reorder(feature_names)
        if self.tree.n_leaves != n_features:
            raise ValueError(f"the tree has {self.tree.n_leaves} leaves but X has "
                             f"{n_features} features")
        return self.tree

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None) -> 'DirichletTreeMixture':
        """
        Fit the Dirichlet-tree mixture model to the data.

        Parameters:
            X (array-like): Count data of shape (n_samples, n_features)
            y (ignored): Present for scikit-learn compatibility

        Returns:
            self
        """
        X_array, sample_names, feature_names = self._validate_input(X)
        X_array = np.ascontiguousarray(X_array)
        validate_temperature(self.temperature)
        tree = self._resolve_tree(X_array, feature_names)

        result_dict = pydmm_core.dirichlet_tree_fit(
            X_array, tree.child_start, tree.child,
            n_components=self.n_components,
            verbose=int(self.verbose),
            seed=self.random_state,
            node_selection=bool(self.node_selection),
        )

        self.tree_ = tree
        self.result_ = DirichletTreeMixtureResult(
            result_dict,
            n_components=self.n_components,
            n_samples=X_array.shape[0],
            n_features=X_array.shape[1],
            tree=tree,
            sample_names=sample_names,
            feature_names=feature_names,
        )

        self.temperature_ = None
        if self.temperature is not None:
            self._apply_temperature(X_array)

        self.is_fitted = True
        self.classes_ = np.arange(self.n_components)
        if feature_names is not None:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        elif hasattr(self, 'feature_names_in_'):
            delattr(self, 'feature_names_in_')
        return self

    def _apply_temperature(self, X_array: np.ndarray) -> None:
        """Replace group_assignments by tempered ones (temperature is not None)."""
        result = self.result_
        log_lik = tree_component_log_likelihood(
            X_array, self.tree_, result.parameter_estimates["Estimate"])
        tau = resolve_temperature(self.temperature, log_lik, result.mixture_weights)
        result.group_assignments_untempered = result.group_assignments
        result.group_assignments = tempered_responsibilities(
            log_lik, result.mixture_weights, tau)
        result.temperature = tau
        self.temperature_ = tau

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Predict component probabilities for new data.

        Parameters:
            X (array-like): Count data of shape (n_samples, n_features). A
                DataFrame is matched to the training features by column name.

        Returns:
            np.ndarray or pd.DataFrame: Component probabilities for each sample.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        if isinstance(X, pd.DataFrame) and hasattr(self, "feature_names_in_"):
            missing = set(map(str, self.feature_names_in_)) - set(map(str, X.columns))
            if missing:
                raise ValueError(f"X is missing features seen during fit: {sorted(missing)[:5]}")
            X = X.loc[:, list(self.feature_names_in_)]
        X_array, _, _ = self._validate_input(X)
        if X_array.shape[1] != self.tree_.n_leaves:
            raise ValueError(f"X has {X_array.shape[1]} features, the model {self.tree_.n_leaves}")

        log_lik = tree_component_log_likelihood(
            X_array, self.tree_, self.result_.parameter_estimates["Estimate"])
        tau = self.temperature_ if getattr(self, "temperature_", None) is not None else 1.0
        responsibilities = tempered_responsibilities(log_lik, self.result_.mixture_weights, tau)

        if self.result_.component_labels is not None:
            columns = [f"{self.result_.component_labels[i]} component"
                       for i in range(self.n_components)]
            return pd.DataFrame(responsibilities, columns=columns)
        return responsibilities

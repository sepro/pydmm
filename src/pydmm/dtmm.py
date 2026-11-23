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

import warnings
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
            "auto" is discouraged for the DTMM and gives a warning: on
            heavy-tailed data its iteration can drift to a tau near zero,
            and the probabilities become almost uniform. Use "one-step".
        n_clades (int or None): Only with tree=None. None (default) builds the
            full binary correlation tree; an integer cuts it into at most that
            many clades, a two-level tree (``Tree.from_correlation(n_clades=)``).
            On real gut microbiome data (400 genera, K=4) the binary tree gave
            clusters that changed a lot between data subsets (78-85% of samples
            kept their cluster); about one clade per 2-3 features (150 for 400)
            was as reproducible as the DMM (91%), with a few clade nodes
            specific to the clusters. Too few clades (10-50) did not help.
        n_bootstrap (int): Refit on this many bootstrap resamples and average
            the assignment probabilities over the fit and the refits (default:
            0, off). The probabilities then include the uncertainty of the fit
            itself: in the same data they were much closer to how often a
            sample's cluster changes between fits than a single fit's, which
            are overconfident. A tree built from the data is rebuilt on every
            resample. Costs n_bootstrap extra fits; 8 is a reasonable choice.

    Attributes:
        result_ (DirichletTreeMixtureResult): Fitting results (after fit)
        tree_ (Tree): The tree used, leaves in the order of the data's features
        classes_ (np.ndarray): Cluster labels (after fit)
        feature_names_in_ (np.ndarray): Feature names (after fit with a DataFrame)
        temperature_ (float or None): Tempering factor used (after fit)
        bootstrap_ (list): One dict per bootstrap refit (estimates, weights,
            tree, tau, order: refit component matched to each component)
    """

    def __init__(self, n_components: int = 2, tree: Optional[Tree] = None,
                 node_selection: bool = True, verbose: bool = False,
                 random_state: int = 42, temperature=None,
                 n_clades: Optional[int] = None, n_bootstrap: int = 0):
        self.n_components = n_components
        self.tree = tree
        self.node_selection = node_selection
        self.n_clades = n_clades
        self.n_bootstrap = n_bootstrap
        self.verbose = verbose
        self.random_state = random_state
        self.temperature = temperature
        self.is_fitted = False
        self.result_ = None

    def _resolve_tree(self, X_array: np.ndarray, feature_names: Optional[list]) -> Tree:
        n_features = X_array.shape[1]
        if self.tree is None:
            return Tree.from_correlation(X_array, leaf_names=feature_names,
                                         n_clades=self.n_clades)
        if not isinstance(self.tree, Tree):
            raise ValueError("tree must be a pydmm.Tree or None")
        if self.n_clades is not None:
            raise ValueError("n_clades only applies when the tree is built from the data (tree=None)")
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
        if isinstance(self.temperature, str) and self.temperature == "auto":
            warnings.warn(
                'temperature="auto" is discouraged for DirichletTreeMixture: on '
                'heavy-tailed data the fixed-point iteration can drift to a '
                'temperature near zero, which makes the assignment probabilities '
                'almost uniform. Use temperature="one-step" instead.',
                UserWarning, stacklevel=2)
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

        self.bootstrap_ = []
        if int(self.n_bootstrap) > 0:
            self._fit_bootstrap(X_array)

        self.is_fitted = True
        self.classes_ = np.arange(self.n_components)
        if feature_names is not None:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        elif hasattr(self, 'feature_names_in_'):
            delattr(self, 'feature_names_in_')
        return self

    def _single_responsibilities(self, X_array, estimates, weights, tau, tree=None):
        log_lik = tree_component_log_likelihood(X_array, self.tree_ if tree is None else tree,
                                                estimates)
        return tempered_responsibilities(log_lik, weights, 1.0 if tau is None else tau)

    def _responsibilities(self, X_array: np.ndarray) -> np.ndarray:
        """Assignment probabilities of the fit, averaged over the bootstrap refits if any."""
        result = self.result_
        p = self._single_responsibilities(X_array, result.parameter_estimates["Estimate"],
                                          result.mixture_weights, getattr(self, "temperature_", None))
        for boot in getattr(self, "bootstrap_", []):
            q = self._single_responsibilities(X_array, boot["estimates"], boot["weights"],
                                              boot["tau"], boot["tree"])
            p = p + q[:, boot["order"]]
        return p / (1 + len(getattr(self, "bootstrap_", [])))

    def _fit_bootstrap(self, X_array: np.ndarray) -> None:
        """Refit on n_bootstrap resamples of the samples and average the assignment
        probabilities. A tree built from the data (tree=None) is rebuilt on every resample; a
        given tree is reused. Each refit's components are matched to the main fit's (Hungarian
        on the overlap of most likely components over the training samples). The averaged
        probabilities include the uncertainty from refitting, which a single fit's
        probabilities leave out."""
        from scipy.optimize import linear_sum_assignment

        K, N = self.n_components, X_array.shape[0]
        result = self.result_
        main = self._single_responsibilities(X_array, result.parameter_estimates["Estimate"],
                                             result.mixture_weights, self.temperature_)
        ref = main.argmax(axis=1)
        rng = np.random.default_rng(self.random_state)
        params = self.get_params()
        params.update(n_bootstrap=0, tree=None if self.tree is None else self.tree_)
        total = main.copy()
        for _ in range(int(self.n_bootstrap)):
            idx = rng.integers(0, N, N)
            refit = DirichletTreeMixture(**params).fit(X_array[idx])
            estimates = refit.result_.parameter_estimates["Estimate"]
            weights = refit.result_.mixture_weights
            q = self._single_responsibilities(X_array, estimates, weights, refit.temperature_,
                                              refit.tree_)
            overlap = np.zeros((K, K))
            np.add.at(overlap, (ref, q.argmax(axis=1)), 1)
            rows, cols = linear_sum_assignment(-overlap)
            order = np.empty(K, dtype=int)
            order[rows] = cols  # main component k <- refit component order[k]
            self.bootstrap_.append({"estimates": estimates, "weights": weights, "tree": refit.tree_,
                                    "tau": refit.temperature_, "order": order})
            total += q[:, order]
        result.group_assignments_single = result.group_assignments
        result.group_assignments = total / (1 + int(self.n_bootstrap))

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

        responsibilities = self._responsibilities(X_array)

        if self.result_.component_labels is not None:
            columns = [f"{self.result_.component_labels[i]} component"
                       for i in range(self.n_components)]
            return pd.DataFrame(responsibilities, columns=columns)
        return responsibilities

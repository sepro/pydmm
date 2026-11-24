"""
Core Python interface for Dirichlet Mixture Model fitting
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional
import warnings
from scipy.special import gammaln
from sklearn.base import BaseEstimator, ClassifierMixin

from .waic import compute_waic
from .calibration import (component_log_likelihood, resolve_temperature,
                          tempered_responsibilities, validate_temperature)

try:
    from . import pydmm_core
except ImportError:  # pragma: no cover
    raise ImportError(  # pragma: no cover
        "pydmm_core module not available. Please install the package with "
        "'pip install -e .' to compile the C extension."
    )


class DirichletMixtureResult:
    """
    Results from Dirichlet Mixture Model fitting.

    Attributes:
        goodness_of_fit (dict): Model fit statistics including NLE, BIC, AIC, etc.
        group_assignments (np.ndarray): Probability assignments to mixture components
        mixture_weights (np.ndarray): Weight of each mixture component
        parameter_estimates (dict): Parameter estimates with confidence intervals
        n_components (int): Number of mixture components used
        n_samples (int): Number of samples in the dataset
        n_features (int): Number of features in the dataset
        pointwise_log_likelihood (np.ndarray or None): Log-likelihood of each
            sample under each posterior draw, shape (n_draws, n_samples).
            Only set when the model was fitted with ``compute_waic=True``.
        temperature (float or None): Tempering factor applied to
            ``group_assignments``. None when the model was fitted without
            ``temperature`` (the default).
        group_assignments_untempered (np.ndarray or None): The usual EM
            responsibilities. Only set when tempering was applied, in which case
            ``group_assignments`` holds the tempered probabilities.
        group_assignments_single (np.ndarray or None): With ``n_bootstrap``,
            the single fit's assignments; ``group_assignments`` then holds the
            average over the fit and its bootstrap refits.
    """

    def __init__(self, result_dict: Dict[str, Any], n_components: int,
                 n_samples: int, n_features: int,
                 sample_names: Optional[list] = None,
                 feature_names: Optional[list] = None):
        self.goodness_of_fit = result_dict["GoodnessOfFit"]
        self.group_assignments = result_dict["Group"]
        self.mixture_weights = result_dict["Mixture"]["Weight"]
        self.parameter_estimates = result_dict["Fit"]
        self.n_components = n_components
        self.n_samples = n_samples
        self.n_features = n_features
        self.sample_names = sample_names
        self.feature_names = feature_names
        self.component_labels = None
        self.pointwise_log_likelihood = None
        self.temperature = None
        self.group_assignments_untempered = None
        self.group_assignments_single = None

    def set_component_labels(self, labels: Dict[int, str]) -> None:
        """
        Set human-readable labels for each component.

        Parameters:
            labels (dict): Dictionary mapping component indices (0 to n_components-1)
                          to string labels. All component indices must be present.

        Raises:
            ValueError: If labels are invalid (missing components, invalid indices,
                       non-string values)

        Example:
            >>> result.set_component_labels({0: 'Healthy', 1: 'Diseased', 2: 'Control'})
        """
        if not isinstance(labels, dict):
            raise ValueError("labels must be a dictionary")

        # Check that all keys are integers (before checking indices)
        if not all(isinstance(k, (int, np.integer)) for k in labels.keys()):
            raise ValueError("All keys in labels dictionary must be integers")

        # Check that all values are strings
        if not all(isinstance(v, str) for v in labels.values()):
            raise ValueError("All values in labels dictionary must be strings")

        # Check that all component indices are present
        expected_indices = set(range(self.n_components))
        provided_indices = set(labels.keys())

        if provided_indices != expected_indices:
            missing = expected_indices - provided_indices
            extra = provided_indices - expected_indices
            error_parts = []
            if missing:
                error_parts.append(f"missing components: {sorted(missing)}")
            if extra:
                error_parts.append(f"invalid component indices: {sorted(extra)}")
            raise ValueError(f"Label dictionary must contain all component indices 0 to {self.n_components-1}. " + ", ".join(error_parts))

        # Store the labels
        self.component_labels = labels.copy()

    def get_best_component(self) -> np.ndarray:
        """
        Get the most likely component for each sample.

        Returns:
            np.ndarray: If component labels are set, returns array of string labels.
                       Otherwise, returns array of integer component indices.
        """
        component_indices = np.argmax(self.group_assignments, axis=1)

        if self.component_labels is not None:
            # Map indices to labels
            return np.array([self.component_labels[idx] for idx in component_indices])
        else:
            return component_indices

    def get_group_assignments_df(self) -> pd.DataFrame:
        """
        Get group assignments as a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame with sample probabilities for each component.
                         Column names are "{label} component" if labels are set,
                         otherwise "Component_{number}".
        """
        index = self.sample_names if self.sample_names is not None else None

        if self.component_labels is not None:
            columns = [f"{self.component_labels[i]} component" for i in range(self.n_components)]
        else:
            columns = [f"Component_{i}" for i in range(self.n_components)]

        return pd.DataFrame(self.group_assignments, index=index, columns=columns)

    def get_parameter_estimates_df(self) -> Dict[str, pd.DataFrame]:
        """Get parameter estimates as pandas DataFrames."""
        index = self.feature_names if self.feature_names is not None else None
        columns = [f"Component_{i}" for i in range(self.n_components)]

        return {
            "Lower": pd.DataFrame(self.parameter_estimates["Lower"], index=index, columns=columns),
            "Estimate": pd.DataFrame(self.parameter_estimates["Estimate"], index=index, columns=columns),
            "Upper": pd.DataFrame(self.parameter_estimates["Upper"], index=index, columns=columns)
        }

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the fitting results."""
        out = {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_components": self.n_components,
            "mixture_weights": self.mixture_weights,
            "goodness_of_fit": self.goodness_of_fit,
            "BIC": self.goodness_of_fit["BIC"],
            "AIC": self.goodness_of_fit["AIC"]
        }
        if "WAIC" in self.goodness_of_fit:
            out["WAIC"] = self.goodness_of_fit["WAIC"]
        if self.temperature is not None:
            out["temperature"] = self.temperature
        return out


def validate_n_bootstrap(n_bootstrap) -> None:
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)) \
            or n_bootstrap < 0:
        raise ValueError("n_bootstrap must be a non-negative integer")


class DirichletMixture(ClassifierMixin, BaseEstimator):
    """
    Dirichlet Mixture Model for compositional data analysis.

    This class provides a scikit-learn style interface to fit Dirichlet
    mixture models to compositional count data such as microbiome data.

    Parameters:
        n_components (int): Number of mixture components (default: 2)
        verbose (bool): Whether to print fitting progress (default: False)
        random_state (int): Random seed for reproducibility (default: 42)
        compute_waic (bool): Also compute WAIC after fitting (default: False).
            Posterior draws come from a joint Laplace approximation around the
            EM solution (observed information of the mixture, Louis' identity).
            Adds 'WAIC', 'elpd_waic', 'p_waic', 'WAIC_se' and
            'WAIC_n_regularised' (number of flat or saddle directions whose
            posterior SD was capped) to ``result_.goodness_of_fit``. WAIC is on
            the same negative log-likelihood scale as BIC/AIC (lower is better).
        n_waic_draws (int): Number of posterior draws for WAIC (default: 1000)
        temperature (None, float, "auto" or "one-step"): Temper the cluster assignment
            probabilities to correct for over-confidence (default: None, off).
            The usual responsibilities treat every feature as independent
            evidence; correlated features and features that vary more than the
            Dirichlet allows make them too close to 0 or 1. With a value tau in
            (0, 1], assignments become z_ik ∝ (w_k * DM(x_i | alpha_k)) ** tau,
            which keeps each sample's most likely component unchanged.
            "auto" estimates tau from the fitted model without labels (Good's
            weight-of-evidence condition, see ``pydmm.calibration``), iterated
            to a fixed point. "one-step" uses only the first step of that
            iteration (the untempered assignments as weights); it cannot drift
            to a too-small tau on heavy-tailed data. Applies to
            ``result_.group_assignments`` and ``predict_proba``; the fitted
            parameters, NLE, BIC, AIC and WAIC are not changed, and the
            untempered assignments stay in
            ``result_.group_assignments_untempered``.
        restart_collapsed (bool): Restart once from hard k-means (k-means++
            seeding) when EM ends with all components equal (default: False).
            The soft k-means start can make all components almost equal when
            the samples have very similar compositions (many features, small
            differences between groups); EM then stays at this symmetric
            saddle point and every assignment probability is 1/K. The restart
            only runs for such collapsed fits, so other fits do not change.
            ``DirichletTreeMixture`` always does this restart.
        n_bootstrap (int): Refit on this many bootstrap resamples of the
            samples and average the assignment probabilities over the fit and
            the refits (default: 0, off). A single fit's probabilities ignore
            that a refit on other samples of the same population would move
            some samples to another cluster; the averaged ones include it.
            Each refit's components are matched to the fit's (Hungarian on the
            overlap of the most likely components). Applies to
            ``result_.group_assignments`` and ``predict_proba``; parameters,
            NLE, BIC, AIC and WAIC stay those of the single fit, whose
            probabilities are kept in ``result_.group_assignments_single``.
            Costs n_bootstrap extra fits, so fitting takes at least
            n_bootstrap + 1 times as long (up to about twice that for the DTMM). Gains are largest when single fits
            are over-confident (small N, misspecified model); nothing is
            gained on well-separated clusters. See the README and
            ``validation/bagging/compare.py``.

    Attributes:
        n_components (int): Number of mixture components
        verbose (bool): Verbosity flag
        random_state (int): Random seed
        is_fitted (bool): Whether the model has been fitted
        result_ (DirichletMixtureResult): Fitting results (available after fit)
        classes_ (np.ndarray): Cluster labels (available after fit)
        feature_names_in_ (np.ndarray): Names of features seen during fit (available after fit with DataFrame)
        temperature_ (float or None): Tempering factor used (available after fit;
            None when ``temperature`` is None)
        bootstrap_ (list): One dict per bootstrap refit (estimates, weights,
            tau, order: the refit component matched to each component; the
            DTMM adds its tree). Empty when ``n_bootstrap`` is 0.
    """

    def __init__(self, n_components: int = 2, verbose: bool = False,
                 random_state: int = 42, compute_waic: bool = False,
                 n_waic_draws: int = 1000, temperature=None,
                 restart_collapsed: bool = False, n_bootstrap: int = 0):
        self.n_components = n_components
        self.verbose = verbose
        self.random_state = random_state
        self.compute_waic = compute_waic
        self.n_waic_draws = n_waic_draws
        self.temperature = temperature
        self.restart_collapsed = restart_collapsed
        self.n_bootstrap = n_bootstrap
        self.is_fitted = False
        self.result_ = None

    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame]) -> tuple:
        """Validate and convert input data."""
        if isinstance(X, pd.DataFrame):
            sample_names = X.index.tolist() if X.index is not None else None
            feature_names = X.columns.tolist() if X.columns is not None else None
            X_array = X.values
        elif isinstance(X, np.ndarray):
            sample_names = None
            feature_names = None
            X_array = X
        else:
            raise ValueError("Input must be a pandas DataFrame or numpy array")

        # Ensure we have a 2D array
        if X_array.ndim != 2:
            raise ValueError("Input must be a 2-dimensional array")

        # Check for negative values
        if np.any(X_array < 0):
            raise ValueError("Input data cannot contain negative values")

        # Convert to int32 for C interface
        if X_array.dtype != np.int32:
            if not np.all(X_array == X_array.astype(int)):
                warnings.warn(
                    "Input data contains non-integer values. Converting to integers.",
                    UserWarning
                )
            X_array = X_array.astype(np.int32)

        return X_array, sample_names, feature_names

    def _compute_dirichlet_multinomial_log_likelihood(self, X: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        Compute log-likelihood of samples under Dirichlet-multinomial distribution.

        This matches the C code's neg_log_evidence_i function, computing the
        compound distribution that arises from drawing probabilities from a
        Dirichlet and then counts from a Multinomial.

        Parameters:
            X (np.ndarray): Count data of shape (n_samples, n_features)
            alpha (np.ndarray): Dirichlet parameters of shape (n_features,)

        Returns:
            np.ndarray: Log-likelihood for each sample of shape (n_samples,)
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # Pre-compute gamma values that don't depend on samples
        # log B(alpha) = sum(log Gamma(alpha_i)) - log Gamma(sum(alpha_i))
        log_beta_alpha = np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))

        # For each sample, compute the log likelihood
        log_likelihoods = np.zeros(n_samples)

        for i in range(n_samples):
            counts = X[i, :]

            # Compute alpha + counts for this sample
            alpha_plus_counts = alpha + counts

            # log B(alpha + counts) = sum(log Gamma(alpha_i + x_i)) - log Gamma(sum(alpha_i + x_i))
            log_beta_alpha_plus_counts = (
                np.sum(gammaln(alpha_plus_counts)) -
                gammaln(np.sum(alpha_plus_counts))
            )

            # The log likelihood is: log B(alpha + counts) - log B(alpha)
            # Note: We're computing the negative of what C code calls "neg_log_evidence"
            # C code: dLogE = -sum(lngamma(alpha+x)) + lngamma(sum(alpha+x)) + sum(lngamma(alpha)) - lngamma(sum(alpha))
            # Python: log_likelihood = sum(lngamma(alpha+x)) - lngamma(sum(alpha+x)) - sum(lngamma(alpha)) + lngamma(sum(alpha))
            log_likelihoods[i] = log_beta_alpha_plus_counts - log_beta_alpha

        return log_likelihoods

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None) -> 'DirichletMixture':
        """
        Fit the Dirichlet mixture model to the data.

        Parameters:
            X (array-like): Input count data of shape (n_samples, n_features)
                          Can be a pandas DataFrame or numpy array
            y (ignored): Not used (clustering is unsupervised), present for sklearn compatibility

        Returns:
            self : DirichletMixture
                Returns self for method chaining
        """
        X_array, sample_names, feature_names = self._validate_input(X)

        # Ensure array is contiguous for C interface
        X_array = np.ascontiguousarray(X_array)

        if self.compute_waic and (not isinstance(self.n_waic_draws, (int, np.integer))
                                  or self.n_waic_draws < 2):
            raise ValueError("n_waic_draws must be an integer >= 2")
        validate_temperature(self.temperature)
        validate_n_bootstrap(self.n_bootstrap)

        # Call the C extension
        result_dict = pydmm_core.dirichlet_fit(
            X_array,
            n_components=self.n_components,
            verbose=int(self.verbose),
            seed=self.random_state,
            return_hessian=bool(self.compute_waic),
            restart_collapsed=bool(self.restart_collapsed)
        )

        # Create result object
        self.result_ = DirichletMixtureResult(
            result_dict,
            n_components=self.n_components,
            n_samples=X_array.shape[0],
            n_features=X_array.shape[1],
            sample_names=sample_names,
            feature_names=feature_names
        )

        if self.compute_waic:
            waic = compute_waic(
                X_array,
                alpha_hat=self.result_.parameter_estimates["Estimate"],
                mixture_weights=self.result_.mixture_weights,
                hessian=result_dict["Hessian"],
                n_draws=int(self.n_waic_draws),
                random_state=self.random_state,
            )
            self.result_.goodness_of_fit["WAIC"] = waic["WAIC"]
            self.result_.goodness_of_fit["elpd_waic"] = waic["elpd_waic"]
            self.result_.goodness_of_fit["p_waic"] = waic["p_waic"]
            self.result_.goodness_of_fit["WAIC_se"] = waic["se"]
            self.result_.goodness_of_fit["WAIC_n_regularised"] = waic["n_regularised"]
            self.result_.pointwise_log_likelihood = waic["log_likelihood"]

        self.temperature_ = None
        if self.temperature is not None:
            self._apply_temperature(X_array)

        self._fit_bootstrap(X_array)

        self.is_fitted = True

        # Set classes_ attribute for sklearn ClassifierMixin compatibility
        self.classes_ = np.arange(self.n_components)

        # Set feature_names_in_ attribute for sklearn compatibility
        if feature_names is not None:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        else:
            # Remove feature_names_in_ if it exists (when refitting with array after DataFrame)
            if hasattr(self, 'feature_names_in_'):
                delattr(self, 'feature_names_in_')

        return self

    def _apply_temperature(self, X_array: np.ndarray) -> None:
        """Replace group_assignments by tempered ones (temperature is not None)."""
        result = self.result_
        alpha = result.parameter_estimates["Estimate"].T  # (K, S)
        log_lik = component_log_likelihood(X_array, alpha)
        tau = resolve_temperature(self.temperature, log_lik, result.mixture_weights)
        result.group_assignments_untempered = result.group_assignments
        result.group_assignments = tempered_responsibilities(
            log_lik, result.mixture_weights, tau)
        result.temperature = tau
        self.temperature_ = tau

    # ------------------------------------------------------------ bagging

    def _component_log_lik(self, X_array: np.ndarray, estimates: np.ndarray,
                           extra: Optional[dict] = None) -> np.ndarray:
        """Per-component log-likelihoods (N, K) for parameter estimates (S, K)."""
        return component_log_likelihood(X_array, np.asarray(estimates).T)

    def _bootstrap_params(self) -> Dict[str, Any]:
        """Constructor parameters for the bootstrap refits."""
        params = self.get_params()
        params.update(n_bootstrap=0)
        if "compute_waic" in params:
            params["compute_waic"] = False
        return params

    def _bootstrap_extra(self, refit) -> dict:
        """Model-specific state of a refit needed to evaluate it (the DTMM's tree)."""
        return {}

    def _single_responsibilities(self, X_array, estimates, weights, tau, extra=None):
        log_lik = self._component_log_lik(X_array, estimates, extra)
        return tempered_responsibilities(log_lik, weights, 1.0 if tau is None else tau)

    def _bagged_responsibilities(self, X_array: np.ndarray) -> np.ndarray:
        """Assignment probabilities averaged over the fit and its bootstrap refits."""
        result = self.result_
        p = self._single_responsibilities(X_array, result.parameter_estimates["Estimate"],
                                          result.mixture_weights, self.temperature_)
        for boot in self.bootstrap_:
            q = self._single_responsibilities(X_array, boot["estimates"], boot["weights"],
                                              boot["tau"], boot)
            p = p + q[:, boot["order"]]
        return p / (1 + len(self.bootstrap_))

    def _fit_bootstrap(self, X_array: np.ndarray) -> None:
        """Refit on n_bootstrap resamples and replace group_assignments by the average
        over the fit and the refits (n_bootstrap > 0)."""
        self.bootstrap_ = []
        B = int(self.n_bootstrap)
        if B == 0:
            return
        from scipy.optimize import linear_sum_assignment

        K, N = self.n_components, X_array.shape[0]
        result = self.result_
        ref = self._single_responsibilities(
            X_array, result.parameter_estimates["Estimate"], result.mixture_weights,
            self.temperature_).argmax(axis=1)
        rng = np.random.default_rng(self.random_state)
        params = self._bootstrap_params()
        for _ in range(B):
            idx = rng.integers(0, N, N)
            refit = type(self)(**params).fit(X_array[idx])
            boot = {"estimates": refit.result_.parameter_estimates["Estimate"],
                    "weights": refit.result_.mixture_weights,
                    "tau": refit.temperature_, **self._bootstrap_extra(refit)}
            q = self._single_responsibilities(X_array, boot["estimates"], boot["weights"],
                                              boot["tau"], boot)
            overlap = np.zeros((K, K))
            np.add.at(overlap, (ref, q.argmax(axis=1)), 1)
            rows, cols = linear_sum_assignment(-overlap)
            order = np.empty(K, dtype=int)
            order[rows] = cols  # component k <- refit component order[k]
            boot["order"] = order
            self.bootstrap_.append(boot)
        result.group_assignments_single = result.group_assignments
        result.group_assignments = self._bagged_responsibilities(X_array)

    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Fit the model and return the most likely component for each sample.

        Parameters:
            X (array-like): Input count data of shape (n_samples, n_features)

        Returns:
            np.ndarray: Component assignments for each sample
        """
        self.fit(X)
        return self.result_.get_best_component()

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Predict component probabilities for new data.

        Parameters:
            X (array-like): Input count data of shape (n_samples, n_features)

        Returns:
            np.ndarray or pd.DataFrame: Component probabilities for each sample.
                If component_labels is set, returns a DataFrame with labeled columns.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_array, _, _ = self._validate_input(X)
        X_array = np.ascontiguousarray(X_array)

        if getattr(self, "bootstrap_", None):
            responsibilities = self._bagged_responsibilities(X_array)
            if self.result_.component_labels is not None:
                columns = [f"{self.result_.component_labels[i]} component"
                           for i in range(self.n_components)]
                return pd.DataFrame(responsibilities, columns=columns)
            return responsibilities

        # Get fitted parameters
        alpha_estimates = self.result_.parameter_estimates["Estimate"]  # shape: (n_features, n_components)
        mixture_weights = self.result_.mixture_weights  # shape: (n_components,)

        n_samples = X_array.shape[0]
        n_components = self.n_components

        # Compute log-likelihood for each component
        log_responsibilities = np.zeros((n_samples, n_components))

        for k in range(n_components):
            alpha_k = alpha_estimates[:, k]  # Parameters for component k
            log_likelihood_k = self._compute_dirichlet_multinomial_log_likelihood(X_array, alpha_k)
            log_responsibilities[:, k] = log_likelihood_k + np.log(mixture_weights[k])
            if getattr(self, "temperature_", None) is not None:
                log_responsibilities[:, k] *= self.temperature_

        # Normalize to get probabilities (subtract max for numerical stability)
        max_log_resp = np.max(log_responsibilities, axis=1, keepdims=True)
        log_responsibilities -= max_log_resp
        responsibilities = np.exp(log_responsibilities)
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

        # Return DataFrame with labeled columns if component_labels is set
        if self.result_.component_labels is not None:
            columns = [f"{self.result_.component_labels[i]} component" for i in range(n_components)]
            return pd.DataFrame(responsibilities, columns=columns)

        return responsibilities

    def predict(self, X: Union[np.ndarray, pd.DataFrame], y=None) -> np.ndarray:
        """
        Predict the most likely component for each sample.

        Parameters:
            X (array-like): Input count data of shape (n_samples, n_features)
            y (ignored): Not used, present here for API consistency by convention

        Returns:
            np.ndarray: Predicted component labels for each sample.
                If component_labels is set, returns an array of string labels.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        probabilities = self.predict_proba(X)

        # Handle DataFrame return from predict_proba when labels are set
        if isinstance(probabilities, pd.DataFrame):
            probabilities = probabilities.values

        component_indices = np.argmax(probabilities, axis=1)

        # Map to labels if component_labels is set
        if self.result_.component_labels is not None:
            return np.array([self.result_.component_labels[idx] for idx in component_indices])

        return component_indices

    def score(self, X: Union[np.ndarray, pd.DataFrame]) -> float:
        """
        Return the log-likelihood of the data under the model.

        Parameters:
            X (array-like): Input count data

        Returns:
            float: Negative log-likelihood (lower is better)
        """
        if not self.is_fitted:
            warnings.warn(
                "Model was not fitted before calling score(). "
                "Fitting the model now on the provided data.",
                UserWarning
            )
            self.fit(X)

        return -self.result_.goodness_of_fit["NLE"]  # Return negative NLE

    @property
    def result(self):
        return self.result_
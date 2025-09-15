"""
Core Python interface for Dirichlet Mixture Model fitting
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional
import warnings

try:
    from . import pydmm_core
except ImportError:
    raise ImportError(
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

    def get_best_component(self) -> np.ndarray:
        """Get the most likely component for each sample."""
        return np.argmax(self.group_assignments, axis=1)

    def get_group_assignments_df(self) -> pd.DataFrame:
        """Get group assignments as a pandas DataFrame."""
        index = self.sample_names if self.sample_names is not None else None
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
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_components": self.n_components,
            "mixture_weights": self.mixture_weights,
            "goodness_of_fit": self.goodness_of_fit,
            "best_model_by_BIC": self.goodness_of_fit["BIC"],
            "best_model_by_AIC": self.goodness_of_fit["AIC"]
        }


class DirichletMixture:
    """
    Dirichlet Mixture Model for compositional data analysis.

    This class provides a scikit-learn style interface to fit Dirichlet
    mixture models to compositional count data such as microbiome data.

    Parameters:
        n_components (int): Number of mixture components (default: 2)
        verbose (bool): Whether to print fitting progress (default: False)
        random_state (int): Random seed for reproducibility (default: 42)

    Attributes:
        n_components (int): Number of mixture components
        verbose (bool): Verbosity flag
        random_state (int): Random seed
        is_fitted (bool): Whether the model has been fitted
        result_ (DirichletMixtureResult): Fitting results (available after fit)
    """

    def __init__(self, n_components: int = 2, verbose: bool = False,
                 random_state: int = 42):
        self.n_components = n_components
        self.verbose = verbose
        self.random_state = random_state
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

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'DirichletMixture':
        """
        Fit the Dirichlet mixture model to the data.

        Parameters:
            X (array-like): Input count data of shape (n_samples, n_features)
                          Can be a pandas DataFrame or numpy array

        Returns:
            self : DirichletMixture
                Returns self for method chaining
        """
        X_array, sample_names, feature_names = self._validate_input(X)

        # Ensure array is contiguous for C interface
        X_array = np.ascontiguousarray(X_array)

        # Call the C extension
        result_dict = pydmm_core.dirichlet_fit(
            X_array,
            n_components=self.n_components,
            verbose=int(self.verbose),
            seed=self.random_state
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

        self.is_fitted = True
        return self

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

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict component probabilities for new data.

        Note: This method currently requires refitting the model as the C
        implementation does not separate training from prediction.

        Parameters:
            X (array-like): Input count data of shape (n_samples, n_features)

        Returns:
            np.ndarray: Component probabilities for each sample
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # For now, we need to refit - this could be optimized in the future
        self.fit(X)
        return self.result_.group_assignments

    def score(self, X: Union[np.ndarray, pd.DataFrame]) -> float:
        """
        Return the log-likelihood of the data under the model.

        Parameters:
            X (array-like): Input count data

        Returns:
            float: Negative log-likelihood (lower is better)
        """
        if not self.is_fitted:
            self.fit(X)

        return -self.result_.goodness_of_fit["NLE"]  # Return negative NLE
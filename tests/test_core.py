"""
Tests for the core DirichletMixture functionality
"""

import pytest
import numpy as np
import pandas as pd
from pydmm import DirichletMixture


class TestDirichletMixture:
    """Test the main DirichletMixture class"""

    def setup_method(self):
        """Set up test data before each test"""
        # Create simple test data with two distinct groups
        np.random.seed(42)

        # Group 1: High in features 0,1, low in features 2,3
        group1_props = np.array([0.6, 0.3, 0.05, 0.05])
        # Group 2: Low in features 0,1, high in features 2,3
        group2_props = np.array([0.05, 0.05, 0.6, 0.3])

        n_samples_per_group = 50
        n_features = 4

        samples = []
        true_labels = []

        # Generate Group 1 samples
        for i in range(n_samples_per_group):
            total_count = np.random.poisson(1000) + 100
            counts = np.random.multinomial(total_count, group1_props)
            samples.append(counts)
            true_labels.append(0)

        # Generate Group 2 samples
        for i in range(n_samples_per_group):
            total_count = np.random.poisson(1000) + 100
            counts = np.random.multinomial(total_count, group2_props)
            samples.append(counts)
            true_labels.append(1)

        self.data_array = np.array(samples, dtype=np.int32)
        self.data_df = pd.DataFrame(
            self.data_array,
            index=[f"Sample_{i:03d}" for i in range(len(samples))],
            columns=[f"Feature_{i}" for i in range(n_features)]
        )
        self.true_labels = np.array(true_labels)
        self.n_samples = len(samples)
        self.n_features = n_features

    def test_initialization(self):
        """Test DirichletMixture initialization"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)

        assert dmm.n_components == 2
        assert dmm.verbose == False
        assert dmm.random_state == 42
        assert dmm.result_ is None

    def test_initialization_invalid_params(self):
        """Test initialization with invalid parameters"""
        # The current implementation doesn't validate n_components in __init__
        # It would fail during fitting. Let's test actual invalid usage.
        dmm = DirichletMixture(n_components=0)

        # Should fail when trying to fit with 0 components
        with pytest.raises((ValueError, RuntimeError)):
            dmm.fit(self.data_array)

    def test_fit_with_numpy_array(self):
        """Test fitting with numpy array input"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(self.data_array)

        assert dmm.result_ is not None
        assert hasattr(dmm.result_, 'goodness_of_fit')
        assert hasattr(dmm.result_, 'group_assignments')
        assert hasattr(dmm.result_, 'mixture_weights')

        # Check dimensions
        assert dmm.result_.group_assignments.shape == (self.n_samples, 2)
        assert len(dmm.result_.mixture_weights) == 2

    def test_fit_with_pandas_dataframe(self):
        """Test fitting with pandas DataFrame input"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(self.data_df)

        assert dmm.result_ is not None
        assert dmm.result_.sample_names is not None
        assert dmm.result_.feature_names is not None
        assert len(dmm.result_.sample_names) == self.n_samples
        assert len(dmm.result_.feature_names) == self.n_features

    def test_fit_invalid_input(self):
        """Test fitting with invalid input data"""
        dmm = DirichletMixture(n_components=2)

        # Test with wrong dimensions
        with pytest.raises(ValueError):
            dmm.fit(np.array([1, 2, 3]))  # 1D array

        # Test with wrong data type - should give warning and convert
        with pytest.warns(UserWarning, match="non-integer"):
            dmm.fit(np.array([[1.5, 2.7], [3.1, 4.8]]))  # float array

        # Test with negative values
        with pytest.raises(ValueError):
            dmm.fit(np.array([[-1, 2], [3, 4]], dtype=np.int32))

    def test_predict_proba(self):
        """Test predict_proba method"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(self.data_array)

        proba = dmm.predict_proba(self.data_array)

        assert proba.shape == (self.n_samples, 2)
        # Probabilities should sum to 1 for each sample
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-10)
        # All probabilities should be non-negative
        assert np.all(proba >= 0)

    def test_fit_predict(self):
        """Test fit_predict method"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)

        labels = dmm.fit_predict(self.data_array)

        assert len(labels) == self.n_samples
        assert set(labels) <= {0, 1}  # Labels should be 0 or 1

    def test_goodness_of_fit_metrics(self):
        """Test that goodness of fit metrics are computed"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(self.data_array)

        gof = dmm.result_.goodness_of_fit

        required_metrics = ['NLE', 'LogDet', 'Laplace', 'BIC', 'AIC']
        for metric in required_metrics:
            assert metric in gof
            assert isinstance(gof[metric], (int, float))
            assert not np.isnan(gof[metric])

    def test_clustering_performance(self):
        """Test that the algorithm achieves reasonable clustering performance"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(self.data_array)

        predicted_labels = dmm.result_.get_best_component()

        # Calculate clustering accuracy (with optimal assignment)
        accuracy_original = np.mean(predicted_labels == self.true_labels)
        accuracy_flipped = np.mean(predicted_labels == (1 - self.true_labels))
        best_accuracy = max(accuracy_original, accuracy_flipped)

        # Should achieve at least 80% accuracy on this well-separated data
        assert best_accuracy >= 0.8

    def test_different_number_of_components(self):
        """Test fitting with different numbers of components"""
        for n_components in [1, 2, 3]:
            dmm = DirichletMixture(n_components=n_components, verbose=False, random_state=42)
            dmm.fit(self.data_array)

            assert dmm.result_ is not None
            assert len(dmm.result_.mixture_weights) == n_components
            assert dmm.result_.group_assignments.shape == (self.n_samples, n_components)

    def test_result_dataframe_methods(self):
        """Test DirichletMixtureResult DataFrame accessor methods"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(self.data_df)  # Use DataFrame to preserve names

        # Test group assignments DataFrame
        group_df = dmm.result_.get_group_assignments_df()
        assert isinstance(group_df, pd.DataFrame)
        assert group_df.shape == (self.n_samples, 2)
        assert list(group_df.index) == list(self.data_df.index)

        # Test parameter estimates DataFrame
        param_dict = dmm.result_.get_parameter_estimates_df()
        assert isinstance(param_dict, dict)
        assert "Estimate" in param_dict
        param_df = param_dict["Estimate"]
        assert isinstance(param_df, pd.DataFrame)
        assert param_df.shape == (self.n_features, 2)
        assert list(param_df.index) == list(self.data_df.columns)

    def test_summary_method(self):
        """Test the summary method"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(self.data_array)

        summary = dmm.result_.summary()

        required_keys = ['n_samples', 'n_features', 'n_components', 'mixture_weights', 'goodness_of_fit']
        for key in required_keys:
            assert key in summary

    def test_reproducibility(self):
        """Test that results are reproducible with same random seed"""
        dmm1 = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm1.fit(self.data_array)

        dmm2 = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm2.fit(self.data_array)

        # Results should be identical
        np.testing.assert_allclose(
            dmm1.result_.group_assignments,
            dmm2.result_.group_assignments,
            rtol=1e-10
        )
        np.testing.assert_allclose(
            dmm1.result_.mixture_weights,
            dmm2.result_.mixture_weights,
            rtol=1e-10
        )


class TestDirichletMixtureResult:
    """Test the DirichletMixtureResult class independently"""

    def setup_method(self):
        """Set up a fitted model for testing result methods"""
        np.random.seed(42)
        data = np.random.multinomial(1000, [0.25, 0.25, 0.25, 0.25], size=20).astype(np.int32)

        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(data)
        self.result = dmm.result_

    def test_get_best_component(self):
        """Test get_best_component method"""
        best_components = self.result.get_best_component()

        assert len(best_components) == self.result.n_samples
        assert all(comp in [0, 1] for comp in best_components)

    def test_attributes(self):
        """Test that all expected attributes are present"""
        expected_attrs = [
            'goodness_of_fit', 'group_assignments', 'mixture_weights',
            'parameter_estimates', 'n_components', 'n_samples', 'n_features'
        ]

        for attr in expected_attrs:
            assert hasattr(self.result, attr)
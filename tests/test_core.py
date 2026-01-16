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
        assert dmm.result is None

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

        assert dmm.result is not None
        assert hasattr(dmm.result, 'goodness_of_fit')
        assert hasattr(dmm.result, 'group_assignments')
        assert hasattr(dmm.result, 'mixture_weights')

        # Check dimensions
        assert dmm.result.group_assignments.shape == (self.n_samples, 2)
        assert len(dmm.result.mixture_weights) == 2

    def test_fit_with_pandas_dataframe(self):
        """Test fitting with pandas DataFrame input"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(self.data_df)

        assert dmm.result is not None
        assert dmm.result.sample_names is not None
        assert dmm.result.feature_names is not None
        assert len(dmm.result.sample_names) == self.n_samples
        assert len(dmm.result.feature_names) == self.n_features

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

    def test_predict(self):
        """Test predict method"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(self.data_array)

        # Test basic functionality
        labels = dmm.predict(self.data_array)
        assert len(labels) == self.n_samples
        assert set(labels) <= {0, 1}  # Labels should be 0 or 1

        # Test with new data (subset of original)
        new_data = self.data_array[:10]
        new_labels = dmm.predict(new_data)
        assert len(new_labels) == 10
        assert set(new_labels) <= {0, 1}

        # Test with pandas DataFrame
        new_labels_df = dmm.predict(self.data_df[:10])
        np.testing.assert_array_equal(new_labels, new_labels_df)

    def test_predict_unfitted_model(self):
        """Test predict method raises error for unfitted model"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)

        with pytest.raises(ValueError, match="Model must be fitted"):
            dmm.predict(self.data_array)

    def test_predict_proba_unfitted_model(self):
        """Test predict_proba method raises error for unfitted model"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)

        with pytest.raises(ValueError, match="Model must be fitted"):
            dmm.predict_proba(self.data_array)

    def test_predict_consistency_with_predict_proba(self):
        """Test that predict results are consistent with predict_proba"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(self.data_array)

        # Test on training data
        proba = dmm.predict_proba(self.data_array)
        labels = dmm.predict(self.data_array)
        expected_labels = np.argmax(proba, axis=1)
        np.testing.assert_array_equal(labels, expected_labels)

        # Test on new data
        new_data = self.data_array[:20]
        proba_new = dmm.predict_proba(new_data)
        labels_new = dmm.predict(new_data)
        expected_labels_new = np.argmax(proba_new, axis=1)
        np.testing.assert_array_equal(labels_new, expected_labels_new)

    def test_predict_with_optional_y_parameter(self):
        """Test predict method accepts optional y parameter for API consistency"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(self.data_array)

        # Should work with y=None (default)
        labels1 = dmm.predict(self.data_array)

        # Should work with y parameter provided (ignored)
        dummy_y = np.zeros(self.n_samples)
        labels2 = dmm.predict(self.data_array, y=dummy_y)

        # Results should be identical
        np.testing.assert_array_equal(labels1, labels2)

    def test_predict_proba_improved_efficiency(self):
        """Test that predict_proba doesn't refit the model"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(self.data_array)

        # Store original result
        original_result = dmm.result_

        # Call predict_proba - should not change the fitted result
        proba = dmm.predict_proba(self.data_array[:10])

        # Result should be the same object (no refitting)
        assert dmm.result_ is original_result

        # Probabilities should still be valid
        assert proba.shape == (10, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-10)
        assert np.all(proba >= 0)

    def test_goodness_of_fit_metrics(self):
        """Test that goodness of fit metrics are computed"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(self.data_array)

        gof = dmm.result.goodness_of_fit

        required_metrics = ['NLE', 'LogDet', 'Laplace', 'BIC', 'AIC']
        for metric in required_metrics:
            assert metric in gof
            assert isinstance(gof[metric], (int, float))
            assert not np.isnan(gof[metric])

    def test_clustering_performance(self):
        """Test that the algorithm achieves reasonable clustering performance"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(self.data_array)

        predicted_labels = dmm.result.get_best_component()

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

            assert dmm.result is not None
            assert len(dmm.result.mixture_weights) == n_components
            assert dmm.result.group_assignments.shape == (self.n_samples, n_components)

    def test_result_dataframe_methods(self):
        """Test DirichletMixtureResult DataFrame accessor methods"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(self.data_df)  # Use DataFrame to preserve names

        # Test group assignments DataFrame
        group_df = dmm.result.get_group_assignments_df()
        assert isinstance(group_df, pd.DataFrame)
        assert group_df.shape == (self.n_samples, 2)
        assert list(group_df.index) == list(self.data_df.index)

        # Test parameter estimates DataFrame
        param_dict = dmm.result.get_parameter_estimates_df()
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

        summary = dmm.result.summary()

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
            dmm1.result.group_assignments,
            dmm2.result.group_assignments,
            rtol=1e-10
        )
        np.testing.assert_allclose(
            dmm1.result.mixture_weights,
            dmm2.result.mixture_weights,
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
        self.result = dmm.result

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


class TestComponentLabeling:
    """Test component labeling functionality"""

    def setup_method(self):
        """Set up a fitted model for testing labeling"""
        np.random.seed(42)

        # Create test data with 3 distinct groups
        group1_props = np.array([0.7, 0.2, 0.1])
        group2_props = np.array([0.2, 0.7, 0.1])
        group3_props = np.array([0.1, 0.2, 0.7])

        n_samples_per_group = 20
        samples = []

        for props in [group1_props, group2_props, group3_props]:
            for _ in range(n_samples_per_group):
                total_count = np.random.poisson(1000) + 100
                counts = np.random.multinomial(total_count, props)
                samples.append(counts)

        self.data = np.array(samples, dtype=np.int32)
        self.data_df = pd.DataFrame(
            self.data,
            index=[f"Sample_{i:03d}" for i in range(len(samples))],
            columns=[f"Feature_{i}" for i in range(3)]
        )

        # Fit model with 3 components
        self.dmm = DirichletMixture(n_components=3, verbose=False, random_state=42)
        self.dmm.fit(self.data)
        self.result = self.dmm.result

    def test_initial_state_no_labels(self):
        """Test that component_labels is None initially"""
        assert self.result.component_labels is None

    def test_set_valid_labels(self):
        """Test setting valid labels"""
        labels = {0: 'Healthy', 1: 'Diseased', 2: 'Control'}
        self.result.set_component_labels(labels)

        assert self.result.component_labels is not None
        assert self.result.component_labels == labels
        # Ensure it's a copy, not the same object
        assert self.result.component_labels is not labels

    def test_set_labels_invalid_type(self):
        """Test that non-dict input raises ValueError"""
        with pytest.raises(ValueError, match="labels must be a dictionary"):
            self.result.set_component_labels(['Healthy', 'Diseased', 'Control'])

    def test_set_labels_missing_components(self):
        """Test that missing component indices raise ValueError"""
        labels = {0: 'Healthy', 1: 'Diseased'}  # Missing component 2

        with pytest.raises(ValueError, match="missing components: \\[2\\]"):
            self.result.set_component_labels(labels)

    def test_set_labels_extra_components(self):
        """Test that extra component indices raise ValueError"""
        labels = {0: 'Healthy', 1: 'Diseased', 2: 'Control', 3: 'Extra'}

        with pytest.raises(ValueError, match="invalid component indices: \\[3\\]"):
            self.result.set_component_labels(labels)

    def test_set_labels_missing_and_extra(self):
        """Test error message when both missing and extra components"""
        labels = {0: 'Healthy', 3: 'Extra'}

        with pytest.raises(ValueError, match="missing components.*invalid component indices"):
            self.result.set_component_labels(labels)

    def test_set_labels_non_integer_keys(self):
        """Test that non-integer keys raise ValueError"""
        labels = {'0': 'Healthy', '1': 'Diseased', '2': 'Control'}

        with pytest.raises(ValueError, match="All keys.*must be integers"):
            self.result.set_component_labels(labels)

    def test_set_labels_non_string_values(self):
        """Test that non-string values raise ValueError"""
        labels = {0: 'Healthy', 1: 123, 2: 'Control'}

        with pytest.raises(ValueError, match="All values.*must be strings"):
            self.result.set_component_labels(labels)

    def test_get_best_component_without_labels(self):
        """Test get_best_component returns integers when no labels are set"""
        best_components = self.result.get_best_component()

        assert isinstance(best_components, np.ndarray)
        assert best_components.dtype in [np.int32, np.int64, np.intp]
        assert len(best_components) == len(self.data)
        assert all(comp in [0, 1, 2] for comp in best_components)

    def test_get_best_component_with_labels(self):
        """Test get_best_component returns labels when labels are set"""
        labels = {0: 'Healthy', 1: 'Diseased', 2: 'Control'}
        self.result.set_component_labels(labels)

        best_components = self.result.get_best_component()

        assert isinstance(best_components, np.ndarray)
        assert best_components.dtype.kind in ['U', 'S', 'O']  # Unicode, byte string, or object
        assert len(best_components) == len(self.data)
        assert all(comp in ['Healthy', 'Diseased', 'Control'] for comp in best_components)

    def test_get_best_component_consistency(self):
        """Test that labeled results map correctly to unlabeled results"""
        labels = {0: 'Healthy', 1: 'Diseased', 2: 'Control'}

        # Get best components without labels
        best_unlabeled = self.result.get_best_component()

        # Set labels and get best components
        self.result.set_component_labels(labels)
        best_labeled = self.result.get_best_component()

        # Verify mapping is correct
        for unlabeled, labeled in zip(best_unlabeled, best_labeled):
            assert labeled == labels[unlabeled]

    def test_get_group_assignments_df_without_labels(self):
        """Test DataFrame column names without labels"""
        df = self.result.get_group_assignments_df()

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['Component_0', 'Component_1', 'Component_2']
        assert df.shape == (len(self.data), 3)

    def test_get_group_assignments_df_with_labels(self):
        """Test DataFrame column names with labels"""
        labels = {0: 'Healthy', 1: 'Diseased', 2: 'Control'}
        self.result.set_component_labels(labels)

        df = self.result.get_group_assignments_df()

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['Healthy component', 'Diseased component', 'Control component']
        assert df.shape == (len(self.data), 3)

    def test_get_group_assignments_df_data_unchanged(self):
        """Test that underlying data is the same with or without labels"""
        labels = {0: 'Healthy', 1: 'Diseased', 2: 'Control'}

        # Get DataFrame without labels
        df_unlabeled = self.result.get_group_assignments_df()

        # Set labels and get DataFrame
        self.result.set_component_labels(labels)
        df_labeled = self.result.get_group_assignments_df()

        # Data should be identical, only column names differ
        np.testing.assert_array_equal(df_unlabeled.values, df_labeled.values)

    def test_get_group_assignments_df_with_sample_names(self):
        """Test that sample names are preserved when using labels"""
        # Fit with DataFrame to get sample names
        dmm = DirichletMixture(n_components=3, verbose=False, random_state=42)
        dmm.fit(self.data_df)

        labels = {0: 'Healthy', 1: 'Diseased', 2: 'Control'}
        dmm.result.set_component_labels(labels)

        df = dmm.result.get_group_assignments_df()

        assert list(df.index) == list(self.data_df.index)
        assert list(df.columns) == ['Healthy component', 'Diseased component', 'Control component']

    def test_labels_persist_across_calls(self):
        """Test that labels persist across multiple method calls"""
        labels = {0: 'Healthy', 1: 'Diseased', 2: 'Control'}
        self.result.set_component_labels(labels)

        # Call methods multiple times
        for _ in range(3):
            best = self.result.get_best_component()
            df = self.result.get_group_assignments_df()

            assert all(comp in ['Healthy', 'Diseased', 'Control'] for comp in best)
            assert list(df.columns) == ['Healthy component', 'Diseased component', 'Control component']

    def test_overwrite_labels(self):
        """Test that labels can be overwritten"""
        labels1 = {0: 'Healthy', 1: 'Diseased', 2: 'Control'}
        labels2 = {0: 'Type A', 1: 'Type B', 2: 'Type C'}

        self.result.set_component_labels(labels1)
        best1 = self.result.get_best_component()

        self.result.set_component_labels(labels2)
        best2 = self.result.get_best_component()

        # Results should use new labels
        assert all(comp in ['Type A', 'Type B', 'Type C'] for comp in best2)
        assert not any(comp in ['Healthy', 'Diseased', 'Control'] for comp in best2)

    def test_user_workflow(self):
        """Test the complete user workflow from the issue description"""
        # This simulates the exact workflow requested
        dmm = DirichletMixture(n_components=3, verbose=False, random_state=42)
        dmm.fit(self.data)

        # Set labels using the result attribute
        dmm.result.set_component_labels({0: 'Healthy', 1: 'Diseased', 2: 'Control'})

        # Verify get_best_component uses labels
        best = dmm.result.get_best_component()
        assert all(comp in ['Healthy', 'Diseased', 'Control'] for comp in best)

        # Verify get_group_assignments_df uses labels in column names
        df = dmm.result.get_group_assignments_df()
        assert list(df.columns) == ['Healthy component', 'Diseased component', 'Control component']
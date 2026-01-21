"""
Tests for sklearn API compatibility of DirichletMixture
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.base import is_classifier, clone
from sklearn.model_selection import cross_val_score, GridSearchCV
from pydmm import DirichletMixture


class TestSklearnCompatibility:
    """Test sklearn API compatibility"""

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

    def test_is_classifier(self):
        """Test that sklearn recognizes DirichletMixture as a classifier"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        assert is_classifier(dmm), "DirichletMixture should be recognized as a classifier by sklearn"

    def test_has_classes_attribute(self):
        """Test that classes_ attribute is set after fitting"""
        dmm = DirichletMixture(n_components=3, verbose=False, random_state=42)

        # Should not have classes_ before fitting
        assert not hasattr(dmm, 'classes_'), "Should not have classes_ before fitting"

        # Fit the model
        dmm.fit(self.data_array)

        # Should have classes_ after fitting
        assert hasattr(dmm, 'classes_'), "Should have classes_ attribute after fitting"

        # classes_ should be correct
        expected_classes = np.arange(3)
        np.testing.assert_array_equal(dmm.classes_, expected_classes)
        assert len(dmm.classes_) == 3, "classes_ should have length equal to n_components"

    def test_get_params(self):
        """Test get_params() inherited from BaseEstimator"""
        dmm = DirichletMixture(n_components=3, verbose=True, random_state=123)

        params = dmm.get_params()

        # Should return a dict with all init parameters
        assert isinstance(params, dict)
        assert 'n_components' in params
        assert 'verbose' in params
        assert 'random_state' in params

        # Values should match what was passed to __init__
        assert params['n_components'] == 3
        assert params['verbose'] == True
        assert params['random_state'] == 123

    def test_get_params_deep(self):
        """Test get_params(deep=True) works correctly"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)

        # get_params with deep=True should work (even though we have no nested estimators)
        params = dmm.get_params(deep=True)

        assert isinstance(params, dict)
        assert params['n_components'] == 2
        assert params['verbose'] == False
        assert params['random_state'] == 42

    def test_set_params(self):
        """Test set_params() inherited from BaseEstimator"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)

        # Verify initial state
        assert dmm.n_components == 2
        assert dmm.verbose == False
        assert dmm.random_state == 42

        # Set new parameters
        dmm.set_params(n_components=4, verbose=True, random_state=999)

        # Verify parameters were updated
        assert dmm.n_components == 4
        assert dmm.verbose == True
        assert dmm.random_state == 999

    def test_set_params_returns_self(self):
        """Test that set_params() returns self for method chaining"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)

        result = dmm.set_params(n_components=3)

        assert result is dmm, "set_params should return self"

    def test_clone(self):
        """Test sklearn's clone() utility works"""
        dmm = DirichletMixture(n_components=3, verbose=True, random_state=123)

        # Clone the estimator
        dmm_clone = clone(dmm)

        # Should be a different object
        assert dmm_clone is not dmm, "Clone should create a new object"

        # But with the same parameters
        assert dmm_clone.n_components == dmm.n_components
        assert dmm_clone.verbose == dmm.verbose
        assert dmm_clone.random_state == dmm.random_state

        # Clone should not be fitted
        assert not dmm_clone.is_fitted

    def test_clone_fitted_estimator(self):
        """Test that cloning a fitted estimator produces an unfitted clone"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(self.data_array)

        # Clone the fitted estimator
        dmm_clone = clone(dmm)

        # Clone should have same parameters but not be fitted
        assert dmm_clone.n_components == dmm.n_components
        assert not dmm_clone.is_fitted
        assert dmm_clone.result_ is None

    def test_fit_with_y_parameter(self):
        """Test that fit(X, y) works and ignores y"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)

        # Create dummy y labels
        dummy_y = np.random.randint(0, 2, size=self.n_samples)

        # Fit with y parameter (should be ignored)
        dmm.fit(self.data_array, y=dummy_y)

        # Should be fitted
        assert dmm.is_fitted
        assert dmm.result_ is not None

        # Now fit another model without y
        dmm2 = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm2.fit(self.data_array)

        # Results should be identical (y was ignored)
        np.testing.assert_allclose(
            dmm.result_.group_assignments,
            dmm2.result_.group_assignments,
            rtol=1e-10
        )

    def test_fit_with_different_y_values(self):
        """Test that different y values produce identical results (y is ignored)"""
        dmm1 = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm2 = DirichletMixture(n_components=2, verbose=False, random_state=42)

        y1 = np.zeros(self.n_samples)
        y2 = np.ones(self.n_samples)

        dmm1.fit(self.data_array, y=y1)
        dmm2.fit(self.data_array, y=y2)

        # Results should be identical (y is ignored)
        np.testing.assert_allclose(
            dmm1.result_.group_assignments,
            dmm2.result_.group_assignments,
            rtol=1e-10
        )

    def test_cross_val_score(self):
        """Test compatibility with cross_val_score"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)

        # cross_val_score requires a scoring function
        # Since we override score() to return negative log-likelihood, we can use it directly
        # Note: cv=2 for small dataset
        try:
            scores = cross_val_score(dmm, self.data_array, cv=2)
            assert len(scores) == 2, "Should return scores for each fold"
            assert all(isinstance(s, (int, float)) for s in scores), "All scores should be numeric"
        except Exception as e:
            # If it fails, it should be a sensible sklearn-related error, not a compatibility issue
            pytest.fail(f"cross_val_score failed with error: {e}")

    def test_grid_search(self):
        """Test compatibility with GridSearchCV"""
        dmm = DirichletMixture(verbose=False, random_state=42)

        # Define a small parameter grid
        param_grid = {
            'n_components': [2, 3]
        }

        # Create GridSearchCV object
        grid_search = GridSearchCV(dmm, param_grid, cv=2, scoring=None)

        # Fit should work
        try:
            grid_search.fit(self.data_array)
            assert hasattr(grid_search, 'best_params_'), "GridSearchCV should find best_params_"
            assert 'n_components' in grid_search.best_params_
            assert grid_search.best_params_['n_components'] in [2, 3]
        except Exception as e:
            # If it fails, it should be a sensible sklearn-related error
            pytest.fail(f"GridSearchCV failed with error: {e}")

    def test_grid_search_best_estimator(self):
        """Test that GridSearchCV produces a fitted best_estimator_"""
        dmm = DirichletMixture(verbose=False, random_state=42)

        param_grid = {'n_components': [2]}

        grid_search = GridSearchCV(dmm, param_grid, cv=2, refit=True)
        grid_search.fit(self.data_array)

        # best_estimator_ should be fitted
        assert hasattr(grid_search, 'best_estimator_')
        assert grid_search.best_estimator_.is_fitted
        assert hasattr(grid_search.best_estimator_, 'classes_')

    def test_backward_compatibility_fit_without_y(self):
        """Ensure existing API still works (fit(X) without y)"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)

        # Old-style fit call without y parameter
        dmm.fit(self.data_array)

        # Should work exactly as before
        assert dmm.is_fitted
        assert dmm.result_ is not None
        assert hasattr(dmm, 'classes_')

    def test_predict_returns_valid_class_labels(self):
        """Test that predict() returns labels from classes_"""
        dmm = DirichletMixture(n_components=3, verbose=False, random_state=42)
        dmm.fit(self.data_array)

        predictions = dmm.predict(self.data_array)

        # All predictions should be in classes_
        assert all(pred in dmm.classes_ for pred in predictions)
        # Predictions should be in range [0, n_components)
        assert all(0 <= pred < 3 for pred in predictions)

    def test_estimator_type(self):
        """Test that estimator has correct _estimator_type"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)

        # ClassifierMixin provides _estimator_type (may be via _get_tags or direct attribute)
        # Check using is_classifier which is the standard way
        from sklearn.base import is_classifier
        assert is_classifier(dmm), "Should be recognized as classifier"

        # Some sklearn versions have _estimator_type directly accessible
        if hasattr(dmm, '_estimator_type'):
            assert dmm._estimator_type == 'classifier'

    def test_sklearn_tags(self):
        """Test that sklearn tags are accessible"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)

        # BaseEstimator provides __sklearn_tags__() method (sklearn >= 1.0)
        # or _get_tags() (sklearn < 1.0)
        # Let's just verify one of them exists
        assert hasattr(dmm, '__sklearn_tags__') or hasattr(dmm, '_get_tags'), \
            "Should have sklearn tags method from BaseEstimator"

    def test_fit_predict_equivalent_to_fit_then_predict(self):
        """Test that fit_predict gives same results as fit followed by predict"""
        dmm1 = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm2 = DirichletMixture(n_components=2, verbose=False, random_state=42)

        # Method 1: fit_predict
        labels1 = dmm1.fit_predict(self.data_array)

        # Method 2: fit then predict
        dmm2.fit(self.data_array)
        labels2 = dmm2.predict(self.data_array)

        # Results should be identical
        np.testing.assert_array_equal(labels1, labels2)

    def test_classes_attribute_matches_predict_output(self):
        """Test that classes_ attribute matches possible predict() outputs"""
        for n_comp in [2, 3, 4]:
            dmm = DirichletMixture(n_components=n_comp, verbose=False, random_state=42)
            dmm.fit(self.data_array)

            predictions = dmm.predict(self.data_array)

            # All unique predictions should be in classes_
            unique_predictions = np.unique(predictions)
            assert all(pred in dmm.classes_ for pred in unique_predictions)

            # classes_ should equal [0, 1, ..., n_components-1]
            expected_classes = np.arange(n_comp)
            np.testing.assert_array_equal(dmm.classes_, expected_classes)

    def test_score_method_works_after_fit(self):
        """Test that score() method works correctly after fitting"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(self.data_array)

        # score() should work on fitted model
        score = dmm.score(self.data_array)

        assert isinstance(score, (int, float))
        assert not np.isnan(score)
        assert not np.isinf(score)

    def test_parameter_constraints(self):
        """Test that get_params/set_params respect parameter constraints"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)

        # get_params should return all parameters
        params = dmm.get_params()
        assert set(params.keys()) == {'n_components', 'verbose', 'random_state'}

        # set_params should allow valid updates
        dmm.set_params(n_components=5, verbose=True)
        assert dmm.n_components == 5
        assert dmm.verbose == True

    def test_feature_names_in_with_dataframe(self):
        """Test that feature_names_in_ is set when fitting with DataFrame"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)

        # Should not have feature_names_in_ before fitting
        assert not hasattr(dmm, 'feature_names_in_'), "Should not have feature_names_in_ before fitting"

        # Fit with DataFrame
        dmm.fit(self.data_df)

        # Should have feature_names_in_ after fitting with DataFrame
        assert hasattr(dmm, 'feature_names_in_'), "Should have feature_names_in_ after fitting with DataFrame"

        # Check type - should be numpy array of object dtype
        assert isinstance(dmm.feature_names_in_, np.ndarray)
        assert dmm.feature_names_in_.dtype == object

        # Check values - should match DataFrame columns
        expected_names = np.array(['Feature_0', 'Feature_1', 'Feature_2', 'Feature_3'], dtype=object)
        np.testing.assert_array_equal(dmm.feature_names_in_, expected_names)

        # Check length
        assert len(dmm.feature_names_in_) == self.n_features

    def test_feature_names_in_not_set_with_array(self):
        """Test that feature_names_in_ is NOT set when fitting with numpy array"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)

        # Fit with numpy array
        dmm.fit(self.data_array)

        # Should NOT have feature_names_in_ when fitted with array
        assert not hasattr(dmm, 'feature_names_in_'), \
            "Should not have feature_names_in_ when fitting with numpy array"

    def test_feature_names_in_with_custom_columns(self):
        """Test feature_names_in_ with custom column names"""
        # Create DataFrame with custom column names
        custom_columns = ['Taxa_A', 'Taxa_B', 'Taxa_C', 'Taxa_D']
        df_custom = pd.DataFrame(self.data_array, columns=custom_columns)

        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(df_custom)

        # Check that custom names are stored
        assert hasattr(dmm, 'feature_names_in_')
        expected_names = np.array(custom_columns, dtype=object)
        np.testing.assert_array_equal(dmm.feature_names_in_, expected_names)

    def test_feature_names_in_preserved_after_multiple_fits(self):
        """Test that feature_names_in_ is updated when refitting"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)

        # First fit with DataFrame
        dmm.fit(self.data_df)
        first_names = dmm.feature_names_in_.copy()

        # Second fit with array (should remove feature_names_in_)
        dmm.fit(self.data_array)
        assert not hasattr(dmm, 'feature_names_in_'), \
            "feature_names_in_ should not exist after fitting with array"

        # Third fit with DataFrame again (should restore)
        dmm.fit(self.data_df)
        assert hasattr(dmm, 'feature_names_in_')
        np.testing.assert_array_equal(dmm.feature_names_in_, first_names)

    def test_feature_names_in_with_different_dataframe(self):
        """Test that feature_names_in_ changes when fitting different DataFrames"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)

        # First DataFrame
        df1 = pd.DataFrame(self.data_array[:20], columns=['A', 'B', 'C', 'D'])
        dmm.fit(df1)
        names1 = dmm.feature_names_in_.copy()

        # Second DataFrame with different columns
        df2 = pd.DataFrame(self.data_array[:20], columns=['W', 'X', 'Y', 'Z'])
        dmm.fit(df2)
        names2 = dmm.feature_names_in_

        # Names should be different
        assert not np.array_equal(names1, names2)
        np.testing.assert_array_equal(names2, np.array(['W', 'X', 'Y', 'Z'], dtype=object))

    def test_feature_names_in_sklearn_compatibility(self):
        """Test that feature_names_in_ follows sklearn conventions"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(self.data_df)

        # Should be a 1D array
        assert dmm.feature_names_in_.ndim == 1

        # Should have object dtype (string array)
        assert dmm.feature_names_in_.dtype == object

        # All elements should be strings
        assert all(isinstance(name, str) for name in dmm.feature_names_in_)

        # Length should match n_features
        assert len(dmm.feature_names_in_) == dmm.result_.n_features


class TestBackwardCompatibility:
    """Test that existing code continues to work"""

    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.data = np.random.multinomial(1000, [0.25, 0.25, 0.25, 0.25], size=50).astype(np.int32)

    def test_old_style_usage_still_works(self):
        """Test that pre-sklearn usage patterns still work"""
        # Old-style usage: fit(X) without y
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(self.data)

        # Old-style predict
        labels = dmm.predict(self.data)

        # Old-style predict_proba
        proba = dmm.predict_proba(self.data)

        # Old-style fit_predict
        dmm2 = DirichletMixture(n_components=2, verbose=False, random_state=42)
        labels2 = dmm2.fit_predict(self.data)

        # All should work
        assert labels is not None
        assert proba is not None
        assert labels2 is not None

    def test_result_attribute_still_accessible(self):
        """Test that result and result_ both work"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(self.data)

        # Both result and result_ should work
        assert dmm.result is not None
        assert dmm.result_ is not None
        assert dmm.result is dmm.result_  # They should be the same object

    def test_existing_tests_compatibility(self):
        """Test that patterns from existing tests still work"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)

        # Pattern from existing tests: fit without y
        dmm.fit(self.data)

        # Access result attributes
        assert hasattr(dmm.result, 'goodness_of_fit')
        assert hasattr(dmm.result, 'group_assignments')
        assert hasattr(dmm.result, 'mixture_weights')

        # Use helper methods
        best_component = dmm.result.get_best_component()
        assert len(best_component) == len(self.data)

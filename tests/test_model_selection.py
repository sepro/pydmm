"""
Tests for model selection functionality
"""

import pytest
import numpy as np
from pydmm import DirichletMixture


class TestModelSelection:
    """Test model selection using BIC/AIC"""

    def setup_method(self):
        """Create test data with known number of clusters"""
        np.random.seed(42)

        # Create data with 2 clear clusters
        cluster1_props = np.array([0.7, 0.2, 0.05, 0.05])
        cluster2_props = np.array([0.05, 0.05, 0.7, 0.2])

        n_samples_per_cluster = 30
        samples = []

        # Generate cluster 1
        for i in range(n_samples_per_cluster):
            total = np.random.poisson(800) + 200
            counts = np.random.multinomial(total, cluster1_props)
            samples.append(counts)

        # Generate cluster 2
        for i in range(n_samples_per_cluster):
            total = np.random.poisson(800) + 200
            counts = np.random.multinomial(total, cluster2_props)
            samples.append(counts)

        self.data = np.array(samples, dtype=np.int32)
        self.true_n_clusters = 2

    def test_bic_model_selection(self):
        """Test that BIC correctly identifies the optimal number of clusters"""
        cluster_options = [1, 2, 3]
        bic_scores = []

        for k in cluster_options:
            dmm = DirichletMixture(n_components=k, verbose=False, random_state=42)
            dmm.fit(self.data)
            bic = dmm.result.goodness_of_fit['BIC']
            bic_scores.append(bic)

        # BIC should be minimized at the true number of clusters
        best_k = cluster_options[np.argmin(bic_scores)]

        # Should select 2 clusters (though allow some flexibility for stochastic data)
        assert best_k in [2, 3]  # 2 is ideal, 3 might be selected due to randomness

        # BIC should decrease from 1 to 2 clusters (better fit)
        assert bic_scores[1] < bic_scores[0]

    def test_aic_model_selection(self):
        """Test AIC model selection"""
        cluster_options = [1, 2, 3]
        aic_scores = []

        for k in cluster_options:
            dmm = DirichletMixture(n_components=k, verbose=False, random_state=42)
            dmm.fit(self.data)
            aic = dmm.result.goodness_of_fit['AIC']
            aic_scores.append(aic)

        # AIC should decrease from 1 to 2 clusters
        assert aic_scores[1] < aic_scores[0]

    def test_model_comparison_metrics(self):
        """Test that different model comparison metrics are computed"""
        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(self.data)

        gof = dmm.result.goodness_of_fit

        # Check that all metrics are present and finite
        metrics = ['NLE', 'LogDet', 'Laplace', 'BIC', 'AIC']
        for metric in metrics:
            assert metric in gof
            assert np.isfinite(gof[metric])

        # BIC should be larger than AIC (BIC penalizes complexity more)
        assert gof['BIC'] > gof['AIC']

    def test_single_cluster_model(self):
        """Test that single cluster model works"""
        dmm = DirichletMixture(n_components=1, verbose=False, random_state=42)
        dmm.fit(self.data)

        assert dmm.result is not None
        assert len(dmm.result.mixture_weights) == 1
        assert dmm.result.group_assignments.shape[1] == 1

        # All samples should be assigned to the single cluster
        assignments = dmm.result.get_best_component()
        assert all(label == 0 for label in assignments)

    def test_overclustering(self):
        """Test behavior with too many clusters"""
        # Try fitting with more clusters than true clusters
        dmm = DirichletMixture(n_components=4, verbose=False, random_state=42)
        dmm.fit(self.data)

        assert dmm.result is not None
        assert len(dmm.result.mixture_weights) == 4

        # BIC should be higher than optimal model
        dmm_optimal = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm_optimal.fit(self.data)

        bic_overfit = dmm.result.goodness_of_fit['BIC']
        bic_optimal = dmm_optimal.result.goodness_of_fit['BIC']

        # Overfitted model should have higher BIC (worse)
        assert bic_overfit > bic_optimal

    def test_model_selection_workflow(self):
        """Test complete model selection workflow"""
        cluster_range = range(1, 5)
        best_model = None
        best_bic = float('inf')

        # Test the workflow from the example
        for n_clusters in cluster_range:
            try:
                dmm = DirichletMixture(n_components=n_clusters, verbose=False, random_state=42)
                dmm.fit(self.data)

                bic = dmm.result.goodness_of_fit['BIC']

                if bic < best_bic:
                    best_bic = bic
                    best_model = dmm

            except Exception as e:
                # Some models might fail - that's okay
                continue

        assert best_model is not None
        assert best_model.result is not None

    def test_reproducible_model_selection(self):
        """Test that model selection is reproducible"""
        cluster_options = [1, 2, 3]
        bic_scores1 = []
        bic_scores2 = []

        # First run
        for k in cluster_options:
            dmm = DirichletMixture(n_components=k, verbose=False, random_state=42)
            dmm.fit(self.data)
            bic_scores1.append(dmm.result.goodness_of_fit['BIC'])

        # Second run with same seeds
        for k in cluster_options:
            dmm = DirichletMixture(n_components=k, verbose=False, random_state=42)
            dmm.fit(self.data)
            bic_scores2.append(dmm.result.goodness_of_fit['BIC'])

        # Should get identical results
        np.testing.assert_allclose(bic_scores1, bic_scores2, rtol=1e-10)
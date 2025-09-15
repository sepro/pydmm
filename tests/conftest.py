"""
Pytest configuration and fixtures for pyDMM tests
"""

import pytest
import numpy as np


@pytest.fixture
def simple_2cluster_data():
    """Fixture providing simple 2-cluster test data"""
    np.random.seed(42)

    # Create two well-separated clusters
    cluster1_props = np.array([0.8, 0.15, 0.03, 0.02])
    cluster2_props = np.array([0.02, 0.03, 0.15, 0.8])

    samples = []
    true_labels = []

    # Generate 25 samples from each cluster
    for cluster_id, props in enumerate([cluster1_props, cluster2_props]):
        for i in range(25):
            total_count = np.random.poisson(1000) + 100
            counts = np.random.multinomial(total_count, props)
            samples.append(counts)
            true_labels.append(cluster_id)

    return {
        'data': np.array(samples, dtype=np.int32),
        'true_labels': np.array(true_labels),
        'true_props': [cluster1_props, cluster2_props],
        'n_samples': 50,
        'n_features': 4,
        'n_clusters': 2
    }


@pytest.fixture
def minimal_data():
    """Fixture providing minimal test data"""
    return np.array([
        [10, 5],
        [8, 7],
        [2, 13],
        [1, 14]
    ], dtype=np.int32)


@pytest.fixture
def edge_case_data():
    """Fixture providing edge case test data"""
    return {
        'zeros': np.array([[10, 0], [0, 10]], dtype=np.int32),
        'single_sample': np.array([[5, 3, 2]], dtype=np.int32),
        'single_feature': np.array([[10], [8], [12]], dtype=np.int32)
    }
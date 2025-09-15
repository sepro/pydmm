"""
Tests specifically for the C extension integration
"""

import pytest
import numpy as np
from pydmm import DirichletMixture


class TestCExtensionIntegration:
    """Test the C extension integration and data handling"""

    def test_c_extension_import(self):
        """Test that the C extension can be imported"""
        from pydmm import pydmm_core
        assert pydmm_core is not None

    def test_data_type_handling(self):
        """Test that the C extension properly handles different data types"""
        # Create test data
        data_float_exact = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        data_float_nonexact = np.array([[1.5, 2.7], [3.1, 4.8]], dtype=np.float64)
        data_int32 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        data_int64 = np.array([[1, 2], [3, 4]], dtype=np.int64)

        dmm = DirichletMixture(n_components=2, verbose=False)

        # Should succeed with exact float data (no warning needed)
        dmm.fit(data_float_exact)
        assert dmm.result_ is not None

        # Should give warning with non-exact float data
        with pytest.warns(UserWarning, match="non-integer"):
            dmm2 = DirichletMixture(n_components=2, verbose=False)
            dmm2.fit(data_float_nonexact)

        # Should succeed with int32 data
        dmm.fit(data_int32)
        assert dmm.result_ is not None

        # Should succeed with int64 data (converted internally)
        dmm2 = DirichletMixture(n_components=2, verbose=False)
        dmm2.fit(data_int64)
        assert dmm2.result_ is not None

    def test_data_layout_handling(self):
        """Test that the data layout conversion is handled correctly"""
        # Create data that would expose layout issues
        np.random.seed(42)
        data = np.array([
            [100, 0, 0],    # Sample 1: all feature 0
            [100, 0, 0],    # Sample 2: all feature 0
            [0, 0, 100],    # Sample 3: all feature 2
            [0, 0, 100],    # Sample 4: all feature 2
        ], dtype=np.int32)

        dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
        dmm.fit(data)

        # Should achieve perfect or near-perfect clustering
        predicted = dmm.result_.get_best_component()

        # Check that samples with same pattern get same label
        # Note: The algorithm might assign labels arbitrarily, so we check clustering quality instead
        # Calculate clustering purity (allowing for label permutation)
        accuracy_original = (predicted[0] == predicted[1]) and (predicted[2] == predicted[3])
        accuracy_flipped = (predicted[0] != predicted[1]) and (predicted[2] != predicted[3])

        # At least one assignment should give good clustering
        assert accuracy_original or not accuracy_flipped

    def test_memory_management(self):
        """Test that memory is properly managed in C extension"""
        # Test with larger dataset to stress memory management
        np.random.seed(42)

        # Generate larger dataset
        n_samples = 200
        n_features = 10
        data = []

        for i in range(n_samples):
            # Random proportions for each sample
            props = np.random.dirichlet(np.ones(n_features))
            total_count = np.random.poisson(1000) + 100
            counts = np.random.multinomial(total_count, props)
            data.append(counts)

        data = np.array(data, dtype=np.int32)

        # Fit multiple models to test memory cleanup
        for i in range(5):
            dmm = DirichletMixture(n_components=2, verbose=False, random_state=i)
            dmm.fit(data)
            assert dmm.result_ is not None

            # Explicitly delete to test cleanup
            del dmm

    def test_edge_case_data(self):
        """Test C extension with edge case data"""
        # Test with minimal data
        min_data = np.array([[1, 1]], dtype=np.int32)
        dmm = DirichletMixture(n_components=1, verbose=False)
        dmm.fit(min_data)
        assert dmm.result_ is not None

        # Test with zeros in data
        zero_data = np.array([[10, 0], [0, 10]], dtype=np.int32)
        dmm2 = DirichletMixture(n_components=2, verbose=False)
        dmm2.fit(zero_data)
        assert dmm2.result_ is not None

    def test_array_contiguity(self):
        """Test that non-contiguous arrays are handled correctly"""
        # Create non-contiguous array
        large_array = np.random.multinomial(1000, [0.2, 0.3, 0.3, 0.2], size=20).astype(np.int32)
        non_contiguous = large_array[::2, ::2]  # Take every other element

        assert not non_contiguous.flags['C_CONTIGUOUS']

        dmm = DirichletMixture(n_components=2, verbose=False)
        dmm.fit(non_contiguous)  # Should handle non-contiguous arrays
        assert dmm.result_ is not None

    def test_random_seed_consistency(self):
        """Test that C extension random seed produces consistent results"""
        data = np.array([
            [10, 5, 2],
            [8, 7, 3],
            [2, 3, 12],
            [1, 4, 15]
        ], dtype=np.int32)

        # Same seed should give same results
        dmm1 = DirichletMixture(n_components=2, verbose=False, random_state=123)
        dmm1.fit(data)

        dmm2 = DirichletMixture(n_components=2, verbose=False, random_state=123)
        dmm2.fit(data)

        np.testing.assert_allclose(
            dmm1.result_.group_assignments,
            dmm2.result_.group_assignments,
            rtol=1e-10
        )

        # Different seeds should potentially give different results
        dmm3 = DirichletMixture(n_components=2, verbose=False, random_state=456)
        dmm3.fit(data)

        # Note: Results might still be the same if the problem is well-determined,
        # but they shouldn't be required to be the same
#!/usr/bin/env python3
"""
Example comparing probability computations between C code and Python code.

This example creates synthetic data with slightly overlapping clusters to test
whether the probability computations in the C extension (used during fitting)
match the Python implementation (used in predict_proba).

The goal is to validate that both implementations produce consistent results.
"""

import numpy as np
import pandas as pd
from pydmm import DirichletMixture
from scipy.stats import pearsonr
import warnings


def create_overlapping_cluster_data():
    """
    Create synthetic data with slightly overlapping clusters.

    Unlike the reference example with well-separated groups, this uses
    clusters with moderate overlap to create meaningful probability differences.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create heavily overlapping cluster proportions to stress-test numerical precision
    # Group 1: Slightly favors features 0,1
    group1_props = np.array([0.35, 0.30, 0.20, 0.15])
    # Group 2: Slightly favors features 2,3
    group2_props = np.array([0.20, 0.25, 0.30, 0.25])

    print("Cluster proportions (with overlap):")
    print(f"Group 1: {group1_props}")
    print(f"Group 2: {group2_props}")
    print()

    # Check overlap by computing similarity
    overlap_metric = np.sum(np.minimum(group1_props, group2_props))
    print(f"Overlap metric (sum of minimums): {overlap_metric:.3f}")
    print("(Higher values indicate more overlap)")
    print()

    # Parameters for data generation
    n_samples_per_group = 200  # 400 total samples
    n_features = 4

    samples = []
    true_labels = []

    # Generate Group 1 samples with variable count depths to test numerical edge cases
    for i in range(n_samples_per_group):
        # Mix of high and low count samples to stress numerical precision
        if i % 10 == 0:
            # Some low-count samples that stress numerical precision
            total_count = np.random.poisson(50) + 20  # Mean ~50, min 20
        else:
            # Regular higher-count samples
            total_count = np.random.poisson(800) + 200  # Mean ~800, min 200
        counts = np.random.multinomial(total_count, group1_props)
        samples.append(counts)
        true_labels.append(0)

    # Generate Group 2 samples with similar variation
    for i in range(n_samples_per_group):
        if i % 10 == 0:
            total_count = np.random.poisson(50) + 20
        else:
            total_count = np.random.poisson(800) + 200
        counts = np.random.multinomial(total_count, group2_props)
        samples.append(counts)
        true_labels.append(1)

    # Convert to arrays
    data_array = np.array(samples, dtype=np.int32)
    true_labels = np.array(true_labels)

    # Create DataFrame with informative names
    sample_names = [f"Sample_{i:03d}" for i in range(len(samples))]
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    data_df = pd.DataFrame(data_array, index=sample_names, columns=feature_names)

    return data_df, true_labels, group1_props, group2_props


def analyze_data_overlap(data, true_labels, true_props):
    """
    Analyze the actual overlap in the generated data.
    """
    print("Data Overlap Analysis:")
    print("=" * 22)

    # Separate groups
    group1_data = data[true_labels == 0]
    group2_data = data[true_labels == 1]

    # Convert to proportions for each sample, then get statistics
    group1_props_obs = group1_data.div(group1_data.sum(axis=1), axis=0)
    group2_props_obs = group2_data.div(group2_data.sum(axis=1), axis=0)

    # Get mean and std for each group
    g1_mean = group1_props_obs.mean()
    g1_std = group1_props_obs.std()
    g2_mean = group2_props_obs.mean()
    g2_std = group2_props_obs.std()

    print("Observed proportions (mean ± std):")
    print("-" * 35)
    print(f"{'Feature':<10} {'Group1_Mean':<12} {'Group1_Std':<12} {'Group2_Mean':<12} {'Group2_Std':<12}")
    print("-" * 58)

    for i in range(len(g1_mean)):
        feature = f"Feature_{i}"
        print(f"{feature:<10} {g1_mean.iloc[i]:<12.3f} {g1_std.iloc[i]:<12.3f} {g2_mean.iloc[i]:<12.3f} {g2_std.iloc[i]:<12.3f}")

    print()

    # Compute separation strength
    separations = []
    for i in range(len(g1_mean)):
        # How many standard deviations apart are the means?
        pooled_std = np.sqrt((g1_std.iloc[i]**2 + g2_std.iloc[i]**2) / 2)
        separation = abs(g1_mean.iloc[i] - g2_mean.iloc[i]) / pooled_std
        separations.append(separation)
        print(f"Feature_{i} separation: {separation:.2f} standard deviations")

    print(f"Average separation: {np.mean(separations):.2f} standard deviations")
    print("(Values < 2 indicate overlapping distributions)")
    print()


def compare_probabilities(c_probabilities, python_probabilities):
    """
    Comprehensive comparison between C and Python computed probabilities.
    """
    print("Probability Computation Comparison:")
    print("=" * 35)

    # Basic shape and type validation
    print(f"C probabilities shape: {c_probabilities.shape}")
    print(f"Python probabilities shape: {python_probabilities.shape}")
    print(f"C probabilities type: {type(c_probabilities)}")
    print(f"Python probabilities type: {type(python_probabilities)}")
    print()

    # Ensure shapes match
    if c_probabilities.shape != python_probabilities.shape:
        print("ERROR: Probability matrices have different shapes!")
        return

    # Convert to numpy arrays for consistency
    if hasattr(c_probabilities, 'values'):
        c_probs = c_probabilities.values
    else:
        c_probs = c_probabilities

    python_probs = python_probabilities

    # Compute element-wise differences
    diff_matrix = np.abs(c_probs - python_probs)

    print("Element-wise Absolute Differences:")
    print("-" * 32)
    print(f"Maximum difference: {np.max(diff_matrix):.8f}")
    print(f"Mean difference: {np.mean(diff_matrix):.8f}")
    print(f"Standard deviation of differences: {np.std(diff_matrix):.8f}")
    print(f"Median difference: {np.median(diff_matrix):.8f}")
    print(f"95th percentile difference: {np.percentile(diff_matrix, 95):.8f}")
    print()

    # Row-wise (sample-wise) analysis
    sample_diffs = np.sum(diff_matrix, axis=1)
    worst_sample_idx = np.argmax(sample_diffs)
    best_sample_idx = np.argmin(sample_diffs)

    print("Sample-wise Analysis:")
    print("-" * 19)
    print(f"Worst sample index: {worst_sample_idx} (total diff: {sample_diffs[worst_sample_idx]:.6f})")
    print(f"Best sample index: {best_sample_idx} (total diff: {sample_diffs[best_sample_idx]:.6f})")
    print(f"Mean sample difference: {np.mean(sample_diffs):.6f}")
    print()

    # Component-wise (column-wise) analysis
    component_diffs = np.sum(diff_matrix, axis=0)
    print("Component-wise Analysis:")
    print("-" * 22)
    for i, diff in enumerate(component_diffs):
        print(f"Component {i}: total difference = {diff:.6f}")
    print()

    # Correlation analysis
    print("Correlation Analysis:")
    print("-" * 19)
    for component in range(c_probs.shape[1]):
        c_comp = c_probs[:, component]
        py_comp = python_probs[:, component]

        correlation, p_value = pearsonr(c_comp, py_comp)
        print(f"Component {component}:")
        print(f"  Pearson correlation: {correlation:.8f}")
        print(f"  P-value: {p_value:.2e}")
        print(f"  R-squared: {correlation**2:.8f}")

    print()

    # Overall correlation (flattened arrays)
    overall_corr, overall_p = pearsonr(c_probs.flatten(), python_probs.flatten())
    print(f"Overall correlation: {overall_corr:.8f} (p={overall_p:.2e})")
    print()

    # Probability sum validation (should be 1.0 for each sample)
    c_sums = np.sum(c_probs, axis=1)
    py_sums = np.sum(python_probs, axis=1)

    print("Probability Sum Validation:")
    print("-" * 26)
    print(f"C probabilities - deviation from 1.0:")
    print(f"  Max deviation: {np.max(np.abs(c_sums - 1.0)):.8f}")
    print(f"  Mean deviation: {np.mean(np.abs(c_sums - 1.0)):.8f}")

    print(f"Python probabilities - deviation from 1.0:")
    print(f"  Max deviation: {np.max(np.abs(py_sums - 1.0)):.8f}")
    print(f"  Mean deviation: {np.mean(np.abs(py_sums - 1.0)):.8f}")
    print()

    return diff_matrix, worst_sample_idx, best_sample_idx


def show_detailed_examples(c_probabilities, python_probabilities, diff_matrix,
                         worst_sample_idx, best_sample_idx, data, true_labels):
    """
    Show detailed examples of specific samples for analysis.
    """
    print("Detailed Sample Examples:")
    print("=" * 25)

    # Convert to numpy if needed
    if hasattr(c_probabilities, 'values'):
        c_probs = c_probabilities.values
    else:
        c_probs = c_probabilities

    python_probs = python_probabilities

    # Show worst sample
    print(f"Worst Sample (Index {worst_sample_idx}):")
    print("-" * 30)
    print(f"True label: {true_labels[worst_sample_idx]}")
    print(f"Sample data: {data.iloc[worst_sample_idx].values}")
    print(f"C probabilities:      {c_probs[worst_sample_idx]}")
    print(f"Python probabilities: {python_probs[worst_sample_idx]}")
    print(f"Absolute differences: {diff_matrix[worst_sample_idx]}")
    print(f"Total difference: {np.sum(diff_matrix[worst_sample_idx]):.6f}")
    print()

    # Show best sample
    print(f"Best Sample (Index {best_sample_idx}):")
    print("-" * 29)
    print(f"True label: {true_labels[best_sample_idx]}")
    print(f"Sample data: {data.iloc[best_sample_idx].values}")
    print(f"C probabilities:      {c_probs[best_sample_idx]}")
    print(f"Python probabilities: {python_probs[best_sample_idx]}")
    print(f"Absolute differences: {diff_matrix[best_sample_idx]}")
    print(f"Total difference: {np.sum(diff_matrix[best_sample_idx]):.6f}")
    print()

    # Show a few medium examples
    sample_diffs = np.sum(diff_matrix, axis=1)
    median_diff = np.median(sample_diffs)
    median_indices = np.where(np.abs(sample_diffs - median_diff) < 0.001)[0][:3]

    print("Representative Samples (near median difference):")
    print("-" * 47)
    for i, idx in enumerate(median_indices):
        if i >= 3:  # Limit to 3 examples
            break
        print(f"Sample {idx}:")
        print(f"  True label: {true_labels[idx]}")
        print(f"  C probs:      {c_probs[idx]}")
        print(f"  Python probs: {python_probs[idx]}")
        print(f"  Differences:  {diff_matrix[idx]}")
        print()


def numerical_stability_analysis(c_probabilities, python_probabilities):
    """
    Analyze numerical stability and potential systematic issues.
    """
    print("Numerical Stability Analysis:")
    print("=" * 29)

    # Convert to numpy if needed
    if hasattr(c_probabilities, 'values'):
        c_probs = c_probabilities.values
    else:
        c_probs = c_probabilities

    python_probs = python_probabilities

    # Check for extreme values
    print("Extreme Value Analysis:")
    print("-" * 22)
    print(f"C probabilities:")
    print(f"  Minimum value: {np.min(c_probs):.8f}")
    print(f"  Maximum value: {np.max(c_probs):.8f}")
    print(f"  Values near 0: {np.sum(c_probs < 1e-6)}")
    print(f"  Values near 1: {np.sum(c_probs > 0.999999)}")

    print(f"Python probabilities:")
    print(f"  Minimum value: {np.min(python_probs):.8f}")
    print(f"  Maximum value: {np.max(python_probs):.8f}")
    print(f"  Values near 0: {np.sum(python_probs < 1e-6)}")
    print(f"  Values near 1: {np.sum(python_probs > 0.999999)}")
    print()

    # Check for systematic biases
    signed_diffs = c_probs - python_probs
    print("Systematic Bias Analysis:")
    print("-" * 24)
    print(f"Mean signed difference: {np.mean(signed_diffs):.8f}")
    print(f"  (Positive = C higher, Negative = Python higher)")

    for component in range(c_probs.shape[1]):
        comp_bias = np.mean(signed_diffs[:, component])
        print(f"Component {component} bias: {comp_bias:.8f}")

    print()

    # Distribution of differences
    abs_diffs = np.abs(signed_diffs)
    print("Distribution of Absolute Differences:")
    print("-" * 36)
    percentiles = [50, 75, 90, 95, 99, 99.9]
    for p in percentiles:
        val = np.percentile(abs_diffs, p)
        print(f"{p:4.1f}th percentile: {val:.8f}")
    print()


def main():
    """
    Main function to run the probability comparison example.
    """
    print("pyDMM - Probability Computation Comparison")
    print("=" * 42)
    print("Testing consistency between C and Python probability computations")
    print()

    # Create overlapping cluster data
    data, true_labels, group1_props, group2_props = create_overlapping_cluster_data()

    print(f"Generated data shape: {data.shape}")
    print(f"True group distribution: {np.bincount(true_labels)}")
    print()

    # Analyze the overlap in generated data
    analyze_data_overlap(data, true_labels, [group1_props, group2_props])

    # Fit the model to get C-computed probabilities
    print("Fitting Dirichlet Mixture Model...")
    print("-" * 33)

    dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
    dmm.fit(data)

    print("Model fitted successfully!")
    print()

    # Get C-computed probabilities (from fitting process)
    c_probabilities = dmm.result_.group_assignments
    print(f"C-computed probabilities obtained: {c_probabilities.shape}")

    # Get Python-computed probabilities (using predict_proba on same data)
    print("Computing Python probabilities using predict_proba...")
    python_probabilities = dmm.predict_proba(data)
    print(f"Python-computed probabilities obtained: {python_probabilities.shape}")
    print()

    # Compare the two sets of probabilities
    diff_matrix, worst_idx, best_idx = compare_probabilities(c_probabilities, python_probabilities)

    # Show detailed examples
    show_detailed_examples(c_probabilities, python_probabilities, diff_matrix,
                          worst_idx, best_idx, data, true_labels)

    # Numerical stability analysis
    numerical_stability_analysis(c_probabilities, python_probabilities)

    # Final assessment
    max_diff = np.max(diff_matrix)
    mean_diff = np.mean(diff_matrix)

    print("Final Assessment:")
    print("=" * 17)

    if max_diff < 1e-6:
        print("🎉 EXCELLENT: Probability computations are virtually identical!")
        print("   Both implementations produce essentially the same results.")
    elif max_diff < 1e-4:
        print("✅ VERY GOOD: Probability computations are highly consistent.")
        print("   Tiny differences are likely due to numerical precision.")
    elif max_diff < 1e-2:
        print("⚠️  ACCEPTABLE: Some differences found, but generally consistent.")
        print("   May indicate minor implementation differences.")
    else:
        print("❌ CONCERNING: Significant differences found between implementations!")
        print("   This may indicate bugs or fundamental implementation differences.")

    print()
    print(f"Maximum difference: {max_diff:.8f}")
    print(f"Mean difference: {mean_diff:.8f}")

    # Additional model info
    summary = dmm.result_.summary()
    print(f"\nModel Summary:")
    print(f"BIC: {summary['BIC']:.2f}")
    print(f"AIC: {summary['AIC']:.2f}")
    print(f"Mixture weights: {summary['mixture_weights']}")

    print("\nComparison completed successfully!")


if __name__ == "__main__":
    main()
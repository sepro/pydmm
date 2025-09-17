#!/usr/bin/env python3
"""
Example comparing classification decisions between C code and Python code.

This example creates synthetic data with overlapping clusters to test whether
the classification decisions from the C extension (used during fitting) match
the Python implementation (used in predict() method).

The goal is to quantify how often the implementations would assign samples to
different clusters, which indicates the practical impact of probability
computation differences.
"""

import numpy as np
import pandas as pd
from pydmm import DirichletMixture
import warnings


def create_challenging_cluster_data():
    """
    Create synthetic data with overlapping clusters and variable count depths
    to maximize the chance of classification disagreements.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create heavily overlapping cluster proportions to maximize classification edge cases
    # These are designed to create the most challenging scenarios
    group1_props = np.array([0.30, 0.28, 0.22, 0.20])
    group2_props = np.array([0.22, 0.24, 0.28, 0.26])

    print("Cluster proportions (designed for classification edge cases):")
    print(f"Group 1: {group1_props}")
    print(f"Group 2: {group2_props}")
    print()

    # Compute overlap metric
    overlap_metric = np.sum(np.minimum(group1_props, group2_props))
    print(f"Overlap metric: {overlap_metric:.3f}")
    print()

    # Parameters for data generation
    n_samples_per_group = 300  # 600 total samples for good statistics
    n_features = 4

    samples = []
    true_labels = []
    count_depths = []

    # Generate Group 1 samples with more low-count samples to stress-test classification
    for i in range(n_samples_per_group):
        # Create many more low-count samples to maximize edge cases
        if i % 5 == 0:
            # Very low count samples (high uncertainty)
            total_count = np.random.poisson(20) + 5  # Mean ~20, min 5
        elif i % 3 == 0:
            # Low count samples
            total_count = np.random.poisson(60) + 20  # Mean ~60, min 20
        elif i % 2 == 0:
            # Medium count samples
            total_count = np.random.poisson(150) + 50  # Mean ~150, min 50
        else:
            # Regular higher count samples
            total_count = np.random.poisson(500) + 100  # Mean ~500, min 100

        counts = np.random.multinomial(total_count, group1_props)
        samples.append(counts)
        true_labels.append(0)
        count_depths.append(total_count)

    # Generate Group 2 samples with similar variation
    for i in range(n_samples_per_group):
        if i % 5 == 0:
            total_count = np.random.poisson(20) + 5
        elif i % 3 == 0:
            total_count = np.random.poisson(60) + 20
        elif i % 2 == 0:
            total_count = np.random.poisson(150) + 50
        else:
            total_count = np.random.poisson(500) + 100

        counts = np.random.multinomial(total_count, group2_props)
        samples.append(counts)
        true_labels.append(1)
        count_depths.append(total_count)

    # Convert to arrays
    data_array = np.array(samples, dtype=np.int32)
    true_labels = np.array(true_labels)
    count_depths = np.array(count_depths)

    # Create DataFrame
    sample_names = [f"Sample_{i:03d}" for i in range(len(samples))]
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    data_df = pd.DataFrame(data_array, index=sample_names, columns=feature_names)

    return data_df, true_labels, count_depths, group1_props, group2_props


def analyze_count_depth_distribution(count_depths, true_labels):
    """
    Analyze the distribution of count depths in the generated data.
    """
    print("Count Depth Distribution Analysis:")
    print("=" * 34)

    group1_depths = count_depths[true_labels == 0]
    group2_depths = count_depths[true_labels == 1]

    print(f"Overall statistics:")
    print(f"  Total samples: {len(count_depths)}")
    print(f"  Count depth range: {np.min(count_depths)} - {np.max(count_depths)}")
    print(f"  Mean count depth: {np.mean(count_depths):.1f}")
    print(f"  Median count depth: {np.median(count_depths):.1f}")
    print()

    # Categorize by count depth
    very_low = np.sum(count_depths < 50)
    low = np.sum((count_depths >= 50) & (count_depths < 150))
    medium = np.sum((count_depths >= 150) & (count_depths < 400))
    high = np.sum(count_depths >= 400)

    print(f"Count depth categories:")
    print(f"  Very low (<50):    {very_low:3d} samples ({very_low/len(count_depths)*100:.1f}%)")
    print(f"  Low (50-149):      {low:3d} samples ({low/len(count_depths)*100:.1f}%)")
    print(f"  Medium (150-399):  {medium:3d} samples ({medium/len(count_depths)*100:.1f}%)")
    print(f"  High (400+):       {high:3d} samples ({high/len(count_depths)*100:.1f}%)")
    print()


def compare_classifications(c_labels, python_labels, data, true_labels, count_depths):
    """
    Comprehensive comparison of classification decisions between C and Python implementations.
    """
    print("Classification Decision Comparison:")
    print("=" * 35)

    # Basic disagreement statistics
    disagreements = c_labels != python_labels
    n_disagreements = np.sum(disagreements)
    disagreement_rate = n_disagreements / len(c_labels)

    print(f"Overall disagreement statistics:")
    print(f"  Total samples: {len(c_labels)}")
    print(f"  Disagreements: {n_disagreements}")
    print(f"  Disagreement rate: {disagreement_rate:.4f} ({disagreement_rate*100:.2f}%)")
    print()

    if n_disagreements == 0:
        print("🎉 Perfect agreement! Both implementations classify all samples identically.")
        return disagreements

    # Per true cluster analysis
    print("Disagreement breakdown by true cluster:")
    print("-" * 38)
    for true_cluster in [0, 1]:
        mask = true_labels == true_cluster
        cluster_disagreements = np.sum(disagreements[mask])
        cluster_total = np.sum(mask)
        cluster_rate = cluster_disagreements / cluster_total if cluster_total > 0 else 0

        print(f"  True cluster {true_cluster}: {cluster_disagreements}/{cluster_total} disagreements ({cluster_rate*100:.2f}%)")
    print()

    # Count depth correlation analysis
    print("Disagreement correlation with count depth:")
    print("-" * 40)

    disagreeing_depths = count_depths[disagreements]
    agreeing_depths = count_depths[~disagreements]

    print(f"  Disagreeing samples - mean depth: {np.mean(disagreeing_depths):.1f}")
    print(f"  Agreeing samples - mean depth: {np.mean(agreeing_depths):.1f}")
    print(f"  Depth difference: {np.mean(agreeing_depths) - np.mean(disagreeing_depths):.1f}")
    print()

    # Count depth category breakdown
    depth_categories = [
        ("Very low (<50)", count_depths < 50),
        ("Low (50-149)", (count_depths >= 50) & (count_depths < 150)),
        ("Medium (150-399)", (count_depths >= 150) & (count_depths < 400)),
        ("High (400+)", count_depths >= 400)
    ]

    print("Disagreement rates by count depth category:")
    print("-" * 43)
    for cat_name, cat_mask in depth_categories:
        if np.sum(cat_mask) > 0:
            cat_disagreements = np.sum(disagreements[cat_mask])
            cat_total = np.sum(cat_mask)
            cat_rate = cat_disagreements / cat_total
            print(f"  {cat_name:<18}: {cat_disagreements}/{cat_total} ({cat_rate*100:.1f}%)")
    print()

    return disagreements


def analyze_probability_patterns(c_probabilities, python_probabilities, disagreements, data, true_labels, count_depths):
    """
    Analyze probability patterns for disagreeing samples.
    """
    print("Probability Pattern Analysis for Disagreements:")
    print("=" * 47)

    if np.sum(disagreements) == 0:
        print("No disagreements to analyze.")
        return

    # Get indices of disagreeing samples
    disagreeing_indices = np.where(disagreements)[0]

    print(f"Analyzing {len(disagreeing_indices)} disagreeing samples...")
    print()

    # Decision boundary analysis
    print("Decision boundary analysis:")
    print("-" * 26)

    # For each disagreeing sample, check how close probabilities were to 0.5
    c_max_probs = np.max(c_probabilities[disagreements], axis=1)
    python_max_probs = np.max(python_probabilities[disagreements], axis=1)

    # Distance from 0.5 (decision boundary for 2 clusters)
    c_boundary_distances = np.abs(c_max_probs - 0.5)
    python_boundary_distances = np.abs(python_max_probs - 0.5)

    print(f"  C implementation - mean distance from boundary: {np.mean(c_boundary_distances):.4f}")
    print(f"  Python implementation - mean distance from boundary: {np.mean(python_boundary_distances):.4f}")
    print()

    # Confidence analysis
    print("Confidence analysis for disagreeing samples:")
    print("-" * 43)

    very_uncertain = np.sum(c_max_probs < 0.6)  # Very close to 50/50
    uncertain = np.sum((c_max_probs >= 0.6) & (c_max_probs < 0.8))
    confident = np.sum((c_max_probs >= 0.8) & (c_max_probs < 0.95))
    very_confident = np.sum(c_max_probs >= 0.95)

    total_disagreeing = len(c_max_probs)

    print(f"  Very uncertain (max prob < 0.6):  {very_uncertain} ({very_uncertain/total_disagreeing*100:.1f}%)")
    print(f"  Uncertain (0.6-0.8):             {uncertain} ({uncertain/total_disagreeing*100:.1f}%)")
    print(f"  Confident (0.8-0.95):            {confident} ({confident/total_disagreeing*100:.1f}%)")
    print(f"  Very confident (>0.95):          {very_confident} ({very_confident/total_disagreeing*100:.1f}%)")
    print()


def show_disagreement_examples(c_labels, python_labels, c_probabilities, python_probabilities,
                              disagreements, data, true_labels, count_depths):
    """
    Show detailed examples of samples where classifications disagree.
    """
    print("Detailed Disagreement Examples:")
    print("=" * 31)

    disagreeing_indices = np.where(disagreements)[0]

    if len(disagreeing_indices) == 0:
        print("No disagreements to show.")
        return

    # Show up to 10 examples, prioritizing interesting cases
    n_examples = min(10, len(disagreeing_indices))

    # Sort by count depth to show low-count examples first
    sorted_indices = disagreeing_indices[np.argsort(count_depths[disagreeing_indices])]

    print(f"Showing {n_examples} disagreement examples (sorted by count depth):")
    print("-" * 60)
    print(f"{'Idx':<4} {'Depth':<6} {'True':<4} {'C_Lab':<5} {'Py_Lab':<6} {'C_Probs':<20} {'Py_Probs':<20} {'Data'}")
    print("-" * 100)

    for i in range(n_examples):
        idx = sorted_indices[i]
        depth = count_depths[idx]
        true_label = true_labels[idx]
        c_label = c_labels[idx]
        py_label = python_labels[idx]
        c_probs = c_probabilities[idx]
        py_probs = python_probabilities[idx]
        sample_data = data.iloc[idx].values

        c_probs_str = f"[{c_probs[0]:.3f}, {c_probs[1]:.3f}]"
        py_probs_str = f"[{py_probs[0]:.3f}, {py_probs[1]:.3f}]"
        data_str = f"{sample_data}"

        print(f"{idx:<4} {depth:<6} {true_label:<4} {c_label:<5} {py_label:<6} {c_probs_str:<20} {py_probs_str:<20} {data_str}")

    print()

    # Summary statistics for examples
    example_depths = count_depths[sorted_indices[:n_examples]]
    example_true = true_labels[sorted_indices[:n_examples]]

    print(f"Example summary:")
    print(f"  Count depth range: {np.min(example_depths)} - {np.max(example_depths)}")
    print(f"  Mean count depth: {np.mean(example_depths):.1f}")
    print(f"  True label distribution: Group 0: {np.sum(example_true == 0)}, Group 1: {np.sum(example_true == 1)}")
    print()


def assess_practical_impact(c_labels, python_labels, true_labels, disagreements):
    """
    Assess the practical impact of classification disagreements.
    """
    print("Practical Impact Assessment:")
    print("=" * 28)

    if np.sum(disagreements) == 0:
        print("No disagreements found - both implementations are practically identical for classification.")
        return

    # Calculate classification accuracy for both implementations
    c_accuracy = np.mean(c_labels == true_labels)
    python_accuracy = np.mean(python_labels == true_labels)

    print(f"Classification accuracy comparison:")
    print(f"  C implementation accuracy: {c_accuracy:.4f} ({c_accuracy*100:.2f}%)")
    print(f"  Python implementation accuracy: {python_accuracy:.4f} ({python_accuracy*100:.2f}%)")
    print(f"  Accuracy difference: {abs(c_accuracy - python_accuracy):.4f}")
    print()

    # Check if disagreements affect the better-performing samples
    disagreeing_indices = np.where(disagreements)[0]
    c_correct_in_disagreements = np.mean(c_labels[disagreeing_indices] == true_labels[disagreeing_indices])
    python_correct_in_disagreements = np.mean(python_labels[disagreeing_indices] == true_labels[disagreeing_indices])

    print(f"Accuracy on disagreeing samples:")
    print(f"  C implementation: {c_correct_in_disagreements:.4f}")
    print(f"  Python implementation: {python_correct_in_disagreements:.4f}")
    print()

    # Impact assessment
    disagreement_rate = np.sum(disagreements) / len(disagreements)

    if disagreement_rate < 0.001:
        impact_level = "NEGLIGIBLE"
        impact_desc = "Disagreements are extremely rare and unlikely to affect practical applications."
    elif disagreement_rate < 0.01:
        impact_level = "MINIMAL"
        impact_desc = "Very few disagreements. Impact on most applications would be minimal."
    elif disagreement_rate < 0.05:
        impact_level = "LOW"
        impact_desc = "Some disagreements present. May affect applications requiring high precision."
    elif disagreement_rate < 0.10:
        impact_level = "MODERATE"
        impact_desc = "Notable disagreements. Could impact results in sensitive applications."
    else:
        impact_level = "HIGH"
        impact_desc = "Significant disagreements. May seriously affect application results."

    print(f"Impact Assessment: {impact_level}")
    print(f"  {impact_desc}")
    print()


def main():
    """
    Main function to run the classification comparison example.
    """
    print("pyDMM - Classification Decision Comparison")
    print("=" * 42)
    print("Testing classification consistency between C and Python implementations")
    print()

    # Create challenging cluster data
    data, true_labels, count_depths, group1_props, group2_props = create_challenging_cluster_data()

    print(f"Generated data shape: {data.shape}")
    print(f"True group distribution: {np.bincount(true_labels)}")
    print()

    # Analyze count depth distribution
    analyze_count_depth_distribution(count_depths, true_labels)

    # Fit the model
    print("Fitting Dirichlet Mixture Model...")
    print("-" * 33)

    dmm = DirichletMixture(n_components=2, verbose=False, random_state=42)
    dmm.fit(data)

    print("Model fitted successfully!")
    print()

    # Get C-based classifications (from fitting process)
    c_labels = dmm.result_.get_best_component()
    print(f"C-based classifications obtained: {len(c_labels)} labels")

    # Get Python-based classifications (using predict method)
    python_labels = dmm.predict(data)
    print(f"Python-based classifications obtained: {len(python_labels)} labels")
    print()

    # Also get probabilities for detailed analysis
    c_probabilities = dmm.result_.group_assignments
    python_probabilities = dmm.predict_proba(data)

    # Compare the classifications
    disagreements = compare_classifications(c_labels, python_labels, data, true_labels, count_depths)

    # Analyze probability patterns if there are disagreements
    if np.sum(disagreements) > 0:
        analyze_probability_patterns(c_probabilities, python_probabilities, disagreements,
                                   data, true_labels, count_depths)

        # Show detailed examples
        show_disagreement_examples(c_labels, python_labels, c_probabilities, python_probabilities,
                                  disagreements, data, true_labels, count_depths)

    # Assess practical impact
    assess_practical_impact(c_labels, python_labels, true_labels, disagreements)

    # Model summary
    summary = dmm.result_.summary()
    print(f"Model Summary:")
    print(f"  BIC: {summary['BIC']:.2f}")
    print(f"  AIC: {summary['AIC']:.2f}")
    print(f"  Mixture weights: {summary['mixture_weights']}")

    print("\nClassification comparison completed successfully!")


if __name__ == "__main__":
    main()
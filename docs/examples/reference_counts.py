#!/usr/bin/env python3
"""
Example with specific reference count patterns for two groups.

This example creates synthetic data based on user-specified reference counts:
- Group 1: [200, 100, 50, 0, 10, 0]
- Group 2: [0, 10, 50, 200, 100, 3]

We convert these to proportions and sample 1000 times to create realistic data.
"""

import numpy as np
import pandas as pd
from pydmm import DirichletMixture

def create_reference_based_data():
    """
    Create synthetic data based on reference count patterns.

    Group 1: High in features 0-2, low in features 3-5
    Group 2: Low in features 0-2, high in features 3-5
    """
    # Reference count patterns
    ref_counts_group1 = np.array([200, 100, 50, 0, 10, 0])
    ref_counts_group2 = np.array([0, 10, 50, 200, 100, 3])

    print("Reference count patterns:")
    print(f"Group 1: {ref_counts_group1}")
    print(f"Group 2: {ref_counts_group2}")

    # Convert to proportions
    props_group1 = ref_counts_group1 / ref_counts_group1.sum()
    props_group2 = ref_counts_group2 / ref_counts_group2.sum()

    print(f"\nGroup 1 proportions: {props_group1}")
    print(f"Group 2 proportions: {props_group2}")

    # Parameters for sampling
    n_samples_per_group = 500  # 500 samples per group = 1000 total
    n_features = 6

    # Generate samples by sampling from multinomial distributions
    np.random.seed(42)

    samples_group1 = []
    samples_group2 = []

    # Generate Group 1 samples
    for i in range(n_samples_per_group):
        # Random total count per sample (simulate sequencing depth variation)
        total_count = np.random.poisson(2000) + 500  # Mean ~2000, minimum 500
        counts = np.random.multinomial(total_count, props_group1)
        samples_group1.append(counts)

    # Generate Group 2 samples
    for i in range(n_samples_per_group):
        total_count = np.random.poisson(2000) + 500
        counts = np.random.multinomial(total_count, props_group2)
        samples_group2.append(counts)

    # Combine data
    all_samples = samples_group1 + samples_group2
    true_labels = np.array([0] * n_samples_per_group + [1] * n_samples_per_group)

    # Create DataFrame
    sample_names = [f"Sample_{i:03d}" for i in range(len(all_samples))]
    feature_names = [f"Feature_{i}" for i in range(n_features)]

    data = pd.DataFrame(all_samples, index=sample_names, columns=feature_names)

    return data, true_labels, props_group1, props_group2

def analyze_group_separation(data, true_labels, true_props):
    """
    Analyze how well separated the groups are in the actual data.
    """
    print("Data Separation Analysis:")
    print("=" * 27)

    # Calculate observed proportions for each group
    group1_data = data[true_labels == 0]
    group2_data = data[true_labels == 1]

    # Convert counts to proportions for each sample, then average
    group1_props = group1_data.div(group1_data.sum(axis=1), axis=0).mean()
    group2_props = group2_data.div(group2_data.sum(axis=1), axis=0).mean()

    print("Observed average proportions:")
    print("-" * 30)
    print(f"{'Feature':<10} {'True_G1':<10} {'Obs_G1':<10} {'True_G2':<10} {'Obs_G2':<10} {'Ratio':<8}")
    print("-" * 58)

    for i in range(len(group1_props)):
        feature = f"Feature_{i}"
        true_g1 = true_props[0][i]
        obs_g1 = group1_props.iloc[i]
        true_g2 = true_props[1][i]
        obs_g2 = group2_props.iloc[i]
        ratio = obs_g1 / obs_g2 if obs_g2 > 0.001 else float('inf')

        print(f"{feature:<10} {true_g1:<10.3f} {obs_g1:<10.3f} {true_g2:<10.3f} {obs_g2:<10.3f} {ratio:<8.1f}")

    print()

    # Show total counts summary
    total_counts = data.sum(axis=1)
    print(f"Total counts per sample:")
    print(f"  Mean: {total_counts.mean():.0f}")
    print(f"  Range: {total_counts.min():.0f} - {total_counts.max():.0f}")
    print(f"  Std: {total_counts.std():.0f}")

def display_classification_performance(predicted_labels, true_labels, group_assignments):
    """
    Display clustering performance using label-invariant metrics.
    Measures how well samples with the same true label cluster together.
    """
    # Calculate clustering purity: for each predicted cluster, what's the most common true label?
    n_samples = len(true_labels)
    n_correct = 0

    # For each predicted cluster, count the most common true label
    for pred_cluster in [0, 1]:
        mask = predicted_labels == pred_cluster
        if np.sum(mask) > 0:
            true_labels_in_cluster = true_labels[mask]
            # Count occurrences of each true label in this predicted cluster
            unique, counts = np.unique(true_labels_in_cluster, return_counts=True)
            # Add the count of the most common true label
            n_correct += np.max(counts)

    purity = n_correct / n_samples

    # Calculate clustering accuracy using the Hungarian/optimal assignment approach
    accuracy_original = np.mean(predicted_labels == true_labels)
    accuracy_flipped = np.mean(predicted_labels == (1 - true_labels))
    optimal_accuracy = max(accuracy_original, accuracy_flipped)

    # Determine which assignment is better
    if accuracy_flipped > accuracy_original:
        predicted_labels_corrected = 1 - predicted_labels
        assignment_note = "flipped (pred_0→true_1, pred_1→true_0)"
    else:
        predicted_labels_corrected = predicted_labels
        assignment_note = "original (pred_0→true_0, pred_1→true_1)"

    print(f"\nClustering Performance (Label-Invariant):")
    print("=" * 42)
    print(f"Clustering Purity: {purity:.3f} ({purity*100:.1f}%)")
    print(f"  - Measures how 'pure' each cluster is (best assignment per cluster)")
    print(f"Optimal Accuracy: {optimal_accuracy:.3f} ({optimal_accuracy*100:.1f}%)")
    print(f"  - Best possible accuracy with {assignment_note} assignment")

    # Confusion matrix using optimal assignment
    confusion = np.zeros((2, 2), dtype=int)
    for true_label, pred_label in zip(true_labels, predicted_labels_corrected):
        confusion[true_label, pred_label] += 1

    print(f"\nConfusion Matrix (Optimal Assignment):")
    print("      Pred_0  Pred_1")
    print(f"True_0   {confusion[0,0]:3d}     {confusion[0,1]:3d}")
    print(f"True_1   {confusion[1,0]:3d}     {confusion[1,1]:3d}")

    # Per-group accuracy
    group1_accuracy = confusion[0,0] / (confusion[0,0] + confusion[0,1]) if (confusion[0,0] + confusion[0,1]) > 0 else 0
    group2_accuracy = confusion[1,1] / (confusion[1,0] + confusion[1,1]) if (confusion[1,0] + confusion[1,1]) > 0 else 0

    print(f"\nPer-group accuracy:")
    print(f"Group 1: {group1_accuracy:.3f} ({group1_accuracy*100:.1f}%)")
    print(f"Group 2: {group2_accuracy:.3f} ({group2_accuracy*100:.1f}%)")

    # Assignment confidence analysis
    max_probs = group_assignments.max(axis=1)
    high_confidence = (max_probs > 0.9).sum()
    medium_confidence = ((max_probs > 0.7) & (max_probs <= 0.9)).sum()
    low_confidence = (max_probs <= 0.7).sum()

    print(f"\nAssignment confidence:")
    print(f"High confidence (>90%): {high_confidence:3d} samples ({high_confidence/len(max_probs)*100:.1f}%)")
    print(f"Medium confidence (70-90%): {medium_confidence:3d} samples ({medium_confidence/len(max_probs)*100:.1f}%)")
    print(f"Low confidence (<70%): {low_confidence:3d} samples ({low_confidence/len(max_probs)*100:.1f}%)")

    # Return corrected labels for use in other parts
    return predicted_labels_corrected

def main():
    print("pyDMM - Reference Count Pattern Example")
    print("=" * 40)
    print("Testing with user-specified reference count patterns")
    print()

    # Create data based on reference counts
    data, true_labels, props_group1, props_group2 = create_reference_based_data()
    true_props = [props_group1, props_group2]

    print(f"\nGenerated data shape: {data.shape}")
    print(f"True group distribution: {np.bincount(true_labels)}")
    print()

    # Analyze separation in the data
    analyze_group_separation(data, true_labels, true_props)

    print(f"\nData preview (first 5 samples):")
    print(data.head())
    print()

    # Model selection: Test different numbers of clusters
    print("Model Selection: Testing different numbers of clusters...")
    print("=" * 57)

    cluster_options = [1, 2, 3, 4]
    bic_scores = []
    models = {}

    for n_clusters in cluster_options:
        print(f"Fitting model with {n_clusters} cluster{'s' if n_clusters != 1 else ''}...", end=" ")
        try:
            dmm_temp = DirichletMixture(n_components=n_clusters, verbose=False, random_state=42)
            dmm_temp.fit(data)

            bic = dmm_temp.result.goodness_of_fit["BIC"]
            bic_scores.append(bic)
            models[n_clusters] = dmm_temp

            print(f"BIC: {bic:.1f}")

        except Exception as e:
            print(f"Failed ({str(e)[:30]}...)")
            bic_scores.append(float('inf'))
            models[n_clusters] = None

    # Find optimal number of clusters (lowest BIC)
    best_k_idx = np.argmin(bic_scores)
    best_k = cluster_options[best_k_idx]
    best_bic = bic_scores[best_k_idx]

    print(f"\nBIC Comparison:")
    print("-" * 25)
    for i, k in enumerate(cluster_options):
        bic = bic_scores[i]
        marker = " ← BEST" if k == best_k else ""
        if bic == float('inf'):
            print(f"{k} cluster{'s' if k != 1 else ''}:  Failed{marker}")
        else:
            print(f"{k} cluster{'s' if k != 1 else ''}:  {bic:.1f}{marker}")

    print(f"\nOptimal number of clusters: {best_k} (BIC: {best_bic:.1f})")
    print("=" * 57)

    # Use the best model for detailed analysis
    print(f"\nUsing best model ({best_k} clusters) for detailed analysis...")
    dmm = models[best_k]

    if dmm is None:
        print("ERROR: Best model failed to fit. Using 2 clusters as fallback.")
        dmm = DirichletMixture(n_components=2, verbose=True, random_state=42)
        dmm.fit(data)

    print("Model analysis ready!")
    print()

    # Model summary
    print("Model Summary:")
    print("-" * 15)
    summary = dmm.result.summary()
    for key, value in summary.items():
        if key == "goodness_of_fit":
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:.3f}")
        elif isinstance(value, np.ndarray):
            if key == "mixture_weights":
                print(f"{key}: {[f'{w:.3f}' for w in value]}")
            else:
                print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")
    print()

    # Get predictions
    predicted_labels = dmm.result.get_best_component()
    group_assignments = dmm.result.get_group_assignments_df()

    # Display performance and get corrected labels
    predicted_labels_corrected = display_classification_performance(predicted_labels, true_labels, group_assignments)

    # Show estimated parameters vs true patterns
    print(f"\nEstimated Parameters vs True Patterns:")
    print("=" * 39)
    param_estimates = dmm.result.get_parameter_estimates_df()["Estimate"]

    print(f"{'Feature':<10} {'True_G1':<10} {'Est_G1':<10} {'True_G2':<10} {'Est_G2':<10}")
    print("-" * 50)

    for i in range(len(props_group1)):
        feature = f"Feature_{i}"
        true_g1 = props_group1[i]
        true_g2 = props_group2[i]
        est_g1 = param_estimates.iloc[i, 0]
        est_g2 = param_estimates.iloc[i, 1]

        print(f"{feature:<10} {true_g1:<10.3f} {est_g1:<10.3f} {true_g2:<10.3f} {est_g2:<10.3f}")

    # Show some sample assignments with confidence
    print(f"\nSample Assignment Examples:")
    print("-" * 28)
    print(f"{'Sample':<12} {'True':<6} {'Pred':<6} {'Conf_0':<8} {'Conf_1':<8} {'Max_Conf':<9} {'Match':<6}")
    print("-" * 57)

    # Show first 10 samples
    for i in range(10):
        sample_name = data.index[i]
        true_label = true_labels[i]
        pred_label = predicted_labels_corrected[i]
        conf_0 = group_assignments.iloc[i, 0]
        conf_1 = group_assignments.iloc[i, 1]
        max_conf = max(conf_0, conf_1)
        match = "✓" if true_label == pred_label else "✗"

        print(f"{sample_name:<12} {true_label:<6} {pred_label:<6} {conf_0:<8.3f} {conf_1:<8.3f} {max_conf:<9.3f} {match:<6}")

    # Show some from the second group
    print("  ... (middle samples omitted) ...")
    for i in range(500, 510):  # Second group starts at index 500
        sample_name = data.index[i]
        true_label = true_labels[i]
        pred_label = predicted_labels_corrected[i]
        conf_0 = group_assignments.iloc[i, 0]
        conf_1 = group_assignments.iloc[i, 1]
        max_conf = max(conf_0, conf_1)
        match = "✓" if true_label == pred_label else "✗"

        print(f"{sample_name:<12} {true_label:<6} {pred_label:<6} {conf_0:<8.3f} {conf_1:<8.3f} {max_conf:<9.3f} {match:<6}")

    print(f"\nExample completed successfully!")

    # Final summary based on clustering performance
    accuracy = np.mean(predicted_labels_corrected == true_labels)

    print(f"\nFinal Clustering Assessment:")
    print("=" * 30)
    if accuracy > 0.95:
        print("🎉 Excellent clustering performance!")
        print("   Algorithm perfectly separated the compositional groups")
    elif accuracy > 0.85:
        print("🎯 Good clustering performance!")
        print("   Algorithm successfully identified most group differences")
    elif accuracy > 0.70:
        print("👍 Decent clustering performance.")
        print("   Algorithm found meaningful group structure")
    else:
        print("⚠️ Poor clustering performance.")
        print("   Groups may not be sufficiently distinct for reliable clustering")

if __name__ == "__main__":
    main()
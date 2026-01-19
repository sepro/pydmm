#!/usr/bin/env python3
"""
Example demonstrating human-readable component labeling.

This example shows how to assign meaningful labels to mixture components
for better interpretability of clustering results, particularly useful
in biological contexts (e.g., 'Healthy', 'Diseased', 'Control').
"""

import numpy as np
import pandas as pd
from pydmm import DirichletMixture


def create_sample_data():
    """
    Create synthetic compositional data representing three distinct groups:
    - Group 0 (Healthy): High in features 0-1
    - Group 1 (Diseased): High in features 2-3
    - Group 2 (Control): Balanced across features
    """
    np.random.seed(42)

    # Define reference proportions for each group
    props_healthy = np.array([0.40, 0.35, 0.15, 0.10])
    props_diseased = np.array([0.10, 0.15, 0.40, 0.35])
    props_control = np.array([0.25, 0.25, 0.25, 0.25])

    n_samples_per_group = 50
    samples = []

    # Generate samples for each group
    for props in [props_healthy, props_diseased, props_control]:
        for _ in range(n_samples_per_group):
            total_count = np.random.poisson(1000) + 500
            counts = np.random.multinomial(total_count, props)
            samples.append(counts)

    # Create DataFrame with sample names
    sample_names = [f"Sample_{i:03d}" for i in range(len(samples))]
    feature_names = ["Gene_A", "Gene_B", "Gene_C", "Gene_D"]

    data = pd.DataFrame(samples, index=sample_names, columns=feature_names)

    return data


def main():
    print("pyDMM - Component Labeling Example")
    print("=" * 40)
    print()

    # Create sample data
    print("Creating synthetic compositional data...")
    data = create_sample_data()
    print(f"Data shape: {data.shape}")
    print(f"\nFirst 5 samples:")
    print(data.head())
    print()

    # Fit the model without labels
    print("Fitting Dirichlet Mixture Model with 3 components...")
    dmm = DirichletMixture(n_components=3, verbose=False, random_state=42)
    dmm.fit(data)
    print("Model fitted successfully!")
    print()

    # Show results WITHOUT labels
    print("=" * 40)
    print("Results WITHOUT component labels:")
    print("=" * 40)

    # Get cluster assignments (numeric)
    labels_numeric = dmm.result.get_best_component()
    print(f"\nCluster assignments (first 10):")
    print(labels_numeric[:10])

    # Get assignment probabilities (numeric column names)
    probs_numeric = dmm.result.get_group_assignments_df()
    print(f"\nAssignment probabilities (first 5 samples):")
    print(probs_numeric.head())
    print()

    # Show predictions without labels
    new_data = np.array([
        [450, 400, 100, 50],   # Healthy-like
        [100, 150, 450, 400],  # Diseased-like
        [250, 250, 250, 250],  # Control-like
    ], dtype=np.int32)

    predictions_numeric = dmm.predict(new_data)
    print(f"Predictions for new samples (numeric): {predictions_numeric}")
    print()

    # Now assign meaningful labels
    print("=" * 40)
    print("Setting human-readable component labels...")
    print("=" * 40)

    dmm.result.set_component_labels({
        0: 'Healthy',
        1: 'Diseased',
        2: 'Control'
    })
    print("Labels set: {0: 'Healthy', 1: 'Diseased', 2: 'Control'}")
    print()

    # Show results WITH labels
    print("=" * 40)
    print("Results WITH component labels:")
    print("=" * 40)

    # Get cluster assignments (labeled)
    labels_named = dmm.result.get_best_component()
    print(f"\nCluster assignments (first 10):")
    print(labels_named[:10])

    # Get assignment probabilities (labeled column names)
    probs_named = dmm.result.get_group_assignments_df()
    print(f"\nAssignment probabilities (first 5 samples):")
    print(probs_named.head())
    print()

    # Show predictions with labels
    predictions_named = dmm.predict(new_data)
    print(f"Predictions for new samples (labeled): {predictions_named}")

    # Get prediction probabilities with labels
    probs_new = dmm.predict_proba(new_data)
    print(f"\nPrediction probabilities for new samples:")
    print(probs_new)
    print()

    # Show cluster distribution with labels
    print("=" * 40)
    print("Cluster Distribution Summary:")
    print("=" * 40)
    unique_labels, counts = np.unique(labels_named, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count} samples ({count/len(labels_named)*100:.1f}%)")
    print()

    # Show model goodness of fit
    print("=" * 40)
    print("Model Goodness of Fit:")
    print("=" * 40)
    gof = dmm.result.goodness_of_fit
    print(f"  BIC: {gof['BIC']:.1f}")
    print(f"  AIC: {gof['AIC']:.1f}")
    print()

    # Practical usage example
    print("=" * 40)
    print("Practical Usage Example:")
    print("=" * 40)
    print("\n# In your analysis pipeline:")
    print("dmm = DirichletMixture(n_components=3)")
    print("dmm.fit(data)")
    print("dmm.result.set_component_labels({")
    print("    0: 'Healthy',")
    print("    1: 'Diseased',")
    print("    2: 'Control'")
    print("})")
    print("\n# All subsequent operations use labels:")
    print("labels = dmm.result.get_best_component()")
    print("# Returns: ['Healthy', 'Diseased', ...]")
    print()
    print("predictions = dmm.predict(new_samples)")
    print("# Returns: ['Healthy', 'Control', ...]")
    print()
    print("probabilities = dmm.predict_proba(new_samples)")
    print("# DataFrame columns: 'Healthy', 'Diseased', 'Control'")
    print()

    print("Example completed successfully!")
    print()

    # Note about label interpretation
    print("=" * 40)
    print("Important Notes:")
    print("=" * 40)
    print("1. Labels should be set AFTER fitting the model")
    print("2. Labels are applied BEFORE making predictions")
    print("3. Component indices (0, 1, 2, ...) map to your labels")
    print("4. Label mapping is preserved across predict() calls")
    print("5. Use meaningful labels for your domain:")
    print("   - Biology: 'Healthy', 'Diseased', 'Treated'")
    print("   - Customer segments: 'High-value', 'Medium', 'Low'")
    print("   - Behavior: 'Active', 'Occasional', 'Inactive'")


if __name__ == "__main__":
    main()

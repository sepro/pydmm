# Assignment calibration

Are pyDMM's cluster assignment probabilities over-confident, and does tempering fix it?
Results below are from `results/` (20 Nov 2026, 10 replicates per setting, N = 300, K = 2).

## Files

| File | Purpose |
|---|---|
| `experiment.py` | Over-confidence vs feature redundancy and correlation; candidate fixes (S_eff tempering, `temperature="auto"`, oracle τ, feature aggregation) |
| `tempered_em.py` | Tempered assignments (`temperature="auto"`) vs tempered EM (τ also in the E-step); max-probability histograms |
| `tempered_em.patch` | Experimental C change adding `temperature` to the E-step. Not part of the package; apply it and rebuild to run `tempered_em.py` |

```bash
venv/bin/python validation/correlation/experiment.py
git apply validation/correlation/tempered_em.patch && python setup.py build_ext --inplace
venv/bin/python validation/correlation/tempered_em.py
git checkout src/pydmm/ && python setup.py build_ext --inplace
```

## Simulations

* **A, duplication**: DMM data (S = 30, depth 3,000), every column copied m times. No information is added, so nothing should change.
* **B, correlated modules**: logistic-normal mixture, 60 features in 10 modules of 6, 4 modules shift between the groups, within-module correlation ρ, depth 5,000.

## Why the probabilities are over-confident

1. The log-odds between components is a sum with one term per feature. Correlated features each add a term, so the same signal is counted several times. Copying every feature m times multiplies the log-odds by m.
2. The Dirichlet fixes each feature's variance on the log scale (≈ ψ′(α_j) − ψ′(α_0)). Abundant features vary about 2× more than the model allows (B, ρ = 0), so their fluctuations are read as strong evidence. This alone produces confident wrong splits: in B, 2–4 of 10 replicates per ρ have accuracy below 70% at about 98% confidence. Eight random restarts give the same partition and NLE, so these are global optima.

## Results

B, mean over 10 replicates:

| ρ | Accuracy | Confidence (original) | Log-loss (original) | τ (`"auto"`) | Confidence (`"auto"`) | Log-loss (`"auto"`) | ECE original → `"auto"` |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.814 | 0.985 | 2.29 | 0.37 | 0.887 | 0.40 | 0.174 → 0.078 |
| 0.3 | 0.839 | 0.984 | 1.61 | 0.25 | 0.878 | 0.34 | 0.150 → 0.056 |
| 0.6 | 0.809 | 0.984 | 2.19 | 0.16 | 0.846 | 0.41 | 0.179 → 0.055 |
| 0.9 | 0.732 | 0.989 | 4.53 | 0.08 | 0.786 | 0.58 | 0.261 → 0.084 |

In A, `"auto"` gives τ = 0.94, 0.48, 0.31, 0.12 for m = 1, 2, 4, 8 (expected ≈ 1/m) and brings log-loss back to the m = 1 value. The same holds for K = 3 and 4.

Share of samples with max probability > 0.99 (B): 89%, 87%, 88%, 92% originally; 46%, 35%, 22%, 11% with `"auto"`. At ρ = 0.9, 749 of 804 misassigned samples had max probability > 0.95 originally, 66 after tempering.

Other findings:

* Tempering the mixture weights as well as the likelihood, z ∝ (w·p)^τ, beat tempering the likelihood only on accuracy, ECE and log-loss at every ρ, and keeps each sample's most likely component. The package uses it.
* τ = S_eff/S (participation ratio of the within-cluster CLR correlation) is exact for duplicated features but misses the variance problem: log-loss 1.54 / 0.89 / 0.83 / 1.27.
* Summing correlated features before fitting did not help in B and lowered accuracy at ρ = 0.9 (0.693): summed modules are more abundant, so the Dirichlet over-trusts them more.
* One replicate (A, m = 4, seed 5) ended with every responsibility at 0.5: EM stalled at the symmetric saddle point.

## Tempered EM (not adopted)

Using τ in the E-step as well, so α and the partition can change:

| ρ | Accuracy: original / `"auto"` / tempered EM | Wrong splits | Log-loss: `"auto"` / tempered EM |
|---|---|---|---|
| 0.0 | 0.814 / 0.814 / 0.753 | 4 / 4 / 5 | 0.40 / 1.69 |
| 0.3 | 0.839 / 0.839 / 0.788 | 2 / 2 / 3 | 0.34 / 0.78 |
| 0.6 | 0.809 / 0.809 / 0.731 | 2 / 2 / 5 | 0.41 / 0.63 |
| 0.9 | 0.732 / 0.732 / 0.594 | 3 / 3 / 8 | 0.58 / 0.94 |

With fixed τ from 0.5 to 0.05, accuracy decreases monotonically in every setting and reaches 0.516 (components merged) at τ ≤ 0.1. Softened responsibilities blend the clusters in the M-step, which makes the components more similar and the next E-step softer. Tempering also scales all features equally, so it cannot change which features drive a wrong split.

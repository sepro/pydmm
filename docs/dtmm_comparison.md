# Dirichlet-tree mixtures versus Dirichlet mixtures

This document compares `DirichletTreeMixture` (DTMM) with `DirichletMixture` (DMM):
what the tree model changes, how it was checked, where it does better and why,
where it does not, and what it costs in runtime. All numbers come from
`validation/dtmm/compare.py` and `validation/dtmm/benchmark.py` (20 Nov 2026).

## Summary

* **Use the DTMM when taxa move together in groups that follow a tree**, such as
  clades that bloom as a whole, or when features differ a lot in how variable
  they are. With the true tree it matched an oracle that knows the generating
  parameters, while the DMM was strongly over-confident (log-loss 0.41 vs 1.45).
  With a tree built from the data it was the only model without wrong splits on
  logistic-normal data with independent features (accuracy 0.945 vs 0.896).
* **Node selection is what makes it work.** Fitting every node separately per
  component lost accuracy in every tree scenario (0.746 vs 0.811 with strong
  blooms). Selection keeps only the nodes that separate the clusters and also
  reports them (`result_.differential_nodes()`).
* **Prefer a real taxonomy over the default data-built tree.** The data-built
  tree is good when features vary independently, but weaker than the true tree
  when whole clades bloom strongly.
* **It is not a universal improvement.** On logistic-normal data with strongly
  correlated modules, and on wide trees where only a few of many clades differ,
  the DTMM clustered worse than the DMM, although BIC preferred it. On plain DMM
  data it is as good as the DMM, and BIC then prefers the DMM.
* **On real data use `n_clades` and `n_bootstrap`.** The default binary data
  tree gave clusters that changed between data subsets (78–85% agreement against
  92% for the DMM). A shallow 150-clade tree was as reproducible as the DMM, and
  bagging made the probabilities realistic. See "Stability on real data".
* **Runtime**: about 1.5× a DMM fit with a taxonomy-like tree and about 2.7×
  with the default data-built binary tree (range 1.1–3.7× over N, S and K).

## The model

A tree has the S features as leaves. In component k, every internal node v
splits the count arriving at v over its children c(v) with its own
Dirichlet-multinomial:

```
log p(x_i | k) = Σ_v log DM(n_i,c(v) | α_kv)        (+ multinomial coefficient)
```

`n_i,c(v)` are sample i's counts summed under each child of v. A star tree (root
with all features as children) gives exactly the DMM. The multinomial
coefficients of the nodes multiply to n!/∏x_j!, so NLE, BIC, AIC and Laplace
are on the same scale as the DMM's and can be compared directly.

Compared with the DMM this changes two things:

1. **Correlation.** A clade's total is drawn once at its parent, so the taxa in
   a clade can rise and fall together. Under a Dirichlet, features can only be
   negatively correlated, so a co-blooming group contributes one log-likelihood
   term per member and is counted several times.
2. **Dispersion.** Every node has its own precision. In the DMM one total α_0
   fixes how much each feature varies (Var log p_j ≈ ψ′(α_j) − ψ′(α_0)), which
   over-trusts abundant features. With a binary tree every split gets its own
   variance.

### Node selection

With K components every node has K parameter vectors. Most nodes usually do not
differ between clusters, and fitting them separately adds noise to every
sample's log-odds. After EM converges, each node is scored:

```
gain_v = Σ_k Σ_i z_ik log DM(n_i,c(v) | α_kv) − Σ_i log DM(n_i,c(v) | α_v,shared)
keep component-specific  if  gain_v > ½ · log N · (K − 1) · |c(v)|
```

The penalty is the BIC cost of the extra parameters. Shared nodes are fitted
once on all samples; their likelihood is the same for every component, so they
drop out of the assignment step. EM is rerun with the new set, until the set
stops changing (at most 5 rounds; 1–3 in practice). If no node passes, the one
closest to passing is kept so the components can still differ. This follows
the idea in Mao & Ma (2022), who select differential nodes with spike-and-slab
priors in a Bayesian model, but stays within pyDMM's EM framework.

## Checks

| Check | Result |
|---|---|
| Star tree without node selection vs `DirichletMixture`, same seed | NLE, BIC, AIC equal to 1e-8 relative; assignments within 1e-4 (`tests/test_dtmm.py`) |
| NLE from C vs likelihood + prior recomputed in Python at the fitted parameters | equal to 1e-9 relative, with and without selection |
| C fit vs NumPy reference (`validation/dtmm/reference.py`) at the same parameters | identical NLE |
| Groups that differ only at the root of a clade tree | node selection kept exactly the root, and nothing else, in all 40 replicates of the S = 60 tree scenarios |
| All components equal after EM (symmetric saddle point) | the DTMM detects it and restarts from hard k-means with k-means++ seeding (`tests/test_dtmm.py::TestCollapsedStart`). The DMM does the same restart only with `restart_collapsed=True` (off by default) |
| `DirichletMixture` output after moving its node routines to `dirichlet_node.c` | bit-identical to before (saved baselines, including WAIC) |
| Existing test suite | all 145 tests pass unchanged; 49 new tests in `tests/test_dtmm.py` |

## Synthetic comparison

N = 300, S = 60, 10 replicates per scenario, K = 2 unless stated.

* **Tree data**: 10 clades of 6 taxa. The groups differ in 4 clade totals
  (shift ±0.8 on the log scale); the composition within each clade is the same
  in both groups. θ_root is the precision of the clade totals: 100 = mild,
  10 = strong blooms of whole clades. *Tree, K = 3* adds a third group with
  shifts in 4 other clades. *Wide tree* has 40 clades (S = 240) with the same
  4 differential clades, so most of the tree carries no signal.
* **DMM data**: generated by a plain DMM, so the tree has nothing to add.
* **Logistic-normal**: the correlated-module data from `validation/correlation/`
  (10 modules of 6 features, within-module correlation ρ). Neither model is
  correct for it.

Fits: the DMM, the DMM with `temperature="auto"`, the DTMM with the generating
clade tree (with and without node selection), the DTMM with its default
data-built tree, and for tree data an oracle using the true parameters.

### Accuracy · log-loss (wrong splits out of 10)

Log-loss is the mean −log probability of the true cluster (lower = better
calibrated). A wrong split is a replicate with accuracy below 70% (K = 2) or
60% (K = 3).

| Scenario | DMM | DMM, `temperature="auto"` | DTMM, true tree | DTMM, true tree, no selection | DTMM, data tree | Oracle |
|---|---|---|---|---|---|---|
| Tree, θ = 100 | .989 · 0.060 | .989 · 0.031 | **.989 · 0.030** | .987 · 0.035 | .987 · 0.034 | .989 · 0.029 |
| Tree, θ = 30 | .909 · 0.611 | .909 · 0.216 | **.912 · 0.215** | .887 · 0.281 | .905 · 0.237 | .915 · 0.205 |
| Tree, θ = 10 | .811 · 1.454 | .811 · 0.426 | **.809 · 0.410** | .746 · 0.756 (3) | .776 · 0.527 (1) | .830 · 0.377 |
| Tree, K = 3, θ = 30 | .889 · 0.795 | .889 · **0.281** | **.890** · 0.285 | .846 · 0.479 | .868 · 0.987 | .900 · 0.259 |
| Wide tree (S = 240), θ = 30 | **.735** · 2.265 (2) | **.735 · 0.783** (2) | .669 · 1.200 (5) | .648 · 2.421 (6) | .629 · 1.465 (6) | .839 · 0.355 |
| DMM data | **.993 · 0.020** | .993 · 0.020 | .993 · 0.020 | .993 · 0.020 | .990 · 0.031 | – |
| Logistic-normal, ρ = 0 | .896 · 1.181 (2) | .896 · 0.283 (2) | .845 · 1.627 (3) | .832 · 1.718 (3) | **.945 · 0.153** | – |
| Logistic-normal, ρ = 0.6 | **.825** · 1.880 (1) | **.825 · 0.417** (1) | .688 · 1.868 (4) | .680 · 2.011 (5) | .737 · 1.112 (3) | – |

For DMM data the "true tree" is the star tree, i.e. the DMM itself.

### Expected calibration error and confidence

| Scenario | DMM: confidence / ECE | DMM auto | DTMM true tree | DTMM data tree | Oracle |
|---|---|---|---|---|---|
| Tree, θ = 30 | 0.986 / 0.081 | 0.917 / 0.027 | 0.921 / 0.026 | 0.920 / 0.034 | 0.918 / 0.025 |
| Tree, θ = 10 | 0.978 / 0.169 | 0.841 / 0.065 | 0.847 / 0.049 | 0.852 / 0.089 | 0.833 / 0.034 |
| Logistic-normal, ρ = 0 | 0.986 / 0.094 | 0.956 / 0.070 | 0.973 / 0.133 | 0.970 / 0.030 | – |
| Logistic-normal, ρ = 0.6 | 0.983 / 0.161 | 0.896 / 0.082 | 0.937 / 0.254 | 0.922 / 0.191 | – |

A calibrated model has confidence ≈ accuracy and ECE near 0.

### Model selection and selected nodes

| Scenario | BIC prefers DTMM (true tree) | BIC prefers DTMM (data tree) | Specific nodes, true tree | Specific nodes, data tree |
|---|---|---|---|---|
| Tree, θ = 100 / 30 / 10 | 10/10 each | 10/10 each | 1.0 of 11 | 7.0 / 6.0 / 6.2 of 59 |
| Tree, K = 3 | 10/10 | 10/10 | 1.0 of 11 | 7.6 of 59 |
| Wide tree (S = 240) | 10/10 | 10/10 | 1.2 of 41 | 6.6 of 239 |
| DMM data | 3/10 (star tree = DMM) | 0/10 | 1 of 1 | 33.7 of 59 |
| Logistic-normal, ρ = 0 | 10/10 | 10/10 | 1.3 of 11 | 9.7 of 59 |
| Logistic-normal, ρ = 0.6 | 10/10 | 10/10 | 1.5 of 11 | 8.0 of 59 |

## Where and why the DTMM does better

**Clade blooms (tree data).** The DMM finds the right clusters (accuracy close
to the oracle) but is far too sure of them: with strong blooms it reports 98%
confidence at 81% accuracy. A clade of 6 taxa that blooms together adds six
roughly equal terms to its log-odds. The DTMM draws the clade total once at the
root, so the bloom counts once, and its probabilities match the oracle's
(ECE 0.049 vs 0.034 for the oracle, 0.169 for the DMM). Node selection kept
exactly one node, the root, in all 40 tree replicates: it found the level at
which the groups differ.

**Uneven dispersion (logistic-normal, ρ = 0).** Here every feature has the same
variance on the log scale, which a Dirichlet cannot represent: it gives abundant
features too little variance, reads their fluctuations as evidence and finds a
wrong split in 2 of 10 replicates with ~98% confidence. Tempering cannot fix a
wrong split. The data-built binary tree gives every split its own precision, so
each feature's variance is fitted separately; node selection then keeps the ~10
splits that separate the groups. Result: no wrong splits, accuracy 0.945 vs
0.896, log-loss 0.153 vs 1.181. The true (6-taxa clade) tree does not help here,
because within each clade one Dirichlet again ties the variance to abundance.

**Interpretation.** `result_.differential_nodes()` lists the clades whose
composition separates the clusters. With the true tree this was always the root,
the level the simulation changed.

**Model comparison.** NLE, BIC and AIC are on the DMM's scale. BIC preferred
the DTMM in every tree and logistic-normal replicate and preferred the DMM on
DMM data when the DTMM used a data-built tree, so BIC can choose between the two
models and between trees on the same data.

## Where it does not

**No selection.** Without node selection the DTMM was worse than the DMM in
accuracy in every tree scenario (e.g. 0.746 vs 0.811), with 3 wrong splits at
θ = 10. Nodes that do not differ between clusters are still fitted per cluster,
and their estimation noise enters every sample's log-odds. Keep
`node_selection=True` unless you need all parameters per component.

**Strongly correlated modules (logistic-normal, ρ = 0.6).** Both trees clustered
worse than the DMM (0.737 with the data tree vs 0.825). The shared module
fluctuations are as large as the group difference, and the tree model, which
fits these module totals well, also finds splits along them. BIC preferred the
DTMM in all 10 replicates because it describes the counts much better, which
shows that a better density fit does not guarantee better clusters. Here the
DMM with `temperature="auto"` was the best option.

**Wide trees with little signal.** With 40 clades of which 4 differ (S = 240),
the DTMM with the true tree found a wrong split in 5 of 10 replicates, the DMM
in 2 (accuracy 0.669 vs 0.735). The oracle reaches 0.839, so the information
is there. Most of the root node's 40 clade totals vary at random, and with
their own dispersion fitted they offer splits that fit the counts nearly as
well as the true one; the likelihood sometimes prefers them (BIC preferred the
DTMM in all 10 replicates). Its probabilities were still better calibrated than
the plain DMM's (log-loss 1.20 vs 2.27), but the DMM with `temperature="auto"`
was better on every measure here. A taxonomy with more levels, so that the
differential clades are not 4 of 40 children of one node, may help; this was
not tested.

**Data-built tree with strong blooms or more clusters.** The default tree is
built from feature correlations over all samples. With strong clade blooms
(θ = 10) or three groups it does not recover the clades perfectly. Its accuracy
then falls below the DMM's (0.776 vs 0.811 at θ = 10; 0.868 vs 0.889 with three
groups), and with three groups its log-loss (0.987) is worse than the DMM with
tempering (0.281). A taxonomy or phylogeny, when available, should be preferred.

**Calibration alone.** If the clusters are right and only the probabilities are
too confident, the DMM with `temperature="auto"` gets close to the DTMM
(log-loss 0.426 vs 0.410 at θ = 10) at the DMM's cost. The DTMM's advantages are
the fixed wrong splits, the differential nodes and the better density fit.

## Stability on real data: clade trees and bagging

*Added 23 Nov 2026.* The synthetic comparison above does not test whether the
clusters are reproducible. On a real gut microbiome cohort (3005 samples, 400
genera, rarefied to 10,000 reads, K = 4; the data are not part of this repository)
the default DTMM was much less stable than the DMM, and its probabilities were
too confident.

**Protocol.** Every model was fitted to two disjoint halves of the samples (5
random splits) and to two disjoint thirds, evaluated on the remaining third
(5 splits). Clusters of the two fits were matched (Hungarian) and compared on
the evaluated samples:
*agreement* (share of samples with the same matched cluster), the lowest
Jaccard index of a matched cluster pair, and *flip ECE*: the expected
calibration error of each fit's maximum probability as a forecast of whether
the other fit puts the sample in the same cluster. Realistic probabilities give
a low flip ECE. Synthetic data with the cohort's properties (logistic-normal
with the cohort's CLR means and covariances, its zero patterns and depth, 4
replicates at full and half the observed separation) checked accuracy and ECE
against the truth.

| model (K = 4) | agreement split / held-out | lowest Jaccard | flip ECE (untempered) | flip ECE (one-step) |
|---|---|---|---|---|
| DMM | 0.918 / 0.918 | 0.81 / 0.81 | 0.053 / 0.052 | 0.047 / 0.056 |
| DTMM, default binary data tree | 0.849 / 0.784 | 0.69 / 0.57 | 0.110 / 0.168 | 0.043 / 0.081 |
| DTMM, `n_clades=150` | 0.908 / 0.909 | 0.79 / 0.77 | 0.061 / 0.057 | 0.043 / 0.047 |
| DTMM, `n_clades=150, n_bootstrap=8` | 0.905 / 0.912 | 0.80 / 0.79 | 0.038 / 0.030 | 0.052 / 0.067 |
| DMM, bagged (8) | 0.921 / 0.920 | 0.82 / 0.82 | 0.025 / 0.028 | 0.059 / 0.066 |

On the synthetic data all of these reached accuracy 0.92 (0.79 at half
separation). The bagged clade DTMM had ECE 0.066 untempered and 0.042 one-step,
against 0.066 and 0.037 for the DMM; at half separation it had 0.137 and 0.080,
against 0.167 and 0.101.

**What worked.**

* **A shallow clade tree** (`n_clades`): cut the correlation dendrogram into
  about one clade per 2–3 features. 10–50 clades did not help (agreement
  0.83–0.88), 100 gave 0.885/0.900, and 150 reached the DMM's stability. With 200
  or more clades only the root stayed specific, so the model became the DMM. With
  150, 6–7 clade nodes stay specific to the clusters (`differential_nodes()`).
  The fit agreed with the DMM on 96% of samples.
* **Bagging** (`n_bootstrap`): average the assignment probabilities over the fit
  and its bootstrap refits. Single fits, DMM or DTMM, give ~87% of samples a
  probability ≥ 0.95, but only 96% of those keep their cluster in another fit.
  After bagging, 72–78% get ≥ 0.95 and 98–99% of them keep it. Untempered bagged
  probabilities were the most realistic of everything tried. On this cohort,
  tempering the bagged probabilities made them under-confident (flip ECE
  0.05–0.07). On misspecified simulated data, however, bagging plus one-step
  tempering was best (`validation/bagging/compare.py`).
* Bagging works just as well for the DMM (last row; now
  `DirichletMixture(n_bootstrap=)`). The DTMM's remaining advantages are the differential clades and the
  better density fit, not better probabilities than a bagged DMM.

**What did not work** (prototyped in the C core, not merged; agreement
split-half / held-out against 0.849 / 0.784 for the default):

* Starting EM from the DMM's assignments: 0.850 / 0.772. Seeds and starts were
  not the problem: different starts converged to the same fit.
* A stronger node-selection penalty (×2: 0.829 / 0.763) or no selection
  (0.846 / 0.811).
* Shrinking each component's node parameters to the pooled fit (Gaussian prior on
  log α; precision 10: 0.827 / 0.793).
* One shared precision per node across components: 0.777 / 0.678.
* Sharing nodes with few expected reads (minimum node count 5 or 20): no change.
* More training data: in learning curves with n = 300–1200 the DTMM's stability
  did not catch up with the DMM's. A fixed tree built from all samples, or K = 5,
  did not help either.

The instability comes from the many parameters of a deep binary tree, each
with its own dispersion: a refit on other samples finds a different set of nodes
that separate the clusters. A shallow tree has far fewer nodes, and bagging turns
the remaining refit variation into probability instead of hiding it.

**Cost.** `n_bootstrap=B` costs B extra fits.

## Runtime

Single process, single thread, Dirichlet-tree data with clades of 6
(`validation/dtmm/benchmark.py`). Median of 3 fits (1 for the largest sizes).

| N | S | K | DMM | DTMM, star tree | DTMM, clade tree | DTMM, data tree | DTMM, data tree, no selection |
|---|---|---|---|---|---|---|---|
| 100 | 60 | 2 | 0.40 s | 0.34 s (0.8×) | 1.07 s (2.7×) | 0.91 s (2.3×) | 0.84 s (2.1×) |
| 300 | 60 | 2 | 1.74 s | 2.00 s (1.1×) | 2.29 s (1.3×) | 4.15 s (2.4×) | 3.19 s (1.8×) |
| 1,000 | 60 | 2 | 4.19 s | 3.82 s (0.9×) | 6.16 s (1.5×) | 13.13 s (3.1×) | 11.16 s (2.7×) |
| 3,000 | 60 | 2 | 16.72 s | 17.33 s (1.0×) | 32.54 s (1.9×) | 46.27 s (2.8×) | 38.75 s (2.3×) |
| 300 | 30 | 2 | 0.48 s | 0.55 s (1.1×) | 0.70 s (1.5×) | 0.98 s (2.0×) | 0.73 s (1.5×) |
| 300 | 120 | 2 | 6.21 s | 6.18 s (1.0×) | 9.32 s (1.5×) | 23.11 s (3.7×) | 18.60 s (3.0×) |
| 300 | 240 | 2 | 19.83 s | 20.96 s (1.1×) | 37.60 s (1.9×) | 54.13 s (2.7×) | 31.98 s (1.6×) |
| 300 | 60 | 3 | 15.58 s | 15.80 s (1.0×) | 16.82 s (1.1×) | 23.48 s (1.5×) | 13.21 s (0.8×) |
| 300 | 60 | 4 | 13.86 s | 13.65 s (1.0×) | 21.45 s (1.5×) | 27.96 s (2.0×) | 20.70 s (1.5×) |

Ratios are relative to the DMM fit on the same data. The clade tree has one
node per 6 taxa plus a root (S/6 + 1 nodes), like a genus-level taxonomy; the
data-built tree is binary (S − 1 nodes). Building the data tree took under
0.02 s in every case.

* **The tree code itself adds no cost.** With a star tree the DTMM takes the
  same time as the DMM (0.8–1.1×).
* **A taxonomy-like tree costs 1.1–2.7× a DMM fit**, typically about 1.5×. The
  ratio is highest for small N, where the fixed cost per node dominates.
* **A data-built binary tree costs 1.5–3.7×**, typically about 2.7×. It has
  S − 1 nodes, each fitted with its own BFGS run, and more features make the
  ratio larger.
* **Node selection adds 8–78%**, because every round refits each node shared
  and per component, then reruns EM. It costs most with many nodes or
  components (S = 240: 54 s vs 32 s; K = 3: 23 s vs 13 s).
* Runtime grows roughly linearly with N for all models (DMM 4.2 s → 16.7 s,
  DTMM clade tree 6.2 s → 32.5 s from N = 1,000 to 3,000).

## Limitations and possible extensions

* The soft k-means start shared with the DMM can put all components at the
  same point when there are many features (seen at N = 300, S = 240). The DTMM
  restarts from hard k-means when that happens; the DMM does so only with
  `restart_collapsed=True` (off by default).
* No WAIC for the DTMM yet. The Hessian is block-diagonal per node, so the
  Laplace approach used for the DMM (`waic.py`) can be extended.
* Node selection is a greedy BIC rule. Mao & Ma (2022) select nodes with
  spike-and-slab priors and report posterior inclusion probabilities.
* Trees can be given as a taxonomy table, SciPy linkage, parent array or built
  from the data; Newick input is not supported yet.
* Most fitting time is in per-element `gsl_sf_lngamma` / `gsl_sf_psi` calls in
  the shared node routines, which the DMM also uses. A dedicated two-parameter
  solver for binary nodes could speed up data-built trees.

## References

* Dennis, S.Y. (1991). On the hyper-Dirichlet type 1 and hyper-Liouville
  distributions. *Communications in Statistics – Theory and Methods* 20(12):
  4069–4081.
* Wang, T. & Zhao, H. (2017). A Dirichlet-tree multinomial regression model for
  associating dietary nutrients with gut microorganisms. *Biometrics* 73(3):
  792–801. doi:10.1111/biom.12654
* Mao, J. & Ma, L. (2022). Dirichlet-tree multinomial mixtures for clustering
  microbiome compositions. *Annals of Applied Statistics* 16(3): 1476–1499.
  doi:10.1214/21-AOAS1552
* Holmes, I., Harris, K. & Quince, C. (2012). Dirichlet multinomial mixtures:
  generative models for microbial metagenomics. *PLoS ONE* 7(2): e30126.

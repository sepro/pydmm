"""
Tempered (calibrated) cluster assignment probabilities.

The Dirichlet-multinomial log-likelihood is a sum with one term per feature.
Features that co-vary (taxa of one guild, ASVs of one strain) each add their own
term, and the Dirichlet ties every feature's variance to its mean, so abundant
features are trusted more than their real variability warrants. Both make the
log-odds between components too large and the assignment probabilities too
close to 0 or 1.

Tempering multiplies every log-odds by tau (0 < tau <= 1):

    z_ik ∝ (w_k * DM(x_i | alpha_k)) ** tau

The fitted alpha and w are not changed, and neither is each sample's most
likely component; only the probabilities move towards uniform. Tempering the
mixture weights as well (rather than only the likelihood) gave higher accuracy,
lower calibration error and lower log-loss in the simulations in
validation/correlation/, because the weights are estimated from the same
over-confident responsibilities.

``estimate_temperature`` chooses tau without labels from I.J. Good's
consistency condition for a weight of evidence: if the log-likelihood ratio
between components k and l is roughly normal for samples from k, and
calibrated, its variance equals twice its mean. A log-likelihood ratio inflated
by a factor c has its mean multiplied by c and its variance by c**2, so
tau = 2 * mean / var estimates 1 / c. ``temperature="auto"`` iterates this to a
fixed point (the weights are the tempered assignments); ``"one-step"`` stops
after the first step (the weights are the untempered assignments).

References:
    Good, I.J. (1985). Weight of evidence: a brief survey. Bayesian Statistics 2.
    Holmes, C.C. & Walker, S.G. (2017). Assigning a value to a power likelihood
    in a general Bayesian model. Biometrika 104(2).
"""

from typing import Optional

import numpy as np
from scipy.special import logsumexp

from .waic import _SparseCounts, _dm_log_likelihood

# Strings accepted for ``temperature`` besides None and a number.
TEMPERATURE_METHODS = ("auto", "one-step")

_TEMPERATURE_ERROR = 'temperature must be None, "auto", "one-step" or a number in (0, 1]'

# Smallest tau "auto" and "one-step" will return; below this the assignments
# are essentially the mixture weights and the estimate is no longer meaningful.
MIN_TEMPERATURE = 1e-3


def validate_temperature(temperature) -> None:
    """Raise ValueError unless temperature is None, "auto", "one-step" or a float in (0, 1]."""
    if temperature is None:
        return
    if isinstance(temperature, str):
        if temperature in TEMPERATURE_METHODS:
            return
        raise ValueError(_TEMPERATURE_ERROR)
    if isinstance(temperature, (bool, np.bool_)) or not isinstance(
        temperature, (int, float, np.integer, np.floating)
    ):
        raise ValueError(_TEMPERATURE_ERROR)
    if not (0.0 < float(temperature) <= 1.0):
        raise ValueError(_TEMPERATURE_ERROR)


def resolve_temperature(temperature, log_lik: np.ndarray, weights: np.ndarray) -> float:
    """
    The tau for a ``temperature`` setting that is not None.

    "auto" runs the fixed-point iteration of ``estimate_temperature`` to
    convergence, "one-step" only its first step; a number is returned as is.
    """
    if isinstance(temperature, str):
        max_iter = 1 if temperature == "one-step" else 200
        return estimate_temperature(log_lik, weights, max_iter=max_iter)
    return float(temperature)


def component_log_likelihood(X: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Dirichlet-multinomial log-likelihood of every sample under every component.

    The multinomial coefficient is left out; it is the same for all components
    and cancels in the assignment probabilities.

    Parameters:
        X: counts, shape (N, S)
        alpha: Dirichlet parameters, shape (K, S)

    Returns:
        np.ndarray of shape (N, K)
    """
    sc = _SparseCounts(np.asarray(X))
    return _dm_log_likelihood(sc, np.asarray(alpha, dtype=np.float64)).T


def tempered_responsibilities(
    log_lik: np.ndarray, weights: np.ndarray, temperature: float
) -> np.ndarray:
    """
    Assignment probabilities z_ik ∝ (w_k * exp(log_lik_ik)) ** temperature.

    Parameters:
        log_lik: per-component log-likelihoods, shape (N, K)
        weights: mixture weights, shape (K,)
        temperature: tau in (0, 1]; 1 gives the usual responsibilities

    Returns:
        np.ndarray of shape (N, K), rows sum to 1
    """
    with np.errstate(divide="ignore"):
        log_w = np.log(np.asarray(weights, dtype=np.float64))
    z = temperature * (np.asarray(log_lik, dtype=np.float64) + log_w[None, :])
    return np.exp(z - logsumexp(z, axis=1, keepdims=True))


def _pooled_weight_of_evidence(log_lik: np.ndarray, z: np.ndarray) -> Optional[float]:
    """
    One evaluation of tau = 2 * mean / var, pooled over ordered component pairs.

    For each component k (samples weighted by z_ik) and each competitor l, the
    log-likelihood ratio log p(x | k) - log p(x | l) gives one estimate. Pairs
    whose mean is not positive (k's samples do not favour k over l) or whose
    variance is zero carry no information and are skipped. Estimates are
    averaged with weight N_k = sum_i z_ik.
    """
    N, K = log_lik.shape
    estimates, weights = [], []
    for k in range(K):
        wk = z[:, k]
        n_k = wk.sum()
        if n_k < 2.0:
            continue
        for l in range(K):
            if l == k:
                continue
            llr = log_lik[:, k] - log_lik[:, l]
            mean = np.dot(wk, llr) / n_k
            var = np.dot(wk, (llr - mean) ** 2) / n_k
            if mean <= 0.0 or var <= 0.0 or not np.isfinite(var):
                continue
            estimates.append(2.0 * mean / var)
            weights.append(n_k)
    if not estimates:
        return None
    return float(np.average(estimates, weights=weights))


def estimate_temperature(
    log_lik: np.ndarray,
    weights: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> float:
    """
    Estimate the tempering factor tau from the data, without labels.

    Fixed-point iteration: compute tempered assignments with the current tau,
    re-estimate tau = 2 * mean / var of the per-pair log-likelihood ratios
    (Good's weight-of-evidence condition), repeat. The result is clipped to
    [MIN_TEMPERATURE, 1]: tau = 1 means no correction is needed.

    ``max_iter=1`` gives the one-step estimate (``temperature="one-step"``):
    Good's condition with the untempered assignments as weights. These weights
    do not depend on tau, so the estimate cannot drift. With heavy-tailed
    log-likelihood ratios (for example a DirichletTreeMixture on real data)
    the full iteration can drift to a tau that is much too small; the one-step
    estimate does not have this problem. DirichletTreeMixture therefore warns
    when it is used with temperature="auto".

    Assumes the log-likelihood ratios are roughly normal within a component.
    Tempering cannot repair a wrong partition, only lower its confidence.

    Parameters:
        log_lik: per-component log-likelihoods, shape (N, K)
        weights: mixture weights, shape (K,)
        max_iter: maximum number of fixed-point steps (1 = one-step estimate)
        tol: stop when tau changes by less than this

    Returns:
        float tau in [MIN_TEMPERATURE, 1]; 1.0 for a single component or when
        no component pair carries information.
    """
    log_lik = np.asarray(log_lik, dtype=np.float64)
    if log_lik.ndim != 2 or log_lik.shape[1] < 2:
        return 1.0
    tau = 1.0
    for _ in range(max_iter):
        z = tempered_responsibilities(log_lik, weights, tau)
        est = _pooled_weight_of_evidence(log_lik, z)
        if est is None:
            return 1.0
        new = float(np.clip(est, MIN_TEMPERATURE, 1.0))
        if abs(new - tau) < tol:
            return new
        tau = new
    return tau

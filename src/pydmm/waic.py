"""
WAIC (widely applicable information criterion) for Dirichlet mixture models.

The EM fit only yields a posterior mode, so posterior draws are obtained from a
Laplace approximation around that mode.

* "full" (default): joint normal approximation over theta = (log alpha_1..K,
  weight logits), with precision equal to the observed information of the
  mixture (Louis' identity, see :func:`observed_information`). This accounts
  for uncertain cluster assignments, which is what makes p_waic grow for
  redundant or split components. Flat or saddle-shaped directions are
  regularised (posterior SD capped at 1).
* "block": independent per-component approximations with assignments held
  fixed, lambda_k ~ N(lambda_hat_k, H_k^-1) and w ~ Dirichlet(N_k + 1). Kept
  for comparison; it underestimates p_waic for overfitted K and makes WAIC
  favour ever more components.

WAIC is then computed from the pointwise log-likelihood matrix exactly as in
Watanabe (2010) / Gelman, Hwang & Vehtari (2014), and matches ``arviz.waic``.
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.special import digamma, gammaln, logsumexp

# Threshold arviz uses to flag unreliable pointwise WAIC contributions
_PWAIC_WARN_THRESHOLD = 0.4

# Smallest precision (inverse variance) allowed along any direction of the joint
# Laplace approximation. Overfitted mixtures have flat or saddle-shaped posterior
# directions (redundant components, split clusters); without a floor those get
# near-infinite variance. 1.0 caps the posterior SD at 1 on the log(alpha) /
# weight-logit scale, which errs towards a larger p_waic, i.e. against extra
# components.
_MIN_PRECISION = 1.0

# Target number of float64 elements per work chunk (~16 MB)
_CHUNK_ELEMENTS = 2_000_000


class _SparseCounts:
    """Non-zero entries of a count matrix, grouped per sample (row)."""

    def __init__(self, X: np.ndarray):
        X = np.asarray(X)
        rows, cols = np.nonzero(X)  # row-major order: sorted by row
        self.n_samples = X.shape[0]
        self.cols = cols
        self.vals = X[rows, cols].astype(np.float64)
        self.depth = X.sum(axis=1).astype(np.float64)
        counts_per_row = np.bincount(rows, minlength=self.n_samples)
        self.nonempty_rows = np.flatnonzero(counts_per_row)
        starts = np.concatenate(([0], np.cumsum(counts_per_row)[:-1]))
        self.starts = starts[self.nonempty_rows]
        # log multinomial coefficient: log n! - sum log x_j!
        self.log_multinom = gammaln(self.depth + 1.0) - np.bincount(
            rows, weights=gammaln(self.vals + 1.0), minlength=self.n_samples
        )


def _dm_log_likelihood(sc: _SparseCounts, alpha: np.ndarray) -> np.ndarray:
    """
    Dirichlet-multinomial log-likelihood for a batch of parameter vectors.

    Parameters:
        sc: sparse representation of the counts (N samples)
        alpha: array of shape (D, S), D parameter vectors

    Returns:
        np.ndarray of shape (D, N), excluding the multinomial coefficient.
    """
    D = alpha.shape[0]
    out = np.zeros((D, sc.n_samples))
    a_sum = alpha.sum(axis=1, keepdims=True)
    out += gammaln(a_sum) - gammaln(a_sum + sc.depth[None, :])
    if sc.vals.size:
        a_nz = alpha[:, sc.cols]
        terms = gammaln(a_nz + sc.vals[None, :]) - gammaln(a_nz)
        out[:, sc.nonempty_rows] += np.add.reduceat(terms, sc.starts, axis=1)
    return out


def dirichlet_mixture_log_likelihood(
    X: np.ndarray, weights: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """
    Pointwise log-likelihood of a Dirichlet-multinomial mixture.

    log p(x_i | w, alpha) = log sum_k w_k DM(x_i | alpha_k), including the
    multinomial coefficient (so values match ``pymc.DirichletMultinomial``).

    Parameters:
        X: counts, shape (N, S)
        weights: mixture weights, shape (K,) or (D, K) for D parameter draws
        alpha: Dirichlet parameters, shape (K, S) or (D, K, S)

    Returns:
        np.ndarray of shape (N,) for a single parameter set, or (D, N).
    """
    weights = np.asarray(weights, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    single = weights.ndim == 1
    if single:
        weights = weights[None, :]
        alpha = alpha[None, :, :]
    sc = _SparseCounts(X)
    out = _pointwise_from_sparse(sc, weights, alpha)
    return out[0] if single else out


def _pointwise_from_sparse(
    sc: _SparseCounts, weights: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    D, K, S = alpha.shape
    out = np.empty((D, sc.n_samples))
    # chunk over draws so the (chunk * K, nnz) work array stays bounded
    per_draw = K * max(sc.vals.size, sc.n_samples)
    chunk = max(1, _CHUNK_ELEMENTS // max(per_draw, 1))
    with np.errstate(divide="ignore"):
        log_w = np.log(weights)
    for start in range(0, D, chunk):
        stop = min(D, start + chunk)
        d = stop - start
        ll = _dm_log_likelihood(sc, alpha[start:stop].reshape(d * K, S))
        ll = ll.reshape(d, K, sc.n_samples) + log_w[start:stop, :, None]
        out[start:stop] = logsumexp(ll, axis=1)
    out += sc.log_multinom[None, :]
    return out


def waic_from_log_likelihood(log_lik: np.ndarray) -> Dict[str, object]:
    """
    WAIC from a pointwise log-likelihood matrix.

    Parameters:
        log_lik: shape (n_draws, n_samples)

    Returns:
        dict with
          elpd_waic   lppd - p_waic (arviz "log" scale, higher is better)
          p_waic      effective number of parameters
          lppd        log pointwise predictive density
          se          standard error of elpd_waic
          WAIC        -elpd_waic, the negative-log-likelihood scale used by
                      pyDMM's BIC/AIC (lower is better)
          elpd_waic_i, p_waic_i  pointwise contributions
    """
    log_lik = np.asarray(log_lik, dtype=np.float64)
    n_draws, n_samples = log_lik.shape
    lppd_i = logsumexp(log_lik, axis=0) - np.log(n_draws)
    p_waic_i = np.var(log_lik, axis=0)  # ddof=0, as in arviz
    elpd_i = lppd_i - p_waic_i
    elpd = float(elpd_i.sum())
    return {
        "WAIC": -elpd,
        "elpd_waic": elpd,
        "p_waic": float(p_waic_i.sum()),
        "lppd": float(lppd_i.sum()),
        "se": float(np.sqrt(n_samples * np.var(elpd_i))),
        "elpd_waic_i": elpd_i,
        "p_waic_i": p_waic_i,
    }


def _precision_cholesky(H: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Cholesky factor of a precision matrix, repairing it if not PD."""
    H = 0.5 * (H + H.T)
    try:
        return np.linalg.cholesky(H), False
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(H)
        floor = max(vals.max(), 1.0) * 1e-10
        vals = np.maximum(vals, floor)
        return np.linalg.cholesky((vecs * vals) @ vecs.T), True


def laplace_posterior_draws(
    alpha_hat: np.ndarray,
    component_counts: np.ndarray,
    hessian: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw mixture weights and Dirichlet parameters from the Laplace approximation.

    Parameters:
        alpha_hat: fitted alpha, shape (K, S)
        component_counts: soft counts N_k = sum_i z_ik, shape (K,)
        hessian: per-component Hessians w.r.t. log(alpha), shape (K, S, S)
        n_draws: number of posterior draws
        rng: NumPy random generator

    Returns:
        weights (n_draws, K), alpha (n_draws, K, S)
    """
    K, S = alpha_hat.shape
    lam_hat = np.log(alpha_hat)
    lam = np.empty((n_draws, K, S))
    repaired = []
    for k in range(K):
        L, fixed = _precision_cholesky(hessian[k])
        if fixed:
            repaired.append(k)
        z = rng.standard_normal((S, n_draws))
        # (L L^T)^-1 covariance: delta = L^-T z
        delta = np.linalg.solve(L.T, z).T
        lam[:, k, :] = lam_hat[k] + delta
    if repaired:
        warnings.warn(
            f"Hessian of component(s) {repaired} is not positive definite; "
            "WAIC draws used a repaired matrix and may be unreliable.",
            UserWarning,
        )
    # Guard against overflow for components the data barely inform
    np.clip(lam, -700.0, 700.0, out=lam)
    weights = rng.dirichlet(np.asarray(component_counts) + 1.0, size=n_draws)
    return weights, np.exp(lam)


def _responsibilities(sc: _SparseCounts, weights: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Posterior component probabilities r_ik, shape (N, K)."""
    with np.errstate(divide="ignore"):
        log_r = _dm_log_likelihood(sc, alpha).T + np.log(weights)[None, :]
    log_r -= logsumexp(log_r, axis=1, keepdims=True)
    return np.exp(log_r)


def _component_score(X: np.ndarray, depth: np.ndarray, alpha_k: np.ndarray) -> np.ndarray:
    """d log DM(x_i | alpha_k) / d log(alpha_k), shape (N, S)."""
    a_sum = alpha_k.sum()
    G = digamma(alpha_k[None, :] + X) - digamma(alpha_k)[None, :]
    G += (digamma(a_sum) - digamma(a_sum + depth))[:, None]
    return G * alpha_k[None, :]


def observed_information(
    X: np.ndarray,
    alpha_hat: np.ndarray,
    weights: np.ndarray,
    hessian: np.ndarray,
) -> np.ndarray:
    """
    Observed information of the mixture (negative log posterior Hessian).

    Parameters are theta = (lambda_1, ..., lambda_K, eta_1, ..., eta_{K-1}) with
    lambda_k = log(alpha_k) and eta_k = log(w_k / w_K). Uses Louis' identity:

        I_obs = H_complete - sum_i Cov_{z_i | x_i}(complete-data score)

    H_complete is block diagonal: the per-component Hessians from the C core
    (which include the Gamma prior on alpha) and N (diag(w) - w w^T) for the
    weight logits. The covariance term couples components through the
    uncertain cluster assignments.

    Parameters:
        X: counts, shape (N, S)
        alpha_hat: fitted alpha, shape (K, S)
        weights: fitted mixture weights, shape (K,)
        hessian: per-component Hessians w.r.t. log(alpha), shape (K, S, S)

    Returns:
        np.ndarray of shape (K*S + K-1, K*S + K-1)
    """
    X = np.asarray(X, dtype=np.float64)
    N, S = X.shape
    K = alpha_hat.shape[0]
    P = K * S + K - 1
    depth = X.sum(axis=1)
    r = _responsibilities(_SparseCounts(X), weights, alpha_hat)  # (N, K)
    w_ref = weights[:K - 1]
    eta = slice(K * S, P)

    info = np.zeros((P, P))
    M = np.empty((N, P))  # rows: expected complete-data score per sample
    M[:, eta] = r[:, :K - 1] - w_ref[None, :]
    for k in range(K):
        blk = slice(k * S, (k + 1) * S)
        G = _component_score(X, depth, alpha_hat[k])
        rG = r[:, k:k + 1] * G
        M[:, blk] = rG
        # complete-data Hessian minus E[s s^T] for the lambda_k block
        info[blk, blk] = hessian[k] - G.T @ rG
        e_k = -w_ref.copy()
        if k < K - 1:
            e_k[k] += 1.0
        cross = -np.outer(rG.sum(axis=0), e_k)
        info[blk, eta] = cross
        info[eta, blk] = cross.T
    # weight logits: complete-data Hessian N (diag(w) - w w^T) minus E[s s^T]
    Wk = r.sum(axis=0)
    E = -np.tile(w_ref, (K, 1))
    E[np.arange(K - 1), np.arange(K - 1)] += 1.0  # rows: e_k - w (k = K has e_K = 0)
    info[eta, eta] = (N * (np.diag(w_ref) - np.outer(w_ref, w_ref))
                      - (E * Wk[:, None]).T @ E)
    # + (sum_k r_ik s_ik)(sum_k r_ik s_ik)^T
    info += M.T @ M
    return 0.5 * (info + info.T)


def _regularised_cholesky(info: np.ndarray) -> Tuple[np.ndarray, int]:
    """Cholesky factor after replacing eigenvalues by max(|ev|, _MIN_PRECISION)."""
    vals, vecs = np.linalg.eigh(info)
    n_fixed = int(np.sum(vals < _MIN_PRECISION))
    if n_fixed:
        vals = np.maximum(np.abs(vals), _MIN_PRECISION)
        info = (vecs * vals) @ vecs.T
        info = 0.5 * (info + info.T)
    return np.linalg.cholesky(info), n_fixed


def laplace_posterior_draws_full(
    X: np.ndarray,
    alpha_hat: np.ndarray,
    weights: np.ndarray,
    hessian: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Joint Laplace draws using the full observed information of the mixture.

    Directions where the observed information is below ``_MIN_PRECISION``
    (including negative curvature at saddle points of overfitted fits) are
    regularised; their count is returned.

    Returns weights (n_draws, K), alpha (n_draws, K, S), n_regularised.
    """
    K, S = alpha_hat.shape
    weights = np.asarray(weights, dtype=np.float64)
    info = observed_information(X, alpha_hat, weights, hessian)
    L, n_fixed = _regularised_cholesky(info)
    z = rng.standard_normal((info.shape[0], n_draws))
    delta = np.linalg.solve(L.T, z).T  # (n_draws, P)
    lam = np.log(alpha_hat)[None] + delta[:, :K * S].reshape(n_draws, K, S)
    np.clip(lam, -700.0, 700.0, out=lam)
    with np.errstate(divide="ignore"):
        eta_hat = np.log(weights[:K - 1]) - np.log(weights[K - 1])
    eta = np.zeros((n_draws, K))
    eta[:, :K - 1] = eta_hat[None, :] + delta[:, K * S:]
    w = np.exp(eta - logsumexp(eta, axis=1, keepdims=True))
    return w, np.exp(lam), n_fixed


def compute_waic(
    X: np.ndarray,
    alpha_hat: np.ndarray,
    mixture_weights: np.ndarray,
    hessian: np.ndarray,
    n_draws: int = 1000,
    random_state: Optional[int] = None,
    method: str = "full",
) -> Dict[str, object]:
    """
    WAIC of a fitted Dirichlet mixture using Laplace-approximation draws.

    Parameters:
        X: counts, shape (N, S)
        alpha_hat: fitted alpha, shape (S, K) as stored in ``Fit["Estimate"]``
        mixture_weights: fitted weights, shape (K,)
        hessian: per-component Hessians, shape (K, S, S)
        n_draws: number of posterior draws
        random_state: seed for the draws
        method: "full" (joint observed information, Louis' identity) or
            "block" (independent per-component Hessians, assignments held fixed;
            underestimates p_waic for overlapping or split clusters)

    Returns:
        dict from :func:`waic_from_log_likelihood` plus ``log_likelihood``
        (n_draws, N) and ``n_draws``.
    """
    rng = np.random.default_rng(random_state)
    N = X.shape[0]
    alpha_kS = np.asarray(alpha_hat).T
    if method == "full":
        weights, alpha, n_fixed = laplace_posterior_draws_full(
            X, alpha_kS, mixture_weights, hessian, n_draws, rng)
    elif method == "block":
        weights, alpha = laplace_posterior_draws(
            alpha_kS, np.asarray(mixture_weights) * N, hessian, n_draws, rng)
        n_fixed = 0
    else:
        raise ValueError(f"unknown method {method!r}")
    log_lik = _pointwise_from_sparse(_SparseCounts(X), weights, alpha)
    res = waic_from_log_likelihood(log_lik)
    n_high = int(np.sum(res["p_waic_i"] > _PWAIC_WARN_THRESHOLD))
    if n_high:
        warnings.warn(
            f"{n_high} of {N} samples have p_waic > {_PWAIC_WARN_THRESHOLD}; "
            "WAIC may be unreliable for this fit.",
            UserWarning,
        )
    res["log_likelihood"] = log_lik
    res["n_draws"] = n_draws
    res["n_regularised"] = n_fixed
    return res

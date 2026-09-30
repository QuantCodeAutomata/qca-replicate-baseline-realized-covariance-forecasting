"""
Realized covariance estimation from high-frequency intraday data.

Implements the standard 5-minute realized covariance estimator
(Andersen et al., 2003), a subsampling-averaged variant (Zhang,
Mykland & Aït-Sahalia, 2005), and related utilities for DRD
decomposition, realized quarticity, and synthetic RC generation.

Config reference (config.yaml):
  intraday.bar_minutes         = 5
  intraday.bars_per_day        = 78   (M)
  intraday.subsampling_offsets = [0,1,2,3,4]
  realized_cov.pd_correction   = higham
"""

import warnings

import numpy as np

from .nearest_pd import ensure_pd, nearest_pd
from .utils import N_ASSETS, N_PAIRS, vech_corr


# ---------------------------------------------------------------------------
# Core estimators
# ---------------------------------------------------------------------------


def compute_rc_simple(returns: np.ndarray) -> np.ndarray:
    """Compute the standard realized covariance matrix.

    Realized covariance is the outer-product sum over M intraday bars:
        H = sum_{l=1}^{M} r_l r_l'  =  R' R
    where R is the (M, N) log-return matrix.

    Parameters
    ----------
    returns : np.ndarray, shape (M, N)
        Intraday log-returns; M = 78 for 5-min bars, N = 27 assets.

    Returns
    -------
    H : np.ndarray, shape (N, N)
        Symmetric positive-semi-definite realized covariance matrix.
    """
    # R' R accumulates the M outer products in one BLAS call
    return returns.T @ returns  # (N, N)


def compute_rc_subsampled(
    returns_grid: np.ndarray,
    n_subsamples: int = 5,
) -> np.ndarray:
    """Compute a subsampling-averaged realized covariance matrix.

    For each offset s in {0, …, n_subsamples-1}, the fine-grid returns are
    aggregated into non-overlapping coarser bars and the standard RC is
    computed.  The n_subsamples estimates are averaged (Zhang et al., 2005).

    Parameters
    ----------
    returns_grid : np.ndarray, shape (M_fine, N)
        Fine-grid (e.g., 1-minute) log-returns over a trading session.
    n_subsamples : int
        Number of offset subsamples (default 5, matches config).

    Returns
    -------
    H_avg : np.ndarray, shape (N, N)
        Subsampling-averaged realized covariance matrix.
    """
    M_fine, N = returns_grid.shape
    H_sum = np.zeros((N, N))
    count = 0

    for s in range(n_subsamples):
        # Complete coarse bars for this offset
        n_complete = (M_fine - s) // n_subsamples
        if n_complete == 0:
            continue

        # Align segment and reshape to (n_complete, n_subsamples, N)
        segment = returns_grid[s : s + n_complete * n_subsamples]
        # Summing n_subsamples consecutive fine returns gives one coarse return
        rets_coarse = segment.reshape(n_complete, n_subsamples, N).sum(axis=1)

        H_sum += compute_rc_simple(rets_coarse)
        count += 1

    return H_sum / count if count > 0 else H_sum


def compute_daily_rc_from_prices(
    prices: np.ndarray,
    bar_minutes: int = 5,
) -> np.ndarray:
    """Compute daily realized covariance from an intraday price grid.

    Converts synchronised prices to log-returns, computes H = R'R, and
    applies Higham's nearest positive-definite correction when needed.

    Parameters
    ----------
    prices : np.ndarray, shape (M+1, N)
        Intraday prices at bar boundaries (M+1 snapshots for M bars).
    bar_minutes : int
        Bar width in minutes (default 5, matches config).

    Returns
    -------
    H : np.ndarray, shape (N, N)
        Realized covariance matrix, guaranteed positive definite.
    """
    # r_l = log(p_l) - log(p_{l-1})
    log_prices = np.log(prices)
    returns = np.diff(log_prices, axis=0)  # (M, N)

    H = compute_rc_simple(returns)

    # Higham nearest-PD if smallest eigenvalue is non-positive
    min_eig = np.linalg.eigvalsh(H).min()
    if min_eig <= 0.0:
        warnings.warn(
            f"RC not PD (min eigenvalue {min_eig:.3e}); applying Higham correction.",
            RuntimeWarning,
            stacklevel=2,
        )
        H = nearest_pd(H)

    return H


# ---------------------------------------------------------------------------
# DRD decomposition
# ---------------------------------------------------------------------------


def extract_drd_components(
    H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a covariance matrix into DRD components (v, D, R).

    The DRD factorization is:
        H = D R D,   D = diag(sqrt(v)),   v = diag(H)
    where R is the implied correlation matrix.

    Parameters
    ----------
    H : np.ndarray, shape (N, N)
        Symmetric positive-semi-definite covariance matrix.

    Returns
    -------
    v : np.ndarray, shape (N,)
        Realized variances, v_i = H_{ii}.
    D : np.ndarray, shape (N, N)
        Diagonal standard-deviation matrix, diag(sqrt(v)).
    R : np.ndarray, shape (N, N)
        Correlation matrix; off-diagonal entries clipped to [-1, 1].
    """
    N = H.shape[0]
    v = np.diag(H)                            # realized variances
    d = np.sqrt(np.maximum(v, 0.0))           # standard deviations (safe sqrt)
    D = np.diag(d)

    # R = D^{-1} H D^{-1}
    d_inv = np.where(d > 0.0, 1.0 / d, 0.0)
    R = (d_inv[:, None] * H) * d_inv[None, :]

    # Enforce exact unit diagonal; clip off-diagonal entries for numerical safety
    off_diag_mask = ~np.eye(N, dtype=bool)
    R[off_diag_mask] = np.clip(R[off_diag_mask], -1.0, 1.0)
    np.fill_diagonal(R, 1.0)

    return v, D, R


# ---------------------------------------------------------------------------
# Realized quarticity
# ---------------------------------------------------------------------------


def compute_realized_quarticity(returns: np.ndarray) -> np.ndarray:
    """Compute the realized quarticity (RQ) for each asset.

    Barndorff-Nielsen & Shephard (2004) consistent estimator:
        RQ_i = (M / 3) * sum_{l=1}^{M} r_{i,l}^4

    Parameters
    ----------
    returns : np.ndarray, shape (M, N)
        Intraday log-return matrix.

    Returns
    -------
    RQ : np.ndarray, shape (N,)
        Realized quarticity vector (one scalar per asset).
    """
    M = returns.shape[0]
    return (M / 3.0) * np.sum(returns ** 4, axis=0)  # (N,)


# ---------------------------------------------------------------------------
# Synthetic RC series generation
# ---------------------------------------------------------------------------


def generate_synthetic_rc_series(
    n_days: int = 500,
    n_assets: int = N_ASSETS,
    bars_per_day: int = 78,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic daily realized covariance series for testing.

    A random walk on log-Cholesky factors guarantees H_t = L_t L_t' > 0:
      - Diagonal Cholesky entries evolve on the log scale so exp(.) > 0.
      - Off-diagonal entries follow an unconstrained random walk.

    Parameters
    ----------
    n_days : int
        Number of trading days T to generate.
    n_assets : int
        Number of assets N.
    bars_per_day : int
        Intraday bars per day (kept for API consistency; not used internally).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    H_series : np.ndarray, shape (T, N, N)
        Daily realized covariance matrices.
    v_series : np.ndarray, shape (T, N)
        Daily realized variance vectors, v_t = diag(H_t).
    x_series : np.ndarray, shape (T, N*(N-1)//2)
        Vech of daily realized correlation matrices (lower off-diagonal).
    """
    rng = np.random.default_rng(seed)

    N = n_assets
    K_chol = N * (N + 1) // 2   # lower-triangle elements: 378 for N=27
    N_corr = N * (N - 1) // 2   # unique correlations: 351 for N=27

    # Lower-triangular indices (row-major order)
    rows, cols = np.tril_indices(N)    # length K_chol
    is_diag = rows == cols             # mask for diagonal positions

    # Initialise log-Cholesky factors:
    #   diagonal (log scale) = 0  →  L_ii = exp(0) = 1
    #   off-diagonal = 0          →  L_ij = 0
    factors = np.zeros(K_chol)

    H_series = np.empty((n_days, N, N))
    v_series = np.empty((n_days, N))
    x_series = np.empty((n_days, N_corr))

    # Smaller step on log-diagonal keeps condition numbers reasonable
    diag_std = 0.02
    offdiag_std = 0.05

    for t in range(n_days):
        # Random-walk step on log-Cholesky factors
        noise = np.zeros(K_chol)
        noise[is_diag]  = rng.normal(0.0, diag_std,   is_diag.sum())
        noise[~is_diag] = rng.normal(0.0, offdiag_std, (~is_diag).sum())
        factors = factors + noise

        # Build lower-triangular L
        L = np.zeros((N, N))
        L[rows[~is_diag], cols[~is_diag]] = factors[~is_diag]       # off-diag (raw)
        L[rows[is_diag],  cols[is_diag]]  = np.exp(factors[is_diag]) # diagonal (exp)

        # Realized covariance H = L L'
        H = L @ L.T
        H_series[t] = H

        # Variance vector v = diag(H)
        v = np.diag(H)
        v_series[t] = v

        # Correlation matrix R and its vech
        d_inv = 1.0 / np.sqrt(np.maximum(v, 1e-14))
        R = (d_inv[:, None] * H) * d_inv[None, :]
        R = np.clip(R, -1.0, 1.0)
        np.fill_diagonal(R, 1.0)
        x_series[t] = vech_corr(R)

    return H_series, v_series, x_series

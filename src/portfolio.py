"""
Portfolio optimization utilities for economic evaluation.

Provides the unconstrained GMVP, the long-only GMVP+ (solved with cvxpy/OSQP),
and helper functions for portfolio return, annualized volatility, and turnover.

Config reference (config.yaml):
  portfolio.annualization = 252
"""

import warnings

import cvxpy as cp
import numpy as np
import scipy.linalg


def gmvp_weights(Sigma: np.ndarray) -> np.ndarray:
    """Compute unconstrained Global Minimum-Variance Portfolio weights.

    Closed-form solution:
        w = Sigma^{-1} 1 / (1' Sigma^{-1} 1)
    where 1 is the all-ones vector.  Uses scipy.linalg.solve for numerical
    stability with a positive-definite system.

    Parameters
    ----------
    Sigma : np.ndarray, shape (N, N)
        Positive-definite covariance matrix.

    Returns
    -------
    w : np.ndarray, shape (N,)
        Portfolio weight vector summing to 1 (may contain negatives).
    """
    N = Sigma.shape[0]
    ones = np.ones(N)
    # Solve Sigma w = 1 (exploiting positive-definiteness)
    w_unnorm = scipy.linalg.solve(Sigma, ones, assume_a="pos")
    denom = w_unnorm.sum()
    return w_unnorm / denom


def gmvp_plus_weights(Sigma: np.ndarray) -> np.ndarray:
    """Compute long-only Global Minimum-Variance Portfolio weights.

    Solves the quadratic programme:
        min  w' Sigma w
        s.t. sum(w) = 1,  w >= 0
    using cvxpy with the OSQP solver.  Adds a small ridge 1e-8 * I if the
    solver fails and retries once before falling back to equal weights.

    Parameters
    ----------
    Sigma : np.ndarray, shape (N, N)
        Positive-definite covariance matrix.

    Returns
    -------
    w : np.ndarray, shape (N,)
        Long-only weight vector summing to 1.
    """
    N = Sigma.shape[0]

    def _solve(S: np.ndarray) -> np.ndarray | None:
        w = cp.Variable(N)
        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(S)))
        constraints = [cp.sum(w) == 1, w >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, warm_starting=True)
        return w.value

    result = _solve(Sigma)
    if result is None:
        # Retry with a small ridge regulariser to improve conditioning
        warnings.warn(
            "OSQP failed on original Sigma; retrying with 1e-8 ridge.",
            RuntimeWarning,
            stacklevel=2,
        )
        result = _solve(Sigma + 1e-8 * np.eye(N))

    if result is None:
        warnings.warn(
            "OSQP failed after ridge regularisation; returning equal weights.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.full(N, 1.0 / N)

    # Clip numerical noise and re-normalise
    result = np.maximum(result, 0.0)
    s = result.sum()
    return result / s if s > 0 else np.full(N, 1.0 / N)


def portfolio_return(weights: np.ndarray, returns: np.ndarray) -> float:
    """Compute the portfolio return for a single period.

    r_p = w' r

    Parameters
    ----------
    weights : np.ndarray, shape (N,)
        Portfolio weights.
    returns : np.ndarray, shape (N,)
        Asset returns over the same period.

    Returns
    -------
    r_p : float
        Portfolio return.
    """
    return float(weights @ returns)


def annualized_volatility(
    returns: np.ndarray,
    annualization: int = 252,
) -> float:
    """Compute annualized portfolio volatility from a daily return series.

    sigma_ann = sqrt(annualization * sample_variance)

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        Daily portfolio returns.
    annualization : int
        Number of trading days per year (default 252, matches config).

    Returns
    -------
    sigma_ann : float
        Annualized volatility.
    """
    if len(returns) < 2:
        return float("nan")
    return float(np.sqrt(annualization * np.var(returns, ddof=1)))


def compute_turnover(
    weights_history: np.ndarray,
    returns_history: np.ndarray,
) -> float:
    """Compute average portfolio turnover accounting for price drift.

    At the end of each period t the portfolio drifts from w_t to:
        w_drift[t] = w_t * (1 + r_t) / (1 + w_t' r_t)
    The one-way turnover at t+1 is then ||w_{t+1} - w_drift||_1.
    Average turnover is the mean over all T-1 rebalancing events.

    Parameters
    ----------
    weights_history : np.ndarray, shape (T, N)
        Portfolio weights at each rebalancing date.
    returns_history : np.ndarray, shape (T, N)
        Asset returns over the period following each rebalancing date.

    Returns
    -------
    avg_turnover : float
        Mean one-way turnover per rebalancing period.
    """
    T = weights_history.shape[0]
    if T < 2:
        return float("nan")

    turnovers = np.empty(T - 1)
    for t in range(T - 1):
        port_ret  = weights_history[t] @ returns_history[t]
        # Drift: each asset weight changes by its return relative to portfolio
        w_drift   = weights_history[t] * (1.0 + returns_history[t]) / (1.0 + port_ret)
        turnovers[t] = np.abs(weights_history[t + 1] - w_drift).sum()

    return float(turnovers.mean())


def equal_weight(N: int = 27) -> np.ndarray:
    """Return the 1/N equal-weight vector.

    Parameters
    ----------
    N : int
        Number of assets (default 27).

    Returns
    -------
    w : np.ndarray, shape (N,)
        Uniform weight vector.
    """
    return np.full(N, 1.0 / N)

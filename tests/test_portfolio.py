"""Tests for portfolio optimization utilities (GMVP, GMVP+, turnover)."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.portfolio import (
    gmvp_weights,
    gmvp_plus_weights,
    portfolio_return,
    annualized_volatility,
    compute_turnover,
    equal_weight,
)
from src.nearest_pd import nearest_pd


def _make_pd_sigma(N: int = 10, seed: int = 0) -> np.ndarray:
    """Return a random positive-definite covariance matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N))
    S = A @ A.T + 1e-4 * np.eye(N)
    return S


# ---------------------------------------------------------------------------
# gmvp_weights
# ---------------------------------------------------------------------------

def test_gmvp_weights_sums_to_one() -> None:
    S = _make_pd_sigma(N=10)
    w = gmvp_weights(S)
    assert abs(w.sum() - 1.0) < 1e-10


def test_gmvp_weights_shape() -> None:
    N = 15
    S = _make_pd_sigma(N)
    w = gmvp_weights(S)
    assert w.shape == (N,)


def test_gmvp_weights_minimizes_variance() -> None:
    """GMVP portfolio variance must be <= variance of any individual asset."""
    N = 5
    S = _make_pd_sigma(N, seed=1)
    w_gmvp = gmvp_weights(S)
    port_var = float(w_gmvp @ S @ w_gmvp)
    for i in range(N):
        e = np.zeros(N)
        e[i] = 1.0
        asset_var = float(e @ S @ e)
        assert port_var <= asset_var + 1e-8, \
            f"GMVP variance ({port_var}) > asset {i} variance ({asset_var})"


def test_gmvp_weights_equal_cov_equal_weights() -> None:
    """When all assets have the same variance and zero covariance, GMVP = equal-weight."""
    N = 4
    S = np.eye(N, dtype=float)
    w = gmvp_weights(S)
    expected = np.full(N, 1.0 / N)
    np.testing.assert_allclose(w, expected, atol=1e-8)


def test_gmvp_weights_finite() -> None:
    S = _make_pd_sigma(N=27, seed=2)
    w = gmvp_weights(S)
    assert np.all(np.isfinite(w))


# ---------------------------------------------------------------------------
# gmvp_plus_weights
# ---------------------------------------------------------------------------

def test_gmvp_plus_weights_sums_to_one() -> None:
    S = _make_pd_sigma(N=10)
    w = gmvp_plus_weights(S)
    assert abs(w.sum() - 1.0) < 1e-6


def test_gmvp_plus_weights_nonnegative() -> None:
    S = _make_pd_sigma(N=10)
    w = gmvp_plus_weights(S)
    assert np.all(w >= -1e-8), "GMVP+ weights must be non-negative"


def test_gmvp_plus_weights_shape() -> None:
    N = 27
    S = _make_pd_sigma(N)
    w = gmvp_plus_weights(S)
    assert w.shape == (N,)


def test_gmvp_plus_variance_le_gmvp() -> None:
    """Long-only GMVP+ variance may be >= unconstrained GMVP variance."""
    N = 10
    S = _make_pd_sigma(N, seed=3)
    w_gmvp  = gmvp_weights(S)
    w_plus  = gmvp_plus_weights(S)
    var_unconstrained = float(w_gmvp @ S @ w_gmvp)
    var_constrained   = float(w_plus @ S @ w_plus)
    # Constrained problem has no smaller feasible set, so var_constrained >= var_unconstrained
    assert var_constrained >= var_unconstrained - 1e-6, \
        "Long-only variance should not be strictly less than unconstrained GMVP variance"


# ---------------------------------------------------------------------------
# portfolio_return
# ---------------------------------------------------------------------------

def test_portfolio_return_scalar() -> None:
    w = np.array([0.5, 0.5])
    r = np.array([0.02, 0.04])
    assert portfolio_return(w, r) == pytest.approx(0.03)


def test_portfolio_return_negative() -> None:
    w = np.array([1.0, 0.0])
    r = np.array([-0.05, 0.10])
    assert portfolio_return(w, r) == pytest.approx(-0.05)


def test_portfolio_return_finite() -> None:
    rng = np.random.default_rng(10)
    w = np.full(27, 1.0 / 27)
    r = rng.standard_normal(27) * 0.01
    result = portfolio_return(w, r)
    assert np.isfinite(result)


# ---------------------------------------------------------------------------
# annualized_volatility
# ---------------------------------------------------------------------------

def test_annualized_volatility_known_series() -> None:
    """Daily returns with std=1% → annualized vol ≈ sqrt(252) * 0.01."""
    rng = np.random.default_rng(11)
    T = 10_000
    r = rng.normal(0, 0.01, T)
    ann_vol = annualized_volatility(r, annualization=252)
    expected = np.sqrt(252) * 0.01
    assert ann_vol == pytest.approx(expected, rel=0.05)


def test_annualized_volatility_nonneg() -> None:
    rng = np.random.default_rng(12)
    r = rng.standard_normal(500) * 0.02
    assert annualized_volatility(r) >= 0.0


def test_annualized_volatility_single_obs_nan() -> None:
    """Single observation → should return NaN (no variance)."""
    result = annualized_volatility(np.array([0.01]))
    assert np.isnan(result)


def test_annualized_volatility_zero_series() -> None:
    """Constant returns → zero volatility."""
    r = np.zeros(100)
    assert annualized_volatility(r) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_turnover
# ---------------------------------------------------------------------------

def test_compute_turnover_zero_for_constant_weights() -> None:
    """If weights don't change and returns are zero, turnover must be zero."""
    T, N = 20, 5
    w = np.tile(np.full(N, 1.0 / N), (T, 1))
    r = np.zeros((T, N))
    to = compute_turnover(w, r)
    assert to == pytest.approx(0.0, abs=1e-10)


def test_compute_turnover_nonneg() -> None:
    rng = np.random.default_rng(13)
    T, N = 30, 10
    w = rng.dirichlet(np.ones(N), T)
    r = rng.normal(0, 0.01, (T, N))
    to = compute_turnover(w, r)
    assert to >= 0.0


def test_compute_turnover_shape_check() -> None:
    T, N = 5, 3
    w = np.tile(np.array([0.5, 0.3, 0.2]), (T, 1))
    r = np.zeros((T, N))
    to = compute_turnover(w, r)
    assert np.isfinite(to)


def test_compute_turnover_single_period_nan() -> None:
    """Only 1 period → no turnover to compute (NaN)."""
    w = np.array([[0.5, 0.5]])
    r = np.array([[0.01, 0.02]])
    to = compute_turnover(w, r)
    assert np.isnan(to)


# ---------------------------------------------------------------------------
# equal_weight
# ---------------------------------------------------------------------------

def test_equal_weight_sums_to_one() -> None:
    w = equal_weight(27)
    assert abs(w.sum() - 1.0) < 1e-10


def test_equal_weight_uniform() -> None:
    N = 27
    w = equal_weight(N)
    np.testing.assert_allclose(w, np.full(N, 1.0 / N), atol=1e-12)


def test_equal_weight_custom_n() -> None:
    w = equal_weight(5)
    np.testing.assert_allclose(w, np.full(5, 0.2), atol=1e-12)

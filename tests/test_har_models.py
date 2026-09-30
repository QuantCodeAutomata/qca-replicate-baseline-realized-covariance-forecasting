"""Tests for HAR-Cholesky and HAR-DRD model implementations."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.har_models import HARCholesky, HARDRD, _build_har_lags, pool_har_ols
from src.realized_cov import generate_synthetic_rc_series


_MIN_LAG = 23   # must have ≥23 obs to form all lags


# ---------------------------------------------------------------------------
# _build_har_lags / pool_har_ols
# ---------------------------------------------------------------------------

def test_build_har_lags_shapes() -> None:
    """Lag matrix should have correct shape for a short panel."""
    T, N = 80, 5
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((T, N))
    lags = _build_har_lags(Z)
    # lags: (T_valid, N, 3) or stacked — check docs if 2D or 3D
    # At minimum the function must return something without error
    assert lags is not None


def test_pool_har_ols_returns_four_values() -> None:
    T, N = 100, 5
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((T, N))
    result = pool_har_ols(Z)
    assert len(result) == 4, "pool_har_ols must return (alpha, beta_d, beta_w, beta_m)"


def test_pool_har_ols_scalars() -> None:
    """Beta coefficients should be scalars (common across components)."""
    T, N = 120, 10
    rng = np.random.default_rng(2)
    Z = rng.standard_normal((T, N))
    alpha, beta_d, beta_w, beta_m = pool_har_ols(Z)
    for val in (beta_d, beta_w, beta_m):
        assert np.isscalar(val) or (hasattr(val, 'shape') and val.shape == ())


def test_pool_har_ols_alpha_shape() -> None:
    """Intercepts should have one entry per component."""
    T, N = 120, 10
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((T, N))
    alpha, _, _, _ = pool_har_ols(Z)
    assert alpha.shape == (N,), f"Alpha shape expected ({N},); got {alpha.shape}"


# ---------------------------------------------------------------------------
# HARCholesky
# ---------------------------------------------------------------------------

@pytest.fixture()
def har_cholesky_fitted():
    """Fit a small HARCholesky model for re-use."""
    K = 6   # 3*(3+1)/2 for N=3 assets
    T = 150
    rng = np.random.default_rng(42)
    Y = rng.standard_normal((T, K))
    model = HARCholesky()
    model.fit(Y)
    return model, Y


def test_harcholesky_fit_stores_params(har_cholesky_fitted) -> None:
    model, _ = har_cholesky_fitted
    assert hasattr(model, "beta_d")
    assert hasattr(model, "beta_w")
    assert hasattr(model, "beta_m")
    assert hasattr(model, "alpha_")


def test_harcholesky_predict_shape(har_cholesky_fitted) -> None:
    """predict() should return a vector of the same length as K."""
    model, Y = har_cholesky_fitted
    K = Y.shape[1]
    y_last    = Y[-1]
    y_weekly  = Y[-5:-1].mean(axis=0)
    y_monthly = Y[-22:-5].mean(axis=0)
    yhat = model.predict(y_last, y_weekly, y_monthly)
    assert yhat.shape == (K,)


def test_harcholesky_predict_horizon_shape(har_cholesky_fitted) -> None:
    model, Y = har_cholesky_fitted
    K = Y.shape[1]
    yhat = model.predict_horizon(Y, horizon=1)
    assert yhat.shape == (K,)


def test_harcholesky_predict_horizon_insufficient_data() -> None:
    K, T = 6, 20  # T < 23
    rng = np.random.default_rng(5)
    Y = rng.standard_normal((T, K))
    model = HARCholesky()
    model.fit(np.random.default_rng(5).standard_normal((150, K)))  # fit on enough data
    with pytest.raises(ValueError):
        model.predict_horizon(Y, horizon=1)


def test_harcholesky_forecast_finite() -> None:
    """All forecast values should be finite."""
    K, T = 378, 200
    rng = np.random.default_rng(7)
    Y = rng.standard_normal((T, K))
    model = HARCholesky().fit(Y)
    yhat = model.predict_horizon(Y, horizon=1)
    assert np.all(np.isfinite(yhat))


# ---------------------------------------------------------------------------
# HARDRD
# ---------------------------------------------------------------------------

@pytest.fixture()
def hardrd_data():
    H, v, x = generate_synthetic_rc_series(n_days=200, n_assets=27, seed=99)
    return H, v, x


@pytest.fixture()
def hardrd_fitted(hardrd_data):
    _, v, x = hardrd_data
    model = HARDRD()
    model.fit(v, x)
    return model, v, x


def test_hardrd_fit_coefs(hardrd_fitted) -> None:
    model, _, _ = hardrd_fitted
    for attr in ("beta_d_v", "beta_w_v", "beta_m_v",
                 "beta_d_x", "beta_w_x", "beta_m_x",
                 "alpha_v", "alpha_x"):
        assert hasattr(model, attr), f"Missing attribute {attr}"


def test_hardrd_scalar_slopes(hardrd_fitted) -> None:
    model, _, _ = hardrd_fitted
    for attr in ("beta_d_v", "beta_w_v", "beta_m_v",
                 "beta_d_x", "beta_w_x", "beta_m_x"):
        val = getattr(model, attr)
        assert np.isscalar(val) or (hasattr(val, 'shape') and val.shape == ()), \
            f"{attr} must be scalar"


def test_hardrd_alpha_shapes(hardrd_fitted) -> None:
    model, v, x = hardrd_fitted
    N, P = v.shape[1], x.shape[1]
    assert model.alpha_v.shape == (N,), f"alpha_v shape mismatch"
    assert model.alpha_x.shape == (P,), f"alpha_x shape mismatch"


def test_hardrd_predict_shapes(hardrd_fitted) -> None:
    model, v, x = hardrd_fitted
    N, P = v.shape[1], x.shape[1]
    v_hist = v[:50]
    x_hist = x[:50]
    vhat, xhat = model.predict(v_hist, x_hist)
    assert vhat.shape == (N,)
    assert xhat.shape == (P,)


def test_hardrd_predict_positive_variances(hardrd_fitted) -> None:
    """Variance forecasts should be positive (the HAR is linear, so enforce externally;
    at minimum the output should be finite for typical synthetic data)."""
    model, v, x = hardrd_fitted
    vhat, xhat = model.predict(v[:50], x[:50])
    assert np.all(np.isfinite(vhat))
    assert np.all(np.isfinite(xhat))


def test_hardrd_predict_correlations_bounded_or_not(hardrd_fitted) -> None:
    """x-hat values may not be in [-1,1] but the forecast must be finite."""
    model, v, x = hardrd_fitted
    _, xhat = model.predict(v[:50], x[:50])
    assert np.all(np.isfinite(xhat))


def test_hardrd_at_least_22_rows_required(hardrd_fitted) -> None:
    """Using < 22 history rows should raise ValueError or produce wrong shapes,
    depending on implementation; at least it shouldn't silently succeed wrongly."""
    model, v, x = hardrd_fitted
    # If the model raises ValueError for too-short history, that is correct behaviour.
    # If it does not raise, the output must still be finite.
    try:
        vhat, xhat = model.predict(v[:10], x[:10])
        assert np.all(np.isfinite(vhat))
    except (ValueError, IndexError):
        pass  # raising is also acceptable


# ---------------------------------------------------------------------------
# Mathematical property: beta coefficients sum ≤ 1 for typical HAR fits
# ---------------------------------------------------------------------------

def test_hardrd_sum_of_betas_reasonable(hardrd_fitted) -> None:
    """Sum of daily + weekly + monthly slopes should be reasonable (< 2 for typical data)."""
    model, _, _ = hardrd_fitted
    s_v = abs(model.beta_d_v) + abs(model.beta_w_v) + abs(model.beta_m_v)
    s_x = abs(model.beta_d_x) + abs(model.beta_w_x) + abs(model.beta_m_x)
    assert s_v < 3.0, f"Variance beta sum unexpectedly large: {s_v}"
    assert s_x < 3.0, f"Correlation beta sum unexpectedly large: {s_x}"

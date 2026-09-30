"""Tests for src/utils.py — vech operators, HAR lag construction, and constants."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.utils import (
    TICKERS,
    SECTOR_MAP,
    N_ASSETS,
    N_PAIRS,
    VECH_PAIRS,
    VECH_CHOL_IDX,
    vech_corr,
    vech_corr_inv,
    vech_chol,
    vech_chol_inv,
    build_drd,
    har_lags,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_tickers_count() -> None:
    assert len(TICKERS) == 27, f"Expected 27 tickers, got {len(TICKERS)}"


def test_n_assets_constant() -> None:
    assert N_ASSETS == 27


def test_n_pairs_constant() -> None:
    assert N_PAIRS == 351, f"N*(N-1)/2 for N=27 is 351, got {N_PAIRS}"


def test_vech_pairs_count() -> None:
    assert len(VECH_PAIRS) == 351, f"Expected 351 VECH_PAIRS, got {len(VECH_PAIRS)}"


def test_vech_chol_idx_count() -> None:
    assert len(VECH_CHOL_IDX) == 378, f"Expected 378 VECH_CHOL_IDX, got {len(VECH_CHOL_IDX)}"


def test_vech_pairs_unique() -> None:
    pairs_set = set(VECH_PAIRS)
    assert len(pairs_set) == len(VECH_PAIRS), "VECH_PAIRS must contain unique entries"


def test_vech_pairs_lower_triangular() -> None:
    for i, j in VECH_PAIRS:
        assert i > j, f"Pair ({i},{j}) violates i > j for lower-triangular"


def test_vech_chol_idx_lower_triangular() -> None:
    for i, j in VECH_CHOL_IDX:
        assert i >= j, f"Chol entry ({i},{j}) violates i >= j"


def test_sector_map_all_tickers() -> None:
    for t in TICKERS:
        assert t in SECTOR_MAP, f"Ticker {t} not in SECTOR_MAP"


# ---------------------------------------------------------------------------
# vech_corr / vech_corr_inv
# ---------------------------------------------------------------------------

def _make_corr(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    S = A @ A.T
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


def test_vech_corr_length() -> None:
    R = _make_corr(27)
    x = vech_corr(R)
    assert x.shape == (351,), f"Expected (351,), got {x.shape}"


def test_vech_corr_roundtrip() -> None:
    """vech_corr followed by vech_corr_inv should recover the off-diagonal lower-triangular part."""
    R = _make_corr(27)
    x = vech_corr(R)
    R_rec = vech_corr_inv(x, N=27)
    # Compare lower-triangular off-diagonal
    rows, cols = zip(*VECH_PAIRS)
    np.testing.assert_allclose(
        R[list(rows), list(cols)],
        R_rec[list(rows), list(cols)],
        atol=1e-12,
        err_msg="Round-trip vech_corr→vech_corr_inv failed",
    )


def test_vech_corr_inv_unit_diagonal() -> None:
    """vech_corr_inv must always produce a unit diagonal."""
    rng = np.random.default_rng(5)
    x = np.clip(rng.standard_normal(351), -0.99, 0.99)
    R = vech_corr_inv(x, N=27)
    np.testing.assert_allclose(
        np.diag(R), np.ones(27), atol=1e-12, err_msg="Diagonal of reconstructed R must be 1"
    )


# ---------------------------------------------------------------------------
# vech_chol / vech_chol_inv
# ---------------------------------------------------------------------------

def _make_chol(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    S = A @ A.T + n * np.eye(n)
    return np.linalg.cholesky(S)


def test_vech_chol_length() -> None:
    C = _make_chol(27)
    y = vech_chol(C)
    assert y.shape == (378,), f"Expected (378,), got {y.shape}"


def test_vech_chol_roundtrip() -> None:
    """vech_chol followed by vech_chol_inv should recover the lower-triangular matrix."""
    C = _make_chol(27)
    y = vech_chol(C)
    C_rec = vech_chol_inv(y, N=27)
    np.testing.assert_allclose(
        np.tril(C), np.tril(C_rec), atol=1e-12, err_msg="Round-trip vech_chol→vech_chol_inv failed"
    )


def test_vech_chol_inv_lower_triangular() -> None:
    """vech_chol_inv must produce a strictly lower-triangular (+ diagonal) matrix."""
    rng = np.random.default_rng(2)
    y = rng.standard_normal(378)
    C = vech_chol_inv(y, N=27)
    # Upper-triangle (above diagonal) must be zero
    assert np.allclose(np.triu(C, k=1), 0), "Upper triangle of reconstructed Cholesky must be zero"


# ---------------------------------------------------------------------------
# build_drd
# ---------------------------------------------------------------------------

def test_build_drd_symmetric() -> None:
    rng = np.random.default_rng(9)
    v = np.abs(rng.standard_normal(27)) + 0.1
    R = _make_corr(27)
    Sigma = build_drd(v, R)
    np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-12, err_msg="build_drd result must be symmetric")


def test_build_drd_diagonal_matches_variance() -> None:
    """Diagonal of Sigma = D R D should equal v_i (since R_ii = 1)."""
    rng = np.random.default_rng(11)
    v = np.abs(rng.standard_normal(27)) + 0.1
    R = _make_corr(27)
    Sigma = build_drd(v, R)
    np.testing.assert_allclose(
        np.diag(Sigma), v, atol=1e-10, err_msg="Diagonal of Sigma=DRD should equal variance v"
    )


def test_build_drd_shape() -> None:
    v = np.ones(27)
    R = np.eye(27)
    Sigma = build_drd(v, R)
    assert Sigma.shape == (27, 27)


# ---------------------------------------------------------------------------
# har_lags
# ---------------------------------------------------------------------------

def test_har_lags_shapes_1d() -> None:
    rng = np.random.default_rng(0)
    z = rng.standard_normal(100)
    d, w, m = har_lags(z, 30)
    assert np.isscalar(d) or d.shape == (), "Daily lag should be scalar for 1D z"


def test_har_lags_shapes_2d() -> None:
    rng = np.random.default_rng(0)
    z = rng.standard_normal((100, 5))
    d, w, m = har_lags(z, 30)
    assert d.shape == (5,), f"Daily lag shape should be (5,), got {d.shape}"
    assert w.shape == (5,), f"Weekly lag shape should be (5,), got {w.shape}"
    assert m.shape == (5,), f"Monthly lag shape should be (5,), got {m.shape}"


def test_har_lags_insufficient_history_returns_none() -> None:
    z = np.ones(10)
    result = har_lags(z, 5)
    assert result is None, "har_lags must return None when t < 23"


def test_har_lags_weekly_mean_correct() -> None:
    """Weekly average should be mean of z[t-5:t-1] (lags 2–5)."""
    z = np.arange(50, dtype=float)
    t = 30
    d, w, m = har_lags(z, t)
    expected_w = np.mean(z[t - 5: t - 1])
    np.testing.assert_allclose(w, expected_w, atol=1e-12)

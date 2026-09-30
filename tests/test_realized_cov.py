"""Tests for realized covariance estimation routines."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.realized_cov import (
    compute_rc_simple,
    extract_drd_components,
    compute_realized_quarticity,
    generate_synthetic_rc_series,
)
from src.nearest_pd import is_positive_definite


# ---------------------------------------------------------------------------
# compute_rc_simple
# ---------------------------------------------------------------------------

def _make_returns(M: int = 78, N: int = 27, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((M, N)) * 0.001


def test_rc_simple_shape() -> None:
    r = _make_returns(78, 27)
    H = compute_rc_simple(r)
    assert H.shape == (27, 27), f"Expected (27, 27), got {H.shape}"


def test_rc_simple_symmetric() -> None:
    r = _make_returns(78, 27, seed=1)
    H = compute_rc_simple(r)
    np.testing.assert_allclose(H, H.T, atol=1e-14, err_msg="H must be symmetric")


def test_rc_simple_psd() -> None:
    """RC must be positive-semidefinite (eigenvalues >= 0)."""
    r = _make_returns(200, 27, seed=2)
    H = compute_rc_simple(r)
    evals = np.linalg.eigvalsh(H)
    assert np.all(evals >= -1e-10), f"Min eigenvalue {evals.min():.3e} violates PSD"


def test_rc_simple_zero_returns() -> None:
    """All-zero returns must produce zero matrix."""
    r = np.zeros((78, 27))
    H = compute_rc_simple(r)
    np.testing.assert_allclose(H, 0, atol=1e-14)


def test_rc_simple_scale_invariance() -> None:
    """H(c*r) = c² * H(r)."""
    r = _make_returns(78, 5, seed=3)
    c = 2.0
    H1 = compute_rc_simple(r)
    H2 = compute_rc_simple(c * r)
    np.testing.assert_allclose(H2, c ** 2 * H1, rtol=1e-10)


def test_rc_simple_single_bar() -> None:
    """RC with M=1 bar should equal outer product of that return vector."""
    r = np.array([[0.1, -0.05, 0.02]])
    H = compute_rc_simple(r)
    expected = np.outer(r[0], r[0])
    np.testing.assert_allclose(H, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# extract_drd_components
# ---------------------------------------------------------------------------

def _make_pd_matrix(n: int = 27, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


def test_extract_drd_shapes() -> None:
    H = _make_pd_matrix(27)
    v, D, R = extract_drd_components(H)
    assert v.shape == (27,), f"v shape {v.shape}"
    assert D.shape == (27, 27), f"D shape {D.shape}"
    assert R.shape == (27, 27), f"R shape {R.shape}"


def test_extract_drd_v_equals_diag_H() -> None:
    H = _make_pd_matrix(5)
    v, D, R = extract_drd_components(H)
    np.testing.assert_allclose(v, np.diag(H), atol=1e-12)


def test_extract_drd_unit_diagonal_R() -> None:
    H = _make_pd_matrix(6)
    v, D, R = extract_drd_components(H)
    np.testing.assert_allclose(np.diag(R), np.ones(6), atol=1e-10,
                                err_msg="R must have unit diagonal")


def test_extract_drd_correlation_bounds() -> None:
    H = _make_pd_matrix(8)
    v, D, R = extract_drd_components(H)
    off_diag = R - np.diag(np.diag(R))
    assert np.all(off_diag >= -1 - 1e-10), "Off-diagonal R entries must be >= -1"
    assert np.all(off_diag <= 1 + 1e-10), "Off-diagonal R entries must be <= 1"


def test_drd_reconstruction() -> None:
    """D @ R @ D should reconstruct H."""
    H = _make_pd_matrix(6)
    v, D, R = extract_drd_components(H)
    H_rec = D @ R @ D
    np.testing.assert_allclose(H_rec, H, rtol=1e-8,
                                err_msg="D @ R @ D must recover H")


def test_extract_drd_symmetric_R() -> None:
    H = _make_pd_matrix(5)
    v, D, R = extract_drd_components(H)
    np.testing.assert_allclose(R, R.T, atol=1e-12, err_msg="R must be symmetric")


# ---------------------------------------------------------------------------
# compute_realized_quarticity
# ---------------------------------------------------------------------------

def test_quarticity_positive() -> None:
    r = _make_returns(78, 10, seed=5)
    RQ = compute_realized_quarticity(r)
    assert np.all(RQ >= 0), "Realized quarticity must be non-negative"


def test_quarticity_shape() -> None:
    r = _make_returns(78, 27, seed=6)
    RQ = compute_realized_quarticity(r)
    assert RQ.shape == (27,), f"Expected (27,), got {RQ.shape}"


def test_quarticity_formula() -> None:
    """Validate against manual calculation for small case."""
    M = 10
    r = _make_returns(M, 3, seed=7)
    RQ = compute_realized_quarticity(r)
    expected = (M / 3) * np.sum(r ** 4, axis=0)
    np.testing.assert_allclose(RQ, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# generate_synthetic_rc_series
# ---------------------------------------------------------------------------

def test_synthetic_series_shapes() -> None:
    H, v, x = generate_synthetic_rc_series(n_days=50, n_assets=27, seed=42)
    assert H.shape == (50, 27, 27), f"H shape {H.shape}"
    assert v.shape == (50, 27), f"v shape {v.shape}"
    assert x.shape == (50, 351), f"x shape {x.shape}"


def test_synthetic_series_pd() -> None:
    """Every H_t in the synthetic series must be positive definite."""
    H, v, x = generate_synthetic_rc_series(n_days=30, n_assets=27, seed=0)
    for t in range(len(H)):
        assert is_positive_definite(H[t]), f"H[{t}] is not PD"


def test_synthetic_series_reproducible() -> None:
    """Two calls with the same seed must produce identical results."""
    H1, v1, x1 = generate_synthetic_rc_series(n_days=20, n_assets=27, seed=7)
    H2, v2, x2 = generate_synthetic_rc_series(n_days=20, n_assets=27, seed=7)
    np.testing.assert_array_equal(H1, H2)
    np.testing.assert_array_equal(v1, v2)
    np.testing.assert_array_equal(x1, x2)


def test_synthetic_series_different_seeds() -> None:
    """Different seeds must produce different results."""
    H1, _, _ = generate_synthetic_rc_series(n_days=20, n_assets=27, seed=1)
    H2, _, _ = generate_synthetic_rc_series(n_days=20, n_assets=27, seed=2)
    assert not np.allclose(H1, H2), "Different seeds should give different results"


def test_synthetic_v_matches_h_diagonal() -> None:
    """v_t should equal diag(H_t)."""
    H, v, x = generate_synthetic_rc_series(n_days=20, n_assets=27, seed=3)
    for t in range(len(H)):
        np.testing.assert_allclose(v[t], np.diag(H[t]), atol=1e-12)

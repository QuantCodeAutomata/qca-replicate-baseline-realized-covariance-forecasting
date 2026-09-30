"""Tests for Euclidean, Frobenius, and QLIKE covariance forecast loss functions."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.losses import (
    euclidean_loss,
    frobenius_loss,
    qlike_loss,
    compute_losses,
    mean_losses,
)


def _make_pd(n: int, scale: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) * scale
    return A @ A.T + n * np.eye(n)


# ---------------------------------------------------------------------------
# euclidean_loss
# ---------------------------------------------------------------------------

def test_euclidean_loss_zero_for_identical_matrices() -> None:
    H = _make_pd(5)
    assert euclidean_loss(H, H) == pytest.approx(0.0, abs=1e-12)


def test_euclidean_loss_positive() -> None:
    H = _make_pd(5, seed=1)
    S = _make_pd(5, seed=2)
    assert euclidean_loss(H, S) > 0


def test_euclidean_loss_symmetric_inputs() -> None:
    """Loss should be the same regardless of argument order symmetry (H - S) vs (S - H)."""
    H = _make_pd(5, seed=3)
    S = _make_pd(5, seed=4)
    l1 = euclidean_loss(H, S)
    l2 = euclidean_loss(S, H)
    assert l1 == pytest.approx(l2, rel=1e-10)


def test_euclidean_loss_known_value() -> None:
    """For a 2×2 case with H=I and S=0, Euclidean loss = sqrt(3) (vech includes diag)."""
    H = np.eye(2)
    S = np.zeros((2, 2))
    # vech lower-tri (including diag) of (H - S) = [1, 0, 1] → sqrt(1+0+1) = sqrt(2)
    # Note: vech of symmetric 2x2: (0,0),(1,0),(1,1) → [1, 0, 1]
    expected = np.sqrt(2.0)
    assert euclidean_loss(H, S) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# frobenius_loss
# ---------------------------------------------------------------------------

def test_frobenius_loss_zero_for_identical_matrices() -> None:
    H = _make_pd(7)
    assert frobenius_loss(H, H) == pytest.approx(0.0, abs=1e-12)


def test_frobenius_loss_positive() -> None:
    H = _make_pd(5, seed=5)
    S = _make_pd(5, seed=6)
    assert frobenius_loss(H, S) > 0


def test_frobenius_loss_equals_matrix_norm() -> None:
    H = _make_pd(4, seed=7)
    S = _make_pd(4, seed=8)
    expected = np.linalg.norm(H - S, "fro")
    assert frobenius_loss(H, S) == pytest.approx(expected, rel=1e-10)


def test_frobenius_loss_symmetric() -> None:
    H = _make_pd(5, seed=9)
    S = _make_pd(5, seed=10)
    assert frobenius_loss(H, S) == pytest.approx(frobenius_loss(S, H), rel=1e-10)


# ---------------------------------------------------------------------------
# qlike_loss
# ---------------------------------------------------------------------------

def test_qlike_loss_finite_for_pd_inputs() -> None:
    H = _make_pd(6, seed=11)
    S = _make_pd(6, seed=12)
    q = qlike_loss(H, S)
    assert np.isfinite(q), f"QLIKE should be finite for PD inputs, got {q}"


def test_qlike_loss_positive_for_mismatched() -> None:
    H = _make_pd(4, seed=13)
    S = _make_pd(4, seed=14) * 2  # deliberately different scale
    q = qlike_loss(H, S)
    assert q > 0, "QLIKE should be positive for mismatched matrices"


def test_qlike_loss_not_symmetric() -> None:
    """QLIKE(H, S) ≠ QLIKE(S, H) in general — it is asymmetric."""
    H = _make_pd(4, seed=15)
    S = _make_pd(4, seed=16)
    q1 = qlike_loss(H, S)
    q2 = qlike_loss(S, H)
    assert not np.isclose(q1, q2), "QLIKE should generally be asymmetric"


def test_qlike_loss_known_formula() -> None:
    """For H = I and S = c*I, QLIKE = N*log(c) + N/c."""
    N = 4
    c = 3.0
    H = np.eye(N)
    S = c * np.eye(N)
    expected = N * np.log(c) + N / c
    q = qlike_loss(H, S)
    assert q == pytest.approx(expected, rel=1e-6), f"QLIKE={q}, expected={expected}"


# ---------------------------------------------------------------------------
# compute_losses / mean_losses
# ---------------------------------------------------------------------------

def test_compute_losses_shape() -> None:
    T = 20
    H_series = [_make_pd(5, seed=i) for i in range(T)]
    S_series = [_make_pd(5, seed=i + 100) for i in range(T)]
    df = compute_losses(H_series, S_series)
    assert df.shape == (T, 3), f"Expected ({T}, 3), got {df.shape}"
    assert set(df.columns) == {"euclidean", "frobenius", "qlike"}


def test_compute_losses_non_negative() -> None:
    T = 10
    H_series = [_make_pd(4, seed=i) for i in range(T)]
    S_series = [_make_pd(4, seed=i + 50) for i in range(T)]
    df = compute_losses(H_series, S_series)
    assert (df[["euclidean", "frobenius"]] >= 0).all().all()


def test_mean_losses_keys() -> None:
    T = 10
    H_series = [_make_pd(4, seed=i) for i in range(T)]
    S_series = [_make_pd(4, seed=i + 50) for i in range(T)]
    df = compute_losses(H_series, S_series)
    ml = mean_losses(df)
    assert "euclidean" in ml
    assert "frobenius" in ml
    assert "qlike" in ml


def test_mean_losses_are_finite() -> None:
    T = 10
    H_series = [_make_pd(4, seed=i) for i in range(T)]
    S_series = [_make_pd(4, seed=i + 20) for i in range(T)]
    df = compute_losses(H_series, S_series)
    ml = mean_losses(df)
    for k, v in ml.items():
        assert np.isfinite(v), f"mean_losses[{k}] = {v} should be finite"

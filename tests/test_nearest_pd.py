"""Tests for Higham's nearest positive-definite matrix algorithm."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.nearest_pd import nearest_pd, is_positive_definite, ensure_pd


def _random_pd(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


def _random_indefinite(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    S = (A + A.T) / 2
    # force smallest eigenvalue negative
    vals, vecs = np.linalg.eigh(S)
    vals[0] = -0.5
    return vecs @ np.diag(vals) @ vecs.T


def test_already_pd_matrix_unchanged() -> None:
    """PD matrix should remain PD; result should be close to the input."""
    A = _random_pd(5)
    B = nearest_pd(A)
    assert is_positive_definite(B), "Output must be PD"
    assert np.allclose(A, B, atol=1e-6), "PD input should not change materially"


def test_non_pd_becomes_pd() -> None:
    """An indefinite matrix must become PD after nearest_pd."""
    A = _random_indefinite(6)
    assert not is_positive_definite(A), "Pre-condition: input is NOT PD"
    B = nearest_pd(A)
    assert is_positive_definite(B), "nearest_pd output must be PD"


def test_symmetric_output() -> None:
    """Output must always be symmetric."""
    rng = np.random.default_rng(7)
    A = rng.standard_normal((8, 8))
    B = nearest_pd(A)
    assert np.allclose(B, B.T, atol=1e-12), "Output must be symmetric"


def test_near_singular_matrix() -> None:
    """Near-singular matrix (rank-deficient + tiny noise) should be repaired."""
    n = 10
    rng = np.random.default_rng(3)
    low_rank = rng.standard_normal((n, 3))
    A = low_rank @ low_rank.T + 1e-14 * np.eye(n)
    B = nearest_pd(A)
    assert is_positive_definite(B, tol=1e-9), "Near-singular matrix must be repaired to PD"


def test_ensure_pd_returns_corrected_flag() -> None:
    """ensure_pd must return (matrix, True) for a non-PD input."""
    A = _random_indefinite(4)
    B, corrected = ensure_pd(A)
    assert corrected is True, "Flag should be True for non-PD input"
    assert is_positive_definite(B), "Corrected output must be PD"


def test_ensure_pd_no_correction_needed() -> None:
    """ensure_pd must return (matrix, False) for an already-PD input."""
    A = _random_pd(5)
    B, corrected = ensure_pd(A)
    assert corrected is False, "Flag should be False for already-PD input"
    assert np.allclose(A, B, atol=1e-10), "PD input should pass through unchanged"


def test_is_positive_definite_true() -> None:
    """is_positive_definite should return True for a PD matrix."""
    A = _random_pd(6)
    assert is_positive_definite(A) is True


def test_is_positive_definite_false() -> None:
    """is_positive_definite should return False for an indefinite matrix."""
    A = _random_indefinite(6)
    assert is_positive_definite(A) is False


def test_single_element() -> None:
    """A 1×1 positive scalar must remain PD."""
    A = np.array([[3.0]])
    B = nearest_pd(A)
    assert is_positive_definite(B), "1x1 PD scalar must stay PD"
    assert B[0, 0] > 0


def test_min_eigenvalue_positive() -> None:
    """All eigenvalues of nearest_pd output must be strictly positive."""
    A = _random_indefinite(7)
    B = nearest_pd(A, epsilon=1e-8)
    evals = np.linalg.eigvalsh(B)
    assert np.all(evals > 0), f"Min eigenvalue {evals.min():.3e} must be > 0"


def test_2x2_known_case() -> None:
    """Validate against a known 2×2 case with a zero eigenvalue."""
    # Symmetric but singular
    A = np.array([[1.0, 1.0], [1.0, 1.0]])
    B = nearest_pd(A)
    assert is_positive_definite(B)
    assert B.shape == (2, 2)

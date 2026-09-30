"""
src/nearest_pd.py
=================
Higham's alternating-projections algorithm for the nearest positive-definite
matrix problem.

Reference
---------
Higham, N. J. (1988). *Computing a nearest symmetric positive semidefinite
matrix*. Linear Algebra and Its Applications, 103, 103–118.
https://doi.org/10.1016/0024-3795(88)90223-6
"""

from __future__ import annotations

import numpy as np


def nearest_pd(A: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Return the nearest symmetric positive-definite (SPD) matrix to *A*.

    Uses Higham's (1988) alternating-projections algorithm with Dykstra's
    correction to guarantee convergence to the *nearest* SPD matrix in the
    Frobenius norm:

    1. **PSD projection** ``P(X)`` — eigen-decompose ``X``, threshold
       negative eigenvalues to *epsilon*, reconstruct.
    2. **Symmetric projection** ``S(X) = (X + Xᵀ) / 2``

    Dykstra's correction is accumulated between iterations to account for the
    non-convexity of the intersection.

    Parameters
    ----------
    A:
        Square (N × N) matrix to project.  Need not be symmetric.
    epsilon:
        Minimum eigenvalue floor; all eigenvalues of the result will be
        ≥ *epsilon*.

    Returns
    -------
    np.ndarray
        Nearest SPD matrix (N × N) in the Frobenius-norm sense, with all
        eigenvalues ≥ *epsilon*.

    Notes
    -----
    Convergence criterion: relative Frobenius change
    ``‖X_k − X_{k−1}‖_F / max(1, ‖X_{k−1}‖_F) < 1e-9`` or 1 000 iterations.
    """
    A = np.array(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected a square 2-D array; got shape {A.shape}.")

    tol = 1e-9
    max_iter = 1_000

    # Dykstra's correction term — initialised to zero
    delta_s = np.zeros_like(A)

    # Working matrices: Y is the symmetrised candidate, X the PSD projection
    Y = (A + A.T) / 2.0
    X = Y.copy()

    for _ in range(max_iter):
        X_prev = X.copy()

        # --- PSD projection (applied to Y minus the Dykstra correction) ---
        R = Y - delta_s
        eigvals, eigvecs = np.linalg.eigh(R)          # eigh exploits symmetry
        eigvals_clipped = np.maximum(eigvals, epsilon)  # eigenvalue floor
        X = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

        # Update Dykstra correction: tracks the cumulative "overshoot"
        delta_s = X - R

        # --- Symmetric projection ---
        Y = (X + X.T) / 2.0

        # Convergence: relative change in Frobenius norm
        norm_change = np.linalg.norm(X - X_prev, "fro")
        norm_prev = max(1.0, np.linalg.norm(X_prev, "fro"))
        if norm_change / norm_prev < tol:
            break

    # Final symmetry enforcement and hard eigenvalue floor
    result = (Y + Y.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(result)
    eigvals = np.maximum(eigvals, epsilon)
    result = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return result


def is_positive_definite(A: np.ndarray, tol: float = 1e-10) -> bool:
    """Test whether *A* is symmetric positive definite.

    Uses an eigenvalue decomposition (via ``eigh``) rather than Cholesky so
    that near-singular matrices do not raise exceptions.

    Parameters
    ----------
    A:
        Square (N × N) matrix to test.
    tol:
        Eigenvalue threshold; every eigenvalue must exceed *tol* for the
        matrix to be considered PD.

    Returns
    -------
    bool
        ``True`` iff *A* is symmetric (within floating-point precision) and
        all eigenvalues are strictly greater than *tol*.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    if not np.allclose(A, A.T, atol=tol, rtol=0.0):
        return False
    eigvals = np.linalg.eigvalsh(A)
    return bool(np.all(eigvals > tol))


def ensure_pd(
    A: np.ndarray,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, bool]:
    """Return *A* if already PD, otherwise return the nearest PD correction.

    Provides a lightweight guard that avoids the cost of :func:`nearest_pd`
    when *A* is already positive definite.

    Parameters
    ----------
    A:
        Square (N × N) matrix to check and possibly correct.
    epsilon:
        Minimum eigenvalue floor passed to :func:`nearest_pd` when correction
        is needed.  Also used as the PD tolerance in :func:`is_positive_definite`.

    Returns
    -------
    tuple[np.ndarray, bool]
        ``(matrix, was_corrected)`` — *matrix* is guaranteed SPD;
        *was_corrected* is ``True`` iff :func:`nearest_pd` was invoked.
    """
    if is_positive_definite(A, tol=epsilon):
        return np.asarray(A, dtype=float), False
    return nearest_pd(A, epsilon=epsilon), True

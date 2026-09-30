"""
src/losses.py
=============
Loss functions for evaluating realized-covariance matrix forecasts.

Three complementary losses are implemented:

* **Euclidean** — L2 norm of the half-vectorisation (unique elements) of the
  forecast error matrix.
* **Frobenius** — standard matrix Frobenius norm of the forecast error.
* **QLIKE**     — Gaussian quasi-likelihood loss (Patton, 2011).

References
----------
Laurent, S., Rombouts, J. V., & Violante, F. (2013). On loss functions and
ranking forecasting performances of multivariate volatility models. *Journal of
Econometrics*, 173(1), 1–10.

Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility
proxies. *Journal of Econometrics*, 160(1), 246–256.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------


def euclidean_loss(H: np.ndarray, Sigma_hat: np.ndarray) -> float:
    r"""Euclidean (half-vectorisation) loss between *H* and *Sigma_hat*.

    Computes:

    .. math::

        L^E = \sqrt{\operatorname{vech}(H - \hat\Sigma)^{\!\top}
                    \operatorname{vech}(H - \hat\Sigma)}

    where :math:`\operatorname{vech}(\cdot)` stacks the lower-triangular
    entries (diagonal included) of its argument into a column vector of
    length :math:`N(N+1)/2`.  This equals the L2 norm of the unique elements
    of the symmetric forecast-error matrix.

    Parameters
    ----------
    H:
        (N × N) realised covariance matrix (proxy for the true variance).
    Sigma_hat:
        (N × N) forecast covariance matrix.

    Returns
    -------
    float
        Non-negative scalar loss.
    """
    H = np.asarray(H, dtype=float)
    Sigma_hat = np.asarray(Sigma_hat, dtype=float)
    diff = H - Sigma_hat
    # Extract lower-triangular entries (row >= col), including diagonal
    rows, cols = np.tril_indices(diff.shape[0])
    vech_diff = diff[rows, cols]
    return float(np.sqrt(np.dot(vech_diff, vech_diff)))


def frobenius_loss(H: np.ndarray, Sigma_hat: np.ndarray) -> float:
    r"""Frobenius norm loss between *H* and *Sigma_hat*.

    Computes:

    .. math::

        L^F = \sqrt{\operatorname{Tr}\!\left[(H - \hat\Sigma)^\top
                                              (H - \hat\Sigma)\right]}
            = \|H - \hat\Sigma\|_F

    Parameters
    ----------
    H:
        (N × N) realised covariance matrix.
    Sigma_hat:
        (N × N) forecast covariance matrix.

    Returns
    -------
    float
        Non-negative scalar loss.
    """
    H = np.asarray(H, dtype=float)
    Sigma_hat = np.asarray(Sigma_hat, dtype=float)
    return float(np.linalg.norm(H - Sigma_hat, "fro"))


def qlike_loss(
    H: np.ndarray,
    Sigma_hat: np.ndarray,
    regularize: float = 1e-8,
) -> float:
    r"""Gaussian QLIKE (quasi-likelihood) loss.

    Computes:

    .. math::

        L^Q = \log\det(\hat\Sigma) + \operatorname{Tr}(\hat\Sigma^{-1} H)

    which is proportional to the negative Gaussian log-likelihood evaluated at
    the realised matrix *H* using the forecast *Sigma_hat* as the covariance.

    Parameters
    ----------
    H:
        (N × N) realised covariance matrix.
    Sigma_hat:
        (N × N) forecast covariance matrix (must be positive definite).
    regularize:
        Small diagonal shift added to *Sigma_hat* when it is near-singular
        (i.e. when ``np.linalg.matrix_rank`` detects rank deficiency or
        ``slogdet`` returns a non-positive sign).

    Returns
    -------
    float
        Scalar loss value, or ``np.nan`` if the computation fails (e.g. if
        *Sigma_hat* is numerically singular even after regularisation).
    """
    H = np.asarray(H, dtype=float)
    Sigma_hat = np.asarray(Sigma_hat, dtype=float)
    n = H.shape[0]

    def _try_compute(S: np.ndarray) -> float | None:
        """Attempt QLIKE computation; return None on failure."""
        try:
            sign, log_det = np.linalg.slogdet(S)
            if sign <= 0:
                return None
            S_inv = np.linalg.inv(S)
            trace_term = float(np.trace(S_inv @ H))
            return float(log_det + trace_term)
        except np.linalg.LinAlgError:
            return None

    result = _try_compute(Sigma_hat)
    if result is None:
        # Regularise and retry once
        result = _try_compute(Sigma_hat + regularize * np.eye(n))
    return result if result is not None else np.nan


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def compute_losses(
    H_series: list[np.ndarray],
    Sigma_hat_series: list[np.ndarray],
) -> pd.DataFrame:
    """Compute all three losses for a sequence of forecast/realised pairs.

    Parameters
    ----------
    H_series:
        Length-T list of (N × N) realised covariance matrices.
    Sigma_hat_series:
        Length-T list of (N × N) forecast covariance matrices.

    Returns
    -------
    pd.DataFrame
        Shape (T, 3) with columns ``euclidean``, ``frobenius``, ``qlike``.
        Entries are ``np.nan`` for time steps where a loss cannot be computed.

    Raises
    ------
    ValueError
        If *H_series* and *Sigma_hat_series* have different lengths.
    """
    if len(H_series) != len(Sigma_hat_series):
        raise ValueError(
            f"H_series length ({len(H_series)}) does not match "
            f"Sigma_hat_series length ({len(Sigma_hat_series)})."
        )
    T = len(H_series)
    records = []
    for t in range(T):
        H = H_series[t]
        S = Sigma_hat_series[t]
        records.append(
            {
                "euclidean": euclidean_loss(H, S),
                "frobenius": frobenius_loss(H, S),
                "qlike":     qlike_loss(H, S),
            }
        )
    return pd.DataFrame(records, columns=["euclidean", "frobenius", "qlike"])


def mean_losses(df: pd.DataFrame) -> dict[str, float]:
    """Compute mean (ignoring NaN) of each loss column.

    Parameters
    ----------
    df:
        DataFrame as returned by :func:`compute_losses`.

    Returns
    -------
    dict[str, float]
        ``{'euclidean': float, 'frobenius': float, 'qlike': float}``.
    """
    return {col: float(df[col].mean(skipna=True)) for col in df.columns}

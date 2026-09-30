"""
src/utils.py
============
Core constants and utility functions for the realized-covariance forecasting
project.

Provides:
- Fixed-order DJIA universe of 27 tickers with sector metadata.
- Vech (half-vectorisation) operators for correlation matrices and lower-
  triangular Cholesky factors.
- HAR lag construction following the config.yaml specification.
- NYSE trading-day calendar builder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Universe constants
# ---------------------------------------------------------------------------

# Tickers are sorted alphabetically within each sector; sectors appear in the
# order: IT, Communication Services, Consumer Discretionary, Consumer Staples,
# Health Care, Financials, Industrials, Energy.
TICKERS: list[str] = [
    # Information Technology (6)
    "AAPL", "CRM", "CSCO", "IBM", "INTC", "MSFT",
    # Communication Services (2)
    "DIS", "VZ",
    # Consumer Discretionary (3)
    "HD", "MCD", "NKE",
    # Consumer Staples (3)
    "KO", "PG", "WMT",
    # Health Care (4)
    "AMGN", "JNJ", "MRK", "UNH",
    # Financials (4)
    "AXP", "GS", "JPM", "TRV",
    # Industrials (4)
    "BA", "CAT", "HON", "MMM",
    # Energy (1)
    "CVX",
]

# Reverse mapping: ticker → sector name
SECTOR_MAP: dict[str, str] = {
    **{t: "Information Technology"   for t in ["AAPL", "CRM", "CSCO", "IBM", "INTC", "MSFT"]},
    **{t: "Communication Services"   for t in ["DIS", "VZ"]},
    **{t: "Consumer Discretionary"   for t in ["HD", "MCD", "NKE"]},
    **{t: "Consumer Staples"         for t in ["KO", "PG", "WMT"]},
    **{t: "Health Care"              for t in ["AMGN", "JNJ", "MRK", "UNH"]},
    **{t: "Financials"               for t in ["AXP", "GS", "JPM", "TRV"]},
    **{t: "Industrials"              for t in ["BA", "CAT", "HON", "MMM"]},
    **{t: "Energy"                   for t in ["CVX"]},
}

N_ASSETS: int = 27
N_PAIRS: int = N_ASSETS * (N_ASSETS - 1) // 2   # 351 — strictly lower-triangular entries
_N_CHOL:  int = N_ASSETS * (N_ASSETS + 1) // 2  # 378 — lower-triangular + diagonal entries

assert len(TICKERS) == N_ASSETS, "TICKERS length mismatch."
assert N_PAIRS == 351

# Pre-computed index tuples (avoids recomputation at call sites)
# (i, j) with i > j in row-major order: (1,0), (2,0), (2,1), ..., (26,25)
VECH_PAIRS: list[tuple[int, int]] = [
    (i, j) for i in range(1, N_ASSETS) for j in range(i)
]

# (i, j) with i >= j in row-major order: (0,0), (1,0), (1,1), ..., (26,26)
VECH_CHOL_IDX: list[tuple[int, int]] = [
    (i, j) for i in range(N_ASSETS) for j in range(i + 1)
]

assert len(VECH_PAIRS) == N_PAIRS
assert len(VECH_CHOL_IDX) == _N_CHOL

# Vectorised index arrays for fast NumPy fancy-indexing
_VP_ROWS = np.array([p[0] for p in VECH_PAIRS], dtype=np.intp)
_VP_COLS = np.array([p[1] for p in VECH_PAIRS], dtype=np.intp)
_VC_ROWS = np.array([p[0] for p in VECH_CHOL_IDX], dtype=np.intp)
_VC_COLS = np.array([p[1] for p in VECH_CHOL_IDX], dtype=np.intp)


# ---------------------------------------------------------------------------
# Vech operators — correlation matrices
# ---------------------------------------------------------------------------


def vech_corr(R: np.ndarray) -> np.ndarray:
    """Extract strictly lower-triangular off-diagonal entries of *R*.

    Parameters
    ----------
    R:
        (N × N) correlation matrix.

    Returns
    -------
    np.ndarray
        1-D array of length ``N_PAIRS = N*(N-1)/2 = 351`` following
        ``VECH_PAIRS`` row-major ordering: (1,0), (2,0), (2,1), …
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"Expected a square 2-D array; got shape {R.shape}.")
    return R[_VP_ROWS, _VP_COLS]


def vech_corr_inv(x: np.ndarray, N: int = N_ASSETS) -> np.ndarray:
    """Reconstruct a symmetric correlation matrix from a vech vector.

    Parameters
    ----------
    x:
        1-D array of length ``N*(N-1)/2`` containing the strictly
        lower-triangular entries in ``VECH_PAIRS`` ordering.
    N:
        Matrix dimension (default 27).

    Returns
    -------
    np.ndarray
        (N × N) symmetric matrix with unit diagonal.
    """
    x = np.asarray(x, dtype=float)
    expected = N * (N - 1) // 2
    if x.shape != (expected,):
        raise ValueError(f"Expected x of length {expected}; got {x.shape}.")
    R = np.eye(N)
    rows = np.array([p[0] for p in [(i, j) for i in range(1, N) for j in range(i)]])
    cols = np.array([p[1] for p in [(i, j) for i in range(1, N) for j in range(i)]])
    R[rows, cols] = x
    R[cols, rows] = x  # enforce symmetry
    return R


# ---------------------------------------------------------------------------
# Vech operators — Cholesky factors
# ---------------------------------------------------------------------------


def vech_chol(C: np.ndarray) -> np.ndarray:
    """Extract lower-triangular entries (including diagonal) from *C*.

    Parameters
    ----------
    C:
        (N × N) lower-triangular matrix (e.g. a Cholesky factor).

    Returns
    -------
    np.ndarray
        1-D array of length ``N*(N+1)/2 = 378`` following ``VECH_CHOL_IDX``
        row-major ordering: (0,0), (1,0), (1,1), …
    """
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"Expected a square 2-D array; got shape {C.shape}.")
    return C[_VC_ROWS, _VC_COLS]


def vech_chol_inv(y: np.ndarray, N: int = N_ASSETS) -> np.ndarray:
    """Reconstruct a lower-triangular matrix from a vech-Cholesky vector.

    Parameters
    ----------
    y:
        1-D array of length ``N*(N+1)/2`` in ``VECH_CHOL_IDX`` ordering.
    N:
        Matrix dimension (default 27).

    Returns
    -------
    np.ndarray
        (N × N) lower-triangular matrix.
    """
    y = np.asarray(y, dtype=float)
    expected = N * (N + 1) // 2
    if y.shape != (expected,):
        raise ValueError(f"Expected y of length {expected}; got {y.shape}.")
    L = np.zeros((N, N))
    rows = np.array([p[0] for p in [(i, j) for i in range(N) for j in range(i + 1)]])
    cols = np.array([p[1] for p in [(i, j) for i in range(N) for j in range(i + 1)]])
    L[rows, cols] = y
    return L


# ---------------------------------------------------------------------------
# Covariance assembly
# ---------------------------------------------------------------------------


def build_drd(v: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Assemble a covariance matrix Σ = D R D where D = diag(√v).

    Parameters
    ----------
    v:
        (N,) array of non-negative variances (diagonal of Σ).
    R:
        (N × N) correlation matrix.

    Returns
    -------
    np.ndarray
        (N × N) symmetric covariance matrix.
    """
    v = np.asarray(v, dtype=float)
    R = np.asarray(R, dtype=float)
    D = np.diag(np.sqrt(v))
    return D @ R @ D


# ---------------------------------------------------------------------------
# HAR lag construction
# ---------------------------------------------------------------------------


def har_lags(
    z: np.ndarray,
    t: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return HAR daily, weekly, and monthly lag components at time index *t*.

    Implements the non-overlapping lag structure from config.yaml:

    * **Daily**   : ``z[t-1]``        — lag 1.
    * **Weekly**  : mean of ``z[t-5:t-1]`` — lags 2–5 (4 observations).
    * **Monthly** : mean of ``z[t-22:t-5]`` — lags 6–22 (17 observations).

    Note: the slice ``z[t-22:t-5]`` gives indices ``t-22, …, t-6`` (17 values,
    i.e. lags 6 through 22 inclusive), which matches the config specification.

    Parameters
    ----------
    z:
        1-D array of shape ``(T,)`` or 2-D array of shape ``(T, K)`` where
        rows are time and columns are features (e.g. vech entries).
    t:
        Current time index (0-based).  Must satisfy ``t >= 23``; returns
        ``None`` if there is insufficient history.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray] | None
        ``(z_daily, z_weekly, z_monthly)`` — each has the same shape as
        ``z[0]`` (scalar or 1-D of length K).  Returns ``None`` if ``t < 23``.
    """
    if t < 23:
        return None

    z_daily   = z[t - 1]                       # lag 1
    z_weekly  = z[t - 5 : t - 1].mean(axis=0)  # lags 2–5  (4 values)
    z_monthly = z[t - 22 : t - 5].mean(axis=0) # lags 6–22 (17 values)
    return z_daily, z_weekly, z_monthly


# ---------------------------------------------------------------------------
# Trading calendar
# ---------------------------------------------------------------------------


def build_trading_calendar(start: str, end: str) -> pd.DatetimeIndex:
    """Build an approximate NYSE trading-day calendar between *start* and *end*.

    Tries to use ``pandas_market_calendars`` for an exact NYSE calendar; falls
    back to pandas business days (``pd.bdate_range``) if the package is not
    installed.

    Parameters
    ----------
    start:
        Inclusive start date in ``YYYY-MM-DD`` format.
    end:
        Inclusive end date in ``YYYY-MM-DD`` format.

    Returns
    -------
    pd.DatetimeIndex
        Sorted index of trading days (midnight timestamps).
    """
    try:
        import pandas_market_calendars as mcal  # optional dependency

        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=start, end_date=end)
        # date_range returns open/close timestamps; take the date part only
        sessions = mcal.date_range(schedule, frequency="1D")
        return sessions.normalize().unique().sort_values()
    except ImportError:
        # pandas business days as fallback — excludes weekends, not holidays
        return pd.bdate_range(start=start, end=end)

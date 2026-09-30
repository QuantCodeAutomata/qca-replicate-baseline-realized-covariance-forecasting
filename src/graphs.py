"""
Graph construction for Graph-HAR (GHAR) models.

Provides five types of adjacency structures used by GHARDRD:
  - Empty  (no edges)
  - Complete (all pairs connected)
  - Sector  (same-GICS-sector pairs connected)
  - Graphical Lasso (data-driven sparse precision graph)
  - Line graph on correlation pairs (for the correlation block)

All returned adjacency matrices are binary with zero diagonal unless noted.
Normalized weighted matrices are produced by :func:`normalize_graph`.

Config reference (config.yaml):
  graphs.vol_graphs      = [empty, complete, sector, graphical_lasso]
  graphs.corr_graphs     = [empty, complete, line]
  graphs.gl_cv_folds     = 5
  graphs.gl_nonzero_tol  = 1.0e-10
"""

import warnings

import numpy as np
from sklearn.covariance import GraphicalLassoCV

from .utils import N_ASSETS, N_PAIRS, SECTOR_MAP, TICKERS, VECH_PAIRS

# Build sector→[tickers] lookup from the flat ticker→sector SECTOR_MAP.
# This allows make_sector_graph to group tickers by sector correctly.
_SECTOR_GROUPS: dict = {}
for _t, _s in SECTOR_MAP.items():
    _SECTOR_GROUPS.setdefault(_s, []).append(_t)


# ---------------------------------------------------------------------------
# Basic topologies
# ---------------------------------------------------------------------------


def make_empty_graph(N: int = N_ASSETS) -> np.ndarray:
    """Return the N×N zero adjacency matrix (no edges).

    Parameters
    ----------
    N : int
        Number of nodes (default 27).

    Returns
    -------
    A : np.ndarray, shape (N, N)
        Zero matrix; used to reduce GHAR to plain HAR.
    """
    return np.zeros((N, N))


def make_complete_graph(N: int = N_ASSETS) -> np.ndarray:
    """Return the N×N complete-graph adjacency (all off-diagonal entries = 1).

    Parameters
    ----------
    N : int
        Number of nodes (default 27).

    Returns
    -------
    A : np.ndarray, shape (N, N)
        Off-diagonal entries are 1, diagonal is 0.
    """
    A = np.ones((N, N))
    np.fill_diagonal(A, 0.0)
    return A


def make_sector_graph(
    tickers: list = None,
    sector_map: dict = None,
) -> np.ndarray:
    """Build sector-membership adjacency: A[i,j]=1 iff i and j share a sector.

    Parameters
    ----------
    tickers : list of str, optional
        Ordered list of N ticker symbols.  Defaults to ``TICKERS`` from utils.
    sector_map : dict, optional
        Flat mapping from ticker to sector name (as in SECTOR_MAP from utils),
        OR mapping from sector name to list of tickers.  Both formats accepted.

    Returns
    -------
    A : np.ndarray, shape (N, N)
        Symmetric binary adjacency; diagonal is 0.
    """
    if tickers is None:
        tickers = TICKERS

    N = len(tickers)
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    # Build sector→[tickers] groups.
    # _SECTOR_GROUPS was built from the flat SECTOR_MAP at module load time.
    if sector_map is None:
        groups = _SECTOR_GROUPS
    else:
        # Accept both flat {ticker: sector} and grouped {sector: [tickers]} formats.
        first_val = next(iter(sector_map.values()))
        if isinstance(first_val, str):
            # flat format: {ticker: sector_name}
            groups = {}
            for t, s in sector_map.items():
                groups.setdefault(s, []).append(t)
        else:
            # already grouped format: {sector_name: [tickers]}
            groups = sector_map

    A = np.zeros((N, N))
    for sector_tickers in groups.values():
        idxs = [ticker_to_idx[t] for t in sector_tickers if t in ticker_to_idx]
        for i in idxs:
            for j in idxs:
                if i != j:
                    A[i, j] = 1.0

    return A  # diagonal is already 0


# ---------------------------------------------------------------------------
# Graphical Lasso
# ---------------------------------------------------------------------------


def estimate_gl_graph(
    X: np.ndarray,
    cv_folds: int = 5,
    tol: float = 1e-10,
    random_state: int = 0,
) -> np.ndarray:
    """Estimate a sparse graph from data using cross-validated Graphical Lasso.

    Fits sklearn's GraphicalLassoCV on the input data and thresholds the
    estimated precision matrix to obtain a binary adjacency.

    Parameters
    ----------
    X : np.ndarray, shape (T, N)
        Input data matrix (demeaned returns or innovations).
    cv_folds : int
        Number of cross-validation folds (default 5, matches config).
    tol : float
        Threshold below which precision entries are treated as zero
        (default 1e-10, matches config ``gl_nonzero_tol``).
    random_state : int
        Seed for reproducibility inside GraphicalLassoCV.

    Returns
    -------
    A : np.ndarray, shape (N, N)
        Binary adjacency matrix; A[i,j]=1 iff |Omega[i,j]|>tol, diagonal=0.
    """
    N = X.shape[1]
    try:
        glcv = GraphicalLassoCV(cv=cv_folds)
        glcv.fit(X)
        Omega = glcv.precision_   # estimated sparse precision matrix
    except Exception as exc:
        warnings.warn(
            f"GraphicalLassoCV failed ({exc}); returning zeros adjacency.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.zeros((N, N))

    # Threshold off-diagonal precision entries
    A = (np.abs(Omega) > tol).astype(float)
    np.fill_diagonal(A, 0.0)
    return A


# ---------------------------------------------------------------------------
# Symmetric normalisation
# ---------------------------------------------------------------------------


def normalize_graph(A: np.ndarray) -> np.ndarray:
    """Symmetrically normalise an adjacency matrix.

    Computes:
        W = O^{-1/2} A O^{-1/2},   O = diag(A @ 1)
    where 1 is the all-ones vector.  Zero-degree nodes receive all-zero rows
    and columns in W.

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
        Binary (or weighted) adjacency matrix with zero diagonal.

    Returns
    -------
    W : np.ndarray, shape (N, N)
        Symmetrically normalised adjacency.
    """
    degree = A.sum(axis=1)                                       # (N,)
    # Safe inverse-square-root: zero for isolated nodes
    d_inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 0.0
    d_inv_sqrt[nonzero] = degree[nonzero] ** (-0.5)
    # W_{ij} = d_i^{-1/2} * A_{ij} * d_j^{-1/2}
    W = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    return W


# ---------------------------------------------------------------------------
# Line graph for correlation pairs
# ---------------------------------------------------------------------------


def make_line_graph(
    vech_pairs: list = None,
    N: int = N_ASSETS,
) -> np.ndarray:
    """Build the line graph on correlation index pairs.

    Two pairs (i, j) and (k, l) share an edge iff they involve at least one
    common asset index, i.e. {i,j} ∩ {k,l} ≠ ∅.  This propagates information
    between correlations that share an asset in the GHAR correlation block.

    Parameters
    ----------
    vech_pairs : list of (int, int), optional
        Ordered pairs (i, j) with i > j defining the correlation vector x_t.
        Defaults to ``VECH_PAIRS`` from utils.
    N : int
        Number of assets (default 27); used only for validation.

    Returns
    -------
    A : np.ndarray, shape (N_pairs, N_pairs)
        Symmetric binary adjacency on correlation pairs; diagonal = 0.
        N_pairs = N*(N-1)//2 = 351 for N=27.
    """
    if vech_pairs is None:
        vech_pairs = VECH_PAIRS

    P = len(vech_pairs)                                    # = 351 for N=27
    A = np.zeros((P, P))

    # Vectorise the inner test via broadcasting
    pairs_arr = np.array(vech_pairs, dtype=np.int32)       # (P, 2)
    i_col = pairs_arr[:, 0]                                # (P,) — first index
    j_col = pairs_arr[:, 1]                                # (P,) — second index

    for p in range(P):
        ip, jp = int(pairs_arr[p, 0]), int(pairs_arr[p, 1])
        # q is adjacent to p iff q shares ip or jp
        connected = (
            (i_col == ip) | (j_col == ip) |
            (i_col == jp) | (j_col == jp)
        )
        connected[p] = False   # no self-loop
        A[p, connected] = 1.0

    return A

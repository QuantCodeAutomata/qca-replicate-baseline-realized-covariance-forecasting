"""Tests for graph construction routines (empty, complete, sector, GL, line, normalize)."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.graphs import (
    make_empty_graph,
    make_complete_graph,
    make_sector_graph,
    normalize_graph,
    make_line_graph,
    estimate_gl_graph,
)
from src.utils import N_ASSETS, N_PAIRS, SECTOR_MAP


# ---------------------------------------------------------------------------
# make_empty_graph
# ---------------------------------------------------------------------------

def test_empty_graph_shape() -> None:
    A = make_empty_graph()
    assert A.shape == (N_ASSETS, N_ASSETS)


def test_empty_graph_all_zeros() -> None:
    A = make_empty_graph()
    np.testing.assert_array_equal(A, np.zeros((N_ASSETS, N_ASSETS)))


def test_empty_graph_custom_n() -> None:
    A = make_empty_graph(N=10)
    assert A.shape == (10, 10)
    assert np.all(A == 0)


# ---------------------------------------------------------------------------
# make_complete_graph
# ---------------------------------------------------------------------------

def test_complete_graph_shape() -> None:
    A = make_complete_graph()
    assert A.shape == (N_ASSETS, N_ASSETS)


def test_complete_graph_zero_diagonal() -> None:
    A = make_complete_graph()
    np.testing.assert_array_equal(np.diag(A), np.zeros(N_ASSETS))


def test_complete_graph_binary() -> None:
    A = make_complete_graph()
    assert set(np.unique(A)).issubset({0, 1}), "Complete graph must be binary"


def test_complete_graph_off_diagonal_ones() -> None:
    A = make_complete_graph(N=5)
    expected = np.ones((5, 5)) - np.eye(5)
    np.testing.assert_array_equal(A, expected)


def test_complete_graph_symmetric() -> None:
    A = make_complete_graph()
    np.testing.assert_array_equal(A, A.T)


# ---------------------------------------------------------------------------
# make_sector_graph
# ---------------------------------------------------------------------------

def test_sector_graph_shape() -> None:
    S = make_sector_graph()
    assert S.shape == (N_ASSETS, N_ASSETS)


def test_sector_graph_symmetric() -> None:
    S = make_sector_graph()
    np.testing.assert_array_equal(S, S.T)


def test_sector_graph_zero_diagonal() -> None:
    S = make_sector_graph()
    np.testing.assert_array_equal(np.diag(S), np.zeros(N_ASSETS))


def test_sector_graph_binary() -> None:
    S = make_sector_graph()
    assert set(np.unique(S)).issubset({0, 1})


def test_sector_graph_within_sector_edges() -> None:
    """Pairs in the same sector must have edge = 1."""
    S = make_sector_graph()
    tickers = list(SECTOR_MAP.keys())
    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            if i != j and SECTOR_MAP[ti] == SECTOR_MAP[tj]:
                assert S[i, j] == 1, f"Expected edge between {ti} and {tj} (same sector)"


def test_sector_graph_cross_sector_no_edges() -> None:
    """Different sectors must have edge = 0."""
    S = make_sector_graph()
    tickers = list(SECTOR_MAP.keys())
    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            if i != j and SECTOR_MAP[ti] != SECTOR_MAP[tj]:
                assert S[i, j] == 0, f"Unexpected edge between {ti} ({SECTOR_MAP[ti]}) and {tj} ({SECTOR_MAP[tj]})"


# ---------------------------------------------------------------------------
# normalize_graph
# ---------------------------------------------------------------------------

def test_normalize_graph_shape() -> None:
    S = make_sector_graph()
    W = normalize_graph(S)
    assert W.shape == S.shape


def test_normalize_empty_graph_is_zero() -> None:
    A = make_empty_graph()
    W = normalize_graph(A)
    np.testing.assert_array_equal(W, np.zeros_like(A))


def test_normalize_graph_symmetric() -> None:
    S = make_sector_graph()
    W = normalize_graph(S)
    np.testing.assert_allclose(W, W.T, atol=1e-12)


def test_normalize_graph_row_degree_formula() -> None:
    """For a path graph, verify W[i,j] = A[i,j]/sqrt(deg_i * deg_j)."""
    A = np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]], dtype=float)
    W = normalize_graph(A)
    # degrees: [1, 2, 1]; W[0,1] = 1/sqrt(1*2), W[1,2] = 1/sqrt(2*1)
    expected_01 = 1 / np.sqrt(1 * 2)
    expected_12 = 1 / np.sqrt(2 * 1)
    assert W[0, 1] == pytest.approx(expected_01, rel=1e-8)
    assert W[1, 2] == pytest.approx(expected_12, rel=1e-8)
    assert W[0, 2] == 0.0


# ---------------------------------------------------------------------------
# make_line_graph
# ---------------------------------------------------------------------------

def test_line_graph_shape() -> None:
    L = make_line_graph()
    assert L.shape == (N_PAIRS, N_PAIRS)


def test_line_graph_symmetric() -> None:
    L = make_line_graph()
    np.testing.assert_array_equal(L, L.T)


def test_line_graph_zero_diagonal() -> None:
    L = make_line_graph()
    np.testing.assert_array_equal(np.diag(L), np.zeros(N_PAIRS))


def test_line_graph_binary() -> None:
    L = make_line_graph()
    assert set(np.unique(L)).issubset({0, 1})


def test_line_graph_small_case() -> None:
    """For N=4 assets, N_pairs=6. Known pair sharing structure.

    make_line_graph requires explicit vech_pairs; N is only for validation.
    """
    N4 = 4
    pairs4: list = [(i, j) for i in range(N4) for j in range(i + 1, N4)]
    # pairs4 = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    L = make_line_graph(vech_pairs=pairs4, N=N4)
    assert L.shape == (6, 6)
    # Build pair→index mapping
    pair_idx = {p: k for k, p in enumerate(pairs4)}
    p01 = pair_idx[(0, 1)]
    p02 = pair_idx[(0, 2)]
    p23 = pair_idx[(2, 3)]
    assert L[p01, p02] == 1, "(0,1) and (0,2) share asset 0 → must be connected"
    assert L[p01, p23] == 0, "(0,1) and (2,3) share no asset → must not be connected"


# ---------------------------------------------------------------------------
# estimate_gl_graph
# ---------------------------------------------------------------------------

def _make_correlated_returns(T: int = 500, N: int = 27, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Simple factor model: 2 factors
    F = rng.standard_normal((T, 2))
    B = rng.standard_normal((N, 2))
    eps = rng.standard_normal((T, N)) * 0.1
    X = F @ B.T + eps
    return X - X.mean(axis=0)


def test_gl_graph_shape() -> None:
    X = _make_correlated_returns(T=300, N=27)
    A = estimate_gl_graph(X)
    assert A.shape == (27, 27)


def test_gl_graph_symmetric() -> None:
    X = _make_correlated_returns(T=300, N=27)
    A = estimate_gl_graph(X)
    np.testing.assert_array_equal(A, A.T)


def test_gl_graph_zero_diagonal() -> None:
    X = _make_correlated_returns(T=300, N=27)
    A = estimate_gl_graph(X)
    np.testing.assert_array_equal(np.diag(A), np.zeros(27))


def test_gl_graph_binary() -> None:
    X = _make_correlated_returns(T=300, N=27)
    A = estimate_gl_graph(X)
    assert set(np.unique(A)).issubset({0, 1})

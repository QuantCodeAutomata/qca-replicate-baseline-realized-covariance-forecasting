"""Tests for Model Confidence Set (MCS) implementation."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.mcs import mcs_test, block_bootstrap


# ---------------------------------------------------------------------------
# block_bootstrap
# ---------------------------------------------------------------------------

def test_block_bootstrap_shape() -> None:
    rng = np.random.default_rng(0)
    T = 100
    x = rng.standard_normal(T)
    boot = block_bootstrap(x, block_length=5, n_replications=200, seed=0)
    assert boot.shape == (200, T)


def test_block_bootstrap_preserves_length() -> None:
    rng = np.random.default_rng(1)
    T = 53
    x = rng.standard_normal(T)
    boot = block_bootstrap(x, block_length=3, n_replications=100, seed=1)
    assert boot.shape[1] == T


def test_block_bootstrap_same_seed_reproducible() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(80)
    b1 = block_bootstrap(x, block_length=5, n_replications=50, seed=42)
    b2 = block_bootstrap(x, block_length=5, n_replications=50, seed=42)
    np.testing.assert_array_equal(b1, b2)


def test_block_bootstrap_values_from_original() -> None:
    """All bootstrapped values must come from the original series."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    boot = block_bootstrap(x, block_length=2, n_replications=200, seed=7)
    for val in boot.flatten():
        assert val in x, f"{val} not in original series"


# ---------------------------------------------------------------------------
# mcs_test: output structure
# ---------------------------------------------------------------------------

def _make_losses(T: int, M: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.exponential(1.0, (T, M))


def test_mcs_returns_dict() -> None:
    L = _make_losses(100, 3, seed=0)
    result = mcs_test(L, n_replications=100, seed=0)
    assert isinstance(result, dict)
    assert "included" in result
    assert "p_values" in result
    assert "eliminated_order" in result


def test_mcs_p_values_shape() -> None:
    M = 4
    L = _make_losses(80, M, seed=1)
    result = mcs_test(L, n_replications=100, seed=1)
    assert len(result["p_values"]) == M


def test_mcs_included_subset_of_models() -> None:
    M = 5
    L = _make_losses(80, M, seed=2)
    result = mcs_test(L, n_replications=100, seed=2)
    for idx in result["included"]:
        assert 0 <= idx < M


def test_mcs_included_plus_eliminated_is_full_set() -> None:
    M = 4
    L = _make_losses(80, M, seed=3)
    result = mcs_test(L, n_replications=100, seed=3)
    all_models = set(result["included"]) | set(result["eliminated_order"])
    assert all_models == set(range(M))


def test_mcs_p_values_in_01() -> None:
    M = 3
    L = _make_losses(100, M, seed=4)
    result = mcs_test(L, n_replications=100, seed=4)
    pv = np.asarray(result["p_values"])
    assert np.all(pv >= 0.0), f"p-values must be >= 0; got min={pv.min()}"
    assert np.all(pv <= 1.0 + 1e-8), f"p-values must be <= 1; got max={pv.max()}"


def test_mcs_clearly_worst_model_excluded() -> None:
    """A model with consistently 10x higher losses should be excluded from MCS."""
    T = 200
    rng = np.random.default_rng(5)
    # Two good models with similar loss, one bad one
    good1 = rng.exponential(1.0, T)
    good2 = rng.exponential(1.0, T) * 1.05
    bad   = rng.exponential(10.0, T)  # clearly worse
    L = np.column_stack([good1, good2, bad])
    result = mcs_test(L, n_replications=500, seed=5)
    # The bad model (index 2) should be eliminated
    assert 2 in result["eliminated_order"], \
        f"Expected worst model (idx=2) eliminated; got {result['eliminated_order']}"


def test_mcs_identical_models_all_included() -> None:
    """If all models have identical loss series, none should be eliminated."""
    T, M = 150, 3
    rng = np.random.default_rng(6)
    base = rng.exponential(1.0, T)
    L = np.tile(base[:, None], M)
    result = mcs_test(L, n_replications=200, seed=6)
    assert len(result["included"]) == M, \
        f"Identical models: all {M} should be in MCS; got {result['included']}"


def test_mcs_wrong_input_raises() -> None:
    """Non-2D input must raise ValueError."""
    with pytest.raises(ValueError):
        mcs_test(np.ones(50), n_replications=100, seed=0)


def test_mcs_single_model_returns_itself() -> None:
    """Single model cannot be eliminated → it is always in MCS."""
    L = np.random.default_rng(7).exponential(1.0, (80, 1))
    result = mcs_test(L, n_replications=100, seed=7)
    assert result["included"] == [0]
    assert len(result["eliminated_order"]) == 0


def test_mcs_p_values_included_models_are_one() -> None:
    """Models remaining in the MCS should have p_value = 1.0 (never eliminated)."""
    M = 4
    L = _make_losses(100, M, seed=8)
    result = mcs_test(L, n_replications=200, seed=8)
    pv = np.asarray(result["p_values"])
    for idx in result["included"]:
        assert pv[idx] == pytest.approx(1.0, abs=1e-12), \
            f"Included model {idx} must have p_value=1.0; got {pv[idx]}"

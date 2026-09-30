"""
src/mcs.py
==========
Model Confidence Set (MCS) test via circular block bootstrap.

Reference
---------
Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set.
*Econometrica*, 79(2), 453–497. https://doi.org/10.3982/ECTA5771
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------


def block_bootstrap(
    data: np.ndarray,
    block_length: int = 2,
    n_replications: int = 10_000,
    seed: int = 0,
) -> np.ndarray:
    """Generate circular block bootstrap samples of a loss series.

    The series *data* is treated as circular (i.e. it wraps around), so every
    starting index is equally likely.  Each bootstrap sample has the same
    length T as *data*.

    Parameters
    ----------
    data:
        1-D array of shape ``(T,)`` — the original time series to resample.
    block_length:
        Length of each contiguous block (denoted *b* in Hansen et al. 2011).
    n_replications:
        Number of independent bootstrap replications (denoted *B*).
    seed:
        Seed for NumPy's default random generator; set to a fixed integer for
        reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_replications, T)`` where row *b* is the *b*-th
        bootstrap resample of *data*.
    """
    data = np.asarray(data, dtype=float)
    T = data.shape[0]
    rng = np.random.default_rng(seed)
    return _block_bootstrap_rng(data, block_length, n_replications, rng)


def _block_bootstrap_rng(
    data: np.ndarray,
    block_length: int,
    n_replications: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Internal block bootstrap that accepts a pre-seeded RNG instance.

    Parameters
    ----------
    data:
        1-D array of shape ``(T,)``.
    block_length:
        Length of each block.
    n_replications:
        Number of bootstrap draws.
    rng:
        Pre-seeded ``np.random.Generator``.

    Returns
    -------
    np.ndarray
        Shape ``(n_replications, T)``.
    """
    T = len(data)
    n_blocks = int(np.ceil(T / block_length))

    # Draw all block start indices at once for efficiency: shape (B, n_blocks)
    starts = rng.integers(0, T, size=(n_replications, n_blocks))

    # Build circular index array: for each start s and offset o, index = (s+o) % T
    # offsets shape: (block_length,); result after broadcast: (B, n_blocks, block_length)
    offsets = np.arange(block_length)
    indices = (starts[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]) % T
    # Flatten blocks → (B, n_blocks * block_length), then trim to T
    flat_indices = indices.reshape(n_replications, n_blocks * block_length)[:, :T]
    return data[flat_indices]  # shape (n_replications, T)


# ---------------------------------------------------------------------------
# MCS test
# ---------------------------------------------------------------------------


def mcs_test(
    loss_matrix: np.ndarray,
    block_length: int = 2,
    n_replications: int = 10_000,
    significance: float = 0.05,
    seed: int = 0,
) -> dict:
    """Model Confidence Set test using the T_R (range) statistic.

    Implements the sequential-elimination procedure of Hansen, Lunde & Nason
    (2011).  At each step, models in the current set *S* are compared via the
    T_R statistic; the worst model is eliminated if the bootstrap p-value is
    below *significance*.  The procedure stops when the null of equal
    predictive ability cannot be rejected.

    Algorithm
    ---------
    1. Start with *S* = {0, …, M−1}.
    2. For each model *i* in *S*, compute the excess-loss series:

       .. math::

           d_{i,t} = L_{i,t} - \\bar L_t, \\quad
           \\bar L_t = \\frac{1}{|S|} \\sum_{j \\in S} L_{j,t}

    3. Compute the T_R statistic:

       .. math::

           T_R = \\max_{i \\in S}\\, t_i, \\quad
           t_i = \\frac{\\bar d_i}{\\widehat{\\mathrm{se}}(\\bar d_i)}

       where :math:`\\widehat{\\mathrm{se}}(\\bar d_i)` is the bootstrap
       standard error of the sample mean :math:`\\bar d_i`.

    4. Obtain a bootstrap p-value by centering the bootstrap excess-loss
       distributions under H₀ (equal predictive ability) and computing
       :math:`\\Pr(T_R^* \\ge T_R)`.

    5. If p-value < *significance*, eliminate the worst model (largest *t_i*)
       from *S* and repeat from step 2.

    Parameters
    ----------
    loss_matrix:
        Array of shape ``(T, M)`` where ``T`` is the number of evaluation
        periods and ``M`` is the number of models.  Each column is a loss
        series for one model.
    block_length:
        Block length for the circular block bootstrap.
    n_replications:
        Number of bootstrap replications.
    significance:
        Nominal size of the sequential test (α).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{'included': list[int], 'p_values': np.ndarray, 'eliminated_order': list[int]}``

        * **included** — list of model indices (columns of *loss_matrix*)
          remaining in the MCS.
        * **p_values** — 1-D array of length M; the p-value at the step when
          model *i* was eliminated, or ``1.0`` if model *i* is in the MCS.
        * **eliminated_order** — model indices in the order they were removed.
    """
    loss_matrix = np.asarray(loss_matrix, dtype=float)
    if loss_matrix.ndim != 2:
        raise ValueError(
            f"loss_matrix must be 2-D (T × M); got shape {loss_matrix.shape}."
        )
    T, M = loss_matrix.shape

    rng = np.random.default_rng(seed)
    active: list[int] = list(range(M))
    p_values: np.ndarray = np.ones(M)  # default 1.0 (included)
    eliminated_order: list[int] = []

    while len(active) > 1:
        n_active = len(active)
        L_active = loss_matrix[:, active]  # (T, n_active)

        # --- Step 1: excess loss d_i,t = L_i,t - mean_j(L_j,t) ------------
        mean_L = L_active.mean(axis=1, keepdims=True)  # (T, 1)
        d = L_active - mean_L                           # (T, n_active)

        # --- Step 2: bootstrap SE of each d_i mean -------------------------
        # Generate bootstrap samples of each excess-loss series
        # boot_d[k] shape: (n_replications, T)
        boot_d = np.stack(
            [_block_bootstrap_rng(d[:, k], block_length, n_replications, rng)
             for k in range(n_active)],
            axis=2,
        )  # (n_replications, T, n_active)

        boot_means = boot_d.mean(axis=1)      # (n_replications, n_active)
        d_means    = d.mean(axis=0)            # (n_active,)

        # Bootstrap SE = std of bootstrap means across replications
        boot_se = boot_means.std(axis=0, ddof=1)        # (n_active,)
        boot_se = np.maximum(boot_se, 1e-14)            # guard zero SE

        # --- Step 3: observed T_R statistic ---------------------------------
        t_stats   = d_means / boot_se          # (n_active,)
        worst_k   = int(np.argmax(t_stats))    # index into active list
        T_R_obs   = t_stats[worst_k]

        # --- Step 4: bootstrap p-value under H₀ ----------------------------
        # Centre bootstrap means at 0 (impose H₀: E[d_i.] = 0)
        boot_t = (boot_means - d_means) / boot_se  # (n_replications, n_active)
        boot_T_R = boot_t.max(axis=1)              # (n_replications,)
        p_val = float(np.mean(boot_T_R >= T_R_obs))

        # --- Step 5: elimination or acceptance ------------------------------
        worst_model = active[worst_k]
        if p_val < significance:
            # Record the p-value at elimination for the worst model
            p_values[worst_model] = p_val
            eliminated_order.append(worst_model)
            active = [m for m in active if m != worst_model]
        else:
            # H₀ not rejected — all remaining models enter the MCS.
            # p_values for included models stay at 1.0 (set at initialisation).
            break

    return {
        "included": active,
        "p_values": p_values,
        "eliminated_order": eliminated_order,
    }

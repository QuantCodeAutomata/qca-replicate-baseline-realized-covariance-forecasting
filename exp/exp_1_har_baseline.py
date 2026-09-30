"""
Exp 1: HAR-Cholesky vs HAR-DRD baseline comparison.
Uses synthetic realized covariance data (fixed seed) to demonstrate the
methodology. In a real deployment, replace with actual 5-minute intraday data.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.realized_cov import generate_synthetic_rc_series
from src.har_models import HARCholesky, HARDRD
from src.ghar_models import reconstruct_covariance, reconstruct_from_cholesky
from src.losses import compute_losses, mean_losses
from src.mcs import mcs_test
from src.utils import vech_chol, N_ASSETS
from src.nearest_pd import ensure_pd

# ── constants ──────────────────────────────────────────────────────────────
T       = 1200
IS      = 1000
OOS     = 200
WINDOW  = 200    # training window size
RE_EST  = 22     # re-estimate every 22 trading days (~monthly)
HIST    = 22     # history rows needed for prediction lags
N_CHOL  = N_ASSETS * (N_ASSETS + 1) // 2   # 378
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
os.makedirs(RESULTS, exist_ok=True)


# ── shared helpers ─────────────────────────────────────────────────────────

def _build_chol_series(H_series: np.ndarray) -> np.ndarray:
    """Convert H_series (T, N, N) → y_series (T, N_CHOL) via Cholesky vech."""
    y = np.empty((H_series.shape[0], N_CHOL))
    for t in range(H_series.shape[0]):
        H_pd, _ = ensure_pd(H_series[t])
        y[t] = vech_chol(np.linalg.cholesky(H_pd))
    return y


def _hist22(series: np.ndarray, t: int) -> np.ndarray:
    """Return the 22-row history ending at index t (inclusive)."""
    return series[max(0, t - HIST + 1) : t + 1]


def _agg_mean(series: np.ndarray, h: int) -> np.ndarray:
    """Non-overlapping h-day block mean: (T, k) → (T//h, k)."""
    n = (series.shape[0] // h) * h
    return series[:n].reshape(-1, h, series.shape[1]).mean(axis=1)


def _agg_H_sum(H_series: np.ndarray, h: int) -> np.ndarray:
    """Non-overlapping h-day block sum: (T, N, N) → (T//h, N, N)."""
    N = H_series.shape[1]
    n = (H_series.shape[0] // h) * h
    return H_series[:n].reshape(-1, h, N, N).sum(axis=1)


def _run_mcs(named_losses: dict, alpha: float = 0.05,
             B: int = 500, seed: int = 42) -> dict:
    """Wrap mcs_test for a dict of {model_name: loss_array}.

    Returns {model_name: {'included': bool, 'p_value': float}}.
    """
    names = list(named_losses.keys())
    mat = np.column_stack([named_losses[n] for n in names])   # (T, M)
    valid = np.all(np.isfinite(mat), axis=1)
    mat = mat[valid]
    res = mcs_test(mat, n_replications=B, significance=alpha, seed=seed)
    incl_set = set(res['included'])
    p_vals   = res['p_values']
    return {n: {'included': (i in incl_set), 'p_value': float(p_vals[i])}
            for i, n in enumerate(names)}


# ── rolling OOS for h=1 ────────────────────────────────────────────────────

def _rolling_h1(H_series, v_series, x_series, y_series):
    """Rolling OOS forecasts for h=1 (daily).

    Returns (fc_drd, fc_chol, H_real) — parallel lists of (27,27) arrays.
    """
    fc_drd, fc_chol, H_real = [], [], []

    for est_day in range(IS, IS + OOS, RE_EST):
        # ---- fit on training window ----
        v_tr = v_series[est_day - WINDOW : est_day]
        x_tr = x_series[est_day - WINDOW : est_day]
        y_tr = y_series[est_day - WINDOW : est_day]

        model_drd  = HARDRD().fit(v_tr, x_tr)
        model_chol = HARCholesky().fit(y_tr)

        # ---- generate forecasts ----
        t_end = min(est_day + RE_EST, IS + OOS - 1)
        for t in range(est_day, t_end):
            v_hat, x_hat = model_drd.predict(_hist22(v_series, t),
                                              _hist22(x_series, t))
            y_hat = model_chol.predict_horizon(_hist22(y_series, t))

            S_drd,  _ = reconstruct_covariance(v_hat, x_hat)
            S_chol, _ = reconstruct_from_cholesky(y_hat)

            fc_drd.append(S_drd)
            fc_chol.append(S_chol)
            H_real.append(H_series[t + 1])   # realized next-day covariance

    return fc_drd, fc_chol, H_real


# ── rolling OOS for h=5 or h=22 ───────────────────────────────────────────

def _rolling_hN(H_series, v_series, x_series, y_series, h: int):
    """Single-fit OOS for horizon h (aggregated direct method).

    Aggregates IS data into h-day blocks, fits models once, then generates
    OOS forecasts on each consecutive h-day block.

    Returns (fc_drd, fc_chol, H_real) — parallel lists of (27,27) arrays.
    """
    n_IS  = IS // h
    n_tot = T  // h

    v_agg = _agg_mean(v_series, h)      # (n_tot, N)
    x_agg = _agg_mean(x_series, h)      # (n_tot, N_PAIRS)
    y_agg = _agg_mean(y_series, h)      # (n_tot, N_CHOL)
    H_agg = _agg_H_sum(H_series, h)     # (n_tot, N, N) — summed realized H

    model_drd  = HARDRD().fit(v_agg[:n_IS], x_agg[:n_IS])
    model_chol = HARCholesky().fit(y_agg[:n_IS])

    fc_drd, fc_chol, H_real = [], [], []
    for k in range(n_IS - 1, n_tot - 1):
        # history: blocks 0..k, predict block k+1
        v_hat, x_hat = model_drd.predict(_hist22(v_agg, k), _hist22(x_agg, k))
        y_hat        = model_chol.predict_horizon(_hist22(y_agg, k))

        S_drd,  _ = reconstruct_covariance(v_hat, x_hat)
        S_chol, _ = reconstruct_from_cholesky(y_hat)

        fc_drd.append(S_drd)
        fc_chol.append(S_chol)
        H_real.append(H_agg[k + 1])   # summed realized H over next h days

    return fc_drd, fc_chol, H_real


# ── main experiment ────────────────────────────────────────────────────────

def run_exp1():
    print("=" * 60)
    print("Exp 1: HAR-Cholesky vs HAR-DRD baseline")
    print("=" * 60)

    # 1. Generate data -------------------------------------------------------
    print("\n[1] Generating synthetic RC data (T=1200, N=27, seed=42)…")
    H_series, v_series, x_series = generate_synthetic_rc_series(
        n_days=T, n_assets=N_ASSETS, seed=42
    )
    y_series = _build_chol_series(H_series)
    print(f"    H: {H_series.shape}, v: {v_series.shape}, "
          f"x: {x_series.shape}, y: {y_series.shape}")

    horizons      = [1, 5, 22]
    horizon_label = {1: "h=1 (daily)", 5: "h=5 (weekly)", 22: "h=22 (monthly)"}
    all_losses_drd  = {}
    all_losses_chol = {}

    # 2–4. Rolling OOS per horizon -------------------------------------------
    for h in horizons:
        print(f"\n[2-4] Rolling OOS — {horizon_label[h]}")
        if h == 1:
            fc_drd, fc_chol, H_real = _rolling_h1(
                H_series, v_series, x_series, y_series)
        else:
            fc_drd, fc_chol, H_real = _rolling_hN(
                H_series, v_series, x_series, y_series, h)
        print(f"    OOS forecasts: {len(H_real)}")

        all_losses_drd[h]  = compute_losses(H_real, fc_drd)
        all_losses_chol[h] = compute_losses(H_real, fc_chol)

    # 5. Mean losses table ---------------------------------------------------
    print("\n[5] Mean losses")
    hdr = f"{'Horizon':<15} {'Model':<12} {'Euclidean':>12} {'Frobenius':>12} {'QLIKE':>12}"
    print(hdr); print("-" * len(hdr))
    for h in horizons:
        for tag, df in [("HAR-DRD", all_losses_drd[h]), ("HAR-Chol", all_losses_chol[h])]:
            ml = mean_losses(df)
            print(f"{horizon_label[h]:<15} {tag:<12} "
                  f"{ml['euclidean']:>12.4f} {ml['frobenius']:>12.4f} {ml['qlike']:>12.4f}")

    # 6. MCS -----------------------------------------------------------------
    print("\n[6] MCS tests (B=500, α=0.05)…")
    mcs_records = []
    for h in horizons:
        for col in ['euclidean', 'frobenius', 'qlike']:
            named = {
                'HAR-DRD':  all_losses_drd[h][col].values,
                'HAR-Chol': all_losses_chol[h][col].values,
            }
            res = _run_mcs(named, alpha=0.05, B=500, seed=42)
            for m, info in res.items():
                mcs_records.append({'horizon': h, 'loss': col, 'model': m, **info})

    mcs_df = pd.DataFrame(mcs_records)

    # 7. Print results -------------------------------------------------------
    print("\n[7] Summary (ratio = HAR-Chol / HAR-DRD, MCS @ α=0.05)")
    for h in horizons:
        ml_d = mean_losses(all_losses_drd[h])
        ml_c = mean_losses(all_losses_chol[h])
        print(f"\n  {horizon_label[h]}")
        print(f"  {'Loss':<12} {'HAR-DRD':>10} {'HAR-Chol':>10} "
              f"{'Ratio':>8} {'DRD-MCS':>9} {'Chol-MCS':>9}")
        print(f"  {'-'*60}")
        for col in ['euclidean', 'frobenius', 'qlike']:
            sub   = mcs_df[(mcs_df.horizon == h) & (mcs_df.loss == col)]
            d_in  = sub[sub.model == 'HAR-DRD']['included'].values[0]
            c_in  = sub[sub.model == 'HAR-Chol']['included'].values[0]
            ratio = ml_c[col] / ml_d[col] if ml_d[col] else np.nan
            print(f"  {col:<12} {ml_d[col]:>10.4f} {ml_c[col]:>10.4f} "
                  f"{ratio:>8.4f} {'yes' if d_in else 'no':>9} "
                  f"{'yes' if c_in else 'no':>9}")

    # 8. Save results --------------------------------------------------------
    print("\n[8] Saving results…")

    loss_rows = []
    for h in horizons:
        for tag, df in [('HAR-DRD', all_losses_drd[h]), ('HAR-Chol', all_losses_chol[h])]:
            ml = mean_losses(df)
            loss_rows.append({'model': tag, 'horizon': h, **ml})
    losses_csv = pd.DataFrame(loss_rows)
    losses_csv.to_csv(os.path.join(RESULTS, "exp1_losses.csv"), index=False)
    print("    Saved exp1_losses.csv")

    mcs_df.to_csv(os.path.join(RESULTS, "exp1_mcs.csv"), index=False)
    print("    Saved exp1_mcs.csv")

    # Bar chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x_pos = np.arange(len(horizons))
    w = 0.35
    for ax_i, col in enumerate(['euclidean', 'frobenius', 'qlike']):
        ax = axes[ax_i]
        drd_v  = [mean_losses(all_losses_drd[h])[col]  for h in horizons]
        chol_v = [mean_losses(all_losses_chol[h])[col] for h in horizons]
        ax.bar(x_pos - w / 2, drd_v,  w, label='HAR-DRD',  color='steelblue')
        ax.bar(x_pos + w / 2, chol_v, w, label='HAR-Chol', color='darkorange')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"h={h}" for h in horizons])
        ax.set_title(f"{col.capitalize()} Loss")
        ax.set_xlabel("Horizon"); ax.set_ylabel("Mean Loss")
        ax.legend()
    fig.suptitle("Exp 1: HAR-DRD vs HAR-Cholesky — Mean Losses by Horizon")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp1_loss_comparison.png"), dpi=120)
    plt.close(fig)
    print("    Saved exp1_loss_comparison.png")

    print("\nExp 1 complete.\n")
    return {
        'losses_drd':  all_losses_drd,
        'losses_chol': all_losses_chol,
        'mcs_df':      mcs_df,
        'losses_csv':  losses_csv,
    }


if __name__ == '__main__':
    run_exp1()

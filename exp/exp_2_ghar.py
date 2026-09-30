"""
Exp 2: GHAR models with volatility and correlation graph augmentation.
Tests 12 combinations of (vol_graph × corr_graph) and compares against
the HAR-DRD baseline (= GHAR with empty+empty graphs).
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
from src.har_models import HARDRD, HARCholesky
from src.ghar_models import GHARDRD, reconstruct_covariance, reconstruct_from_cholesky
from src.losses import compute_losses, mean_losses
from src.mcs import mcs_test
from src.graphs import (make_empty_graph, make_complete_graph, make_sector_graph,
                         estimate_gl_graph, normalize_graph, make_line_graph)
from src.utils import N_ASSETS, N_PAIRS, vech_chol
from src.nearest_pd import ensure_pd

# ── constants ──────────────────────────────────────────────────────────────
T       = 1200
IS      = 1000
OOS     = 200
WINDOW  = 200
RE_EST  = 22
HIST    = 22
N_CHOL  = N_ASSETS * (N_ASSETS + 1) // 2
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
os.makedirs(RESULTS, exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────

def _build_chol_series(H_series):
    y = np.empty((H_series.shape[0], N_CHOL))
    for t in range(H_series.shape[0]):
        H_pd, _ = ensure_pd(H_series[t])
        y[t] = vech_chol(np.linalg.cholesky(H_pd))
    return y


def _hist22(series, t):
    return series[max(0, t - HIST + 1) : t + 1]


def _agg_mean(series, h):
    n = (series.shape[0] // h) * h
    return series[:n].reshape(-1, h, series.shape[1]).mean(axis=1)


def _agg_H_sum(H_series, h):
    N = H_series.shape[1]
    n = (H_series.shape[0] // h) * h
    return H_series[:n].reshape(-1, h, N, N).sum(axis=1)


def _run_mcs(named_losses, alpha=0.05, B=500, seed=42):
    names = list(named_losses.keys())
    mat   = np.column_stack([named_losses[n] for n in names])
    valid = np.all(np.isfinite(mat), axis=1)
    mat   = mat[valid]
    res   = mcs_test(mat, n_replications=B, significance=alpha, seed=seed)
    incl  = set(res['included'])
    pv    = res['p_values']
    return {n: {'included': (i in incl), 'p_value': float(pv[i])}
            for i, n in enumerate(names)}


def _rolling_oos_ghar(H_series, v_series, x_series, W_vol, W_corr):
    """Rolling OOS h=1 using GHARDRD with the given graph pair."""
    fc, H_real = [], []
    for est_day in range(IS, IS + OOS, RE_EST):
        model = GHARDRD(W_vol, W_corr)
        model.fit(v_series[est_day - WINDOW : est_day],
                  x_series[est_day - WINDOW : est_day])
        t_end = min(est_day + RE_EST, IS + OOS - 1)
        for t in range(est_day, t_end):
            v_hat, x_hat = model.predict(_hist22(v_series, t), _hist22(x_series, t))
            S, _ = reconstruct_covariance(v_hat, x_hat)
            fc.append(S)
            H_real.append(H_series[t + 1])
    return fc, H_real


def _rolling_oos_hardrd(H_series, v_series, x_series):
    """Explicit HAR-DRD rolling OOS h=1 (structural baseline)."""
    fc, H_real = [], []
    for est_day in range(IS, IS + OOS, RE_EST):
        model = HARDRD().fit(v_series[est_day - WINDOW : est_day],
                             x_series[est_day - WINDOW : est_day])
        t_end = min(est_day + RE_EST, IS + OOS - 1)
        for t in range(est_day, t_end):
            v_hat, x_hat = model.predict(_hist22(v_series, t), _hist22(x_series, t))
            S, _ = reconstruct_covariance(v_hat, x_hat)
            fc.append(S)
            H_real.append(H_series[t + 1])
    return fc, H_real


def _oos_hN_ghar(H_series, v_series, x_series, W_vol, W_corr, h):
    """Single-fit OOS for horizon h (aggregated direct method)."""
    n_IS  = IS // h
    n_tot = T  // h
    v_agg = _agg_mean(v_series, h)
    x_agg = _agg_mean(x_series, h)
    H_agg = _agg_H_sum(H_series, h)

    model = GHARDRD(W_vol, W_corr)
    model.fit(v_agg[:n_IS], x_agg[:n_IS])

    fc, H_real = [], []
    for k in range(n_IS - 1, n_tot - 1):
        v_hat, x_hat = model.predict(_hist22(v_agg, k), _hist22(x_agg, k))
        S, _ = reconstruct_covariance(v_hat, x_hat)
        fc.append(S)
        H_real.append(H_agg[k + 1])
    return fc, H_real


# ── main experiment ────────────────────────────────────────────────────────

def run_exp2():
    print("=" * 60)
    print("Exp 2: GHAR-DRD — graph structure comparison")
    print("=" * 60)

    # 1. Generate data -------------------------------------------------------
    print("\n[1] Generating synthetic RC data…")
    H_series, v_series, x_series = generate_synthetic_rc_series(
        n_days=T, n_assets=N_ASSETS, seed=42
    )
    print(f"    H: {H_series.shape}, v: {v_series.shape}, x: {x_series.shape}")

    # 2. Build all graphs ----------------------------------------------------
    print("\n[2] Building graphs…")
    A_vol_empty    = make_empty_graph(N_ASSETS)
    A_vol_complete = make_complete_graph(N_ASSETS)
    A_vol_sector   = make_sector_graph()

    v_win = v_series[:WINDOW] - v_series[:WINDOW].mean(axis=0)
    print("    Fitting GraphicalLassoCV for GL volatility graph…")
    A_vol_gl = estimate_gl_graph(v_win, cv_folds=5, tol=1e-10)

    A_corr_empty    = make_empty_graph(N_PAIRS)
    A_corr_complete = make_complete_graph(N_PAIRS)
    A_corr_line     = make_line_graph()

    W_vol = {
        'E':  normalize_graph(A_vol_empty),
        'K':  normalize_graph(A_vol_complete),
        'S':  normalize_graph(A_vol_sector),
        'GL': normalize_graph(A_vol_gl),
    }
    W_corr = {
        'E': normalize_graph(A_corr_empty),
        'K': normalize_graph(A_corr_complete),
        'L': normalize_graph(A_corr_line),
    }

    for name, A in [('vol_empty', A_vol_empty), ('vol_complete', A_vol_complete),
                    ('vol_sector', A_vol_sector), ('vol_GL', A_vol_gl)]:
        print(f"    {name}: {int(A.sum())} edges")
    for name, A in [('corr_empty', A_corr_empty), ('corr_complete', A_corr_complete),
                    ('corr_line', A_corr_line)]:
        print(f"    {name}: {int(A.sum())} edges")

    # 3–4. Rolling OOS h=1 for all 12 GHAR combinations ---------------------
    print("\n[3] Rolling OOS h=1 for all models (4 vol × 3 corr + HAR-DRD)…")
    combinations = [(vn, cn) for vn in ['E', 'K', 'S', 'GL'] for cn in ['E', 'K', 'L']]

    all_fc   = {}
    all_H    = None   # shared realized H (same for all models)

    # HAR-DRD explicit baseline
    print("    HAR-DRD baseline…")
    fc_drd, H_real = _rolling_oos_hardrd(H_series, v_series, x_series)
    all_fc['HAR-DRD'] = fc_drd
    all_H = H_real

    # GHAR(E,E): should match HAR-DRD numerically
    print("    GHAR(E,E) [should match HAR-DRD]…")
    fc_ee, _ = _rolling_oos_ghar(H_series, v_series, x_series, W_vol['E'], W_corr['E'])
    all_fc['GHAR(E,E)'] = fc_ee

    frob_diff = np.mean([np.linalg.norm(fc_drd[i] - fc_ee[i], 'fro')
                         for i in range(min(len(fc_drd), len(fc_ee)))])
    print(f"    GHAR(E,E) vs HAR-DRD mean Frobenius diff: {frob_diff:.6f}")

    for vn, cn in combinations:
        key = f"GHAR({vn},{cn})"
        if key in all_fc:
            continue
        print(f"    {key}…")
        fc, _ = _rolling_oos_ghar(H_series, v_series, x_series, W_vol[vn], W_corr[cn])
        all_fc[key] = fc

    # 5. Compute losses ------------------------------------------------------
    print("\n[5] Computing losses…")
    losses_by_model = {m: compute_losses(all_H, fc) for m, fc in all_fc.items()}
    ml_drd = mean_losses(losses_by_model['HAR-DRD'])

    # 6. Report, rankings, MCS -----------------------------------------------
    print("\n[6] Results table (ratio to HAR-DRD, h=1):")
    ordered = ['HAR-DRD'] + [f"GHAR({vn},{cn})" for vn, cn in combinations]
    loss_rows = []
    for m in ordered:
        ml = mean_losses(losses_by_model[m])
        row = {'model': m}
        for col in ['euclidean', 'frobenius', 'qlike']:
            row[col] = ml[col]
            row[f"{col}_ratio"] = ml[col] / ml_drd[col] if ml_drd[col] else np.nan
        loss_rows.append(row)
    results_df = pd.DataFrame(loss_rows)
    for col in ['euclidean', 'frobenius', 'qlike']:
        results_df[f"{col}_rank"] = results_df[col].rank().astype(int)

    print(results_df[['model', 'euclidean', 'euclidean_ratio', 'euclidean_rank',
                       'frobenius_ratio', 'qlike_ratio']].to_string(index=False))

    # MCS test
    print("\n    MCS test (α=0.05)…")
    mcs_records = []
    for col in ['euclidean', 'frobenius', 'qlike']:
        named = {m: losses_by_model[m][col].values for m in ordered}
        res = _run_mcs(named, alpha=0.05, B=200, seed=42)
        for m, info in res.items():
            mcs_records.append({'model': m, 'loss': col, **info})
    mcs_df = pd.DataFrame(mcs_records)

    # 7. Multi-horizon abbreviated -------------------------------------------
    print("\n[7] Abbreviated multi-horizon: GHAR(GL,L) vs HAR-DRD")
    for h, lbl in [(5, "h=5 weekly"), (22, "h=22 monthly")]:
        n_IS  = IS // h
        n_tot = T  // h
        v_agg = _agg_mean(v_series, h)
        x_agg = _agg_mean(x_series, h)
        H_agg = _agg_H_sum(H_series, h)

        drd_m  = HARDRD().fit(v_agg[:n_IS], x_agg[:n_IS])
        ghar_m = GHARDRD(W_vol['GL'], W_corr['L'])
        ghar_m.fit(v_agg[:n_IS], x_agg[:n_IS])

        fc_d, fc_g, H_r = [], [], []
        for k in range(n_IS - 1, n_tot - 1):
            vd, xd = drd_m.predict(_hist22(v_agg, k), _hist22(x_agg, k))
            vg, xg = ghar_m.predict(_hist22(v_agg, k), _hist22(x_agg, k))
            Sd, _ = reconstruct_covariance(vd, xd)
            Sg, _ = reconstruct_covariance(vg, xg)
            fc_d.append(Sd); fc_g.append(Sg); H_r.append(H_agg[k + 1])

        if fc_d:
            ld = mean_losses(compute_losses(H_r, fc_d))
            lg = mean_losses(compute_losses(H_r, fc_g))
            print(f"  {lbl} ({len(H_r)} forecasts):")
            for col in ['euclidean', 'frobenius', 'qlike']:
                r = lg[col] / ld[col] if ld[col] else np.nan
                print(f"    {col:<12}: GHAR(GL,L)={lg[col]:.4f} "
                      f"HAR-DRD={ld[col]:.4f} ratio={r:.4f}")

    # 8. Save ----------------------------------------------------------------
    print("\n[8] Saving results…")
    results_df.to_csv(os.path.join(RESULTS, "exp2_losses.csv"), index=False)
    print("    Saved exp2_losses.csv")

    rank_cols = ['model', 'euclidean_rank', 'frobenius_rank', 'qlike_rank']
    results_df[rank_cols].to_csv(os.path.join(RESULTS, "exp2_rankings.csv"), index=False)
    print("    Saved exp2_rankings.csv")

    # Heatmap of euclidean loss ratios
    vol_order  = ['E', 'K', 'S', 'GL']
    corr_order = ['E', 'K', 'L']
    ratio_mat  = np.array([[mean_losses(losses_by_model[f"GHAR({vn},{cn})"])['euclidean']
                             / ml_drd['euclidean']
                             for cn in corr_order]
                            for vn in vol_order])

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(ratio_mat, cmap='RdYlGn_r', vmin=0.8, vmax=1.2)
    ax.set_xticks(range(3)); ax.set_yticks(range(4))
    ax.set_xticklabels(corr_order); ax.set_yticklabels(vol_order)
    ax.set_xlabel("Correlation graph"); ax.set_ylabel("Volatility graph")
    ax.set_title("Exp 2: Euclidean loss ratio GHAR / HAR-DRD\n(< 1 = better)")
    plt.colorbar(im, ax=ax, label='Ratio to HAR-DRD')
    for vi in range(4):
        for ci in range(3):
            ax.text(ci, vi, f"{ratio_mat[vi, ci]:.3f}",
                    ha='center', va='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp2_model_comparison.png"), dpi=120)
    plt.close(fig)
    print("    Saved exp2_model_comparison.png")

    print("\nExp 2 complete.\n")
    return {
        'losses_by_model': losses_by_model,
        'results_df':      results_df,
        'mcs_df':          mcs_df,
        'W_vol':           W_vol,
        'W_corr':          W_corr,
    }


if __name__ == '__main__':
    run_exp2()

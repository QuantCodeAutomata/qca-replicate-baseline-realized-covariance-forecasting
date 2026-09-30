"""
Exp 3: Global minimum variance portfolio economic evaluation.
Compares HAR-DRD, HAR-Cholesky, GHAR(GL,L), GHAR(GL,K), and 1/N benchmark
using annualized portfolio volatility and daily turnover.
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
from src.ghar_models import GHARDRD, reconstruct_covariance, reconstruct_from_cholesky
from src.graphs import (estimate_gl_graph, make_line_graph, make_complete_graph,
                         make_empty_graph, normalize_graph)
from src.portfolio import (gmvp_weights, gmvp_plus_weights, equal_weight,
                            annualized_volatility, compute_turnover)
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


def _gen_daily_returns(H_series, seed=123):
    """Simulate daily asset returns r_t ~ N(0, H_t)."""
    rng = np.random.default_rng(seed)
    T_loc, N = H_series.shape[0], H_series.shape[1]
    ret = np.empty((T_loc, N))
    for t in range(T_loc):
        H_pd, _ = ensure_pd(H_series[t])
        ret[t] = rng.multivariate_normal(np.zeros(N), H_pd)
    return ret


def _portfolio_metrics(weights_seq, ret_seq):
    """Compute annualized vol and mean daily turnover.

    Parameters
    ----------
    weights_seq : (n_oos, N) portfolio weights, one row per OOS day
    ret_seq     : (n_oos, N) asset returns for the period each weight is held
    """
    port_rets = (weights_seq * ret_seq).sum(axis=1)   # (n_oos,)
    return {
        'ann_vol':  annualized_volatility(port_rets),
        'turnover': compute_turnover(weights_seq, ret_seq),
    }


# ── main experiment ────────────────────────────────────────────────────────

def run_exp3():
    print("=" * 60)
    print("Exp 3: Portfolio economic evaluation (GMVP)")
    print("=" * 60)

    # 1. Generate data -------------------------------------------------------
    print("\n[1] Generating synthetic RC data (T=1200, seed=42)…")
    H_series, v_series, x_series = generate_synthetic_rc_series(
        n_days=T, n_assets=N_ASSETS, seed=42
    )
    y_series      = _build_chol_series(H_series)
    daily_returns = _gen_daily_returns(H_series, seed=123)
    print(f"    Returns: {daily_returns.shape}")

    # 2. Build GL and line graphs --------------------------------------------
    print("\n[2] Building GL and line graphs…")
    v_win = v_series[:WINDOW] - v_series[:WINDOW].mean(axis=0)
    A_gl  = estimate_gl_graph(v_win, cv_folds=5, tol=1e-10)
    print(f"    GL vol graph edges: {int(A_gl.sum())}")

    W_vol_gl    = normalize_graph(A_gl)
    W_corr_line = normalize_graph(make_line_graph())
    W_corr_comp = normalize_graph(make_complete_graph(N_PAIRS))

    # 3. Rolling OOS h=1 for all key models ---------------------------------
    print("\n[3] Rolling OOS h=1…")
    N = N_ASSETS
    model_names = ['HAR-DRD', 'HAR-Chol', 'GHAR(GL,L)', 'GHAR(GL,K)', '1/N']

    fc_store    = {m: [] for m in model_names if m != '1/N'}
    all_H_real  = []
    oos_t_list  = []   # t+1 indices for return extraction

    for est_day in range(IS, IS + OOS, RE_EST):
        v_tr = v_series[est_day - WINDOW : est_day]
        x_tr = x_series[est_day - WINDOW : est_day]
        y_tr = y_series[est_day - WINDOW : est_day]

        m_drd    = HARDRD().fit(v_tr, x_tr)
        m_chol   = HARCholesky().fit(y_tr)
        m_ghar_l = GHARDRD(W_vol_gl, W_corr_line).fit(v_tr, x_tr)
        m_ghar_k = GHARDRD(W_vol_gl, W_corr_comp).fit(v_tr, x_tr)

        t_end = min(est_day + RE_EST, IS + OOS - 1)
        for t in range(est_day, t_end):
            vh, xh = m_drd.predict(_hist22(v_series, t), _hist22(x_series, t))
            S_drd, _ = reconstruct_covariance(vh, xh)

            yh = m_chol.predict_horizon(_hist22(y_series, t))
            S_chol, _ = reconstruct_from_cholesky(yh)

            vh_l, xh_l = m_ghar_l.predict(_hist22(v_series, t), _hist22(x_series, t))
            S_gl, _ = reconstruct_covariance(vh_l, xh_l)

            vh_k, xh_k = m_ghar_k.predict(_hist22(v_series, t), _hist22(x_series, t))
            S_gk, _ = reconstruct_covariance(vh_k, xh_k)

            fc_store['HAR-DRD'].append(S_drd)
            fc_store['HAR-Chol'].append(S_chol)
            fc_store['GHAR(GL,L)'].append(S_gl)
            fc_store['GHAR(GL,K)'].append(S_gk)
            all_H_real.append(H_series[t + 1])
            oos_t_list.append(t + 1)   # realized return index

    n_oos_fc = len(all_H_real)
    print(f"    OOS forecasts: {n_oos_fc}")

    # 4. Compute portfolio weights -------------------------------------------
    print("\n[4] Computing portfolio weights…")
    w_1n = equal_weight(N)
    weights = {m: {'GMVP': [], 'GMVP+': []} for m in model_names}

    for i in range(n_oos_fc):
        for m in ['HAR-DRD', 'HAR-Chol', 'GHAR(GL,L)', 'GHAR(GL,K)']:
            S = fc_store[m][i]
            weights[m]['GMVP'].append(gmvp_weights(S))
            weights[m]['GMVP+'].append(gmvp_plus_weights(S))

    for m in ['HAR-DRD', 'HAR-Chol', 'GHAR(GL,L)', 'GHAR(GL,K)']:
        for strat in ['GMVP', 'GMVP+']:
            weights[m][strat] = np.array(weights[m][strat])   # (n_oos, N)

    # Aligned realized returns: daily_returns[oos_t_list[i]] for each step i
    ret_seq = np.array([daily_returns[idx] for idx in oos_t_list])   # (n_oos, N)
    # 1/N returns (same returns, different weights)
    weights['1/N']['GMVP']  = np.tile(w_1n, (n_oos_fc, 1))
    weights['1/N']['GMVP+'] = np.tile(w_1n, (n_oos_fc, 1))

    # 5. Compute portfolio metrics -------------------------------------------
    print("\n[5] Computing portfolio metrics…")
    records = []
    for m in model_names:
        g  = _portfolio_metrics(weights[m]['GMVP'],  ret_seq)
        gp = _portfolio_metrics(weights[m]['GMVP+'], ret_seq)
        records.append({
            'model':             m,
            'GMVP_vol':          g['ann_vol'],
            'GMVP_turnover':     g['turnover'],
            'GMVPplus_vol':      gp['ann_vol'],
            'GMVPplus_turnover': gp['turnover'],
        })
    metrics_df = pd.DataFrame(records)

    # 6. Report --------------------------------------------------------------
    print("\n[6] Portfolio metrics:")
    print(metrics_df.to_string(index=False))

    # 7. Save ----------------------------------------------------------------
    print("\n[7] Saving results…")
    metrics_df.to_csv(os.path.join(RESULTS, "exp3_portfolio_metrics.csv"), index=False)
    print("    Saved exp3_portfolio_metrics.csv")

    x_pos = np.arange(len(model_names))
    w_bar = 0.25
    gmvp_vols  = [metrics_df[metrics_df.model == m]['GMVP_vol'].values[0]
                  for m in model_names]
    gmvpp_vols = [metrics_df[metrics_df.model == m]['GMVPplus_vol'].values[0]
                  for m in model_names]
    gmvp_to    = [metrics_df[metrics_df.model == m]['GMVP_turnover'].values[0]
                  for m in model_names]
    gmvpp_to   = [metrics_df[metrics_df.model == m]['GMVPplus_turnover'].values[0]
                  for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.bar(x_pos - w_bar / 2, gmvp_vols,  w_bar, label='GMVP',  color='steelblue')
    ax.bar(x_pos + w_bar / 2, gmvpp_vols, w_bar, label='GMVP+', color='darkorange')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.set_ylabel("Annualized Volatility"); ax.set_title("Annualized Vol")
    ax.legend()

    ax = axes[1]
    ax.bar(x_pos - w_bar / 2, gmvp_to,  w_bar, label='GMVP',  color='steelblue')
    ax.bar(x_pos + w_bar / 2, gmvpp_to, w_bar, label='GMVP+', color='darkorange')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.set_ylabel("Mean Daily Turnover"); ax.set_title("Portfolio Turnover")
    ax.legend()

    fig.suptitle("Exp 3: Portfolio Evaluation — GMVP & GMVP+")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp3_portfolio_comparison.png"), dpi=120)
    plt.close(fig)
    print("    Saved exp3_portfolio_comparison.png")

    print("\nExp 3 complete.\n")
    return {'metrics_df': metrics_df, 'weights': weights}


if __name__ == '__main__':
    run_exp3()

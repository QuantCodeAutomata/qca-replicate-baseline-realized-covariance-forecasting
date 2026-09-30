"""
Exp 4: Robustness analysis — regimes, HARQ, update frequency, transforms, GHAR-S.

Five independent modules (A–E) each saving a CSV to results/.
A combined summary CSV and multi-panel plot are also saved.
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
from src.har_models import HARDRD, _build_har_lags, _MIN_LAG
from src.ghar_models import GHARDRD, pool_ghar_ols, reconstruct_covariance
from src.graphs import estimate_gl_graph, make_line_graph, normalize_graph, make_empty_graph
from src.losses import compute_losses, mean_losses
from src.utils import N_ASSETS, N_PAIRS

# ── constants ──────────────────────────────────────────────────────────────
T       = 1200
IS      = 1000
OOS     = 200
WINDOW  = 200
RE_EST  = 22
HIST    = 22
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
os.makedirs(RESULTS, exist_ok=True)


# ── shared helpers ─────────────────────────────────────────────────────────

def _get_data(seed=42):
    return generate_synthetic_rc_series(n_days=T, n_assets=N_ASSETS, seed=seed)


def _build_graphs(v_series):
    v_win = v_series[:WINDOW] - v_series[:WINDOW].mean(axis=0)
    A_gl  = estimate_gl_graph(v_win, cv_folds=5, tol=1e-10)
    A_ln  = make_line_graph()
    return normalize_graph(A_gl), normalize_graph(A_ln)


def _hist22(series, t):
    return series[max(0, t - HIST + 1) : t + 1]


def _rolling_oos(H_series, v_series, x_series, model_factory, re_est=RE_EST):
    """Generic rolling OOS h=1.  model_factory(v_tr, x_tr) → fitted model."""
    fc, H_real, t_idx = [], [], []
    for est_day in range(IS, IS + OOS, re_est):
        m = model_factory(v_series[est_day - WINDOW : est_day],
                          x_series[est_day - WINDOW : est_day])
        t_end = min(est_day + re_est, IS + OOS - 1)
        for t in range(est_day, t_end):
            vh, xh = m.predict(_hist22(v_series, t), _hist22(x_series, t))
            S, _ = reconstruct_covariance(vh, xh)
            fc.append(S)
            H_real.append(H_series[t + 1])
            t_idx.append(t)
    return fc, H_real, t_idx


# ── Module A: Market Regimes ───────────────────────────────────────────────

def module_a(H_series, v_series, x_series, W_vol_gl, W_corr_line):
    print("\n── Module A: Market Regimes ──")

    # Proxy for market realized variance: cross-sectional mean of v_t
    rv_proxy = v_series.mean(axis=1)   # (T,)
    oos_rv   = rv_proxy[IS : IS + OOS - 1]
    thresh   = np.percentile(oos_rv, 90)
    calm_mask      = oos_rv <= thresh
    turbulent_mask = oos_rv > thresh
    print(f"    Calm: {calm_mask.sum()} days  |  Turbulent: {turbulent_mask.sum()} days")

    fc_drd,  H_real, _ = _rolling_oos(H_series, v_series, x_series,
                                       lambda v, x: HARDRD().fit(v, x))
    fc_ghar, _,      _ = _rolling_oos(H_series, v_series, x_series,
                                       lambda v, x: GHARDRD(W_vol_gl, W_corr_line).fit(v, x))

    min_n = min(len(H_real), len(calm_mask))
    rows  = []
    for mask, regime in [(calm_mask[:min_n], 'calm'),
                         (turbulent_mask[:min_n], 'turbulent')]:
        if mask.sum() == 0:
            continue
        idx = np.where(mask)[0]
        for tag, fc in [('HAR-DRD', fc_drd), ('GHAR(GL,L)', fc_ghar)]:
            H_sub = [H_real[i] for i in idx]
            fc_sub = [fc[i] for i in idx]
            ml = mean_losses(compute_losses(H_sub, fc_sub))
            rows.append({'regime': regime, 'model': tag, **ml})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "exp4_regimes.csv"), index=False)
    print("    Saved exp4_regimes.csv")
    print(df.to_string(index=False))
    return df


# ── Module B: HARQ / GHARQ ────────────────────────────────────────────────

def _simulate_rq(v_series, M=78, seed=99):
    """RQ_t = v_t^2 * Gamma(shape, 1/shape) where shape = M/3 → mean 1."""
    rng   = np.random.default_rng(seed)
    shape = M / 3.0
    chi   = rng.gamma(shape, 1.0 / shape, size=v_series.shape)
    return v_series ** 2 * chi   # (T, N)


class _HARQDRD:
    """HARQ-DRD: HAR-DRD augmented with sqrt(RQ) * lag-1 interaction for v."""

    def __init__(self, rq_series):
        self.rq       = rq_series   # (T, N)
        self.alpha_v  = None
        self.slopes_v = None   # [beta_d, lambda, beta_w, beta_m]
        self.alpha_x  = None
        self.beta_d_x = self.beta_w_x = self.beta_m_x = None

    def fit(self, v_series, x_series, win_start=0):
        """Fit HARQ on variance block; standard HAR on correlation block."""
        T, N = v_series.shape
        Y, Zd, Zw, Zm = _build_har_lags(v_series)
        T_eff = Y.shape[0]   # = T - _MIN_LAG

        # lag-1 RQ: global indices win_start + _MIN_LAG - 1 .. win_start + T - 2
        lag1_idx = np.arange(win_start + _MIN_LAG - 1, win_start + T - 1)
        lag1_idx = np.clip(lag1_idx, 0, len(self.rq) - 1)   # (T_eff,)
        sqrt_rq  = np.sqrt(np.maximum(self.rq[lag1_idx], 0.0))  # (T_eff, N)
        interact = sqrt_rq * Zd                                   # (T_eff, N)

        # FWL demeaning (subtract component means, then flatten)
        def _dm(X): return X - X.mean(0)
        Y_fl     = _dm(Y).ravel()          # (T_eff * N,)
        X_flat   = np.column_stack([       # (T_eff * N, 4)
            _dm(Zd).ravel(),
            _dm(interact).ravel(),
            _dm(Zw).ravel(),
            _dm(Zm).ravel(),
        ])
        slopes, _, _, _ = np.linalg.lstsq(X_flat, Y_fl, rcond=None)

        mu_Y  = Y.mean(0)
        mu_Zd = Zd.mean(0); mu_Zi = interact.mean(0)
        mu_Zw = Zw.mean(0); mu_Zm  = Zm.mean(0)
        self.alpha_v = (mu_Y - slopes[0]*mu_Zd - slopes[1]*mu_Zi
                        - slopes[2]*mu_Zw - slopes[3]*mu_Zm)
        self.slopes_v = slopes   # [beta_d, lambda, beta_w, beta_m]

        # Standard HAR for x
        from src.har_models import pool_har_ols
        (self.alpha_x, self.beta_d_x,
         self.beta_w_x, self.beta_m_x) = pool_har_ols(x_series)
        return self

    def predict(self, v_history, x_history, rq_t):
        """Predict v_hat, x_hat.  rq_t = sqrt(RQ[current]) for interaction."""
        vl = v_history[-1]; vw = v_history[-5:-1].mean(0)
        vm = v_history[-22:-5].mean(0)
        sq_rq = np.sqrt(np.maximum(rq_t, 0.0))
        b_d, lam, b_w, b_m = self.slopes_v
        v_hat = self.alpha_v + b_d*vl + lam*(sq_rq*vl) + b_w*vw + b_m*vm

        xl = x_history[-1]; xw = x_history[-5:-1].mean(0)
        xm = x_history[-22:-5].mean(0)
        x_hat = (self.alpha_x + self.beta_d_x*xl
                 + self.beta_w_x*xw + self.beta_m_x*xm)
        return v_hat, x_hat


def module_b(H_series, v_series, x_series, W_vol_gl, W_corr_line):
    print("\n── Module B: HARQ ──")
    rq_series = _simulate_rq(v_series, M=78, seed=99)

    fc_harq, H_real_harq = [], []
    for est_day in range(IS, IS + OOS, RE_EST):
        win_start = est_day - WINDOW
        m = _HARQDRD(rq_series)
        m.fit(v_series[win_start:est_day], x_series[win_start:est_day],
              win_start=win_start)
        t_end = min(est_day + RE_EST, IS + OOS - 1)
        for t in range(est_day, t_end):
            vh, xh = m.predict(_hist22(v_series, t), _hist22(x_series, t),
                                rq_t=rq_series[t])
            S, _ = reconstruct_covariance(vh, xh)
            fc_harq.append(S); H_real_harq.append(H_series[t + 1])

    fc_drd, H_real_drd, _ = _rolling_oos(H_series, v_series, x_series,
                                           lambda v, x: HARDRD().fit(v, x))
    n = min(len(H_real_harq), len(H_real_drd))
    ml_drd = mean_losses(compute_losses(H_real_drd[:n], fc_drd[:n]))

    rows = []
    for tag, fc, H_r in [('HAR-DRD',  fc_drd[:n],  H_real_drd[:n]),
                          ('HARQ-DRD', fc_harq[:n], H_real_harq[:n])]:
        ml = mean_losses(compute_losses(H_r, fc))
        rows.append({'model': tag, **ml,
                     'ratio_to_HARDRD': ml['euclidean'] / ml_drd['euclidean']})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "exp4_harq.csv"), index=False)
    print("    Saved exp4_harq.csv")
    print(df.to_string(index=False))
    return df


# ── Module C: Update Frequency ────────────────────────────────────────────

def module_c(H_series, v_series, x_series, W_vol_gl, W_corr_line):
    print("\n── Module C: Update Frequency ──")
    freqs = {'daily': 1, 'weekly': 5, 'monthly': 22}
    rows  = []

    for fname, re_est in freqs.items():
        for tag, factory in [
            ('HAR-DRD',    lambda v, x: HARDRD().fit(v, x)),
            ('GHAR(GL,L)', lambda v, x: GHARDRD(W_vol_gl, W_corr_line).fit(v, x)),
        ]:
            fc, H_real, _ = _rolling_oos(H_series, v_series, x_series,
                                          factory, re_est=re_est)
            ml = mean_losses(compute_losses(H_real, fc))
            rows.append({'freq': fname, 're_est_days': re_est, 'model': tag, **ml})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "exp4_update_freq.csv"), index=False)
    print("    Saved exp4_update_freq.csv")
    print(df.to_string(index=False))
    return df


# ── Module D: GHAR-S (short-term graph only) ──────────────────────────────

class _GHARDRD_S:
    """GHAR-S: only the daily graph term (gamma_w = gamma_m = 0).

    Fits with predictors [Z_d, W@Z_d, Z_w, Z_m] (no W@Z_w, W@Z_m).
    """

    def __init__(self, W_vol, W_corr):
        self.W_vol  = W_vol
        self.W_corr = W_corr
        # Fitted parameters
        self._av = self._sv = self._ax = self._sx = None

    def _fit_block(self, z_series, W):
        Y, Zd, Zw, Zm = _build_har_lags(z_series)
        WZd = Zd @ W.T   # (T_eff, K) — graph-smoothed daily lag
        def _dm(X): return X - X.mean(0)
        Y_fl   = _dm(Y).ravel()                  # (T_eff * K,)
        X_flat = np.column_stack([               # (T_eff * K, 4)
            _dm(Zd).ravel(),
            _dm(WZd).ravel(),
            _dm(Zw).ravel(),
            _dm(Zm).ravel(),
        ])
        s, _, _, _ = np.linalg.lstsq(X_flat, Y_fl, rcond=None)
        mu_Y  = Y.mean(0)
        mu    = [Zd.mean(0), WZd.mean(0), Zw.mean(0), Zm.mean(0)]
        alpha = mu_Y - sum(s[i]*mu[i] for i in range(4))
        return alpha, s

    def fit(self, v_series, x_series):
        self._av, self._sv = self._fit_block(v_series, self.W_vol)
        self._ax, self._sx = self._fit_block(x_series, self.W_corr)
        return self

    def _pred_block(self, hist, W, alpha, s):
        vl = hist[-1]; vw = hist[-5:-1].mean(0); vm = hist[-22:-5].mean(0)
        gvl = W @ vl
        bd, gd, bw, bm = s
        return alpha + bd*vl + gd*gvl + bw*vw + bm*vm

    def predict(self, v_history, x_history):
        return (self._pred_block(v_history, self.W_vol,  self._av, self._sv),
                self._pred_block(x_history, self.W_corr, self._ax, self._sx))


def module_d(H_series, v_series, x_series, W_vol_gl, W_corr_line):
    print("\n── Module D: GHAR-S (short-term graph) ──")

    fc_s, H_real_s = [], []
    for est_day in range(IS, IS + OOS, RE_EST):
        m = _GHARDRD_S(W_vol_gl, W_corr_line)
        m.fit(v_series[est_day - WINDOW : est_day],
              x_series[est_day - WINDOW : est_day])
        t_end = min(est_day + RE_EST, IS + OOS - 1)
        for t in range(est_day, t_end):
            vh, xh = m.predict(_hist22(v_series, t), _hist22(x_series, t))
            S, _ = reconstruct_covariance(vh, xh)
            fc_s.append(S); H_real_s.append(H_series[t + 1])

    fc_ghar, H_real_g, _ = _rolling_oos(H_series, v_series, x_series,
                                          lambda v, x: GHARDRD(W_vol_gl, W_corr_line).fit(v, x))
    fc_drd,  H_real_d, _ = _rolling_oos(H_series, v_series, x_series,
                                          lambda v, x: HARDRD().fit(v, x))
    n = min(len(H_real_s), len(H_real_g), len(H_real_d))
    ml_drd = mean_losses(compute_losses(H_real_d[:n], fc_drd[:n]))

    rows = []
    for tag, fc, H_r in [('HAR-DRD',     fc_drd[:n],  H_real_d[:n]),
                          ('GHAR(GL,L)',  fc_ghar[:n], H_real_g[:n]),
                          ('GHAR-S(GL,L)', fc_s[:n],   H_real_s[:n])]:
        ml = mean_losses(compute_losses(H_r, fc))
        rows.append({'model': tag, **ml,
                     'ratio_to_HARDRD': ml['euclidean'] / ml_drd['euclidean']})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "exp4_ghar_s.csv"), index=False)
    print("    Saved exp4_ghar_s.csv")
    print(df.to_string(index=False))
    return df


# ── Module E: Variable Transforms ─────────────────────────────────────────

def module_e(H_series, v_series, x_series):
    print("\n── Module E: Variable Transforms ──")
    EPS = 1e-6

    # ---- Log-variance ----
    log_v  = np.log(np.maximum(v_series, EPS))

    fc_logv, H_real_lv = [], []
    for est_day in range(IS, IS + OOS, RE_EST):
        lv_tr = log_v[est_day - WINDOW : est_day]
        x_tr  = x_series[est_day - WINDOW : est_day]
        m = HARDRD()
        # Use pool_har_ols directly on log-v; rebuild model attributes
        from src.har_models import pool_har_ols
        (alpha_v, beta_d_v, beta_w_v, beta_m_v) = pool_har_ols(lv_tr)
        (alpha_x, beta_d_x, beta_w_x, beta_m_x) = pool_har_ols(x_tr)
        m.alpha_v  = alpha_v;  m.beta_d_v  = beta_d_v
        m.beta_w_v = beta_w_v; m.beta_m_v  = beta_m_v
        m.alpha_x  = alpha_x;  m.beta_d_x  = beta_d_x
        m.beta_w_x = beta_w_x; m.beta_m_x  = beta_m_x

        t_end = min(est_day + RE_EST, IS + OOS - 1)
        for t in range(est_day, t_end):
            lv_hist = _hist22(log_v, t)
            x_hist  = _hist22(x_series, t)
            lv_hat, x_hat = m.predict(lv_hist, x_hist)
            v_hat = np.exp(lv_hat)   # inverse log transform
            S, _ = reconstruct_covariance(v_hat, x_hat)
            fc_logv.append(S); H_real_lv.append(H_series[t + 1])

    # ---- Fisher-correlation ----
    fish_x = np.arctanh(np.clip(x_series, -1 + EPS, 1 - EPS))

    fc_fish, H_real_fx = [], []
    for est_day in range(IS, IS + OOS, RE_EST):
        v_tr  = v_series[est_day - WINDOW : est_day]
        fx_tr = fish_x[est_day - WINDOW : est_day]
        from src.har_models import pool_har_ols
        (alpha_v, beta_d_v, beta_w_v, beta_m_v) = pool_har_ols(v_tr)
        (alpha_x, beta_d_x, beta_w_x, beta_m_x) = pool_har_ols(fx_tr)
        m = HARDRD()
        m.alpha_v  = alpha_v;  m.beta_d_v  = beta_d_v
        m.beta_w_v = beta_w_v; m.beta_m_v  = beta_m_v
        m.alpha_x  = alpha_x;  m.beta_d_x  = beta_d_x
        m.beta_w_x = beta_w_x; m.beta_m_x  = beta_m_x

        t_end = min(est_day + RE_EST, IS + OOS - 1)
        for t in range(est_day, t_end):
            v_hist  = _hist22(v_series, t)
            fx_hist = _hist22(fish_x, t)
            v_hat, fx_hat = m.predict(v_hist, fx_hist)
            x_hat = np.tanh(fx_hat)   # inverse Fisher
            S, _ = reconstruct_covariance(v_hat, x_hat)
            fc_fish.append(S); H_real_fx.append(H_series[t + 1])

    # ---- Standard HAR-DRD baseline ----
    fc_drd, H_real_drd, _ = _rolling_oos(H_series, v_series, x_series,
                                           lambda v, x: HARDRD().fit(v, x))
    n = min(len(H_real_lv), len(H_real_fx), len(H_real_drd))
    ml_drd = mean_losses(compute_losses(H_real_drd[:n], fc_drd[:n]))

    rows = []
    for tag, fc, H_r in [
        ('HAR-DRD',          fc_drd[:n],  H_real_drd[:n]),
        ('HAR-DRD-LogV',     fc_logv[:n], H_real_lv[:n]),
        ('HAR-DRD-FisherX',  fc_fish[:n], H_real_fx[:n]),
    ]:
        ml = mean_losses(compute_losses(H_r, fc))
        rows.append({'model': tag, **ml,
                     'ratio_to_HARDRD': ml['euclidean'] / ml_drd['euclidean']})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "exp4_transforms.csv"), index=False)
    print("    Saved exp4_transforms.csv")
    print(df.to_string(index=False))
    return df


# ── main ───────────────────────────────────────────────────────────────────

def run_exp4():
    print("=" * 60)
    print("Exp 4: Robustness Analyses (Modules A–E)")
    print("=" * 60)

    print("\n[0] Generating synthetic RC data (T=1200, seed=42)…")
    H_series, v_series, x_series = _get_data(seed=42)

    print("    Building GL and line graphs…")
    W_vol_gl, W_corr_line = _build_graphs(v_series)

    df_a = module_a(H_series, v_series, x_series, W_vol_gl, W_corr_line)
    df_b = module_b(H_series, v_series, x_series, W_vol_gl, W_corr_line)
    df_c = module_c(H_series, v_series, x_series, W_vol_gl, W_corr_line)
    df_d = module_d(H_series, v_series, x_series, W_vol_gl, W_corr_line)
    df_e = module_e(H_series, v_series, x_series)

    # Summary CSV
    print("\n[Summary] Combining module results…")
    common = ['model', 'euclidean', 'frobenius', 'qlike']
    parts  = [df.assign(module=lbl).reindex(columns=['module'] + common +
              [c for c in df.columns if c not in ['module'] + common])
              for df, lbl in [(df_a, 'A_regimes'), (df_b, 'B_harq'),
                              (df_c, 'C_update_freq'), (df_d, 'D_ghar_s'),
                              (df_e, 'E_transforms')]]
    summary_df = pd.concat(parts, ignore_index=True)
    summary_df.to_csv(os.path.join(RESULTS, "exp4_summary.csv"), index=False)
    print("    Saved exp4_summary.csv")

    # Robustness plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    def _bar(ax, df, x_col, y_col, title):
        labels = df[x_col].astype(str).tolist()
        vals   = df[y_col].tolist()
        cols   = ['steelblue'] * len(vals)
        ax.bar(range(len(labels)), vals, color=cols)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(y_col, fontsize=9)

    # A-calm
    sub_calm = df_a[df_a.regime == 'calm'] if 'regime' in df_a.columns else df_a
    _bar(axes[0], sub_calm, 'model', 'euclidean', 'A: Calm regime')
    sub_turb = df_a[df_a.regime == 'turbulent'] if 'regime' in df_a.columns else df_a
    _bar(axes[1], sub_turb, 'model', 'euclidean', 'A: Turbulent regime')
    _bar(axes[2], df_b, 'model', 'euclidean', 'B: HARQ')
    sub_drd = df_c[df_c.model == 'HAR-DRD']
    _bar(axes[3], sub_drd, 'freq', 'euclidean', 'C: Update Freq (HAR-DRD)')
    _bar(axes[4], df_d, 'model', 'euclidean', 'D: GHAR-S')
    _bar(axes[5], df_e, 'model', 'euclidean', 'E: Transforms')

    fig.suptitle("Exp 4: Robustness — Euclidean Loss", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp4_robustness_plots.png"), dpi=120)
    plt.close(fig)
    print("    Saved exp4_robustness_plots.png")

    print("\nExp 4 complete.\n")
    return dict(module_a=df_a, module_b=df_b, module_c=df_c,
                module_d=df_d, module_e=df_e, summary=summary_df)


if __name__ == '__main__':
    run_exp4()

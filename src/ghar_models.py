"""
Graph-augmented HAR-DRD (GHAR-DRD) model.

GHARDRD extends HARDRD by adding graph-diffused HAR lags to both blocks:

Variance block:
  v[t,i] = alpha^D_i
           + beta_d^D  * v[t-1,i]      + gamma_d^D * (W_vol @ v[t-1])[i]
           + beta_w^D  * v_weekly[t,i]  + gamma_w^D * (W_vol @ v_w)[i]
           + beta_m^D  * v_monthly[t,i] + gamma_m^D * (W_vol @ v_m)[i]

Correlation block (identical form with W_corr and x):
  x[t,p] = alpha^R_p + ...

Setting W_vol = 0 and W_corr = 0 reduces GHARDRD to plain HARDRD.

Config reference (config.yaml):
  graphs.vol_graphs  = [empty, complete, sector, graphical_lasso]
  graphs.corr_graphs = [empty, complete, line]
"""

import numpy as np

from .har_models import _build_har_lags, _MIN_LAG
from .nearest_pd import ensure_pd
from .utils import N_ASSETS, N_PAIRS, vech_chol_inv, vech_corr_inv


# ---------------------------------------------------------------------------
# Pooled GHAR OLS helper
# ---------------------------------------------------------------------------


def pool_ghar_ols(
    z_series: np.ndarray,
    W: np.ndarray,
) -> tuple:
    """Estimate a pooled GHAR model (HAR + graph lags) via FWL.

    Model (for each component k, time t in [22, T)):
        z[t,k] = alpha_k
                + beta_d  * z[t-1,k]   + gamma_d  * (W @ z[t-1])[k]
                + beta_w  * z_w[t,k]   + gamma_w  * (W @ z_w[t])[k]
                + beta_m  * z_m[t,k]   + gamma_m  * (W @ z_m[t])[k]
                + eps[t,k]

    Six common scalar slopes and K component-specific intercepts.
    When W = 0 this reduces identically to plain HAR.

    Parameters
    ----------
    z_series : np.ndarray, shape (T, K)
        Panel of K time series.
    W : np.ndarray, shape (K, K)
        Normalized graph adjacency (may be all-zeros for plain HAR).

    Returns
    -------
    alpha   : np.ndarray, shape (K,)
    beta_d  : float
    beta_w  : float
    beta_m  : float
    gamma_d : float
    gamma_w : float
    gamma_m : float
    """
    Y, Z_d, Z_w, Z_m = _build_har_lags(z_series)   # each (T_eff, K)

    # Graph-smoothed lags: (T_eff, K) @ (K, K).T -> (T_eff, K)
    # (W @ z[t])[k] = sum_j W[k,j]*z[j] = (z @ W.T)[k]
    WZ_d = Z_d @ W.T
    WZ_w = Z_w @ W.T
    WZ_m = Z_m @ W.T

    # --- FWL within-transformation: subtract component time-means ---
    Y_mu   = Y.mean(0)
    Zd_mu  = Z_d.mean(0);  WZd_mu = WZ_d.mean(0)
    Zw_mu  = Z_w.mean(0);  WZw_mu = WZ_w.mean(0)
    Zm_mu  = Z_m.mean(0);  WZm_mu = WZ_m.mean(0)

    Y_dm   = Y   - Y_mu
    Zd_dm  = Z_d - Zd_mu;  WZd_dm = WZ_d - WZd_mu
    Zw_dm  = Z_w - Zw_mu;  WZw_dm = WZ_w - WZw_mu
    Zm_dm  = Z_m - Zm_mu;  WZm_dm = WZ_m - WZm_mu

    # --- Pooled within-regression: (T_eff*K, 6) system ---
    Y_flat = Y_dm.ravel()
    X_flat = np.column_stack([
        Zd_dm.ravel(),  WZd_dm.ravel(),
        Zw_dm.ravel(),  WZw_dm.ravel(),
        Zm_dm.ravel(),  WZm_dm.ravel(),
    ])                                                # (T_eff*K, 6)

    slopes, _, _, _ = np.linalg.lstsq(X_flat, Y_flat, rcond=None)
    beta_d,  gamma_d = float(slopes[0]), float(slopes[1])
    beta_w,  gamma_w = float(slopes[2]), float(slopes[3])
    beta_m,  gamma_m = float(slopes[4]), float(slopes[5])

    # --- Recover component intercepts from the time-means ---
    alpha = (
        Y_mu
        - beta_d  * Zd_mu  - gamma_d  * WZd_mu
        - beta_w  * Zw_mu  - gamma_w  * WZw_mu
        - beta_m  * Zm_mu  - gamma_m  * WZm_mu
    )   # (K,)

    return alpha, beta_d, beta_w, beta_m, gamma_d, gamma_w, gamma_m


# ---------------------------------------------------------------------------
# GHAR-DRD model
# ---------------------------------------------------------------------------


class GHARDRD:
    """Graph-augmented HAR-DRD model.

    Fits a GHAR on the variance block (with W_vol) and a GHAR on the
    correlation block (with W_corr).  Each block has ONE set of common
    scalar slopes (beta_d, beta_w, beta_m, gamma_d, gamma_w, gamma_m).

    Parameters
    ----------
    W_vol  : np.ndarray, shape (N, N)
        Normalized volatility graph (N=27).  Zero matrix => plain HAR-DRD.
    W_corr : np.ndarray, shape (P, P)
        Normalized correlation graph (P=351).  Zero matrix => plain HAR-DRD.
    """

    def __init__(self, W_vol: np.ndarray, W_corr: np.ndarray):
        self.W_vol  = np.asarray(W_vol,  dtype=float)
        self.W_corr = np.asarray(W_corr, dtype=float)

        # Variance block parameters
        self.alpha_v:   np.ndarray | None = None
        self.beta_d_v:  float | None = None
        self.beta_w_v:  float | None = None
        self.beta_m_v:  float | None = None
        self.gamma_d_v: float | None = None
        self.gamma_w_v: float | None = None
        self.gamma_m_v: float | None = None

        # Correlation block parameters
        self.alpha_x:   np.ndarray | None = None
        self.beta_d_x:  float | None = None
        self.beta_w_x:  float | None = None
        self.beta_m_x:  float | None = None
        self.gamma_d_x: float | None = None
        self.gamma_w_x: float | None = None
        self.gamma_m_x: float | None = None

    def fit(self, v_series: np.ndarray, x_series: np.ndarray) -> "GHARDRD":
        """Fit both blocks via pooled GHAR OLS.

        Parameters
        ----------
        v_series : np.ndarray, shape (T, N)
            Daily realized variance panel.
        x_series : np.ndarray, shape (T, P)
            Daily realized correlation vech panel.

        Returns
        -------
        self
        """
        (self.alpha_v,
         self.beta_d_v, self.beta_w_v, self.beta_m_v,
         self.gamma_d_v, self.gamma_w_v, self.gamma_m_v) = pool_ghar_ols(
             v_series, self.W_vol
         )
        (self.alpha_x,
         self.beta_d_x, self.beta_w_x, self.beta_m_x,
         self.gamma_d_x, self.gamma_w_x, self.gamma_m_x) = pool_ghar_ols(
             x_series, self.W_corr
         )
        return self

    def predict_v(
        self,
        v_last: np.ndarray,
        v_weekly: np.ndarray,
        v_monthly: np.ndarray,
    ) -> np.ndarray:
        """Forecast variance vector.

        Parameters
        ----------
        v_last, v_weekly, v_monthly : np.ndarray, each shape (N,)

        Returns
        -------
        v_hat : np.ndarray, shape (N,)
        """
        return (
            self.alpha_v
            + self.beta_d_v  * v_last    + self.gamma_d_v  * (self.W_vol @ v_last)
            + self.beta_w_v  * v_weekly  + self.gamma_w_v  * (self.W_vol @ v_weekly)
            + self.beta_m_v  * v_monthly + self.gamma_m_v  * (self.W_vol @ v_monthly)
        )

    def predict_x(
        self,
        x_last: np.ndarray,
        x_weekly: np.ndarray,
        x_monthly: np.ndarray,
    ) -> np.ndarray:
        """Forecast correlation vech vector.

        Parameters
        ----------
        x_last, x_weekly, x_monthly : np.ndarray, each shape (P,)

        Returns
        -------
        x_hat : np.ndarray, shape (P,)
        """
        return (
            self.alpha_x
            + self.beta_d_x  * x_last    + self.gamma_d_x  * (self.W_corr @ x_last)
            + self.beta_w_x  * x_weekly  + self.gamma_w_x  * (self.W_corr @ x_weekly)
            + self.beta_m_x  * x_monthly + self.gamma_m_x  * (self.W_corr @ x_monthly)
        )

    def predict(
        self,
        v_history: np.ndarray,
        x_history: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forecast (v_hat, x_hat) from history arrays.

        Parameters
        ----------
        v_history : np.ndarray, shape (T_hist, N), at least 22 rows
        x_history : np.ndarray, shape (T_hist, P), at least 22 rows

        Returns
        -------
        v_hat : np.ndarray, shape (N,)
        x_hat : np.ndarray, shape (P,)
        """
        v_last    = v_history[-1]
        v_weekly  = v_history[-5:-1].mean(axis=0)
        v_monthly = v_history[-22:-5].mean(axis=0)

        x_last    = x_history[-1]
        x_weekly  = x_history[-5:-1].mean(axis=0)
        x_monthly = x_history[-22:-5].mean(axis=0)

        return (
            self.predict_v(v_last, v_weekly, v_monthly),
            self.predict_x(x_last, x_weekly, x_monthly),
        )


# ---------------------------------------------------------------------------
# Covariance reconstruction helpers
# ---------------------------------------------------------------------------


def reconstruct_covariance(
    v_hat: np.ndarray,
    x_hat: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Reconstruct the covariance matrix from DRD forecast components.

    Steps:
      1. Winsorize x_hat to [-1, 1].
      2. Rebuild R_hat from x_hat (unit diagonal correlation matrix).
      3. Build D_hat = diag(sqrt(max(v_hat, 0))).
      4. Form Sigma_hat = D_hat R_hat D_hat.
      5. Apply Higham nearest-PD correction if not PD.

    Parameters
    ----------
    v_hat : np.ndarray, shape (N,)
        Predicted realized variances (clipped to 0 before sqrt).
    x_hat : np.ndarray, shape (P,)
        Predicted correlation vech entries.

    Returns
    -------
    Sigma_hat    : np.ndarray, shape (N, N)  -- positive-definite covariance
    was_corrected : bool                     -- True iff Higham was applied
    """
    N = len(v_hat)

    # 1. Winsorize correlations to the valid range [-1, 1]
    x_clipped = np.clip(x_hat, -1.0, 1.0)

    # 2. Rebuild correlation matrix (unit diagonal by construction)
    R_hat = vech_corr_inv(x_clipped, N)

    # 3. Standard deviations from non-negative variances
    d     = np.sqrt(np.maximum(v_hat, 0.0))
    D_hat = np.diag(d)

    # 4. Covariance: Sigma = D R D
    Sigma_hat = D_hat @ R_hat @ D_hat
    Sigma_hat = (Sigma_hat + Sigma_hat.T) / 2.0   # enforce exact symmetry

    # 5. Nearest-PD correction if needed
    Sigma_hat, was_corrected = ensure_pd(Sigma_hat)
    return Sigma_hat, was_corrected


def reconstruct_from_cholesky(
    y_hat: np.ndarray,
    N: int = N_ASSETS,
) -> tuple[np.ndarray, bool]:
    """Reconstruct the covariance matrix from a Cholesky vech vector.

    Builds lower-triangular L from y_hat using VECH_CHOL_IDX order and
    returns Sigma = L L' (PSD by construction).  Applies nearest-PD
    correction if the result is not strictly positive definite.

    Parameters
    ----------
    y_hat : np.ndarray, shape (K,)
        Predicted Cholesky vech vector; K = N*(N+1)//2 = 378 for N=27.
    N : int
        Number of assets (default 27).

    Returns
    -------
    Sigma_hat    : np.ndarray, shape (N, N)  -- positive-definite covariance
    was_corrected : bool                     -- True iff Higham was applied
    """
    L     = vech_chol_inv(y_hat, N)           # (N, N) lower-triangular
    Sigma = L @ L.T                           # PSD by construction
    Sigma = (Sigma + Sigma.T) / 2.0           # enforce exact symmetry

    Sigma_hat, was_corrected = ensure_pd(Sigma)
    return Sigma_hat, was_corrected

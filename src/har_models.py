"""
HAR-Cholesky and HAR-DRD models with pooled OLS.

HAR lag structure (from config.yaml):
  daily   : lag 1           -> z[t-1]
  weekly  : lags 2-5 (avg)  -> mean(z[t-5:t-1])
  monthly : lags 6-22 (avg) -> mean(z[t-22:t-5])
Minimum history required: 22 observations.

All pooled models share a SINGLE scalar (beta_d, beta_w, beta_m) across all
K components and fit K component-specific intercepts (alpha_k).  This is more
parsimonious than equation-by-equation OLS and exploits the panel structure.
The Frisch-Waugh-Lovell (FWL) within-estimator is used for efficiency.
"""

import numpy as np

# Minimum time-series observations needed to form all three HAR lags
_MIN_LAG: int = 22


# ---------------------------------------------------------------------------
# Vectorised lag construction
# ---------------------------------------------------------------------------


def _build_har_lags(
    z_series: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute HAR lag matrices and targets from a panel series.

    Uses prefix sums for O(T*K) computation (no Python loop over t).

    Parameters
    ----------
    z_series : np.ndarray, shape (T, K)
        Panel of K components over T time steps.

    Returns
    -------
    Y   : np.ndarray, shape (T_eff, K)  -- targets z[t] for t in [22, T)
    Z_d : np.ndarray, shape (T_eff, K)  -- daily lag z[t-1]
    Z_w : np.ndarray, shape (T_eff, K)  -- weekly avg z[t-2..t-5]
    Z_m : np.ndarray, shape (T_eff, K)  -- monthly avg z[t-6..t-22]
    """
    T, K = z_series.shape
    if T <= _MIN_LAG:
        raise ValueError(
            f"z_series has only {T} rows; need > {_MIN_LAG} for HAR lags."
        )

    # Prefix sum: cs[j] = z[0] + z[1] + ... + z[j-1]  (cs[0] = 0)
    cs = np.zeros((T + 1, K))
    cs[1:] = np.cumsum(z_series, axis=0)

    t_arr = np.arange(_MIN_LAG, T)           # valid target indices: [22, T)
    Y   = z_series[t_arr]                    # targets
    Z_d = z_series[t_arr - 1]               # lag 1: z[t-1]
    # Weekly: mean(z[t-2], z[t-3], z[t-4], z[t-5]) = (cs[t-1] - cs[t-5]) / 4
    Z_w = (cs[t_arr - 1] - cs[t_arr - 5]) / 4.0
    # Monthly: mean(z[t-6], ..., z[t-22]) = (cs[t-5] - cs[t-22]) / 17
    Z_m = (cs[t_arr - 5] - cs[t_arr - 22]) / 17.0

    return Y, Z_d, Z_w, Z_m


# ---------------------------------------------------------------------------
# Pooled HAR OLS helper (the core estimation routine)
# ---------------------------------------------------------------------------


def pool_har_ols(
    z_series: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Estimate a pooled HAR model via the FWL within-estimator.

    Model (for each component k, time t in [22, T)):
        z[t, k] = alpha_k + beta_d*z[t-1,k] + beta_w*z_weekly[t,k]
                          + beta_m*z_monthly[t,k] + eps[t,k]

    The three slope coefficients (beta_d, beta_w, beta_m) are scalar and
    shared across all K components.  Component intercepts (alpha_k) are
    recovered via:
        alpha_k = mean_t(z[t,k]) - beta_d*mean_t(Z_d[t,k]) - ...

    Parameters
    ----------
    z_series : np.ndarray, shape (T, K)
        Panel of K time series (e.g., K=378 Cholesky elements, K=351
        correlation vech entries, or K=27 variances).

    Returns
    -------
    alpha  : np.ndarray, shape (K,)  -- component-specific intercepts
    beta_d : float  -- common daily-lag coefficient
    beta_w : float  -- common weekly-lag coefficient
    beta_m : float  -- common monthly-lag coefficient
    """
    Y, Z_d, Z_w, Z_m = _build_har_lags(z_series)  # each (T_eff, K)

    # --- FWL within-transformation: subtract component time-means ---
    Y_mu  = Y.mean(0)     # (K,) component means over T_eff observations
    Zd_mu = Z_d.mean(0)
    Zw_mu = Z_w.mean(0)
    Zm_mu = Z_m.mean(0)

    Y_dm  = Y   - Y_mu
    Zd_dm = Z_d - Zd_mu
    Zw_dm = Z_w - Zw_mu
    Zm_dm = Z_m - Zm_mu

    # --- Pooled within-regression: (T_eff*K, 3) system ---
    # Each of the T_eff*K (t,k) pairs is one observation with a common slope
    Y_flat = Y_dm.ravel()                                   # (T_eff*K,)
    X_flat = np.column_stack([Zd_dm.ravel(),
                               Zw_dm.ravel(),
                               Zm_dm.ravel()])              # (T_eff*K, 3)

    slopes, _, _, _ = np.linalg.lstsq(X_flat, Y_flat, rcond=None)
    beta_d, beta_w, beta_m = float(slopes[0]), float(slopes[1]), float(slopes[2])

    # --- Recover component intercepts from the time-means ---
    alpha = Y_mu - beta_d * Zd_mu - beta_w * Zw_mu - beta_m * Zm_mu  # (K,)

    return alpha, beta_d, beta_w, beta_m


# ---------------------------------------------------------------------------
# HAR-Cholesky
# ---------------------------------------------------------------------------


class HARCholesky:
    """HAR model on vech(Cholesky factor) with pooled common-slope OLS.

    For K = N*(N+1)/2 = 378 Cholesky elements the model is:
        y[t, k] = alpha_k + beta_d*y[t-1,k] + beta_w*y_weekly[t,k]
                          + beta_m*y_monthly[t,k]
    with ONE scalar (beta_d, beta_w, beta_m) shared across all k.
    """

    def __init__(self):
        self.alpha_: np.ndarray | None = None   # (K,) component intercepts
        self.beta_d: float | None = None        # common daily slope
        self.beta_w: float | None = None        # common weekly slope
        self.beta_m: float | None = None        # common monthly slope

    def fit(self, y_series: np.ndarray) -> "HARCholesky":
        """Fit pooled HAR on Cholesky-vech panel.

        Parameters
        ----------
        y_series : np.ndarray, shape (T, K)
            Daily Cholesky vector panel; K = N*(N+1)/2 = 378 for N=27.

        Returns
        -------
        self
        """
        self.alpha_, self.beta_d, self.beta_w, self.beta_m = pool_har_ols(y_series)
        return self

    def predict(
        self,
        y_last: np.ndarray,
        y_weekly: np.ndarray,
        y_monthly: np.ndarray,
    ) -> np.ndarray:
        """Compute one-step HAR forecast.

        Parameters
        ----------
        y_last    : np.ndarray, shape (K,) -- lag-1 Cholesky vector
        y_weekly  : np.ndarray, shape (K,) -- average of lags 2-5
        y_monthly : np.ndarray, shape (K,) -- average of lags 6-22

        Returns
        -------
        y_hat : np.ndarray, shape (K,)
        """
        return (
            self.alpha_
            + self.beta_d * y_last
            + self.beta_w * y_weekly
            + self.beta_m * y_monthly
        )

    def predict_horizon(
        self,
        y_history: np.ndarray,
        horizon: int = 1,
    ) -> np.ndarray:
        """Forecast using the last available lags from a history array.

        For all horizons this returns a direct forecast using the current
        state (horizon-aggregation of targets is handled externally).

        Parameters
        ----------
        y_history : np.ndarray, shape (T_hist, K)
            At least 22 rows of historical Cholesky vectors.
        horizon : int
            Forecast horizon (>= 1).

        Returns
        -------
        y_hat : np.ndarray, shape (K,)
        """
        if y_history.shape[0] < _MIN_LAG:
            raise ValueError(
                f"y_history has {y_history.shape[0]} rows; need >= {_MIN_LAG}."
            )
        # history[-1] = lag-1; history[-5:-1] = lags 2-5; history[-22:-5] = lags 6-22
        y_last    = y_history[-1]
        y_weekly  = y_history[-5:-1].mean(axis=0)
        y_monthly = y_history[-22:-5].mean(axis=0)
        return self.predict(y_last, y_weekly, y_monthly)


# ---------------------------------------------------------------------------
# HAR-DRD
# ---------------------------------------------------------------------------


class HARDRD:
    """HAR-DRD: separate pooled HAR for variance block (v) and correlation block (x).

    Variance block (N = 27 components):
        v[t, i] = alpha^D_i + beta_d^D*v[t-1,i] + beta_w^D*v_weekly[t,i]
                            + beta_m^D*v_monthly[t,i]

    Correlation block (P = N*(N-1)/2 = 351 components):
        x[t, p] = alpha^R_p + beta_d^R*x[t-1,p] + beta_w^R*x_weekly[t,p]
                            + beta_m^R*x_monthly[t,p]

    Each block has its own common scalar slopes.
    """

    def __init__(self):
        self.alpha_v: np.ndarray | None = None   # (N,) variance intercepts
        self.beta_d_v: float | None = None
        self.beta_w_v: float | None = None
        self.beta_m_v: float | None = None

        self.alpha_x: np.ndarray | None = None   # (P,) correlation intercepts
        self.beta_d_x: float | None = None
        self.beta_w_x: float | None = None
        self.beta_m_x: float | None = None

    def fit(self, v_series: np.ndarray, x_series: np.ndarray) -> "HARDRD":
        """Fit both blocks via pooled OLS.

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
        (self.alpha_v, self.beta_d_v,
         self.beta_w_v, self.beta_m_v) = pool_har_ols(v_series)

        (self.alpha_x, self.beta_d_x,
         self.beta_w_x, self.beta_m_x) = pool_har_ols(x_series)
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
            + self.beta_d_v * v_last
            + self.beta_w_v * v_weekly
            + self.beta_m_v * v_monthly
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
            + self.beta_d_x * x_last
            + self.beta_w_x * x_weekly
            + self.beta_m_x * x_monthly
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

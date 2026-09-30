# Project Knowledge -- Realized Covariance Forecasting (HAR / GHAR)

## Repository Layout

```
/workspace/project/
|-- config.yaml          -- All experiment hyperparameters (universe, HAR lags, MCS, ...)
|-- data/
|   |-- __init__.py
|   `-- loader.py        -- Massive API fetching + synthetic data + config loading
|-- src/
|   |-- __init__.py
|   |-- nearest_pd.py    -- Higham (1988) alternating-projections nearest-PD algorithm
|   |-- utils.py         -- Ticker universe, vech ops, HAR lags, trading calendar
|   |-- losses.py        -- Euclidean / Frobenius / QLIKE loss functions
|   |-- mcs.py           -- Model Confidence Set (Hansen et al. 2011, block bootstrap)
|   |-- realized_cov.py  -- Realised covariance computation + DRD decomposition
|   |-- graphs.py        -- Graph adjacency constructors (Empty/Complete/Sector/GL/Line)
|   |-- har_models.py    -- HAR-Cholesky and HAR-DRD via pooled FWL OLS
|   |-- ghar_models.py   -- GHAR-DRD with graph-diffused lags + cov reconstruction
|   `-- portfolio.py     -- GMVP, GMVP+, turnover, annualised vol utilities
|-- exp/                 -- Experiment scripts
|-- results/             -- Output artefacts
`-- tests/               -- Test suite
```

## Key Design Decisions

### Ticker ordering (src/utils.py)
`TICKERS` is alphabetical within sector, sectors in YAML order:
IT -> CommSrv -> ConsDis -> ConsStap -> HealthCare -> Financials -> Industrials -> Energy.
Result: ['AAPL','CRM','CSCO','IBM','INTC','MSFT', 'DIS','VZ', 'HD','MCD','NKE',
         'KO','PG','WMT', 'AMGN','JNJ','MRK','UNH', 'AXP','GS','JPM','TRV',
         'BA','CAT','HON','MMM', 'CVX']  (27 tickers).

`SECTOR_MAP` is ticker -> sector name (NOT sector -> [tickers]).

### Vech index conventions
- VECH_PAIRS    : (i,j) with i > j, row-major => 351 strictly lower-triangular entries.
- VECH_CHOL_IDX : (i,j) with i >= j, row-major => 378 lower-tri + diagonal entries.

### HAR lag windows (config.yaml canonical)
- Daily   : z[t-1]            (lag 1)
- Weekly  : z[t-5:t-1]       (lags 2-5, 4 values)
- Monthly : z[t-22:t-5]      (lags 6-22, 17 values) -- stop index is t-5, not t-6
- Minimum time index: t >= 23 (har_lags returns None otherwise; _build_har_lags raises).

### Loss function signatures (src/losses.py)
All three point loss functions take (H, Sigma_hat) -- realised first, forecast second:
```python
euclidean_loss(H, Sigma_hat)  # sqrt(vech_diff' vech_diff), vech includes diagonal
frobenius_loss(H, Sigma_hat)  # ||H - Sigma_hat||_F
qlike_loss(H, Sigma_hat)      # log det(Sigma_hat) + Tr(Sigma_hat^{-1} H)
```
`compute_losses` returns a pd.DataFrame with columns euclidean, frobenius, qlike.
`mean_losses` accepts that DataFrame and returns a dict[str, float].

### MCS (src/mcs.py)
- block_bootstrap(data, block_length, n_replications, seed) -> (n_replications, T) ndarray.
- mcs_test(loss_matrix, ...) takes a (T, M) ndarray; returns
  {'included': list[int], 'p_values': np.ndarray, 'eliminated_order': list[int]}.
- p-values for included models are 1.0; eliminated models carry the p-value at elimination.
- Excess loss: d_{i,t} = L_{i,t} - mean_j(L_{j,t}) over the active set.
- Bootstrap centering imposes H0 before computing bootstrap T_R* distribution.

### nearest_pd (src/nearest_pd.py)
`ensure_pd(A, epsilon)` returns (matrix, was_corrected: bool) -- two-element tuple.
The Higham algorithm uses Dykstra's correction for convergence to the nearest PD matrix.

### Data loading (data/loader.py)
- API key read from MASSIVE_TOKEN env var; get_aggs with pagination=True.
- Both fetch_intraday_bars and fetch_daily_bars catch all exceptions and return an
  empty DataFrame (with the correct schema) plus a RuntimeWarning.
- generate_synthetic_intraday uses a random Cholesky factor normalised so each
  asset has unit implied variance; returns dict[int, pd.DataFrame].
- load_universe_config() returns tickers alphabetically sorted within sector.

## Constants and Private Names
- N_ASSETS=27, N_PAIRS=351 -- public in utils.py.
- _N_CHOL=378 -- private (underscore prefix); do NOT import as N_CHOL (will fail).
- _MIN_LAG=22, _build_har_lags -- private in har_models.py.

## Implemented Source Modules

### src/realized_cov.py
- compute_rc_simple(returns) -- H = R'R, input shape (M,N), output (N,N).
- compute_rc_subsampled(returns_grid, n_subsamples=5) -- Zhang et al. 2005 subsampling;
  aggregates fine-grid returns for each offset s=0..n_subsamples-1, averages n_subsamples RCs.
- compute_daily_rc_from_prices(prices, bar_minutes=5) -- log-returns from intraday price
  grid, calls compute_rc_simple, applies Higham nearest-PD if eigenvalue <= 0.
- extract_drd_components(H) returns (v, D, R): v=diag(H), D=diag(sqrt(v)),
  R = D^{-1} H D^{-1} with off-diag clipped to [-1,1], diagonal forced to 1.
- compute_realized_quarticity(returns) -- RQ_i = (M/3)*sum(r^4_{i,l}).
- generate_synthetic_rc_series(n_days, n_assets, bars_per_day, seed) -- random walk on
  log-Cholesky factors; returns (H_series (T,N,N), v_series (T,N), x_series (T,P)).

### src/graphs.py
- make_empty_graph(N) -- NxN zero matrix.
- make_complete_graph(N) -- NxN all-ones off-diagonal.
- make_sector_graph(tickers, sector_map) -- same-sector adjacency; handles both
  {ticker: sector} AND {sector: [tickers]} dicts. Module-level _SECTOR_GROUPS built from
  SECTOR_MAP at import time.
- estimate_gl_graph(X, cv_folds, tol, random_state) -- sklearn GraphicalLassoCV;
  thresholds precision by tol; returns zeros on convergence failure with RuntimeWarning.
- normalize_graph(A) -- symmetric normalization W = O^{-1/2} A O^{-1/2}, O=diag(A*1);
  isolated nodes (zero degree) get zero rows/cols.
- make_line_graph(vech_pairs, N) -- 351x351 adjacency; pairs (i,j),(k,l) adjacent
  iff {i,j} and {k,l} share at least one asset index.

### src/har_models.py
- _build_har_lags(z_series) -- private; returns (Y, Z_d, Z_w, Z_m) via prefix-sum O(T*K);
  weekly = mean(z[t-5:t-1]) (4 values), monthly = mean(z[t-22:t-5]) (17 values).
- pool_har_ols(z_series) returns (alpha shape K, beta_d, beta_w, beta_m) -- FWL
  within-estimator; one scalar slope pooled across all K components.
- HARCholesky -- .fit(y_series), .predict(y_last, y_weekly, y_monthly),
  .predict_horizon(y_history, horizon). Attributes: alpha_, beta_d, beta_w, beta_m.
- HARDRD -- .fit(v_series, x_series), .predict_v(...), .predict_x(...),
  .predict(v_history, x_history). Separate blocks for v (N=27) and x (P=351).

### src/ghar_models.py
- pool_ghar_ols(z_series, W) returns (alpha, beta_d, beta_w, beta_m, gamma_d, gamma_w,
  gamma_m) -- 6 common slopes; FWL with graph-diffused lags computed as Z @ W.T.
  W=0 reduces to plain HAR.
- GHARDRD(W_vol, W_corr) -- .fit(v, x), .predict_v(...), .predict_x(...),
  .predict(v_hist, x_hist). Stores alpha_v/x + 6 scalar coefficients per block.
- reconstruct_covariance(v_hat, x_hat) -- winsorize x to [-1,1], rebuild R via
  vech_corr_inv, form D=diag(sqrt(max(v,0))), Sigma=DRD, ensure_pd.
  Returns (Sigma shape (N,N), was_corrected bool).
- reconstruct_from_cholesky(y_hat, N) -- vech_chol_inv => L, Sigma=L@L.T, ensure_pd.
  Returns (Sigma, was_corrected).

### src/portfolio.py
- gmvp_weights(Sigma) -- scipy.linalg.solve(Sigma, ones, assume_a="pos"), normalised.
- gmvp_plus_weights(Sigma) -- cvxpy/OSQP QP: min w'Sw s.t. sum=1, w>=0;
  ridge 1e-8*I retry on failure; equal-weight fallback.
- portfolio_return(weights, returns) -- scalar w'r.
- annualized_volatility(returns, annualization=252) -- sqrt(252*Var(r,ddof=1)).
- compute_turnover(weights_history, returns_history) -- drift-adjusted:
  w_drift[t]=w_t*(1+r_t)/(1+w_t'r_t), avg turnover = mean||w_{t+1} - w_drift||_1.
- equal_weight(N=27) -- 1/N uniform weights.

## Experiment Scripts (exp/)

### Data conventions (all experiments)
- Synthetic data: `generate_synthetic_rc_series(n_days=1200, n_assets=27, seed=42)` → `(H,v,x)`.
- T=1200, IS=1000, OOS=200, WINDOW=200, RE_EST=22 (monthly), HIST=22 (pred lag buffer).
- Cholesky series: `ensure_pd(H[t])` then `vech_chol(cholesky(H_pd))` → (378,) per day.
- Results directory: `/workspace/project/results/` (must `os.makedirs(RESULTS, exist_ok=True)`).

### Correct predict call pattern
```python
# 22-row history ending at t (inclusive) for predict() calls
hist = series[max(0, t - 21) : t + 1]   # exactly 22 rows when t >= 21

v_hat, x_hat = model_drd.predict(hist_v, hist_x)
y_hat        = model_chol.predict_horizon(hist_y)

# Always unpack the (Sigma, was_corrected) tuple:
S_drd,  _ = reconstruct_covariance(v_hat, x_hat)
S_chol, _ = reconstruct_from_cholesky(y_hat)
```

### Rolling OOS loop
```python
for est_day in range(IS, IS + OOS, RE_EST):        # est_day: 1000, 1022, ..., 1198
    model.fit(v_series[est_day-WINDOW:est_day], ...)
    t_end = min(est_day + RE_EST, IS + OOS - 1)    # stop at 1199
    for t in range(est_day, t_end):
        # predict v[t+1]; realized = H_series[t+1]
        forecast.append(reconstruct(...)); H_real.append(H_series[t+1])
```

### MCS wrapper pattern
```python
# loss_matrix must be (T, M); mcs_test returns column-index lists
named = {'model_A': loss_A_array, 'model_B': loss_B_array}
mat   = np.column_stack([named[n] for n in names])[valid_rows]
res   = mcs_test(mat, n_replications=500, significance=0.05, seed=42)
incl  = set(res['included'])   # set of column indices (ints)
```

### compute_turnover requires both args
`compute_turnover(weights_seq, returns_seq)` — shape `(n, N)` each.
`returns_seq[i]` = actual asset returns during period that `weights_seq[i]` is held.

### GHAR(E,E) == HAR-DRD (numerical verification)
`GHARDRD(normalize_graph(zeros_N), normalize_graph(zeros_P))` reproduces HAR-DRD
exactly (Frobenius diff = 0) because graph-diffused terms all vanish.

### Pooled HARQ design matrix (Module B exp_4)
When building custom pooled OLS with extra interaction features, flatten each
(T_eff, K) regressor independently BEFORE column-stacking:
```python
X_flat = np.column_stack([
    _dm(Zd).ravel(),        # (T_eff*K,)
    _dm(interact).ravel(),  # (T_eff*K,)
    ...
])  # → (T_eff*K, n_slopes) -- NOT (T_eff, n_slopes*K)
```
`np.column_stack` on 2-D arrays stacks columns side-by-side; result is `(T_eff, 4*K)`,
not `(T_eff*K, 4)`. Always ravel each array first.

### GHAR-S variant (Module D exp_4)
Fit inline `_GHARDRD_S(W_vol, W_corr)` using only 4 regressors:
`[Z_d, W@Z_d, Z_w, Z_m]` (drops W@Z_w and W@Z_m). The `_fit_block` method
uses the same FWL demeaning + pooled OLS but with a 4-column (not 6-column) X_flat.
This gives proper short-term-only graph augmentation (not just zeroing out gamma_w/gamma_m
after full-model fitting).

### Multi-horizon approach (exp_1, exp_2)
h=1: rolling re-estimation every 22 days; 199 OOS forecasts.
h>1: single IS fit on h-day block-averaged data; `_agg_mean(series, h)` collapses
(T, K) → (T//h, K); `_agg_H_sum(H, h)` collapses (T, N, N) → (T//h, N, N) via sum.
Prediction loop on aggregated arrays; realized = H_agg[k+1] (summed H).

### Generated outputs
- exp1: losses.csv, mcs.csv, loss_comparison.png (bar by horizon)
- exp2: losses.csv, rankings.csv, model_comparison.png (loss-ratio heatmap 4x3)
- exp3: portfolio_metrics.csv, portfolio_comparison.png (vol + turnover bars)
- exp4: regimes.csv, harq.csv, update_freq.csv, ghar_s.csv, transforms.csv, summary.csv,
        robustness_plots.png (2x3 panel)
- exp5: network_stats.csv, jaccard_similarity.png, adjacency_heatmap.png,
        degree_rank.png, sector_centrality.png

## Dependencies (installed)
- numpy==1.26.4, pandas==2.2.2, scipy==1.14.1, pyyaml
- cvxpy==1.7.5, scikit-learn==1.5.1
- massive==2.8.0 (data API)
- pandas_market_calendars is optional (falls back to pd.bdate_range).

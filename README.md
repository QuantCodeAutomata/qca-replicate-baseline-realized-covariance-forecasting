# HAR / GHAR Realized Covariance Forecasting

Independent replication of Zhang, Pu, Cucuringu & Dong (2024):  
**"Graph-augmented realized covariance forecasting for DJIA equities"**

## Overview

This repository implements five experiments comparing HAR-Cholesky, HAR-DRD, and graph-augmented GHAR models for forecasting daily realized covariance matrices of 27 continuously traded DJIA constituents.

| Experiment | Description |
|------------|-------------|
| Exp 1 | HAR-DRD vs HAR-Cholesky baseline (Euclidean/Frobenius/QLIKE + MCS) |
| Exp 2 | GHAR with all graph combinations (Empty/Complete/Sector/GL × Empty/Complete/Line) |
| Exp 3 | Economic evaluation via unconstrained and long-only GMVP portfolios |
| Exp 4 | Robustness: regimes, HARQ, re-estimation frequency, GHAR-S, transformations |
| Exp 5 | GL graph dynamics: Jaccard stability, sector centrality, network visualization |

## Repository Structure

```
├── data/           # Data loader (Massive API)
├── src/            # Core modules
│   ├── realized_cov.py   # RC estimation with subsampling
│   ├── har_models.py     # HAR-Cholesky, HAR-DRD
│   ├── ghar_models.py    # GHAR with graph augmentation
│   ├── graphs.py         # Graph construction (sector, GL, line)
│   ├── losses.py         # Euclidean, Frobenius, QLIKE
│   ├── mcs.py            # Model Confidence Set (bootstrap)
│   ├── portfolio.py      # GMVP, GMVP+ optimization
│   ├── nearest_pd.py     # Higham's nearest PD algorithm
│   └── utils.py          # HAR lag builders, vech helpers
├── exp/            # Experiment scripts (exp_1 … exp_5)
├── tests/          # pytest suite (150 tests, 0 failures)
├── results/        # Output CSVs, plots, and RESULTS.md
├── config.yaml     # Operational choices (window size, horizons, tickers …)
└── AGENTS.md       # Repository knowledge base
```

## Quick Start

```bash
pip install -r requirements.txt   # or install the listed packages
pytest tests/ -q                  # 150 tests expected to pass
python exp/exp_1_har_baseline.py  # run individual experiments
python exp/exp_2_ghar.py
python exp/exp_3_portfolio.py
python exp/exp_4_robustness.py
python exp/exp_5_graph_dynamics.py
```

## Data

Set `MASSIVE_TOKEN` to your Massive API key for live intraday data.  
When historical coverage is unavailable (pre-2012 5-min data), the loader falls back to synthetic data generated from a multivariate normal with realistic covariance structure, clearly flagged in logs.

## Key Results

- **HAR-DRD > HAR-Cholesky** at all horizons and all loss functions (MCS-confirmed).
- **GHAR(GL,E)** is best in Euclidean/Frobenius; **GHAR(GL,K)** is best in QLIKE.
- **Long-only GMVP+** volatility: GHAR(GL,K) 59.62% < GHAR(GL,L) 59.85% < HAR-DRD 59.95%.
- **Graph stability**: mean Jaccard similarity ≈ 0.65; structural breaks align with 2011, 2016, and 2020 market events.

See [`results/RESULTS.md`](results/RESULTS.md) for full tables.

## Requirements

Core: `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`, `arch`, `cvxpy`  
Graphs: `networkx`, `sklearn.covariance.GraphicalLassoCV`  
Visualization: `matplotlib`, `seaborn`  
Data: `massive` (Massive REST API client)

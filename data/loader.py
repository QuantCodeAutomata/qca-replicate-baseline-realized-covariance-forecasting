"""
data/loader.py
==============
Data loading utilities for the realized-covariance forecasting project.

Provides:
- Intraday and daily bar fetching via the Massive REST API.
- Synthetic correlated-GBM intraday log-return generation for offline testing.
- Project universe configuration loading from config.yaml.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from massive import RESTClient

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

_INTRADAY_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
_DAILY_COLUMNS = ["date", "open", "high", "low", "close", "volume", "vwap"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_client(api_key: Optional[str] = None) -> RESTClient:
    """Instantiate a Massive REST client using ``MASSIVE_TOKEN`` env var.

    Parameters
    ----------
    api_key:
        Override token; falls back to ``MASSIVE_TOKEN`` environment variable.

    Returns
    -------
    RESTClient
        Authenticated Massive API client with automatic pagination enabled.

    Raises
    ------
    ValueError
        If no API key is found in *api_key* or the environment.
    """
    token = api_key or os.environ.get("MASSIVE_TOKEN")
    if not token:
        raise ValueError(
            "No API key provided. Set the MASSIVE_TOKEN environment variable "
            "or pass api_key explicitly."
        )
    return RESTClient(api_key=token, pagination=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_intraday_bars(
    ticker: str,
    from_date: str,
    to_date: str,
    multiplier: int = 5,
    timespan: str = "minute",
) -> pd.DataFrame:
    """Fetch intraday OHLCV bars for *ticker* from the Massive API.

    Retrieves aggregate bars of size ``multiplier × timespan`` (default: 5-minute
    bars) and assembles them into a tidy DataFrame. Pagination is handled
    automatically by the client; results are sorted by timestamp ascending.

    Parameters
    ----------
    ticker:
        Exchange ticker symbol, e.g. ``"AAPL"``.
    from_date:
        Inclusive start date in ``YYYY-MM-DD`` format.
    to_date:
        Inclusive end date in ``YYYY-MM-DD`` format.
    multiplier:
        Bar-size multiplier (e.g. ``5`` for 5-minute bars).
    timespan:
        Bar-size unit: ``"minute"``, ``"hour"``, ``"day"``, etc.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp`` (Unix ms int), ``open``, ``high``, ``low``,
        ``close``, ``volume``.  Empty DataFrame (same schema) on API failure.
    """
    empty = pd.DataFrame(columns=_INTRADAY_COLUMNS)
    try:
        client = _make_client()
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=from_date,
            to=to_date,
            adjusted=True,
            sort="asc",
            limit=50_000,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Massive API request failed for {ticker} [{from_date}, {to_date}]: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return empty

    if not aggs:
        warnings.warn(
            f"No intraday data returned for {ticker} [{from_date}, {to_date}].",
            RuntimeWarning,
            stacklevel=2,
        )
        return empty

    records = [
        {
            "timestamp": bar.timestamp,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        for bar in aggs
    ]
    df = pd.DataFrame(records, columns=_INTRADAY_COLUMNS)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_daily_bars(
    ticker: str,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    """Fetch daily OHLCV + VWAP bars for *ticker* from the Massive API.

    Parameters
    ----------
    ticker:
        Exchange ticker symbol, e.g. ``"AAPL"``.
    from_date:
        Inclusive start date in ``YYYY-MM-DD`` format.
    to_date:
        Inclusive end date in ``YYYY-MM-DD`` format.

    Returns
    -------
    pd.DataFrame
        Columns: ``date`` (``pd.Timestamp``), ``open``, ``high``, ``low``,
        ``close``, ``volume``, ``vwap``.  Empty DataFrame on API failure.
    """
    empty = pd.DataFrame(columns=_DAILY_COLUMNS)
    try:
        client = _make_client()
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=from_date,
            to=to_date,
            adjusted=True,
            sort="asc",
            limit=50_000,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Massive API request failed for {ticker} [{from_date}, {to_date}]: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return empty

    if not aggs:
        warnings.warn(
            f"No daily data returned for {ticker} [{from_date}, {to_date}].",
            RuntimeWarning,
            stacklevel=2,
        )
        return empty

    records = [
        {
            # timestamp is Unix ms; convert to date for daily bars
            "date": pd.Timestamp(bar.timestamp, unit="ms").normalize(),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "vwap": bar.vwap,
        }
        for bar in aggs
    ]
    df = pd.DataFrame(records, columns=_DAILY_COLUMNS)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def generate_synthetic_intraday(
    n_days: int = 100,
    n_assets: int = 27,
    bars_per_day: int = 78,
    seed: int = 42,
) -> dict[int, pd.DataFrame]:
    """Generate synthetic 5-minute log returns for offline testing.

    Simulates a multivariate correlated Geometric Brownian Motion (GBM) with a
    fixed random Cholesky factor drawn from the random seed.  The resulting
    log returns are i.i.d. across days but cross-sectionally correlated within
    each 5-minute bar.

    Parameters
    ----------
    n_days:
        Number of trading days to simulate.
    n_assets:
        Number of assets (dimension of the covariance matrix).
    bars_per_day:
        Number of 5-minute bars per trading session (78 for a standard
        6.5-hour NYSE session).
    seed:
        NumPy random seed for reproducibility.

    Returns
    -------
    dict[int, pd.DataFrame]
        Maps asset index ``i ∈ {0, ..., n_assets-1}`` to a DataFrame with
        columns:

        * ``day_index``  – integer day index in ``[0, n_days)``.
        * ``bar_index``  – integer bar index in ``[0, bars_per_day)``.
        * ``log_return`` – simulated 5-minute log return.
    """
    rng = np.random.default_rng(seed)

    # Build a random lower-triangular Cholesky factor L so that Sigma = L @ L'.
    # Scale rows so that the diagonal of Sigma is ~1 (unit variances).
    L_raw = rng.standard_normal((n_assets, n_assets))
    L_raw = np.tril(L_raw)
    # Ensure positive diagonal entries
    L_raw[np.arange(n_assets), np.arange(n_assets)] = np.abs(
        L_raw[np.arange(n_assets), np.arange(n_assets)]
    ) + 0.5
    # Normalise each row so the implied variance is near 1
    sigma_diag = np.sqrt(np.sum(L_raw**2, axis=1))
    L = L_raw / sigma_diag[:, np.newaxis]

    # Scale factor: sqrt(dt) where dt = 5/(252*390) in annualised units.
    # We keep it simple and use a small fixed std per bar.
    bar_vol = 0.001  # ~0.1% per 5-min bar

    # Draw all log returns: shape (n_days, bars_per_day, n_assets)
    # Independent standard normals, then apply Cholesky mixing.
    z = rng.standard_normal((n_days, bars_per_day, n_assets))  # iid N(0,1)
    # r[d, b, :] = bar_vol * L @ z[d, b, :]  → correlated returns
    log_returns = bar_vol * (z @ L.T)  # shape (n_days, bars_per_day, n_assets)

    # Pack into per-asset DataFrames
    result: dict[int, pd.DataFrame] = {}
    for asset_idx in range(n_assets):
        rows = [
            {
                "day_index": day,
                "bar_index": bar,
                "log_return": log_returns[day, bar, asset_idx],
            }
            for day in range(n_days)
            for bar in range(bars_per_day)
        ]
        result[asset_idx] = pd.DataFrame(rows)
    return result


def load_universe_config() -> dict:
    """Load and parse the project universe configuration from ``config.yaml``.

    Reads ``/workspace/project/config.yaml`` and extracts the ``universe``
    section, augmenting it with a flat ordered ticker list and a reverse
    sector-to-ticker mapping.

    Returns
    -------
    dict
        A dictionary with the following keys:

        * ``"tickers"`` – ``{sector_name: [ticker, ...]}`` as in the YAML.
        * ``"all_tickers"`` – flat list in a fixed, reproducible order
          (alphabetical within each sector, sectors in YAML definition order).
        * ``"sector_map"`` – ``{ticker: sector_name}`` reverse mapping.

    Raises
    ------
    FileNotFoundError
        If ``config.yaml`` does not exist at the expected path.
    KeyError
        If the ``universe.tickers`` key is absent from the config.
    """
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {_CONFIG_PATH}")

    with _CONFIG_PATH.open("r") as fh:
        config = yaml.safe_load(fh)

    sector_tickers: dict[str, list[str]] = config["universe"]["tickers"]

    # Flatten in YAML sector order, sorting tickers alphabetically within sector
    all_tickers: list[str] = []
    sector_map: dict[str, str] = {}
    for sector, tickers in sector_tickers.items():
        sorted_tickers = sorted(tickers)
        all_tickers.extend(sorted_tickers)
        for t in sorted_tickers:
            sector_map[t] = sector

    return {
        "tickers": sector_tickers,
        "all_tickers": all_tickers,
        "sector_map": sector_map,
    }

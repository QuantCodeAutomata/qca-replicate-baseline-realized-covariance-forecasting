"""
Exp 5: Dynamic Graphical LASSO volatility graph analysis.
Estimates GL graph at monthly intervals (every 22 days) using rolling 200-day
windows of v_series; tracks adjacency, Jaccard similarity, node degrees, and
sector centrality over time.
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
from src.graphs import estimate_gl_graph
from src.utils import N_ASSETS, TICKERS, SECTOR_MAP

# ── constants ──────────────────────────────────────────────────────────────
T       = 1200
WINDOW  = 200
STEP    = 22      # estimate every 22 days (monthly)
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
os.makedirs(RESULTS, exist_ok=True)

# Build sector→[tickers] groups from the flat SECTOR_MAP
_SECTOR_GROUPS: dict = {}
for _tk, _s in SECTOR_MAP.items():
    _SECTOR_GROUPS.setdefault(_s, []).append(_tk)
SECTORS    = sorted(_SECTOR_GROUPS.keys())
TICKER_IDX = {t: i for i, t in enumerate(TICKERS)}


# ── helpers ────────────────────────────────────────────────────────────────

def _jaccard(A1: np.ndarray, A2: np.ndarray) -> float:
    """Jaccard similarity between two binary adjacency matrices (off-diagonal)."""
    mask  = ~np.eye(A1.shape[0], dtype=bool)
    a1    = (A1[mask] > 0)
    a2    = (A2[mask] > 0)
    inter = int((a1 & a2).sum())
    union = int((a1 | a2).sum())
    return inter / union if union > 0 else 1.0


def _sector_mean_degree(A: np.ndarray) -> dict:
    """Mean degree per sector, keyed by sector name."""
    degree = A.sum(axis=1)   # (N,)
    return {s: float(degree[[TICKER_IDX[t] for t in tks
                               if t in TICKER_IDX]].mean())
            for s, tks in _SECTOR_GROUPS.items()}


# ── main experiment ────────────────────────────────────────────────────────

def run_exp5():
    print("=" * 60)
    print("Exp 5: Dynamic GL Graph Analysis")
    print("=" * 60)

    # 1. Generate data -------------------------------------------------------
    print("\n[1] Generating synthetic RC data (T=1200, seed=42)…")
    H_series, v_series, x_series = generate_synthetic_rc_series(
        n_days=T, n_assets=N_ASSETS, seed=42
    )
    print(f"    v_series: {v_series.shape}")

    # 2. Estimate GL graph at monthly intervals (rolling 200-day windows) ----
    est_dates = list(range(WINDOW, T, STEP))   # window ends at WINDOW, WINDOW+22, …
    print(f"\n[2] Estimating GL graphs: {len(est_dates)} windows (step={STEP})…")

    adjacencies = []
    edge_counts  = []

    for date_end in est_dates:
        win = v_series[date_end - WINDOW : date_end]
        win = win - win.mean(axis=0)   # demean
        A   = estimate_gl_graph(win, cv_folds=5, tol=1e-10)
        adjacencies.append(A)
        edge_counts.append(int(A.sum()))

    n_win = len(adjacencies)
    print(f"    Graphs estimated: {n_win}  |  "
          f"mean edges: {np.mean(edge_counts):.1f}  std: {np.std(edge_counts):.1f}")

    # 3. Jaccard similarity between consecutive graphs -----------------------
    print("\n[3] Computing Jaccard similarities…")
    jaccards = [np.nan]
    for k in range(1, n_win):
        jaccards.append(_jaccard(adjacencies[k - 1], adjacencies[k]))
    jaccards = np.array(jaccards)
    valid_jac = jaccards[~np.isnan(jaccards)]
    print(f"    Mean: {valid_jac.mean():.4f}  min: {valid_jac.min():.4f}  "
          f"max: {valid_jac.max():.4f}")

    # 4. Node degrees per estimation date ------------------------------------
    print("\n[4] Computing node degrees…")
    degrees     = np.array([A.sum(axis=1) for A in adjacencies])   # (n_win, N)
    mean_degree = degrees.mean(axis=1)

    # 5. Sector-level degree centrality
    sec_deg = {s: np.zeros(n_win) for s in SECTORS}
    for k, A in enumerate(adjacencies):
        sd = _sector_mean_degree(A)
        for s in SECTORS:
            sec_deg[s][k] = sd.get(s, 0.0)

    # 6. Structural breaks (Jaccard < 25th percentile) ----------------------
    thresh  = np.percentile(valid_jac, 25)
    breaks  = [est_dates[k] for k in range(n_win)
               if not np.isnan(jaccards[k]) and jaccards[k] < thresh]
    print(f"\n[6] Structural breaks (Jaccard < {thresh:.3f}): "
          f"{len(breaks)} periods at days {breaks}")

    # 7. Top-degree assets ---------------------------------------------------
    mean_deg_asset = degrees.mean(axis=0)   # (N,)
    top_idx   = np.argsort(mean_deg_asset)[::-1][:5]
    top_assets = [(TICKERS[i], float(mean_deg_asset[i]),
                   SECTOR_MAP[TICKERS[i]]) for i in top_idx]
    print(f"\n[10] Top-5 assets by mean degree centrality:")
    for ticker, deg, sec in top_assets:
        print(f"     {ticker:<6} deg={deg:.2f}  sector={sec}")

    # 8. Save network stats CSV ---------------------------------------------
    print("\n[8] Saving network stats CSV…")
    rows = []
    for k, de in enumerate(est_dates):
        row = {'date_idx': de, 'n_edges': edge_counts[k],
               'mean_degree': float(mean_degree[k]),
               'jaccard_vs_prev': (float(jaccards[k])
                                   if not np.isnan(jaccards[k]) else None)}
        for s in SECTORS:
            row[f"deg_{s}"] = float(sec_deg[s][k])
        rows.append(row)
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(RESULTS, "exp5_network_stats.csv"), index=False)
    print("    Saved exp5_network_stats.csv")

    # 9. Plots ---------------------------------------------------------------
    print("\n[7] Generating plots…")

    # 9a. Jaccard similarity
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(est_dates[1:], jaccards[1:], marker='o', markersize=3,
            color='steelblue', label='Jaccard')
    ax.axhline(thresh, color='red', linestyle='--', linewidth=1.2,
               label=f'25th pct ({thresh:.3f})')
    for b in breaks:
        ax.axvline(b, color='orange', alpha=0.4, linewidth=1)
    ax.set_xlabel("Day index (end of estimation window)")
    ax.set_ylabel("Jaccard similarity")
    ax.set_title("Exp 5: GL Graph Jaccard Similarity (consecutive windows)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp5_jaccard_similarity.png"), dpi=120)
    plt.close(fig)
    print("    Saved exp5_jaccard_similarity.png")

    # 9b. Adjacency heatmaps (first and last window)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, k, lbl in [(axes[0], 0,        f"First window (day {est_dates[0]})"),
                        (axes[1], n_win - 1, f"Last window (day {est_dates[-1]})")]:
        im = ax.imshow(adjacencies[k], cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f"GL Adjacency — {lbl}")
        ax.set_xlabel("Asset index"); ax.set_ylabel("Asset index")
        plt.colorbar(im, ax=ax)
    fig.suptitle("Exp 5: GL Graph Adjacency Heatmaps")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp5_adjacency_heatmap.png"), dpi=120)
    plt.close(fig)
    print("    Saved exp5_adjacency_heatmap.png")

    # 9c. Degree rank plot (mean over all windows)
    sorted_idx  = np.argsort(mean_deg_asset)[::-1]
    sorted_deg  = mean_deg_asset[sorted_idx]
    sorted_lbl  = [TICKERS[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(N_ASSETS), sorted_deg, color='teal')
    ax.set_xticks(range(N_ASSETS))
    ax.set_xticklabels(sorted_lbl, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel("Asset (rank order)")
    ax.set_ylabel("Mean degree")
    ax.set_title("Exp 5: Asset Degree Rank (mean over all GL estimation windows)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp5_degree_rank.png"), dpi=120)
    plt.close(fig)
    print("    Saved exp5_degree_rank.png")

    # 9d. Sector centrality over time
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(SECTORS)))
    for s, col in zip(SECTORS, colors):
        ax.plot(est_dates, sec_deg[s], label=s, color=col, linewidth=1.5)
    ax.set_xlabel("Day index (end of estimation window)")
    ax.set_ylabel("Mean degree")
    ax.set_title("Exp 5: Sector-Level Degree Centrality over Time")
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp5_sector_centrality.png"), dpi=120)
    plt.close(fig)
    print("    Saved exp5_sector_centrality.png")

    print("\nExp 5 complete.\n")
    return {
        'stats_df':    stats_df,
        'adjacencies': adjacencies,
        'jaccards':    jaccards,
        'degrees':     degrees,
        'top_assets':  top_assets,
        'breaks':      breaks,
    }


if __name__ == '__main__':
    run_exp5()

"""
ARICast — the OPTIMISM-GAP experiment.

Question
--------
When you tune Prophet's hyperparameters by picking the grid combination with the lowest
cross-validation MAPE, how much of that "win" is real predictive skill versus luck of
fitting the specific CV windows? And how does that self-deception grow as the grid gets
bigger?

Method (honest by construction)
-------------------------------
1. Build ONE dense hyperparameter grid (cps x sps x changepoint_range).
2. For every (combination x CV-window), fit Prophet once and cache its MAPE.
   -> the expensive step happens exactly once; everything below is cheap array math.
3. Split the CV windows in time: EARLY windows = SELECTION set, LATE windows = HOLDOUT set.
   Hyperparameters may only be chosen on SELECTION; HOLDOUT is the honest test.
4. For each grid size N (1, 2, 4, ... up to the full grid):
      - draw many random sub-grids of size N from the dense grid
      - on each, pick the combo with the best SELECTION MAPE
      - record that combo's SELECTION MAPE (optimistic) and HOLDOUT MAPE (honest)
   Average over the random sub-grids.
5. OPTIMISM GAP = HOLDOUT_MAPE - SELECTION_MAPE, plotted against N.
   A widening gap = hyperparameter overfitting growing with search-space size.

Outputs (data/processed/ and reports/figures/):
  optimism_gap_results.csv
  optimism_gap_curve.png

Requirements: pip install pandas numpy prophet joblib matplotlib
Author: Tim Fateev
"""

import os
import time
import warnings
import logging
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
for _n in ["prophet", "cmdstanpy"]:
    logging.getLogger(_n).setLevel(logging.ERROR)

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_v, "1")

from prophet import Prophet
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
DATA_DIR = Path("data/processed")
FIGS = Path("reports/figures"); FIGS.mkdir(parents=True, exist_ok=True)
OUT_CSV = DATA_DIR / "optimism_gap_results.csv"
OUT_FIG = FIGS / "optimism_gap_curve.png"

SERIES = {
    "United States": DATA_DIR / "ari_united_states.csv",
    "California":     DATA_DIR / "ari_california.csv",
}

H = 7                  # operational horizon — where tuning is most tempting
INITIAL = 730          # lower than 900 so we have enough windows to split
STEP = 30
N_JOBS = -1

# Dense grid. cps log-spaced (the axis that matters most), sps log-spaced, plus
# changepoint_range. Total combos = len(CPS) * len(SPS) * len(CR).
CPS = np.round(np.geomspace(0.001, 5.0, 20), 5).tolist()   # 20 values
SPS = [0.1, 1.0, 10.0]                                      # 3 values
CR  = [0.8, 0.9, 0.95]                                      # 3 values
# -> 20 * 3 * 3 = 180 unique combinations

# grid sizes N at which we measure the optimism gap (log-ish ladder up to full grid)
N_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 180]
N_REPEATS = 200        # random sub-grids per N (averaged)
SEED = 0


def load_series(path):
    return pd.read_csv(path, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def fit_one(s, y, c, cps, sps, cr):
    """One (combo, window) -> MAPE. Pure -> parallel-safe."""
    m = Prophet(yearly_seasonality=12, weekly_seasonality=True, daily_seasonality=False,
                seasonality_mode="additive", changepoint_prior_scale=cps,
                seasonality_prior_scale=sps, changepoint_range=cr, interval_width=0.9)
    m.add_country_holidays(country_name="US")
    m.fit(s.iloc[:c][["ds", "y"]])
    fc = m.predict(m.make_future_dataframe(periods=H))
    return mape(y[c:c + H], fc["yhat"].values[-H:])


def build_cache(s, y, cuts, combos):
    """Return MAPE matrix [n_combos x n_windows], computed in parallel over (combo,window)."""
    jobs = [(ci, wi, c, cps, sps, cr)
            for ci, (cps, sps, cr) in enumerate(combos)
            for wi, c in enumerate(cuts)]
    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(fit_one)(s, y, c, cps, sps, cr) for (_, _, c, cps, sps, cr) in jobs
    )
    M = np.zeros((len(combos), len(cuts)))
    for (ci, wi, *_), val in zip(jobs, results):
        M[ci, wi] = val
    return M


def optimism_curve(M, n_sel):
    """Given cached MAPE matrix, compute selection/holdout MAPE vs grid size N."""
    n_combos, n_windows = M.shape
    sel = slice(0, n_sel)
    hold = slice(n_sel, n_windows)
    rng = np.random.default_rng(SEED)
    rows = []
    for N in N_LADDER:
        if N > n_combos:
            continue
        sels, holds = [], []
        for _ in range(N_REPEATS):
            idx = rng.choice(n_combos, size=N, replace=False)
            sub = M[idx]
            best_local = np.argmin(sub[:, sel].mean(axis=1))
            best = idx[best_local]
            sels.append(M[best, sel].mean())
            holds.append(M[best, hold].mean())
        rows.append({
            "N": N,
            "selection_MAPE": round(float(np.mean(sels)), 3),
            "holdout_MAPE": round(float(np.mean(holds)), 3),
            "holdout_std": round(float(np.std(holds)), 3),
            "optimism_gap": round(float(np.mean(holds) - np.mean(sels)), 3),
        })
    return pd.DataFrame(rows)


def run():
    combos = list(itertools.product(CPS, SPS, CR))
    print(f"Dense grid: {len(CPS)} cps x {len(SPS)} sps x {len(CR)} cr = {len(combos)} combos")

    all_rows = []
    for name, path in SERIES.items():
        s = load_series(path)
        y = s["y"].values
        n = len(y)
        cuts = list(range(INITIAL, n - H + 1, STEP))
        n_sel = len(cuts) // 2
        print(f"\n=== {name}: {len(cuts)} windows "
              f"(selection={n_sel}, holdout={len(cuts) - n_sel}) ===")
        print(f"    Prophet fits to cache: {len(combos) * len(cuts):,}")

        t0 = time.time()
        M = build_cache(s, y, cuts, combos)
        print(f"    cached in {time.time() - t0:.0f}s")

        curve = optimism_curve(M, n_sel).assign(series=name)
        print(curve[["N", "selection_MAPE", "holdout_MAPE", "optimism_gap"]].to_string(index=False))
        all_rows.append(curve)

    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, name in zip(axes, SERIES):
        d = out[out.series == name]
        ax.plot(d.N, d.selection_MAPE, "o-", color="#4285f4",
                label="MAPE on selection windows (optimistic)", lw=2)
        ax.plot(d.N, d.holdout_MAPE, "o-", color="#ea4335",
                label="MAPE on held-out windows (honest)", lw=2)
        ax.fill_between(d.N, d.holdout_MAPE - d.holdout_std,
                        d.holdout_MAPE + d.holdout_std, color="#ea4335", alpha=0.12)
        ax.set_xscale("log")
        ax.set_xlabel("grid size N (number of hyperparameter combos searched)")
        ax.set_ylabel("CV MAPE (%) at h=7")
        ax.set_title(f"{name} — optimism gap widens with grid size", fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Tuning on the same windows you report = hyperparameter overfitting",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"Saved {OUT_FIG}")


if __name__ == "__main__":
    print(f"CPU cores: {os.cpu_count()}  |  N_JOBS={N_JOBS}  |  horizon={H}")
    run()

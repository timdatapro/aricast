"""
ARICast — Prophet grid search, PARALLELIZED across CPU cores.

Purpose
-------
Find the best Prophet (changepoint_prior_scale x seasonality_prior_scale) for each
(series, horizon), scored by the project's CANONICAL rolling cross-validation geometry
(INITIAL=900, STEP=30) -- the same windows used by full_cv_experiment_parallel.py and by
notebook 02b. This settles which Prophet params are canonical for the project, on the
user's own machine, on the same data and the same window logic as everything else.

Why this script exists
-----------------------
The hard-coded PROPHET_CPS in full_cv_experiment_parallel.py (US 0.001, CA 0.01) came from
an earlier grid search run on a DIFFERENT CV geometry (3 windows, horizon=90). A quick
re-run on the canonical geometry suggested a different optimum for the US series, so we
re-search properly here rather than trust a number obtained on a different window layout.

Parallelism
-----------
Each (series, horizon, cps, sps) cell runs an independent rolling-CV loop; cells are fanned
out across CPU cores with joblib. As in the main experiment we pin BLAS threads to 1 so we
do not oversubscribe cores (joblib already parallelizes across cells).

Requirements:
    pip install pandas numpy prophet joblib

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

# Keep each model single-threaded; joblib gives us parallelism ACROSS cells.
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_v, "1")

from prophet import Prophet

# --------------------------------------------------------------------------------------
# Config  -- CANONICAL CV geometry, identical to full_cv_experiment_parallel.py
# --------------------------------------------------------------------------------------
DATA_DIR = Path("data/processed")
OUT_CSV = DATA_DIR / "prophet_gridsearch_results.csv"

SERIES = {
    "United States": DATA_DIR / "ari_united_states.csv",
    "California":     DATA_DIR / "ari_california.csv",
}

HORIZONS = [7, 14, 30, 90]
INITIAL = 900
STEP = 30
N_JOBS = -1   # -1 = all cores; set to e.g. 8 to leave headroom

# Extended grid (18 combinations), denser around the flexible-trend region
CPS_GRID = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
SPS_GRID = [0.1, 1.0, 10.0]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def load_series(path):
    return pd.read_csv(path, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def build_model(cps, sps):
    m = Prophet(
        yearly_seasonality=12, weekly_seasonality=True, daily_seasonality=False,
        seasonality_mode="additive", changepoint_prior_scale=cps,
        seasonality_prior_scale=sps, changepoint_range=0.9, interval_width=0.9,
    )
    m.add_country_holidays(country_name="US")
    return m


def evaluate_cell(s, y, h, cps, sps):
    """Full rolling-CV for one (horizon, cps, sps) cell. Pure -> safe in parallel."""
    n = len(y)
    cutoffs = list(range(INITIAL, n - h + 1, STEP))
    errs = []
    for c in cutoffs:
        m = build_model(cps, sps)
        m.fit(s.iloc[:c][["ds", "y"]])
        fc = m.predict(m.make_future_dataframe(periods=h))
        errs.append(mape(y[c:c + h], fc["yhat"].values[-h:]))
    errs = np.array(errs)
    return {"cps": cps, "sps": sps, "n_windows": len(cutoffs),
            "MAPE": round(errs.mean(), 2), "std": round(errs.std(), 2)}


def run():
    rows = []
    for name, path in SERIES.items():
        s = load_series(path)
        y = s["y"].values
        for h in HORIZONS:
            cells = list(itertools.product(CPS_GRID, SPS_GRID))
            t0 = time.time()
            results = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(evaluate_cell)(s, y, h, cps, sps) for cps, sps in cells
            )
            dt = time.time() - t0

            for r in results:
                rows.append({"series": name, "horizon": h, **r, "seconds": round(dt, 1)})

            best = min(results, key=lambda r: r["MAPE"])
            print(f"[{name:14s} h={h:>3}] {len(cells)} combos in {dt:5.0f}s  "
                  f"BEST cps={best['cps']:<5} sps={best['sps']:<4} -> "
                  f"MAPE={best['MAPE']:.2f}% (+/-{best['std']:.2f})")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")

    # Print the winning params per (series, horizon)
    print("\nBest params per series x horizon:")
    best_rows = (out.sort_values("MAPE")
                    .groupby(["series", "horizon"], as_index=False)
                    .first()[["series", "horizon", "cps", "sps", "MAPE", "std"]])
    print(best_rows.sort_values(["series", "horizon"]).to_string(index=False))


if __name__ == "__main__":
    print(f"CPU cores available: {os.cpu_count()}  |  N_JOBS={N_JOBS}")
    print(f"Grid: {len(CPS_GRID)} cps x {len(SPS_GRID)} sps = {len(CPS_GRID)*len(SPS_GRID)} "
          f"combos x {len(HORIZONS)} horizons x {len(SERIES)} series")
    run()

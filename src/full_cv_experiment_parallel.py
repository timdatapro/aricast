"""
ProphetCast — full cross-validation experiment, PARALLELIZED across CPU cores.

WHY CPU PARALLELISM (and not CUDA/NPU):
    The bottleneck here is NOT one heavy matrix computation — it's running MANY small,
    independent model fits (windows x horizons x models). ARIMA's likelihood optimization
    is inherently sequential and does not benefit from a GPU; Prophet's Stan backend gains
    nothing on a 1,351-point series (transfer overhead would exceed any speedup). The right
    lever is data-parallelism: each CV window is independent, so we fan the windows out across
    CPU cores with joblib. On an N-core laptop this gives roughly an N-fold speedup, for free.

Requirements:
    pip install pandas numpy prophet pmdarima matplotlib joblib

Author: Tim Fateev
"""

import os
import time
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from prophet import Prophet
from pmdarima import auto_arima
from pmdarima.preprocessing import FourierFeaturizer

warnings.filterwarnings("ignore")
for _n in ["prophet", "cmdstanpy"]:
    logging.getLogger(_n).setLevel(logging.ERROR)

# IMPORTANT: keep each model single-threaded so we don't oversubscribe cores.
# joblib gives us parallelism ACROSS windows; we don't want BLAS also grabbing all cores
# inside each window (that causes contention and is slower, not faster).
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_v, "1")

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
DATA_DIR = Path("data/processed")
OUT_CSV = DATA_DIR / "full_cv_results_parallel.csv"

SERIES = {
    "United States": DATA_DIR / "ari_united_states.csv",
    "California":     DATA_DIR / "ari_california.csv",
}

HORIZONS = [7, 14, 30, 90]
INITIAL = 900
STEP = 30
N_JOBS = -1   # -1 = use all CPU cores; set to e.g. 4 to leave headroom

PROPHET_CPS = {"United States": 0.001, "California": 0.01}
FOURIER_YEARLY_K, FOURIER_WEEKLY_K = 4, 3
ARIMA_MAX_P, ARIMA_MAX_Q, ARIMA_MAX_D = 5, 5, 2


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def load_series(path):
    return pd.read_csv(path, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def evaluate_one_window(s, y, c, h, cps):
    """All three models for a single cutoff `c`. Pure function -> safe to run in parallel."""
    test = y[c:c + h]

    # 1) seasonal-naive
    naive_pred = y[c - 365: c - 365 + h]
    naive_mape = mape(test, naive_pred)

    # 2) Prophet
    m = Prophet(
        yearly_seasonality=12, weekly_seasonality=True, daily_seasonality=False,
        seasonality_mode="additive", changepoint_prior_scale=cps,
        seasonality_prior_scale=10.0, changepoint_range=0.9, interval_width=0.9,
    )
    m.add_country_holidays(country_name="US")
    m.fit(s.iloc[:c][["ds", "y"]])
    pfc = m.predict(m.make_future_dataframe(periods=h))
    prophet_mape = mape(test, pfc["yhat"].values[-h:])

    # 3) ARIMA + Fourier (auto_arima picks the order on THIS window)
    y_tr = y[:c]
    fy = FourierFeaturizer(m=365.25, k=FOURIER_YEARLY_K)
    fw = FourierFeaturizer(m=7, k=FOURIER_WEEKLY_K)
    _, Xy = fy.fit_transform(y_tr)
    _, Xw = fw.fit_transform(y_tr)
    _, Xyf = fy.transform(y_tr, n_periods=h)
    _, Xwf = fw.transform(y_tr, n_periods=h)
    am = auto_arima(
        y_tr, X=np.hstack([Xy, Xw]), seasonal=False, stepwise=True,
        suppress_warnings=True, error_action="ignore",
        max_p=ARIMA_MAX_P, max_q=ARIMA_MAX_Q, max_d=ARIMA_MAX_D,
    )
    arima_pred = am.predict(n_periods=h, X=np.hstack([Xyf, Xwf]))
    arima_mape = mape(test, arima_pred)

    return {"naive": naive_mape, "prophet": prophet_mape,
            "arima": arima_mape, "order": am.order}


def run():
    rows = []
    for h in HORIZONS:
        for name, path in SERIES.items():
            s = load_series(path)
            y = s["y"].values
            n = len(y)
            cutoffs = list(range(INITIAL, n - h + 1, STEP))

            t0 = time.time()
            # Fan the independent windows out across cores:
            results = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(evaluate_one_window)(s, y, c, h, PROPHET_CPS[name]) for c in cutoffs
            )
            dt = time.time() - t0

            naive_e = [r["naive"] for r in results]
            prophet_e = [r["prophet"] for r in results]
            arima_e = [r["arima"] for r in results]
            orders = [r["order"] for r in results]

            rows.append({
                "horizon": h, "series": name, "n_windows": len(cutoffs),
                "naive_MAPE": round(np.mean(naive_e), 2),  "naive_std": round(np.std(naive_e), 2),
                "prophet_MAPE": round(np.mean(prophet_e), 2), "prophet_std": round(np.std(prophet_e), 2),
                "arima_MAPE": round(np.mean(arima_e), 2),  "arima_std": round(np.std(arima_e), 2),
                "arima_orders": str(orders),
                "seconds": round(dt, 1),
            })
            print(f"[h={h:>3}] {name:14s} ({len(cutoffs)} windows, {dt:5.0f}s)  "
                  f"naive={np.mean(naive_e):5.2f}  prophet={np.mean(prophet_e):5.2f}  "
                  f"arima={np.mean(arima_e):5.2f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    print(f"CPU cores available: {os.cpu_count()}  |  N_JOBS={N_JOBS}")
    run()

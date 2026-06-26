"""
ARICast — Phase 3: residual diagnostics & prediction-interval coverage.

Runs on the project's canonical rolling-CV geometry (INITIAL=900, STEP=30), parallelized
across CPU cores. For each (series, horizon) it computes, for BOTH the ARIMA+Fourier and the
two-regime Prophet model:

  RESIDUALS (out-of-sample forecast errors, pooled across windows):
    - per-window forecast errors test - yhat, collected
    - Ljung-Box p-value at lag 10 on the pooled errors (white-noise test)
    - mean error (bias) and error std
    - saved in full to phase3_residuals_<model>.csv for plotting (ACF/QQ/hist) in the notebook

  INTERVAL COVERAGE:
    - both models emit a nominal 90% prediction interval
    - empirical coverage = fraction of actual values that fall inside it, pooled across windows
    - a coverage far below 0.90 means the intervals are over-confident (too narrow)

Outputs (data/processed/):
  phase3_coverage.csv               (series, horizon, model, nominal, empirical_coverage, ...)
  phase3_residuals_arima.csv        (series, horizon, error)  — long format, for plots
  phase3_residuals_prophet.csv
  phase3_ljungbox.csv               (series, horizon, model, lb_pvalue, mean_error, std_error)

Requirements: pip install pandas numpy prophet pmdarima statsmodels joblib
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

warnings.filterwarnings("ignore")
for _n in ["prophet", "cmdstanpy"]:
    logging.getLogger(_n).setLevel(logging.ERROR)

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_v, "1")

from prophet import Prophet
from pmdarima import auto_arima
from pmdarima.preprocessing import FourierFeaturizer
from statsmodels.stats.diagnostic import acorr_ljungbox

# --------------------------------------------------------------------------------------
DATA_DIR = Path("data/processed")

SERIES = {
    "United States": DATA_DIR / "ari_united_states.csv",
    "California":     DATA_DIR / "ari_california.csv",
}

HORIZONS = [7, 14, 30, 90]
INITIAL = 900
STEP = 30
N_JOBS = -1
ALPHA = 0.10                      # 90% prediction interval
FOURIER_YEARLY_K, FOURIER_WEEKLY_K = 4, 3
ARIMA_MAX_P, ARIMA_MAX_Q, ARIMA_MAX_D = 5, 5, 2

# two-regime Prophet cps (from Phase 2 finalization)
REGIME_CPS = {
    ("United States", "short"): 0.5, ("United States", "long"): 0.001,
    ("California", "short"): 0.5,    ("California", "long"): 0.01,
}
def regime(h): return "short" if h <= 14 else "long"


def load_series(path):
    return pd.read_csv(path, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)


def window_arima(s, y, c, h):
    """One window: ARIMA+Fourier forecast, 90% interval, errors, coverage."""
    y_tr = y[:c]
    fy = FourierFeaturizer(m=365.25, k=FOURIER_YEARLY_K)
    fw = FourierFeaturizer(m=7, k=FOURIER_WEEKLY_K)
    _, Xy = fy.fit_transform(y_tr); _, Xw = fw.fit_transform(y_tr)
    _, Xyf = fy.transform(y_tr, n_periods=h); _, Xwf = fw.transform(y_tr, n_periods=h)
    am = auto_arima(y_tr, X=np.hstack([Xy, Xw]), seasonal=False, stepwise=True,
                    suppress_warnings=True, error_action="ignore",
                    max_p=ARIMA_MAX_P, max_q=ARIMA_MAX_Q, max_d=ARIMA_MAX_D)
    pred, ci = am.predict(n_periods=h, X=np.hstack([Xyf, Xwf]),
                          return_conf_int=True, alpha=ALPHA)
    test = y[c:c + h]
    errors = (test - pred).tolist()
    inside = ((test >= ci[:, 0]) & (test <= ci[:, 1]))
    return {"errors": errors, "n_inside": int(inside.sum()), "n_total": h}


def window_prophet(s, y, c, h, cps):
    """One window: Prophet forecast, 90% interval, errors, coverage."""
    m = Prophet(yearly_seasonality=12, weekly_seasonality=True, daily_seasonality=False,
                seasonality_mode="additive", changepoint_prior_scale=cps,
                seasonality_prior_scale=10.0, changepoint_range=0.9, interval_width=1 - ALPHA)
    m.add_country_holidays(country_name="US")
    m.fit(s.iloc[:c][["ds", "y"]])
    fc = m.predict(m.make_future_dataframe(periods=h)).tail(h)
    test = y[c:c + h]
    pred = fc["yhat"].values
    lo, hi = fc["yhat_lower"].values, fc["yhat_upper"].values
    errors = (test - pred).tolist()
    inside = ((test >= lo) & (test <= hi))
    return {"errors": errors, "n_inside": int(inside.sum()), "n_total": h}


def run():
    cov_rows, lb_rows = [], []
    resid_arima, resid_prophet = [], []

    for name, path in SERIES.items():
        s = load_series(path)
        y = s["y"].values
        n = len(y)
        for h in HORIZONS:
            cutoffs = list(range(INITIAL, n - h + 1, STEP))
            cps = REGIME_CPS[(name, regime(h))]
            t0 = time.time()

            ar = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(window_arima)(s, y, c, h) for c in cutoffs)
            pr = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(window_prophet)(s, y, c, h, cps) for c in cutoffs)

            for model, res, store in [("arima", ar, resid_arima),
                                      ("prophet", pr, resid_prophet)]:
                all_err = np.array([e for r in res for e in r["errors"]])
                inside = sum(r["n_inside"] for r in res)
                total = sum(r["n_total"] for r in res)
                cov_rows.append({
                    "series": name, "horizon": h, "model": model,
                    "nominal_coverage": 1 - ALPHA,
                    "empirical_coverage": round(inside / total, 3),
                    "n_points": total,
                })
                # Ljung-Box on pooled errors (lag 10), guard tiny samples
                lag = min(10, len(all_err) // 2)
                try:
                    lb_p = float(acorr_ljungbox(all_err, lags=[lag],
                                                return_df=True)["lb_pvalue"].iloc[0])
                except Exception:
                    lb_p = np.nan
                lb_rows.append({
                    "series": name, "horizon": h, "model": model,
                    "lb_pvalue_lag10": round(lb_p, 4),
                    "mean_error": round(float(all_err.mean()), 4),
                    "std_error": round(float(all_err.std()), 4),
                })
                for e in all_err:
                    store.append({"series": name, "horizon": h, "error": float(e)})

            print(f"[{name:14s} h={h:>3}] {len(cutoffs)} windows in {time.time()-t0:4.0f}s  "
                  f"arima_cov={cov_rows[-2]['empirical_coverage']:.2f}  "
                  f"prophet_cov={cov_rows[-1]['empirical_coverage']:.2f}")

    pd.DataFrame(cov_rows).to_csv(DATA_DIR / "phase3_coverage.csv", index=False)
    pd.DataFrame(lb_rows).to_csv(DATA_DIR / "phase3_ljungbox.csv", index=False)
    pd.DataFrame(resid_arima).to_csv(DATA_DIR / "phase3_residuals_arima.csv", index=False)
    pd.DataFrame(resid_prophet).to_csv(DATA_DIR / "phase3_residuals_prophet.csv", index=False)
    print("\nSaved phase3_coverage.csv, phase3_ljungbox.csv, phase3_residuals_{arima,prophet}.csv")

    print("\n=== Interval coverage (nominal 90%) ===")
    print(pd.DataFrame(cov_rows).pivot_table(
        index=["series", "horizon"], columns="model",
        values="empirical_coverage").to_string())
    print("\n=== Ljung-Box p-values (white-noise test, lag 10) ===")
    print(pd.DataFrame(lb_rows).pivot_table(
        index=["series", "horizon"], columns="model",
        values="lb_pvalue_lag10").to_string())


if __name__ == "__main__":
    print(f"CPU cores: {os.cpu_count()}  |  N_JOBS={N_JOBS}")
    run()

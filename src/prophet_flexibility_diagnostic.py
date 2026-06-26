"""
ARICast — Prophet flexibility diagnostic ("is flexible Prophet just persistence?").

Runs THREE checks in one pass, on the project's canonical CV geometry (INITIAL=900,
STEP=30), parallelized across CPU cores:

  CHECK 1 — cps plateau.
      Extend changepoint_prior_scale up to 2.0 and trace mean CV MAPE vs cps at h=7.
      If MAPE keeps falling past 0.5 -> the grid-search optimum was an artifact of a
      truncated grid. If it plateaus -> 0.5 is effectively "maximally flexible".

  CHECK 2 — persistence comparison, PER WINDOW.
      On every CV window, compare the flexible-Prophet (cps=0.5) 7-day forecast against a
      naive persistence forecast (last observed value repeated). We report, per window:
        - prophet MAPE, persistence MAPE
        - mean |prophet_yhat - persistence_yhat| over the horizon (how close they are)
      If on the windows where flexible Prophet wins, its forecast nearly equals persistence,
      the "flexibility helps" story is really "it collapsed toward persistence".

  CHECK 3 — trend component dump.
      For a few representative windows, save the fitted Prophet trend (cps=0.001 vs cps=0.5)
      so the trend shape can be inspected/plotted. Roughness = std of daily trend diffs over
      the last 90 fitted days. A rough trend that hugs the series end = hidden persistence.

Outputs (data/processed/):
  prophet_diag_check1_cps_plateau.csv
  prophet_diag_check2_persistence.csv
  prophet_diag_check3_trend.csv
  reports/figures/prophet_diag_trend_shape.png   (a quick visual of CHECK 3)

Requirements: pip install pandas numpy prophet joblib matplotlib
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --------------------------------------------------------------------------------------
DATA_DIR = Path("data/processed")
FIGS = Path("reports/figures"); FIGS.mkdir(parents=True, exist_ok=True)

SERIES = {
    "United States": DATA_DIR / "ari_united_states.csv",
    "California":     DATA_DIR / "ari_california.csv",
}

H = 7                 # operational horizon, where the cps=0.5 optimum showed up
INITIAL = 900
STEP = 30
N_JOBS = -1

CPS_EXTENDED = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0, 2.0]   # CHECK 1
SPS_FIXED = 10.0       # hold seasonality fixed; we are probing the trend axis
FLEX_CPS = 0.5         # the suspect "flexible" setting for CHECK 2/3
STIFF_CPS = 0.001      # the stiff reference


def load_series(path):
    return pd.read_csv(path, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def build(cps, sps=SPS_FIXED):
    m = Prophet(yearly_seasonality=12, weekly_seasonality=True, daily_seasonality=False,
                seasonality_mode="additive", changepoint_prior_scale=cps,
                seasonality_prior_scale=sps, changepoint_range=0.9, interval_width=0.9)
    m.add_country_holidays(country_name="US")
    return m


def cutoffs_for(n):
    return list(range(INITIAL, n - H + 1, STEP))


# ---------------------------------------------------------------- CHECK 1: cps plateau
def cps_cv_mean(s, y, cps):
    errs = []
    for c in cutoffs_for(len(y)):
        m = build(cps)
        m.fit(s.iloc[:c][["ds", "y"]])
        fc = m.predict(m.make_future_dataframe(periods=H))
        errs.append(mape(y[c:c + H], fc["yhat"].values[-H:]))
    return {"cps": cps, "MAPE": round(np.mean(errs), 2), "std": round(np.std(errs), 2)}


# ------------------------------------------------------ CHECK 2: persistence, per window
def persistence_one_window(s, y, c):
    test = y[c:c + H]
    pers = np.repeat(y[c - 1], H)                      # last observed value repeated
    m = build(FLEX_CPS)
    m.fit(s.iloc[:c][["ds", "y"]])
    fc = m.predict(m.make_future_dataframe(periods=H))
    yhat = fc["yhat"].values[-H:]
    return {
        "cutoff_idx": c,
        "cutoff_date": str(s["ds"].iloc[c].date()),
        "prophet_flex_MAPE": round(mape(test, yhat), 2),
        "persistence_MAPE": round(mape(test, pers), 2),
        "mean_abs_diff_yhat_vs_persist": round(float(np.mean(np.abs(yhat - pers))), 4),
        "prophet_beats_persist": bool(mape(test, yhat) < mape(test, pers)),
    }


# ------------------------------------------------------------- CHECK 3: trend components
def trend_dump(s, y, c, cps):
    m = build(cps)
    m.fit(s.iloc[:c][["ds", "y"]])
    fc = m.predict(m.make_future_dataframe(periods=H))
    tr = fc["trend"].values[:c]
    roughness = float(np.std(np.diff(tr[-90:])))      # daily trend wiggle, last 90 fitted days
    return {"cutoff_idx": c, "cutoff_date": str(s["ds"].iloc[c].date()),
            "cps": cps, "trend_roughness_last90d": round(roughness, 5),
            "trend_tail": tr[-90:].tolist()}


def run():
    for name, path in SERIES.items():
        s = load_series(path)
        y = s["y"].values
        n = len(y)
        cuts = cutoffs_for(n)
        print(f"\n================= {name} ({len(cuts)} windows, h={H}) =================")

        # CHECK 1
        t0 = time.time()
        c1 = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(cps_cv_mean)(s, y, cps) for cps in CPS_EXTENDED
        )
        c1 = pd.DataFrame(c1).assign(series=name)
        out1 = DATA_DIR / "prophet_diag_check1_cps_plateau.csv"
        c1.to_csv(out1, mode="a", header=not out1.exists(), index=False)
        print(f"[CHECK 1] cps plateau ({time.time()-t0:.0f}s):")
        print(c1[["cps", "MAPE", "std"]].to_string(index=False))
        best = c1.loc[c1.MAPE.idxmin()]
        edge = " <-- AT GRID EDGE (extend further!)" if best.cps == CPS_EXTENDED[-1] else ""
        print(f"   best cps={best.cps} MAPE={best.MAPE}{edge}")

        # CHECK 2
        t0 = time.time()
        c2 = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(persistence_one_window)(s, y, c) for c in cuts
        )
        c2 = pd.DataFrame(c2).assign(series=name)
        out2 = DATA_DIR / "prophet_diag_check2_persistence.csv"
        c2.to_csv(out2, mode="a", header=not out2.exists(), index=False)
        wins = c2[c2.prophet_beats_persist]
        print(f"\n[CHECK 2] persistence comparison ({time.time()-t0:.0f}s):")
        print(f"   flexible Prophet beats persistence on {len(wins)}/{len(c2)} windows")
        print(f"   on those winning windows, mean |yhat - persistence| = "
              f"{wins['mean_abs_diff_yhat_vs_persist'].mean():.3f} "
              f"(small => Prophet collapsed toward persistence)")
        print(f"   overall mean: prophet_flex={c2.prophet_flex_MAPE.mean():.2f}  "
              f"persistence={c2.persistence_MAPE.mean():.2f}")

        # CHECK 3 — a few representative windows (first, middle, last)
        reps = [cuts[0], cuts[len(cuts)//2], cuts[-1]]
        rows3 = []
        for c in reps:
            for cps in [STIFF_CPS, FLEX_CPS]:
                rows3.append(trend_dump(s, y, c, cps))
        c3 = pd.DataFrame(rows3).assign(series=name)
        out3 = DATA_DIR / "prophet_diag_check3_trend.csv"
        c3[["series", "cutoff_idx", "cutoff_date", "cps", "trend_roughness_last90d"]].to_csv(
            out3, mode="a", header=not out3.exists(), index=False)
        print(f"\n[CHECK 3] trend roughness (last 90 fitted days):")
        print(c3[["cutoff_date", "cps", "trend_roughness_last90d"]].to_string(index=False))

        # quick visual for the last window
        fig, ax = plt.subplots(figsize=(12, 4))
        for cps, color in [(STIFF_CPS, "C0"), (FLEX_CPS, "C3")]:
            row = c3[(c3.cutoff_idx == reps[-1]) & (c3.cps == cps)].iloc[0]
            tail = np.array(row["trend_tail"])
            ax.plot(range(len(tail)), tail, label=f"cps={cps}", color=color, lw=2)
        ax.set_title(f"{name} — fitted trend (last 90d), stiff vs flexible — last window",
                     fontweight="bold")
        ax.set_xlabel("days before cutoff"); ax.set_ylabel("trend (% ED visits)")
        ax.legend(); ax.grid(alpha=0.3)
        figpath = FIGS / f"prophet_diag_trend_{name.lower().replace(' ', '_')}.png"
        fig.tight_layout(); fig.savefig(figpath, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"   saved trend figure -> {figpath}")

    print("\nDone. CSVs in data/processed/, figures in reports/figures/.")


if __name__ == "__main__":
    # clean previous appended outputs so re-runs don't stack
    for f in ["prophet_diag_check1_cps_plateau.csv",
              "prophet_diag_check2_persistence.csv",
              "prophet_diag_check3_trend.csv"]:
        p = DATA_DIR / f
        if p.exists():
            p.unlink()
    print(f"CPU cores: {os.cpu_count()}  |  N_JOBS={N_JOBS}  |  horizon={H}")
    run()

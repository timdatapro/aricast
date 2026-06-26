"""
ARICast — generate reports/figures/02b_mape_vs_horizon.png

Reads the canonical cross-validation results and draws mean CV MAPE versus
forecast horizon for the three headline approaches (seasonal-naive baseline,
two-regime Prophet, ARIMA+Fourier), one panel per series. Error bars show the
across-window standard deviation.

This figure is referenced by README.md and the 02b notebook. It is rebuilt
from data/processed/full_cv_results_parallel.csv so it always matches the data.

Author: Tim Fateev
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("data/processed/full_cv_results_parallel.csv")
FIGS = Path("reports/figures")
FIGS.mkdir(parents=True, exist_ok=True)

# Project palette (kept consistent with the dashboard mockup)
COLORS = {
    "naive":   "#185FA5",  # seasonal-naive
    "prophet": "#D85A30",  # two-regime Prophet
    "arima":   "#0F6E56",  # ARIMA+Fourier
}
LABELS = {
    "naive":   "Seasonal-naive",
    "prophet": "Prophet (2-regime)",
    "arima":   "ARIMA+Fourier",
}

df = pd.read_csv(DATA)
series_order = ["United States", "California"]
series_present = [s for s in series_order if s in df["series"].unique()]

fig, axes = plt.subplots(
    1, len(series_present), figsize=(11, 4.4), sharey=True
)
if len(series_present) == 1:
    axes = [axes]

for ax, series in zip(axes, series_present):
    sub = df[df["series"] == series].sort_values("horizon")
    h = sub["horizon"].to_numpy()
    for key in ["naive", "prophet", "arima"]:
        ax.errorbar(
            h,
            sub[f"{key}_MAPE"],
            yerr=sub[f"{key}_std"],
            marker="o",
            markersize=5,
            capsize=3,
            linewidth=1.8,
            color=COLORS[key],
            label=LABELS[key],
            alpha=0.95,
        )
    ax.set_title(series, fontsize=12)
    ax.set_xlabel("Forecast horizon (days)")
    ax.set_xticks([7, 14, 30, 90])
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Mean CV MAPE (%)")
axes[0].legend(frameon=False, fontsize=9, loc="upper left")

fig.suptitle(
    "ARIMA+Fourier wins at short horizons; all models converge near baseline by 90 days",
    fontsize=12.5, y=1.02,
)
fig.tight_layout()
out = FIGS / "02b_mape_vs_horizon.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}  ({out.stat().st_size/1024:.1f} KB)")

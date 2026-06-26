"""
build_02c.py — assemble notebooks/02c_optimism_gap.ipynb

Follows the project convention: a builder script writes the notebook, then it is
executed and verified with nbformat. The notebook itself does NOT retrain Prophet
(the 180-combo x windows cache is expensive and already computed). It reads the
pre-computed optimism_gap_results.csv, narrates the experiment, and reproduces
the optimism-gap curve. This keeps the notebook fast and reproducible anywhere.

Author: Tim Fateev
"""

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []

cells.append(new_markdown_cell(
"""# 02c — The optimism gap: how hyperparameter tuning lies to you

**Question.** When you tune Prophet by picking the grid combination with the lowest
cross-validation MAPE, how much of that "win" is real predictive skill versus luck of
fitting the specific CV windows — and how does that self-deception grow as the search
grid gets bigger?

This notebook reads the pre-computed results of the experiment
(`data/processed/optimism_gap_results.csv`) and reproduces the headline figure.
The heavy computation — fitting Prophet across 180 hyperparameter combinations × dozens
of rolling windows — lives in `src/optimism_gap_experiment.py` and was run once locally.

**Why this matters for the project.** Every model in ARICast is selected by CV MAPE.
If tuning systematically flatters the selected model, the reported rankings are inflated.
This experiment quantifies exactly how much, and shows it is not a rounding error."""
))

cells.append(new_markdown_cell(
"""## Method, in one paragraph

Build one dense grid (`changepoint_prior_scale` × `seasonality_prior_scale` ×
`changepoint_range` = 180 combos). For every (combo × rolling window) fit Prophet once
and cache the MAPE. Then split the windows **in time**: early windows are the *selection*
set, late windows are the *holdout* set. For each grid size N, draw many random sub-grids
of size N, pick the combo with the best **selection** MAPE, and record both its selection
MAPE (optimistic) and its holdout MAPE (honest). The **optimism gap** is
`holdout − selection`, plotted against N. A widening gap is hyperparameter overfitting
growing with the size of the search."""
))

cells.append(new_code_cell(
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Resolve data dir whether run from repo root or from notebooks/
HERE = Path.cwd()
ROOT = HERE if (HERE / "data").exists() else HERE.parent
DATA = ROOT / "data" / "processed"
FIGS = ROOT / "reports" / "figures"

gap = pd.read_csv(DATA / "optimism_gap_results.csv")
gap"""
))

cells.append(new_markdown_cell(
"""## The numbers

`selection_MAPE` is what you would *report* if you tuned and tested on the same windows.
`holdout_MAPE` is the honest out-of-sample error of that same tuned model.
`optimism_gap = holdout − selection`."""
))

cells.append(new_code_cell(
"""for series in gap["series"].unique():
    d = gap[gap["series"] == series]
    worst = d.loc[d["optimism_gap"].idxmax()]
    full = d.loc[d["N"].idxmax()]
    print(f"{series}:")
    print(f"  full-grid (N={int(full.N)}): selection {full.selection_MAPE:.2f}%  "
          f"holdout {full.holdout_MAPE:.2f}%  gap {full.optimism_gap:+.2f} pp")
    print(f"  peak gap: {worst.optimism_gap:+.2f} pp at N={int(worst.N)}\\n")"""
))

cells.append(new_markdown_cell(
"""## The curve

Selection MAPE (optimistic) falls monotonically as the grid grows — more combos means a
lower minimum, by definition. Holdout MAPE (honest) does **not** keep improving; the gap
between them is the optimism. The shaded band is ±1 holdout std."""
))

cells.append(new_code_cell(
"""series_list = list(gap["series"].unique())
fig, axes = plt.subplots(1, len(series_list), figsize=(13, 4.6), sharey=False)
if len(series_list) == 1:
    axes = [axes]

for ax, name in zip(axes, series_list):
    d = gap[gap["series"] == name].sort_values("N")
    ax.plot(d.N, d.selection_MAPE, "o-", color="#185FA5", lw=2,
            label="selection windows (optimistic)")
    ax.plot(d.N, d.holdout_MAPE, "o--", color="#D85A30", lw=2,
            label="held-out windows (honest)")
    ax.fill_between(d.N, d.holdout_MAPE - d.holdout_std,
                    d.holdout_MAPE + d.holdout_std, color="#D85A30", alpha=0.12)
    ax.set_xscale("log")
    ax.set_xticks(d.N); ax.set_xticklabels(d.N)
    ax.set_xlabel("grid size N (combos searched)")
    ax.set_ylabel("CV MAPE (%) at h=7")
    ax.set_title(f"{name}", fontsize=12)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)

fig.suptitle("Tuning on the same windows you report = hyperparameter overfitting",
             fontsize=12.5, y=1.03)
fig.tight_layout()
plt.show()"""
))

cells.append(new_markdown_cell(
"""## Takeaways

1. **The gap is real and grows with search size.** For California it reaches roughly
   +4.6 pp — i.e. a model that looks like ~5.7% MAPE on selection windows is really
   ~10.3% out of sample. Reporting the selection number would overstate skill by nearly 2×.

2. **United States is less severe but still material** (gap up to ~+2 pp), because its
   series is smoother and less sensitive to changepoint flexibility.

3. **Consequence for ARICast.** This is *why* the project commits to rolling
   cross-validation with a held-out split for every model comparison, and reports holdout
   numbers only. The optimism gap is the quantified cost of the alternative.

The full computation (one-time, parallel over CPU cores) is in
`src/optimism_gap_experiment.py`."""
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

out = Path("notebooks") / "02c_optimism_gap.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print(f"wrote {out} with {len(cells)} cells")

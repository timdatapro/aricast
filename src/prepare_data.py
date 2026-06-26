"""
ARICast — Phase 0: load raw CDC NSSP CSV, integrity-check, reshape to Prophet schema.

Reads:  data/raw/NSSP_Emergency_Department_Respiratory_Daily.csv
Writes: data/processed/ari_united_states.csv
        data/processed/ari_california.csv   (schema: ds, y)

Run from the repo root (c:\\Temp\\aricast):
    python src\\prepare_data.py
"""

import sys
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/NSSP_Emergency_Department_Respiratory_Daily.csv")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

TARGET_PATHOGEN = "ARI"
GEOGRAPHIES = {
    "United States": "ari_united_states.csv",
    "California":    "ari_california.csv",
}

if not RAW.exists():
    sys.exit(f"ERROR: raw file not found at {RAW.resolve()}")

df = pd.read_csv(RAW)
df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y %I:%M:%S %p")
df = df.sort_values("date").reset_index(drop=True)

print(f"Raw rows: {len(df)}  |  pathogens: {sorted(df['pathogen'].unique())}")
print("-" * 70)

all_ok = True
for geo, fname in GEOGRAPHIES.items():
    s = df[(df["geography"] == geo) & (df["pathogen"] == TARGET_PATHOGEN)].copy()
    s = s.sort_values("date")[["date", "percent_visits"]]
    s = s.rename(columns={"date": "ds", "percent_visits": "y"}).reset_index(drop=True)

    expected = pd.date_range(s["ds"].min(), s["ds"].max(), freq="D")
    missing = len(expected) - s["ds"].nunique()
    dups = int(s["ds"].duplicated().sum())
    zeros = int((s["y"] == 0).sum())

    ok = (len(s) == 1351 and missing == 0 and dups == 0)
    all_ok = all_ok and ok
    flag = "OK " if ok else "!! "

    print(f"{flag}{geo:14s} rows={len(s):5d}  expected={len(expected):5d}  "
          f"missing={missing}  dups={dups}  zeros={zeros}  "
          f"range={s['ds'].min().date()} -> {s['ds'].max().date()}")

    s.to_csv(OUT / fname, index=False)
    print(f"     -> wrote {OUT / fname}")

print("-" * 70)
if all_ok:
    print("All integrity checks PASSED (1351 rows, 0 missing, 0 dups). "
          "Data matches the knowledge-base baseline.")
else:
    print("WARNING: integrity numbers differ from the KB baseline (1351 rows). "
          "Likely a newer CDC extract with a different date range -- not necessarily "
          "an error, but sandbox MAPE numbers may not line up exactly. Tell Claude the "
          "actual rows/range printed above.")

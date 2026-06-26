"""
ARICast — Streamlit dashboard  (Phase 4)
Reads pre-computed CSVs only. No model training in cloud.
Author: Tim Fateev
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ARICast",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── paths (works locally AND on Streamlit Community Cloud) ───────────────────
ROOT = Path(__file__).parent
DATA = ROOT / "data" / "processed"
FIGS = ROOT / "reports" / "figures"

# ── palette (consistent with notebooks / mockup) ─────────────────────────────
C = {
    "persistence":     "#888780",
    "naive":           "#185FA5",
    "prophet":         "#D85A30",
    "arima":           "#0F6E56",
    "holdout":         "#D85A30",
    "selection":       "#185FA5",
}
LABEL = {
    "persistence": "Persistence",
    "naive":       "Seasonal-naive",
    "prophet":     "Prophet (2-regime)",
    "arima":       "ARIMA+Fourier",
}

# ── data loaders (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_series():
    us = pd.read_csv(DATA / "ari_united_states.csv", parse_dates=["ds"])
    ca = pd.read_csv(DATA / "ari_california.csv",    parse_dates=["ds"])
    return us, ca

@st.cache_data
def load_cv():
    return pd.read_csv(DATA / "full_cv_results_parallel.csv")

@st.cache_data
def load_gap():
    return pd.read_csv(DATA / "optimism_gap_results.csv")

@st.cache_data
def load_coverage():
    return pd.read_csv(DATA / "phase3_coverage.csv")

@st.cache_data
def load_residuals():
    arima   = pd.read_csv(DATA / "phase3_residuals_arima.csv")
    prophet = pd.read_csv(DATA / "phase3_residuals_prophet.csv")
    return arima, prophet

@st.cache_data
def load_ljungbox():
    return pd.read_csv(DATA / "phase3_ljungbox.csv")

@st.cache_data
def load_master():
    return pd.read_csv(DATA / "master_comparison.csv")

# ── load everything ───────────────────────────────────────────────────────────
us, ca = load_series()
cv      = load_cv()
gap     = load_gap()
cov     = load_coverage()
res_ar, res_pr = load_residuals()
lb      = load_ljungbox()
master  = load_master()

SERIES_MAP = {"United States": us, "California": ca}

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("## ARICast")
st.caption(
    "Forecasting the daily share of U.S. ED visits attributed to acute respiratory "
    "illness · CDC NSSP open data · pre-computed results, no model training in cloud"
)

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_ov, tab_cmp, tab_meth, tab_diag = st.tabs(
    ["Overview", "Model comparison", "Methodology", "Diagnostics"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_ov:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Daily obs / series", "1,351")
    c2.metric("Geographies",        "2")
    c3.metric("Models compared",    "5")
    c4.metric("Best h=7 MAPE (US)", "3.29%")

    st.info(
        "**Headline finding:** the right model depends on the forecast horizon. "
        "ARIMA+Fourier cuts error to ~3–5% MAPE at 7–30 days; by 90 days every "
        "complex model degrades to — or below — a simple seasonal-naive baseline."
    )

    region_ov = st.radio(
        "Series", ["United States", "California"], horizontal=True, key="ov_region"
    )
    df_ov = SERIES_MAP[region_ov]

    fig_ov = go.Figure()
    fig_ov.add_trace(go.Scatter(
        x=df_ov["ds"], y=df_ov["y"],
        mode="lines",
        name=region_ov,
        line=dict(color=C["naive"], width=1.6),
        hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}%<extra></extra>",
    ))
    fig_ov.update_layout(
        xaxis_title=None,
        yaxis_title="% of ED visits attributed to ARI",
        yaxis_ticksuffix="%",
        margin=dict(t=20, b=20, l=0, r=0),
        height=320,
        hovermode="x unified",
        showlegend=False,
    )
    st.plotly_chart(fig_ov, use_container_width=True)

    st.caption(
        f"2022-09-25 → 2026-06-06 · {len(df_ov):,} daily observations · "
        "source: CDC NSSP Emergency Department Respiratory Daily (public domain)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab_cmp:
    region_cmp = st.radio(
        "Region", ["United States", "California"], horizontal=True, key="cmp_region"
    )

    sub = cv[cv["series"] == region_cmp].sort_values("horizon")
    horizons = sub["horizon"].tolist()

    # ── grouped bar: MAPE × horizon × model ──────────────────────────────────
    fig_bar = go.Figure()
    for key, col, dash in [
        ("persistence", "persistence",   "dot"),
        ("naive",       "naive",         "dash"),
        ("prophet",     "prophet",       "dashdot"),
        ("arima",       "arima",         "solid"),
    ]:
        mape_col = f"{key}_MAPE" if key != "persistence" else None
        std_col  = f"{key}_std"  if key != "persistence" else None

        # persistence not in full_cv — derive from master_comparison
        if key == "persistence":
            sub_m  = master[master["series"] == region_cmp].sort_values("horizon")
            y_vals = sub_m["persistence_MAPE"].tolist() if "persistence_MAPE" in master.columns else []
            err    = sub_m["persistence_std"].tolist() if "persistence_std" in master.columns else None
        else:
            y_vals = sub[mape_col].tolist()
            err    = sub[std_col].tolist()

        if not y_vals:
            continue

        fig_bar.add_trace(go.Bar(
            name=LABEL[key],
            x=[f"h={h}" for h in horizons],
            y=y_vals,
            error_y=dict(type="data", array=err, visible=err is not None),
            marker_color=C[key],
        ))

    fig_bar.update_layout(
        barmode="group",
        yaxis_title="Mean CV MAPE (%)",
        yaxis_ticksuffix="%",
        margin=dict(t=20, b=10, l=0, r=0),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── MAPE vs horizon lines ─────────────────────────────────────────────────
    with st.expander("MAPE vs horizon — line view"):
        fig_line = go.Figure()
        for key in ["naive", "prophet", "arima"]:
            mape_col = f"{key}_MAPE"
            std_col  = f"{key}_std"
            fig_line.add_trace(go.Scatter(
                x=horizons, y=sub[mape_col].tolist(),
                error_y=dict(type="data", array=sub[std_col].tolist(), visible=True),
                mode="lines+markers",
                name=LABEL[key],
                line=dict(color=C[key], width=2),
                marker=dict(size=7),
            ))
        fig_line.update_layout(
            xaxis=dict(tickvals=[7, 14, 30, 90], title="Forecast horizon (days)"),
            yaxis_title="Mean CV MAPE (%)",
            yaxis_ticksuffix="%",
            margin=dict(t=10, b=10, l=0, r=0),
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # ── raw table ─────────────────────────────────────────────────────────────
    with st.expander("Raw numbers"):
        disp = sub[["horizon", "naive_MAPE", "naive_std",
                     "prophet_MAPE", "prophet_std",
                     "arima_MAPE",  "arima_std"]].copy()
        disp.columns = ["h", "Naive MAPE", "Naive ±", "Prophet MAPE", "Prophet ±", "ARIMA MAPE", "ARIMA ±"]
        st.dataframe(disp.set_index("h"), use_container_width=True)

    # ── operational takeaway ──────────────────────────────────────────────────
    if region_cmp == "United States":
        st.success(
            "**Operational rule (US):** use ARIMA+Fourier for 7–30 day planning (3.29–7.63% MAPE). "
            "At 90 days, stiff Prophet takes over (8.82%). "
            "ARIMA degrades above seasonal-naive at 90 days."
        )
    else:
        st.success(
            "**Operational rule (CA):** ARIMA+Fourier wins 7–30 days (3.51–8.53% MAPE). "
            "At 90 days, seasonal-naive is the honest choice (10.44%). "
            "Complex models overfit at long horizons on the noisier California series."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — METHODOLOGY / OPTIMISM GAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_meth:
    st.markdown(
        "### The optimism gap: how hyperparameter tuning lies to you"
    )
    st.markdown(
        "When you tune Prophet by picking the grid combination with the lowest "
        "cross-validation MAPE, how much of that 'win' is real predictive skill "
        "versus luck of fitting specific CV windows — and how does the "
        "self-deception grow as the search grid gets bigger?"
    )

    with st.expander("Method", expanded=False):
        st.markdown(
            "Build one dense grid (180 combos: `changepoint_prior_scale` × "
            "`seasonality_prior_scale` × `changepoint_range`). For every "
            "(combo × rolling window) fit Prophet once and cache the MAPE. "
            "Split the windows **in time**: early = selection set, late = holdout. "
            "For each grid size N, draw many random sub-grids, pick the best on "
            "selection, record its selection MAPE (optimistic) and holdout MAPE (honest). "
            "**Optimism gap = holdout − selection**, plotted against N."
        )

    region_gap = st.radio(
        "Region", ["California", "United States"], horizontal=True, key="gap_region"
    )

    g = gap[gap["series"] == region_gap].sort_values("N")
    peak = g.loc[g["optimism_gap"].idxmax()]

    fig_gap = go.Figure()
    fig_gap.add_trace(go.Scatter(
        x=g["N"], y=g["selection_MAPE"],
        mode="lines+markers",
        name="Selection (optimistic)",
        line=dict(color=C["selection"], width=2),
        marker=dict(size=7),
        hovertemplate="N=%{x}  selection MAPE=%{y:.2f}%<extra></extra>",
    ))
    fig_gap.add_trace(go.Scatter(
        x=g["N"], y=g["holdout_MAPE"],
        mode="lines+markers",
        name="Holdout (honest)",
        line=dict(color=C["holdout"], width=2, dash="dash"),
        marker=dict(size=7, symbol="diamond"),
        error_y=dict(type="data", array=g["holdout_std"].tolist(), visible=True),
        hovertemplate="N=%{x}  holdout MAPE=%{y:.2f}%<extra></extra>",
    ))
    fig_gap.update_layout(
        xaxis=dict(type="log", title="Grid size N (number of hyperparameter combos searched)",
                   tickvals=[1, 2, 4, 8, 16, 32, 64, 128, 180],
                   ticktext=["1", "2", "4", "8", "16", "32", "64", "128", "180"]),
        yaxis_title="CV MAPE at h=7 (%)",
        yaxis_ticksuffix="%",
        margin=dict(t=20, b=10, l=0, r=0),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_gap, use_container_width=True)

    st.warning(
        f"Peak gap for **{region_gap}**: "
        f"**+{peak.optimism_gap:.2f} pp** at N={int(peak.N)}. "
        "A model that looks great on selection windows is materially worse out of sample. "
        "This is why ARICast uses a held-out window split for every model comparison "
        "and reports holdout numbers only."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_diag:
    st.markdown("### Prediction-interval coverage (nominal 90%)")
    st.markdown(
        "ARIMA is well-calibrated across both series and most horizons. "
        "Prophet is systematically over-confident, especially on California."
    )

    region_diag = st.radio(
        "Region", ["United States", "California"], horizontal=True, key="diag_region"
    )
    cov_sub = cov[cov["series"] == region_diag].sort_values(["model", "horizon"])

    fig_cov = go.Figure()
    for model, color in [("arima", C["arima"]), ("prophet", C["prophet"])]:
        d = cov_sub[cov_sub["model"] == model]
        fig_cov.add_trace(go.Bar(
            name=LABEL.get(model, model),
            x=[f"h={h}" for h in d["horizon"]],
            y=(d["empirical_coverage"] * 100).round(1).tolist(),
            marker_color=color,
        ))
    fig_cov.add_hline(
        y=90, line_dash="dot", line_color="gray",
        annotation_text="nominal 90%", annotation_position="top right",
    )
    fig_cov.update_layout(
        barmode="group",
        yaxis=dict(title="Empirical coverage (%)", range=[50, 110], ticksuffix="%"),
        margin=dict(t=20, b=10, l=0, r=0),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_cov, use_container_width=True)

    # ── residual distributions ────────────────────────────────────────────────
    st.markdown("### Residual distributions")
    horizon_diag = st.select_slider(
        "Forecast horizon", options=[7, 14, 30, 90], key="diag_h"
    )

    col_ar, col_pr = st.columns(2)
    for col, res_df, model_name, color in [
        (col_ar, res_ar, "ARIMA+Fourier", C["arima"]),
        (col_pr, res_pr, "Prophet",       C["prophet"]),
    ]:
        with col:
            r = res_df[
                (res_df["series"] == region_diag) &
                (res_df["horizon"] == horizon_diag)
            ]["error"]
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=r, nbinsx=40,
                marker_color=color, opacity=0.8,
                hovertemplate="error=%{x:.2f}  count=%{y}<extra></extra>",
            ))
            fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_hist.update_layout(
                title_text=model_name,
                xaxis_title="Forecast error (pp)",
                yaxis_title="Count",
                margin=dict(t=40, b=10, l=0, r=0),
                height=260,
                showlegend=False,
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption(
                f"n={len(r):,}  mean={r.mean():.3f}  std={r.std():.3f}"
            )

    # ── Ljung-Box note ────────────────────────────────────────────────────────
    st.warning(
        "**Ljung-Box p ≈ 0** for both models at every horizon — residuals retain "
        "day-of-week autocorrelation (lag 7). This is an honest ceiling: "
        "the weekly pattern is not fully captured by either model."
    )

    with st.expander("Ljung-Box detail"):
        st.dataframe(
            lb[lb["series"] == region_diag][[
                "horizon", "model", "lb_pvalue_lag10", "mean_error", "std_error"
            ]].rename(columns={
                "horizon": "h", "model": "model",
                "lb_pvalue_lag10": "LB p-value (lag 10)",
                "mean_error": "mean error", "std_error": "std error",
            }).set_index("h"),
            use_container_width=True,
        )

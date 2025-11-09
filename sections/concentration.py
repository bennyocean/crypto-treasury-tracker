# sections/concentration.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from modules.filters import apply_filters
from modules.kpi_helpers import render_plotly
from modules.charts import lorenz_curve_chart
from modules.emojis import country_emoji_map
from modules.ui import render_ticker

# ---------- concentration metrics ----------
def _top_share(s: pd.Series, n: int = 10) -> float:
    s = s.astype(float)
    tot = s.sum()
    if tot <= 0:
        return 0.0
    return float(s.sort_values(ascending=False).head(n).sum() / tot)

def _hhi(s: pd.Series) -> float:
    """HHI in 0..1 space (multiply by 10,000 for 'index points')."""
    s = s.astype(float)
    tot = s.sum()
    if tot <= 0:
        return 0.0
    w = s / tot
    return float((w ** 2).sum())

def _gini(s: pd.Series) -> float:
    """Gini coefficient in [0,1]."""
    x = np.sort(s.astype(float).values)
    S = x.sum()
    n = x.size
    if n == 0 or S == 0:
        return 0.0
    i = np.arange(1, n + 1)
    # G = 2*sum(i*x)/(n*sum(x)) - (n+1)/n
    return float((2.0 * np.sum(i * x)) / (n * S) - (n + 1.0) / n)

def _lorenz_points(s: pd.Series):
    """Return (p, L(p)) arrays for Lorenz curve."""
    x = np.sort(s.astype(float).values)
    S = x.sum()
    if S == 0:
        return np.array([0,1]), np.array([0,1])
    cum = np.cumsum(x)
    p = np.arange(1, len(x) + 1) / len(x)
    L = cum / S
    # prepend origin
    p = np.insert(p, 0, 0.0)
    L = np.insert(L, 0, 0.0)
    return p, L


def render_concentration():
    #st.title("Crypto Treasury Concentration")
    render_ticker()
    
    base_df = st.session_state["data_df"]
    df_view = apply_filters(base_df)
    
    # Map country column to emoji
    flag_series = (
        df_view["Country"]
        .astype("string")
        .map(lambda c: country_emoji_map.get(c, "🏳️"))
    )

    # Prepend flag to Entity Name
    df_view["Country"] = flag_series.fillna("🏳️") + " " + df_view["Country"].astype("string")

    # guards
    if df_view.empty:
        st.info("No data for the current filters.")
        return

    with st.container(border=True):
        st.markdown("#### Gini Index & Lorenz Curve", help= "HHI and Gini coefficient tell if reserves are concentrated in a handful of players.")

        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

        group_by = c1.segmented_control(
            "Group by",
            options=["Holder", "Country", "Holder Type"],
            default="Holder",
            help="Choose the population over which concentration is measured."
        )

        # detect multi-asset selection from global filters
        assets_sel = st.session_state.get("flt_assets", [])
        multi_asset = len(assets_sel) != 1  # True if 0 or >1 assets selected

        measure_opts = ["USD"] if multi_asset else ["USD", "Units"]
        weight_mode = c2.segmented_control(
            "Measure",
            options=measure_opts,
            default="USD",
            help="USD works for single/multi-asset; Units only makes sense for a single asset."
        )

        top_n = c3.selectbox(
            "Top-N",
            options=[5, 10, 25, 50, 100],
            index=1,
            help="Share captured by the top N groups."
        )

        show_table = c4.checkbox("Show Top Table", value=True)

        # map group key
        if group_by == "Holder":
            key = "Entity Name"
        elif group_by == "Country":
            key = "Country"
        else:
            key = "Entity Type"

        # weights
        value_col = "USD Value" if weight_mode == "USD" else "Holdings (Unit)"
        if weight_mode == "Units" and multi_asset:
            st.info("Units across multiple assets aren’t comparable. Falling back to USD.")
            weight_mode = "USD"
            value_col = "USD Value"

        weights = df_view.groupby(key)[value_col].sum()
        # drop zeros/NA
        weights = weights[weights > 0].sort_values(ascending=False)

        if weights.empty or len(weights) < 2:
            st.info("Not enough data to compute concentration metrics for this selection.")
            return

        # metrics
        m_top = _top_share(weights, int(top_n))
        m_hhi = _hhi(weights)
        m_gini = _gini(weights)

        st.markdown("")

        k1, k2, k3 = st.columns(3)
        with k1:
            with st.container(border=True):
                st.metric(f"Top {top_n} Share", f"{m_top*100:,.1f}%", help="Sum of the top-N weights divided by total.")
        with k2:
            with st.container(border=True):
                st.metric("HHI (0–10,000)", f"{m_hhi*10000:,.0f}", help="Herfindahl–Hirschman Index in points. 0=perfectly dispersed; 10,000=monopoly.")
        with k3:
            with st.container(border=True):
                st.metric("Gini (0–1)", f"{m_gini:,.3f}", help="0=equal distribution; 1=single holder dominates.")

        assets_sel = st.session_state.get("flt_assets", [])
        asset_for_color = assets_sel[0] if len(assets_sel) == 1 else None

        st.markdown("")

        # Lorenz curve
        p, L = _lorenz_points(weights)
        fig = lorenz_curve_chart(p, L, asset=asset_for_color)
        #with st.container(border=True):
        render_plotly(fig, filename=f"lorenz_{group_by.lower()}_{weight_mode.lower()}")

    if show_table:
        top_tbl = (weights
                .head(int(top_n))
                .rename("Weight")
                .to_frame())
        total = weights.sum()
        top_tbl["Share"] = top_tbl["Weight"] / total  # 0..1

        # display copy (keep raw cols for math, add pretty/percent for UI)
        disp = top_tbl.copy()
        disp["Weight_fmt"] = disp["Weight"].map(lambda v: f"${v:,.0f}" if value_col == "USD Value" else f"{v:,.0f}")
        disp["SharePct"] = (disp["Share"] * 100).round(4)  # 0..100 for ProgressColumn


        st.dataframe(
            disp[["Weight_fmt", "SharePct"]],
            width="stretch",
            height=min(400, 38*(len(disp)+1)+6),
            column_config={
                "Entity Name": st.column_config.TextColumn("Holder"),
                "Weight_fmt": st.column_config.TextColumn("Weight (USD)"),
                "SharePct": st.column_config.ProgressColumn("Share", min_value=0, max_value=100, format="%.2f%%"),
            },
        )

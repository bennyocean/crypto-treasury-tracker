import streamlit as st
import pandas as pd

from modules.filters import apply_filters_historic
from modules.charts import historic_chart, cumulative_market_cap_chart, dominance_area_chart_usd
from modules.kpi_helpers import render_historic_kpis, render_flow_decomposition
from modules.ui import render_plotly


def render_historic_holdings():
    st.title("Crypto Treasury History & Trends")

    df = st.session_state["historic_df"]
    df_filtered = apply_filters_historic(df)

    if df_filtered.empty:
        st.info("No data for the current filters")
        return

    render_historic_kpis(df_filtered)

    st.markdown("")
    
    row1_col1, row1_col2 = st.columns([1, 1])

    with row1_col1:
        with st.container(border=False):
            st.markdown("#### Cumulative Market Cap of Crypto Treasuries", help="Total USD value of selected assets over time. If one asset is selected, shows units (area, left axis) + USD (line, right axis).")
            fig_cap = cumulative_market_cap_chart(df_filtered, current_df=st.session_state.get("data_df"))
            render_plotly(fig_cap, "cumulative_market_cap")

    with row1_col2:
        with st.container(border=False):
            st.markdown("#### Crypto Treasury Dominance (USD)", help="Stacked area of USD value by asset. Shows how each asset contributes to the total over time.")
            fig_dom = dominance_area_chart_usd(df_filtered, current_df=st.session_state.get("data_df"))
            render_plotly(fig_dom, "dominance_usd_area")

    st.divider()
    render_flow_decomposition(df_filtered)
    st.divider()

    with st.container(border=False):
        st.markdown("#### Historic Crypto Treasury Holdings Breakdown", help="Shows the historic development of aggregated and individual crypto asset holdings across all entities")
        st.markdown("")

        metric = st.segmented_control("Display mode", options = ["USD Value", "Unit Count"], default="USD Value", label_visibility="collapsed")
        by = "USD" if metric == "USD Value" else "Holdings (Unit)"
        st.markdown("")

        render_plotly(historic_chart(df_filtered, by=by), "historic_crypto_reserves")
        
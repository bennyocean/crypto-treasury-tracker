import streamlit as st
import pandas as pd

from modules.filters import apply_filters_historic
from modules.charts import historic_chart, cumulative_market_cap_chart, dominance_area_chart_usd, historic_changes_chart
from modules.kpi_helpers import render_historic_kpis, render_flow_decomposition
from modules.ui import render_plotly


def render_historic_holdings():
    st.title("Crypto Treasury History & Trends")

    df = st.session_state["historic_df"]
    df_filtered, display_start = apply_filters_historic(df)

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

    with st.container(border=False):
        st.markdown("#### Historic Crypto Treasury Holdings Breakdown", help="Shows the historic development of aggregated and individual crypto asset holdings across all entities")
        st.markdown("")

        metric = st.segmented_control(
            "Metric",
            options=["USD Value (Total)", "Unit Count (Total)", "Monthly Change (Units)"],
            default="USD Value (Total)",
            label_visibility="collapsed"
        )

        if metric == "Monthly Change (Units)":
            assets_in_scope = sorted(df_filtered["Crypto Asset"].dropna().unique())
            if not assets_in_scope:
                st.info("No assets available in current selection.")
            else:
                chosen_asset = st.pills(
                    "Asset",
                    assets_in_scope,
                    default=assets_in_scope[0],
                    key="historic_changes_asset_picker"
                )
                
                df_single = df_filtered[df_filtered["Crypto Asset"] == chosen_asset]
                ticker = df_single["Crypto Asset"].iloc[0]

                render_plotly(
                    historic_changes_chart(df_single, start=display_start, end=df_single["Date"].max()), f"historic_unit_changes_{ticker}")

        elif metric == "USD Value (Total)":
            render_plotly(historic_chart(df_filtered, by="USD"), "historic_usd_total")

        elif metric == "Unit Count (Total)":
            if df_filtered['Crypto Asset'].nunique() > 1:
                st.info("Select a single crypto asset to view unit totals.")
            else:
                ticker = df_filtered["Crypto Asset"].iloc[0]

                render_plotly(historic_chart(df_filtered, by="Unit"),f"historic_units_{ticker}")

    st.divider()
    render_flow_decomposition(df_filtered)

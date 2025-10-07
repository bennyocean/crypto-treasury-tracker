import streamlit as st
import pandas as pd

from modules.filters import apply_filters_historic
from modules.charts import historic_chart, cumulative_market_cap_chart, dominance_area_chart_usd, historic_changes_chart
from modules.kpi_helpers import render_historic_kpis, render_flow_decomposition
from modules.ui import render_plotly

def cumulative_viewer(df_display, current_df):
    st.markdown(
        "#### Cumulative Market Overview",
        help=(
            "Switch between Total Value view (USD and units if one asset), "
            "Market Share (USD dominance by asset), "
            "and Historic USD totals as stacked bars."
        )
    )

    view = st.segmented_control(
        "View",
        options=["Total Value", "Market Share", "Historic USD", "Historic Units"],
        default="Total Value",
        label_visibility="collapsed",
        key="cumulative_view_selector",
    )

    if view == "Total Value":
        fig = cumulative_market_cap_chart(df_display, current_df=current_df)
        render_plotly(fig, "cumulative_market_cap")

    elif view == "Market Share":
        fig = dominance_area_chart_usd(df_display, current_df=current_df)
        render_plotly(fig, "dominance_usd_area")

    elif view == "Historic USD":
        fig = historic_chart(df_display, by="USD")
        render_plotly(fig, "historic_usd_total")

    elif view == "Historic Units":
        if df_display['Crypto Asset'].nunique() > 1:
            st.info("Select a single crypto asset to view unit totals.")
        else:
            fig = historic_chart(df_display, by="Unit")
            ticker = df_display["Crypto Asset"].iloc[0]
            render_plotly(fig, f"historic_units_{ticker}")


def render_historic_holdings():
    st.title("Crypto Treasury History & Trends")

    df = st.session_state["historic_df"]
    df_display, df_changes, display_start = apply_filters_historic(df)

    if df_display.empty:
        st.info("No data for the current filters")
        return

    render_historic_kpis(df_display)
 
    row1_col1, row1_col2 = st.columns([1, 1])
    with row1_col1:
        with st.container(border=False):
            cumulative_viewer(df_display, current_df=st.session_state.get("data_df"))
    
    with row1_col2:
        with st.container(border=False):
            st.markdown(
                "#### Monthly Changes, Net Flows & Trends", 
                help="Month-over-month net change in holdings; Units are always per asset, USD can aggregate."
            )

            # --- toggle between Units and USD ---
            metric = st.segmented_control(
                "Metric",
                options=["USD", "Units"],
                default="USD",
                key="changes_metric_toggle",
                label_visibility="collapsed"
            )

            if metric == "Units":
                if df_display['Crypto Asset'].nunique() > 1:
                    st.info("Select a single crypto asset to view unit totals.")

                elif df_changes.empty:
                    st.info("No data for this asset.")
                else:
                    ticker = df_changes["Crypto Asset"].iloc[0]

                    render_plotly(
                        historic_changes_chart(
                            df_changes, by="Units",
                            start=display_start, end=df_changes["Date"].max()
                        ),
                        f"historic_changes_units_{ticker}"
                    )

            else:  # USD mode
                if df_changes.empty:
                    st.info("No data available for current selection.")
                else:
                    render_plotly(
                        historic_changes_chart(
                            df_changes, by="USD",
                            start=display_start, end=df_changes["Date"].max()
                        ),
                        "historic_changes_usd_all"
                    )

    st.divider()
    render_flow_decomposition(df_display)

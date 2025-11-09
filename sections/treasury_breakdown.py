import streamlit as st
from modules.filters import apply_filters
from modules import charts
from modules.ui import render_plotly, render_ticker

def render_treasury_breakdown():
    #st.markdown("#### Treasury Breakdown & Distribution")
    #st.title("Treasury Breakdown & Distribution")
    
    render_ticker()

    df = st.session_state["data_df"]
    df_filtered = apply_filters(df)
    print("number of countries:", df_filtered["Country"].nunique())

    if df_filtered.empty:
        st.info("No data for the current filters")
        return
    # Summary KPIs
    with st.container(border=False):
        col1, col2, col3 = st.columns(3)
        total_value = df_filtered['USD Value'].sum()
        entity_count = df_filtered['Entity Name'].nunique()
        avg_value = df_filtered.groupby('Entity Name')['USD Value'].sum().mean()

        with col1:
            with st.container(border=True):
                st.metric(
                    "Total USD Value",
                    f"${total_value:,.0f}",
                    help="Sum of all reported crypto holdings for the selected filters (in USD)."
                )
        with col2:
            with st.container(border=True):
                st.metric(
                    "Number of Holders",
                    f"{entity_count}",
                    help="Count of unique entities reporting crypto holdings under the selected filters."
                )
        with col3:
            with st.container(border=True):
                st.metric(
                    "Avg. USD Value per Holder",
                    f"${avg_value:,.0f}",
                    help="Average USD value of treasury or reserve amount held per reporting entity under the selected filters."
                )
    
    st.markdown("")

    # Charts row
    row1_col1, row1_col2= st.columns(2)

    with row1_col1:
        with st.container(border=True):
            st.markdown("#### Treasury Type Ranking", help="Ranking of treasury holder types by aggregated USD value.")

            render_plotly(charts.holdings_by_entity_type_bar(df_filtered), "holdings_by_entity_type")

    with row1_col2:
        with st.container(border=True):
            st.markdown("#### Holder Distribution",help="Share of treasury holders by aggregated USD value or total count per entity type. Other includes DeFi protocols, L1/L2 networks, AI agents, and community-led projects")

            mode_lbl = st.segmented_control(
                "Metric",
                options=["USD Value", "Holder Count"],
                default="USD Value",
                key="entity_pie_mode",
                label_visibility="collapsed",
            )

            mode_arg = "count" if mode_lbl == "Holder Count" else "usd"

            render_plotly(charts.entity_type_distribution_pie(df_filtered, mode=mode_arg), "entity_type_distribution")

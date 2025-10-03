import streamlit as st
from modules.filters import apply_filters
from modules import charts
from modules.ui import render_plotly

def render_treasury_breakdown():
    #st.markdown("#### Treasury Breakdown & Distribution")
    st.title("Treasury Breakdown & Distribution")
    
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
    row1_col1, row1_col2, row1_col3= st.columns([1, 1, 1])

    with row1_col1:
        with st.container(border=False):
            st.markdown("#### Holdings by Holder Type", help="USD value of selected crypto holdings by entity category.")

            render_plotly(charts.holdings_by_entity_type_bar(df_filtered), "holdings_by_entity_type")

    with row1_col2:
        with st.container(border=False):
            st.markdown("#### Crypto Treasury Holder Share",help="Share of entities by aggregated USD value or total count per entity type. Other includes protocols, L1/L2 networks, AI agents, and community-led projects")

            mode_lbl = st.segmented_control(
                "Metric",
                options=["USD Value", "Holder Count"],
                default="USD Value",
                key="entity_pie_mode",
                label_visibility="collapsed",
            )

            mode_arg = "count" if mode_lbl == "Holder Count" else "usd"

            render_plotly(charts.entity_type_distribution_pie(df_filtered, mode=mode_arg),
                        "entity_type_distribution")

    with row1_col3:
        with st.container(border=False):
            st.markdown("#### Top 10 Countries",help="Countries with the largest reported crypto treasury holdings, ranked by the selected metric. Note: 'Decentralized' refers to globally running networks and protocols without a headquarter and/or legal registration.")

            display_mode = st.segmented_control("Display mode", options=["USD Value", "Holder Count"], default="Holder Count", label_visibility="collapsed")

            if display_mode == "Holder Count":
                fig_country = charts.top_countries_by_entity_count(df_filtered)
            else:
                fig_country = charts.top_countries_by_usd_value(df_filtered)

            render_plotly(fig_country, "top_5_countries")

    st.divider()

    with st.container(border=False):
        st.markdown(
            "#### Crypto Treasury Treemap",
            help=" Shows area size based on USD value. Switch between entity- and regional-level views."
        )
        layout_choice = st.segmented_control(
            "Treemap layout",
            options=["Holder Distribution", "Geographic Distribution"],
            default="Holder Distribution", label_visibility="collapsed"
        )
        mode = "country_type" if "Geographic Distribution" in layout_choice else "type_entity"
        render_plotly(charts.treemap_composition(df_filtered, mode=mode), "treemap_composition")

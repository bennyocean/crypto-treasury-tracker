import streamlit as st
from modules.filters import apply_filters
from modules import charts
from modules.ui import render_plotly

def render_treasury_breakdown():

    df = st.session_state["data_df"]
    df_filtered = apply_filters(df)
    print("number of countries:", df_filtered["Country"].nunique())

    if df_filtered.empty:
        st.info("No data for the current filters")
        return

    # Summary KPIs
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        total_value = df_filtered['USD Value'].sum()
        entity_count = df_filtered['Entity Name'].nunique()
        avg_value = df_filtered.groupby('Entity Name')['USD Value'].sum().mean()

        with col1:
            with st.container(border=True):
                st.metric(
                    "Total USD Value",
                    f"${total_value:,.0f}",
                    help="Sum of all reported crypto reserves for the selected filters, converted to USD using the latest available prices."
                )
        with col2:
            with st.container(border=True):
                st.metric(
                    "Number of Entities",
                    f"{entity_count}",
                    help="Count of unique entities reporting crypto reserves under the selected filters."
                )
        with col3:
            with st.container(border=True):
                st.metric(
                    "Avg. USD Value per Entity",
                    f"${avg_value:,.0f}",
                    help="Average USD value of reserves held per reporting entity under the selected filters."
                )

    # Charts row
    row1_col1, row1_col2, row1_col3= st.columns([1, 1, 1])

    with row1_col1:
        with st.container(border=True):
            st.markdown("#### Holdings by Entity Type", help="USD value of selected crypto holdings by entity category.")

            render_plotly(charts.holdings_by_entity_type_bar(df_filtered), "holdings_by_entity_type")

    with row1_col2:
        with st.container(border=True):
            st.markdown("#### Entity Type Distribution",help="Share of entities by aggregated USD value or total count per entity type. Other includes protocols, L1/L2 networks, AI agents, and community-led projects")

            mode_lbl = st.segmented_control(
                "Metric",
                options=["USD Value", "Entity Count"],
                default="USD Value",
                key="entity_pie_mode",
                label_visibility="collapsed",
            )

            mode_arg = "count" if mode_lbl == "Entity Count" else "usd"

            render_plotly(charts.entity_type_distribution_pie(df_filtered, mode=mode_arg),
                        "entity_type_distribution")

    with row1_col3:
        with st.container(border=True):
            st.markdown("#### Top 5 Countries",help="Countries or regions with the largest reported crypto reserves, ranked by the selected metric. Note: 'Decentralized' refers to entities (e.g., DAOs) without a specific country affiliation.")

            display_mode = st.segmented_control("Display mode", options=["USD Value", "Entity Count"], default="USD Value", label_visibility="collapsed")

            if display_mode == "Entity Count":
                fig_country = charts.top_countries_by_entity_count(df_filtered)
            else:
                fig_country = charts.top_countries_by_usd_value(df_filtered)

            render_plotly(fig_country, "top_5_countries")
        
    with st.container(border=True):
        st.markdown(
            "#### Current Crypto Treasury Distribution",
            help=" Shows area size based on USD value. Switch between entity- and regional-level views."
        )
        layout_choice = st.segmented_control(
            "Treemap layout",
            options=["Entity Distribution", "Geographic Distribution"],
            default="Entity Distribution", label_visibility="collapsed"
        )
        mode = "country_type" if "Geographic Distribution" in layout_choice else "type_entity"
        render_plotly(charts.treemap_composition(df_filtered, mode=mode), "treemap_composition")

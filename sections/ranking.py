import streamlit as st
from modules.filters import apply_filters
from modules import charts
from modules.ui import render_plotly


def render_entity_ranking():
    #st.title("Crypto Treasury Ranking")

    df = st.session_state["data_df"]
    df_filtered = apply_filters(df)
    if df_filtered.empty:
        st.info("No data for the current filters")
        return
    
    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        with st.container(border=True):
            st.markdown("#### Crypto Treasury Ranking", help="Rank leading entities by total crypto holdings (USD value) or number of units held.")

            col_toggle, col_n, = st.columns(2)

            metric = col_toggle.segmented_control(label = "Select View", options=["USD Value", "Unit Count"], default="USD Value")
            top_n = col_n.number_input("Expand Ranking", min_value=1, max_value=100, value=5, step=1)

            by = "USD" if metric == "USD Value" else "units"

            render_plotly(charts.entity_ranking(df_filtered, by=by, top_n=top_n), "entity_ranking")
    with row1_c2:

        with st.container(border=True):
            st.markdown("#### Share of Circulating Supply", help="Top entities by share of circulating supply per asset.")
            top_n_share = st.number_input("Expand Ranking", min_value=1, max_value=100, value=5, step=1, key="share_topn")
            fig_share = charts.entity_supply_share_ranking(df_filtered, top_n=int(top_n_share))
            render_plotly(fig_share, "entity_supply_share_ranking")

    col_a, col_b = st.columns(2)

    with col_a:
        with st.container(border=True):
            st.markdown("#### Top assets by treasury USD")
            topn_assets_usd = st.number_input("Expand Ranking", min_value=1, max_value=30, value=5, step=1, key="asset_usd_topn")
            fig_assets_usd = charts.asset_totals_usd_bar(df_filtered, top_n=int(topn_assets_usd))
            render_plotly(fig_assets_usd, "asset_totals_usd_bar")

    with col_b:
        with st.container(border=True):
            st.markdown("#### Top assets by share of supply")
            topn_assets_share = st.number_input("Expand Ranking", min_value=1, max_value=30, value=5, step=1, key="asset_share_topn")
            fig_assets_share = charts.asset_totals_supply_share_bar(df_filtered, top_n=int(topn_assets_share))

            render_plotly(fig_assets_share, "asset_totals_supply_share_bar")

    with st.container(border=True):
        st.markdown(
            "#### Treasury Holder Distribution",
            help=" Shows area size based on USD value."
        )
        render_plotly(charts.treemap_composition(df_filtered, mode="type_entity"), "treemap_composition")
"""

    col_btc, col_eth, col_sol = st.columns(3)

    with col_btc:
        top_5_holders(df, asset="BTC", key_prefix="btc")

    with col_eth:
        top_5_holders(df, asset="ETH", key_prefix="eth")

    with col_sol:
        top_5_holders(df, asset="SOL", key_prefix="sol")
"""
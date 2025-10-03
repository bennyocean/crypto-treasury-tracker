import streamlit as st
from modules.filters import apply_filters
from modules.charts import entity_ranking
from modules.ui import render_plotly
from modules.kpi_helpers import top_5_holders


def render_entity_ranking():
    st.title("Crypto Treasury Ranking")

    df = st.session_state["data_df"]
    df_filtered = apply_filters(df)
    if df_filtered.empty:
        st.info("No data for the current filters")
        return

    st.markdown("#### Top 5 Crypto Treasury Holders for BTC, ETH & SOL")
    st.markdown("")

    # Top 5 Crypto Asset Charts
    col_btc, col_eth, col_sol = st.columns(3)

    with col_btc:
        top_5_holders(df, asset="BTC", key_prefix="btc")

    with col_eth:
        top_5_holders(df, asset="ETH", key_prefix="eth")

    with col_sol:
        top_5_holders(df, asset="SOL", key_prefix="sol")

    st.divider()
    
    with st.container(border=False):
        st.markdown("#### Global Crypto Treasury Ranking", help="Rank leading entities by total crypto holdings (USD value) or number of units held.")
        st.markdown("")

        col_toggle, col_n, _ = st.columns([1, 1, 1])

        metric = col_toggle.segmented_control(label = "", options=["USD Value", "Unit Count"], default="USD Value", label_visibility="collapsed")
        top_n = col_n.number_input("Max. Holders Displayed", min_value=1, max_value=100, value=10, step=1)

        by = "USD" if metric == "USD Value" else "units"

        render_plotly(entity_ranking(df_filtered, by=by, top_n=top_n), "entity_ranking")

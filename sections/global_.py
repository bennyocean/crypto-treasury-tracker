import streamlit as st
from modules.ui import render_plotly
from analytics import log_filter_if_changed
from modules.filters import apply_filters
from modules import charts


def render_global():
    #st.title("Global Crypto Treasury Map")

    df = st.session_state["data_df"]

    filtered = apply_filters(df)

    value_opts = ["All", "0–100M", "100M–1B", ">1B"]
    sel_v = st.selectbox(
        "Value Range (USD)",
        options=value_opts,
        key="flt_value_range_global"
    )
    st.session_state["flt_value_range"] = sel_v

    if sel_v != "All" and "USD Value" in filtered.columns:
        vals = filtered["USD Value"].astype(float)
        if sel_v == "0–100M":
            filtered = filtered[vals.between(0, 100_000_000)]
        elif sel_v == "100M–1B":
            filtered = filtered[vals.between(100_000_000, 1_000_000_000)]
        elif sel_v == ">1B":
            filtered = filtered[vals > 1_000_000_000]

    log_filter_if_changed("global_summary", {
        "asset": st.session_state.get("flt_asset_choice", "All Assets"),
        "entity_type": st.session_state.get("flt_entity_type", "All"),
        "value_range": sel_v or "All",
    })

    # --- render map ---
    if filtered.empty:
        st.info("No entities found for the selected filters.")
    else:
        assets = filtered["Crypto Asset"].dropna().unique().tolist()
        fig = charts.render_world_map(filtered, assets, st.session_state.get("flt_entity_type"), sel_v)
        if fig is not None:

            #row1_col1, row1_col2= st.columns(2)

            with st.container(border=True):
                st.markdown(
                    "#### World Map",
                    help=" Shows global distribution based on USD value."
                )
                render_plotly(fig, "crypto_reserve_world_map", extra_config={"scrollZoom": False})

        #with row1_col2:
            with st.container(border=True):
                st.markdown("#### Top 10 Countries",help="Countries with the largest reported crypto treasury holdings, ranked by the selected metric. Note: 'Decentralized' refers to globally running networks and protocols without a headquarter and/or legal registration.")

                display_mode = st.segmented_control("Display mode", options=["USD Value", "Holder Count"], default="Holder Count", label_visibility="collapsed")

                if display_mode == "Holder Count":
                    fig_country = charts.top_countries_by_entity_count(filtered)
                else:
                    fig_country = charts.top_countries_by_usd_value(filtered)

                render_plotly(fig_country, "top_5_countries")

            with st.container(border=True):

                st.markdown(
                    "#### Regional Distribution",
                    help=" Shows area size based on USD value. Switch between entity- and regional-level views."
                )
                
                render_plotly(charts.treemap_composition(filtered, mode="country_type"), "treemap_composition")

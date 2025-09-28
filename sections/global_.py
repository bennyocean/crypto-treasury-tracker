import streamlit as st
from modules.charts import render_world_map
from modules.ui import render_plotly
from analytics import log_filter_if_changed


def render_global():
    df = st.session_state["data_df"]

    with st.container(border=True):
        #c1, c2, c3 = st.columns(3)
        c1, c2, c3 = st.columns([2, 1, 1])

        asset_opts   = st.session_state["opt_assets"]
        type_opts    = st.session_state["opt_entity_types"]
        value_opts   = ["All", "0–100M", "100M–1B", ">1B"]

        # assets
        if "ui_assets_map" not in st.session_state:
            st.session_state["ui_assets_map"] = st.session_state.get("flt_assets", asset_opts)
        sel_assets = c1.pills(
            "Select Crypto Asset(s)",
            options=asset_opts,
            selection_mode="multi",
            key="ui_assets_map",
            label_visibility="visible",
            width="stretch",
        )
        st.session_state["flt_assets"] = sel_assets

        # entity type
        if "ui_entity_type_map" not in st.session_state:
            st.session_state["ui_entity_type_map"] = st.session_state.get("flt_entity_type", "All")
        sel_et = c2.selectbox(
            "Entity Type",
            options=type_opts,
            key="ui_entity_type_map"   # no index on reruns
        )
        st.session_state["flt_entity_type"] = sel_et

        # value bucket
        if "ui_value_range_map" not in st.session_state:
            st.session_state["ui_value_range_map"] = st.session_state.get("flt_value_range", "All")
        sel_v = c3.selectbox(
            "Value Range (USD)",
            options=value_opts,
            key="ui_value_range_map"   # no index on reruns
        )
        st.session_state["flt_value_range"] = sel_v
    
        # Log filters if changed
        log_filter_if_changed("global_summary", {
            "asset": sel_assets or ["All"],
            "entity_type": sel_et or "All",
            "value_range": sel_v or "All",
        })

    
    # Global Map
    with st.container(border=True):
        st.markdown("#### Global Crypto Treasury Map", help="Geographic distribution of crypto treasuries, filtered by crypto asset, entity type, and value range.")

        if not sel_assets:
            st.info("Select at least one Crypto Asset to display the map")
        else:
            fig = render_world_map(df, sel_assets, sel_et, sel_v)
            if fig is not None:
                render_plotly(fig, "crypto_reserve_world_map", extra_config={"scrollZoom": False})

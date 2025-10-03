import warnings, logging
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("py.warnings").setLevel(logging.ERROR)

import streamlit as st

from modules import ui
from modules.data_loader import get_prices, load_units, attach_usd_values, load_historic_data, load_planned_data, load_kpi_snapshots, load_entities_reference_ctt, load_assets_reference_ctt, load_sources_reference_ctt, load_holdings_events_ctt
from modules.filters import _init_global_filters, _opts
from modules.sidebar_info import render_sidebar
from analytics import init_analytics


st.set_page_config(page_title="Crypto Treasury Tracker", layout="wide")

# init analytics
init_analytics()

# one-time init
if "initialized" not in st.session_state:
    loader = ui.show_global_loader("Initializing Crypto Treasury Tracker")

    # fetch prices once with cache ttl
    st.session_state["prices"] = get_prices()  # (BTC, ETH, XRP, BNB, SOL)

    # load units once with cache ttl
    units_df = load_units()
    st.session_state["units_df"] = units_df

    # compute USD values once per price snapshot
    st.session_state["data_df"] = attach_usd_values(units_df, st.session_state["prices"])
    _init_global_filters(st.session_state["data_df"])
    print(st.session_state["data_df"].head(10))

    # load kpi snapshots
    st.session_state["kpi_snapshots"] = load_kpi_snapshots()
    # CTT relational sources
    st.session_state["ctt_entities"] = load_entities_reference_ctt()
    st.session_state["ctt_assets"]   = load_assets_reference_ctt()
    st.session_state["ctt_sources"]  = load_sources_reference_ctt()
    st.session_state["ctt_events"]   = load_holdings_events_ctt()

    # canonical option lists used by ALL pages
    st.session_state["opt_assets"] = _opts(st.session_state["data_df"]["Crypto Asset"])
    st.session_state["opt_entity_types"] = ["All"] + _opts(st.session_state["data_df"]["Entity Type"])
    st.session_state["opt_countries"] = ["All"] + _opts(st.session_state["data_df"]["Country"])

    # make sure current selections are valid against canonical lists
    if st.session_state.get("flt_entity_type") not in st.session_state["opt_entity_types"]:
        st.session_state["flt_entity_type"] = "All"
    if st.session_state.get("flt_country") not in st.session_state["opt_countries"]:
        st.session_state["flt_country"] = "All"
    if not st.session_state.get("flt_assets"):
        st.session_state["flt_assets"] = st.session_state["opt_assets"]
    else:
        st.session_state["flt_assets"] = [
            a for a in st.session_state["flt_assets"] if a in st.session_state["opt_assets"]
        ] or st.session_state["opt_assets"]

    # load historic data
    st.session_state["historic_df"] = load_historic_data()

    # load planned / announced data
    st.session_state["planned_df"] = load_planned_data()

    st.session_state["initialized"] = True
    loader.empty()

render_sidebar()
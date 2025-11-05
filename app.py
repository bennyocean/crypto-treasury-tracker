import warnings, logging
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("py.warnings").setLevel(logging.ERROR)

import streamlit as st
import pandas as pd

from modules import ui
from modules.data_loader import (
    get_prices, load_units, attach_usd_values, load_historic_data, #load_planned_data,
    load_kpi_snapshots, load_entities_reference_ctt, load_assets_reference_ctt,
    load_sources_reference_ctt, load_holdings_events_ctt
)
from modules.filters import _init_global_filters, _opts
from modules.sidebar_info import render_sidebar
from analytics import init_analytics

st.set_page_config(page_title="Crypto Treasury Tracker", layout="wide")

init_analytics()

if "initialized" not in st.session_state:
    loader = ui.show_global_loader("Initializing Crypto Treasury Tracker")

    # prices
    st.session_state["prices"] = get_prices()

    # units
    units_df = load_units()
    st.session_state["units_df"] = units_df

    # compute USD values
    # st.session_state["data_df"] = attach_usd_values(units_df, st.session_state["prices"])
    st.session_state["data_df"] = attach_usd_values(
        units_df,
        prices_input=st.session_state["prices"],
        use_default_prices=False
    )
    # normalize DAT to int 0/1 BEFORE filters
    df0 = st.session_state["data_df"].copy()
    df0["DAT"] = pd.to_numeric(df0.get("DAT", 0), errors="coerce").fillna(0).astype(int)
    st.session_state["data_df"] = df0
    
    # initialize global filter option lists
    _init_global_filters(st.session_state["data_df"])
    print(st.session_state["data_df"].head(10))

    # snapshots and references
    st.session_state["kpi_snapshots"] = load_kpi_snapshots()
    st.session_state["ctt_entities"]  = load_entities_reference_ctt()
    st.session_state["ctt_assets"]    = load_assets_reference_ctt()
    st.session_state["ctt_sources"]   = load_sources_reference_ctt()
    st.session_state["ctt_events"]    = load_holdings_events_ctt()

    # keep your canonical lists block if you want  it matches _init_global_filters
    st.session_state["opt_assets"] = _opts(st.session_state["data_df"]["Crypto Asset"])
    base_types = _opts(st.session_state["data_df"]["Entity Type"])
    st.session_state["opt_entity_types"] = ["All", "DAT Wrappers"] + base_types
    st.session_state["opt_countries"] = ["All"] + _opts(st.session_state["data_df"]["Country"])

    # validate persisted selections
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

    # historic and planned
    st.session_state["historic_df"] = load_historic_data()
    #st.session_state["planned_df"]  = load_planned_data()

    st.session_state["initialized"] = True
    loader.empty()

render_sidebar()

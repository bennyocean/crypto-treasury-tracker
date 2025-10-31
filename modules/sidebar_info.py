import streamlit as st
import base64, mimetypes
import time

from sections import overview, global_, historic, ranking, treasury_breakdown, about, concentration, valuation, planned
from modules.ui import render_header, render_subscribe_cta, render_support, show_global_loader
from analytics import log_page_once


def render_sidebar():

    st.sidebar.image("assets/ctt-logo.svg", width=250)
    st.sidebar.markdown("_The most comprehensive Digital Asset Treasury Terminal!_")

    # global top header on every page
    #render_header()

    # Crypto Reserve Report Link
    render_subscribe_cta()

    # section switcher
    section = st.sidebar.radio("Explore The Tracker", 
                               [
                                    "Dashboard",
                                    #"Treasury Pipeline",
                                    "Market Cap & Flows",
                                    "Treasury Ranking",
                                    "Regional Analysis",
                                    "Holder Analysis",
                                    "DAT Analysis",
                                    "Concentration Analysis",
                                    "About"
                                ],
                                #default = "Dashboard",
                                width = "stretch",
                                label_visibility = "visible")
    
    st.sidebar.write(" ")

    # --- Reset filters ---
    if st.sidebar.button("Reset Filters", type="primary", width="stretch"):
        st.session_state["flt_assets"]       = st.session_state["opt_assets"]
        st.session_state["flt_entity_type"]  = "All"
        st.session_state["flt_country"]      = "All"
        st.session_state["flt_value_range"]  = "All"
        st.session_state["flt_time_range"]   = "All"
        st.session_state["flt_asset_choice"] = "All assets combined"

        # bump UI revision so widget keys change on next render
        st.session_state["ui_rev"] = st.session_state.get("ui_rev", 0) + 1

        # clear old UI keys so they visually snap back
        for k in [
            "ui_asset_choice_0","ui_asset_choice_1","ui_asset_choice_2",  # harmless if absent
            "ui_assets","ui_entity_type","ui_country",
            "ui_assets_map","ui_entity_type_map","ui_value_range_map",
            "ui_assets_hist","ui_time_range_hist",
        ]:
            st.session_state.pop(k, None)

        st.rerun()


    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2.8rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


    # Support
    render_support()


    # External Links / Contact
    def data_uri(path):
        mime, _ = mimetypes.guess_type(path)
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        return f"data:{mime};base64,{b64}"

    linkedin = data_uri("assets/linkedin-logo.png")
    xicon = data_uri("assets/x-logo.svg")
    linktree_icon = data_uri("assets/linktree-logo.svg")
    substack_icon = data_uri("assets/substack-logo.png")

    st.sidebar.markdown(
        f"""
        <div style="display:flex; gap:20px; align-items:center;">
        <a href="https://www.linkedin.com/in/benjaminschellinger/" target="_blank" rel="LinkedIn">
            <img src="{linkedin}" alt="LinkedIn" style="width:20px;height:20px;vertical-align:middle;">
        </a>
        <a href="https://x.com/CTTbyBen" target="_blank" rel="X">
            <img src="{xicon}" alt="X" style="width:20px;height:20px;vertical-align:middle;">
        </a>
        <a href="https://linktr.ee/benjaminschellinger" target="_blank" rel="noopener" title="Linktree">
            <img src="{linktree_icon}" alt="Linktree" style="width:20px;height:20px;vertical-align:middle;">
        </a>
        <a href="https://digitalfinancebriefing.substack.com/" target="_blank" rel="noopener" title="Substack">
            <img src="{substack_icon}" alt="Linktree" style="width:20px;height:20px;vertical-align:middle;">
        </a>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Version and brand footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<p style='font-size: 0.75rem; color: gray;'>"
        "v2.0 • © 2025 Crypto Treasury Tracker"
        "</p>", unsafe_allow_html=True
    )

    # Track section changes
    st.session_state.setdefault("current_section", None)
    st.session_state.setdefault("active_dialog", None)            # dialog queue used by the Dashboard
    st.session_state.setdefault("overview_editor_rev", 0)         # editor key rotator used by the Dashboard

    prev_section = st.session_state["current_section"]
    loader = None

    if prev_section != section:
        # We navigated to a new section
        st.session_state["current_section"] = section

        loader = show_global_loader(f"Loading {section}")

        # If we just arrived on the Dashboard, clear any stale dialog queue
        if section == "Dashboard":
            st.session_state["active_dialog"] = None
            # rotate the editor key so any lingering checkboxes are reset
            st.session_state["overview_editor_rev"] += 1

    # render selected page & log info

    if section == "Dashboard":
        log_page_once("overview")
        overview.render_overview()

    """    if section == "Treasury Pipeline":
            log_page_once("planned")
            planned.render_planned()"""

    if section == "Regional Analysis":
        log_page_once("world_map")
        global_.render_global()

    if section == "Market Cap & Flows":
        log_page_once("history")
        historic.render_historic_holdings()

    if section == "Treasury Ranking":
        log_page_once("leaderboard")
        ranking.render_entity_ranking()

    if section == "Holder Analysis":
        log_page_once("treasury_breakdown")
        treasury_breakdown.render_treasury_breakdown()

    if section == "Concentration Analysis":
        log_page_once("concentration")
        concentration.render_concentration()

    if section == "DAT Analysis":
        log_page_once("valuation")
        valuation.render_valuation_insights()
        
    if section == "About":
        log_page_once("about")
        about.render_about()

    if loader is not None:
        loader.empty()
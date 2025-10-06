import streamlit as st
import pandas as pd


def _opts(series):
    return sorted({str(x).strip() for x in series.dropna().tolist()})


def _init_global_filters(df):
    asset_opts = _opts(df["Crypto Asset"])
    type_opts = ["All"] + _opts(df["Entity Type"])
    country_opts = ["All"] + _opts(df["Country"])

    if "flt_assets" not in st.session_state:
        st.session_state["flt_assets"] = asset_opts
    else:
        # coerce to valid values in case schema changed
        st.session_state["flt_assets"] = [a for a in st.session_state["flt_assets"] if a in asset_opts] or asset_opts

    if "flt_entity_type" not in st.session_state or st.session_state["flt_entity_type"] not in type_opts:
        st.session_state["flt_entity_type"] = "All"

    if "flt_country" not in st.session_state or st.session_state["flt_country"] not in country_opts:
        st.session_state["flt_country"] = "All"

    if "flt_value_range" not in st.session_state:
        st.session_state["flt_value_range"] = "All"

    if "flt_time_range" not in st.session_state:
        st.session_state["flt_time_range"] = "All"


def apply_filters(df):
    with st.container(border=False):

        st.markdown("")

        col1, col2, col3 = st.columns([2, 1, 1])

        asset_opts   = st.session_state["opt_assets"]
        type_opts    = st.session_state["opt_entity_types"]
        country_opts = st.session_state["opt_countries"]

        # --- Assets pills ---
        if "ui_assets" not in st.session_state:
            st.session_state["ui_assets"] = st.session_state.get("flt_assets", asset_opts)
        sel_assets = col1.pills(
            "Select Crypto Asset(s)",
            options=asset_opts,
            selection_mode="multi",
            key="ui_assets",
            label_visibility="visible",
            width="stretch",
        )
        st.session_state["flt_assets"] = sel_assets

        # entity type
        if "ui_entity_type" not in st.session_state:
            st.session_state["ui_entity_type"] = st.session_state.get("flt_entity_type", "All")
        sel_et = col2.selectbox(
            "Select Holder Type",
            options=type_opts,
            key="ui_entity_type"
        )
        st.session_state["flt_entity_type"] = sel_et

        
        # country
        if "ui_country" not in st.session_state:
            st.session_state["ui_country"] = st.session_state.get("flt_country", "All")
        sel_co = col3.selectbox(
            "Select Country/Region",
            options=country_opts,
            key="ui_country"
        )
        st.session_state["flt_country"] = sel_co


        # Guards and filtering
        if not sel_assets:
            st.info("Select at least one Crypto Asset to display data")
            return df.iloc[0:0]

        out = df[df["Crypto Asset"].isin(sel_assets)]
        if sel_et != "All":
            out = out[out["Entity Type"] == sel_et]
        if sel_co != "All":
            out = out[out["Country"] == sel_co]
        out = out[out["USD Value"] > 0]
    
        st.markdown("")

        return out



def apply_filters_historic(df: pd.DataFrame):
    with st.container(border=False):

        st.markdown("")

        col1, col2 = st.columns(2)

        # ensure a valid default exists in state
        if st.session_state.get("flt_time_range") not in {"All","3 Months", "6 Months","12 Months", "YTD"}:
            st.session_state["flt_time_range"] = "All"

        # --- Asset pills ---
        asset_opts = _opts(df["Crypto Asset"])
        if "ui_assets_hist" not in st.session_state:
            st.session_state["ui_assets_hist"] = st.session_state.get("flt_assets", asset_opts)

        sel_assets = col1.pills(
            "Select Crypto Asset(s)",
            options=asset_opts,
            selection_mode="multi",
            key="ui_assets_hist",
            label_visibility="visible",
            width="stretch",
        )
        st.session_state["flt_assets"] = sel_assets

        # --- Time range widget: initialize UI key once, then no index on reruns
        time_opts = ["All","3 Months", "6 Months","12 Months", "YTD"]
        if "ui_time_range_hist" not in st.session_state:
            st.session_state["ui_time_range_hist"] = st.session_state.get("flt_time_range", "All")
        sel_tr = col2.selectbox(
            "Select Time Range",
            options=time_opts,
            key="ui_time_range_hist"          # no index on reruns
        )
        st.session_state["flt_time_range"] = sel_tr

        # --- Filtering logic (unchanged)
        df_sel = df[df["Crypto Asset"].isin(sel_assets)] if sel_assets else df.iloc[0:0]

        display_start = None
        df_for_changes = df_sel

        if not df_sel.empty and sel_tr != "All":
            latest_date = df_sel["Date"].max()
            if sel_tr == "3 Months":
                cutoff_date = latest_date - pd.DateOffset(months=2)   # show 3 incl latest
            elif sel_tr == "6 Months":
                cutoff_date = latest_date - pd.DateOffset(months=5)   # show 6 incl latest
            elif sel_tr == "12 Months":
                cutoff_date = latest_date - pd.DateOffset(months=11)  # show 12 incl latest
            elif sel_tr == "YTD":
                cutoff_date = pd.Timestamp(year=latest_date.year - 1, month=12, day=31)

            display_start = cutoff_date

            # normal charts -> EXACT window
            df_sel = df_sel[df_sel["Date"] >= cutoff_date]

            # changes chart -> one baseline month before
            df_for_changes = df_for_changes[df_for_changes["Date"] >= (cutoff_date - pd.DateOffset(months=1))]

        return df_sel, df_for_changes, display_start
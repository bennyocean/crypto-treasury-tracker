import streamlit as st
import pandas as pd


def _opts(series: pd.Series) -> list[str]:
    return sorted({str(x).strip() for x in series.dropna().tolist()})

def _init_global_filters(df: pd.DataFrame) -> None:
    asset_opts   = _opts(df["Crypto Asset"])
    base_types   = _opts(df["Entity Type"])
    type_opts    = ["All", "DAT Wrappers"] + [t for t in base_types if t != "DAT Wrappers"]
    country_opts = ["All"] + _opts(df["Country"])

    st.session_state["opt_assets"]       = asset_opts
    st.session_state["opt_entity_types"] = type_opts
    st.session_state["opt_countries"]    = country_opts

    if "flt_asset_choice" not in st.session_state:
        st.session_state["flt_asset_choice"] = "All Assets"
    if st.session_state.get("flt_entity_type") not in type_opts:
        st.session_state["flt_entity_type"] = "All"
    if st.session_state.get("flt_country") not in country_opts:
        st.session_state["flt_country"] = "All"
    if "flt_time_range" not in st.session_state:
        st.session_state["flt_time_range"] = "All"

# ---------- main filter (entity ranking) ----------
def apply_filters(df: pd.DataFrame):
    with st.container(border=False):
        st.markdown("")
        col1, col2, col3 = st.columns([2, 0.5, 0.5])

        type_opts    = st.session_state["opt_entity_types"]
        country_opts = st.session_state["opt_countries"]

        # assets with any live values
        _live_mask = (
            pd.to_numeric(df["USD Value"], errors="coerce").fillna(0.0) > 0
        ) | (
            pd.to_numeric(df["Holdings (Unit)"], errors="coerce").fillna(0.0) > 0
        )
        base_assets = _opts(df.loc[_live_mask, "Crypto Asset"])

        asset_select_opts = ["All Assets", "Pending"] + base_assets

        # single-select pills with stable key
        stored_choice = st.session_state.get("flt_asset_choice", "All Assets")
        if stored_choice not in asset_select_opts:
            stored_choice = "All Assets"
        st.session_state["flt_asset_choice"] = stored_choice

        ui_rev = st.session_state.get("ui_rev", 0)
        asset_key = f"ui_asset_choice_{ui_rev}"
        if asset_key not in st.session_state:
            st.session_state[asset_key] = stored_choice

        sel_asset = col1.pills(
            "Select Crypto Asset",
            options=asset_select_opts,
            selection_mode="single",
            key=asset_key,
            label_visibility="visible",
        )
        st.session_state["flt_asset_choice"] = sel_asset

        # Entity Type
        if "ui_entity_type" not in st.session_state:
            st.session_state["ui_entity_type"] = st.session_state.get("flt_entity_type", "All")
        sel_et = col2.selectbox("Select Holder Type", options=type_opts, key="ui_entity_type")
        st.session_state["flt_entity_type"] = sel_et

        # Country
        if "ui_country" not in st.session_state:
            st.session_state["ui_country"] = st.session_state.get("flt_country", "All")
        sel_co = col3.selectbox("Select Country/Region", options=country_opts, key="ui_country")
        st.session_state["flt_country"] = sel_co

        # filtering
        pending_mode = False
        if sel_asset == "All Assets":
            out = df.copy()
        elif sel_asset == "Pending":
            pending_mode = True
            out = df[df["status"].astype(str).str.contains("pending", case=False, na=False)].copy()
            out["USD Value"]       = pd.to_numeric(out.get("target_usd"), errors="coerce").fillna(0.0)
            out["Holdings (Unit)"] = pd.to_numeric(out.get("target_units"), errors="coerce").fillna(0.0)
        else:
            out = df[df["Crypto Asset"] == sel_asset].copy()
            if ((pd.to_numeric(out["USD Value"], errors="coerce").fillna(0.0) <= 0) &
                (pd.to_numeric(out["Holdings (Unit)"], errors="coerce").fillna(0.0) <= 0)).all():
                pending_rows = df[
                    (df["Crypto Asset"] == sel_asset)
                    & (df["status"].astype(str).str.contains("pending", case=False, na=False))
                ].copy()
                if not pending_rows.empty:
                    pending_mode = True
                    out = pending_rows
                    out["USD Value"]       = pd.to_numeric(out.get("target_usd"), errors="coerce").fillna(0.0)
                    out["Holdings (Unit)"] = pd.to_numeric(out.get("target_units"), errors="coerce").fillna(0.0)

        if "DAT" not in df.columns:
            df["DAT"] = 0
        df["DAT"] = pd.to_numeric(df["DAT"], errors="coerce").fillna(0).astype(int)

        # Entity type with DAT Wrappers virtual class
        if sel_et == "DAT Wrappers":
            out = out[out["DAT"] == 1]
        elif sel_et != "All":
            out = out[out["Entity Type"] == sel_et]

        # Country filter
        if sel_co != "All":
            out = out[out["Country"] == sel_co]

        # drop zeroes except in pending-mode
        if not pending_mode:
            out = out[
                (pd.to_numeric(out["USD Value"], errors="coerce").fillna(0.0) > 0)
                | (pd.to_numeric(out["Holdings (Unit)"], errors="coerce").fillna(0.0) > 0)
            ]

        if pending_mode:
            st.session_state["pending_context"] = {"active": True, "scope": sel_asset}
        else:
            st.session_state["pending_context"] = {"active": False, "scope": None}

        st.markdown("")
        return out



def apply_filters_historic(df: pd.DataFrame):
    with st.container(border=False):

        st.markdown("")

        col1, col2 = st.columns([0.8,0.2])

        # ensure a valid default exists in state
        if st.session_state.get("flt_time_range") not in {"All","3 Months", "6 Months","12 Months", "YTD"}:
            st.session_state["flt_time_range"] = "All"

        # --- Asset pills ---
        asset_opts = list(_opts(df["Crypto Asset"]))
        pill_options = ["All Assets"] + asset_opts

        # remember last choice, fall back to "All Assets"
        default_choice = st.session_state.get("hist_asset_choice", "All Assets")
        if default_choice not in pill_options:
            default_choice = "All Assets"

        # revisioned key so resets remount the widget
        ui_rev   = st.session_state.get("ui_rev", 0)
        hist_key = f"ui_assets_hist_{ui_rev}"

        # seed widget state once (string, not list). Do NOT pass default= to the widget.
        if hist_key not in st.session_state:
            st.session_state[hist_key] = default_choice

        sel_choice = col1.pills(
            "Select Crypto Asset",
            options=pill_options,
            selection_mode="single",
            key=hist_key,
            label_visibility="visible",
        )

        # map single choice -> list for downstream filters
        if sel_choice == "All Assets":
            sel_assets = asset_opts
        else:
            sel_assets = [sel_choice]

        # persist cleanly
        st.session_state["hist_asset_choice"] = sel_choice
        st.session_state["flt_assets"] = sel_assets



        # --- Time range widget: initialize UI key once, then no index on reruns
        time_opts = ["All","3 Months", "6 Months","12 Months", "YTD"]
        if "ui_time_range_hist" not in st.session_state:
            st.session_state["ui_time_range_hist"] = st.session_state.get("flt_time_range", "All")
        sel_tr = col2.selectbox(
            "Select Time Range",
            options=time_opts,
            key="ui_time_range_hist"
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
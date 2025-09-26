# sections/planned.py
import streamlit as st
import pandas as pd
import numpy as np

from modules.emojis import country_emoji_map
from sections.overview import _badge_svg_uri


def _opts(series: pd.Series):
    return sorted({str(x).strip() for x in series.dropna().tolist() if str(x).strip()})

def _pretty_usd(x):
    try:
        v = float(x)
    except Exception:
        return "-"
    if np.isnan(v):
        return "-"
    ax = abs(v)
    if ax >= 1e12: return f"${v/1e12:.2f}T"
    if ax >= 1e9:  return f"${v/1e9:.2f}B"
    if ax >= 1e6:  return f"${v/1e6:.2f}M"
    if ax >= 1e3:  return f"${v/1e3:.2f}K"
    return f"${v:,.0f}"

def _df_auto_height(n_rows: int, row_px: int = 35) -> int:
    # header ≈ one row + thin borders
    return int((n_rows + 1) * row_px + 3)

def render_planned():
    #st.markdown("### Planned / Announced Crypto Treasury Activity")

    df = st.session_state.get("planned_df")
    if df is None or df.empty:
        st.info("No planned/announced data available yet.")
        return

    # ---------- Filters ----------
    with st.container(border=True):
        c1, c2, c3 = st.columns([1,1,1])

        asset_opts   = ["All"] + _opts(df["Crypto Asset"])
        country_opts = ["All"] + _opts(df["Country"])
        status_opts  = ["All"] + _opts(df["Status"])

        sel_asset = c1.selectbox("Crypto Asset", options=asset_opts, index=0, key="pln_asset")
        sel_country = c2.selectbox("Country/Region", options=country_opts, index=0, key="pln_country")
        sel_status = c3.selectbox("Status", options=status_opts, index=0, key="pln_status")

    # Apply filters
    filt = df.copy()
    if sel_asset != "All":
        filt = filt[filt["Crypto Asset"] == sel_asset]
    if sel_country != "All":
        filt = filt[filt["Country"] == sel_country]
    if sel_status != "All":
        filt = filt[filt["Status"] == sel_status]

    # ---------- KPIs ----------
    total_planned = float(filt["Planned USD"].fillna(0).sum())
    total_invested = float(filt["Invested/Cost USD"].fillna(0).sum())
    n_entities = int(filt["Entity Name"].nunique())
    
    pct_invested = (total_invested / total_planned * 100.0) if total_planned > 0 else 0

    k1,k2,k3 = st.columns(3)
    with st.container(border=True):
        st.markdown("#### Treasury Pipeline", help = "List of announced digital asset treasury strategies.")

        with k1:
            with st.container(border=True):
                st.metric("Total Pipeline Value (USD)", _pretty_usd(total_planned))
        
        with k2:
            with st.container(border=True):
                st.metric("% Already Invested", f"{pct_invested:.1f}%")
                
        with k3:
            with st.container(border=True):
                st.metric("Number of Unique Entities", n_entities)

        # One-shot dialog
        if st.session_state.pop("show_planned_submit", False):
            @st.dialog("Submit a planned/announced entry", width="large")
            def _submit_dialog():
                with st.form("planned_submit_form", clear_on_submit=True):
                    c1, c2, c3 = st.columns(3)
                    name   = c1.text_input("Entity Name *")
                    ticker = c2.text_input("Ticker")
                    etype  = c3.selectbox("Entity Type", ["Public Company","Private Company","DAO","Non-Profit Organisation","Government","Other"])

                    c4, c5, c6 = st.columns(3)
                    country = c4.text_input("Country / Region")
                    asset   = c5.text_input("Crypto Asset (e.g., BTC)")
                    status  = c6.selectbox("Status", ["Announced","Approved","Funded","Allocating"])

                    c7, c8 = st.columns(2)
                    usd_planned   = c7.number_input("Planned USDd", min_value=0.0, step=1000.0, format="%.0f")
                    usd_cost      = c8.number_input("Invested/Cost USD", min_value=0.0, step=1000.0, format="%.0f")

                    timeline = st.text_input("Timeline")
                    source   = st.text_input("Data Source (URL)")
                    date_src = st.text_input("Date Source (e.g., Aug 15, 2025)")
                    comments = st.text_area("Comments / Notes")

                    submitted = st.form_submit_button("Send")
                if submitted:
                    # For now: put into an in-memory inbox (you can wire to email/Sheets later)
                    inbox = st.session_state.setdefault("planned_inbox", [])
                    inbox.append({
                        "Entity Name": name, "Ticker": ticker, "Entity Type": etype, "Country": country,
                        "Crypto Asset": asset, "Status": status,
                        "Planned USD": usd_planned, "Invested/Cost USD": usd_cost,
                        "Timeline": timeline, "Data Source": source,
                        "Date Source": date_src, "Comments": comments,
                    })
                    st.success("Thanks! We’ve received your submission for review.")
                    if st.button("Close"):
                        st.rerun(scope="app")
            _submit_dialog()

        _status_map = {
            "Announced": "Announced",
            "Approved": "Approved",
            "Funded": "Funded",
            "Allocating": "Investing",
        }

        # ---------- Table ----------
        # Build flag + name
        flag_series = (
            filt["Country"]
            .astype("string")
            .map(lambda c: country_emoji_map.get(c, "🏳️"))
        )

        disp = filt.copy()
        disp["Entity"] = flag_series.fillna("🏳️") + " " + disp["Entity Name"].astype("string")

        _type_palette = {"Public Company": (123, 197, 237), # blue 
                        "Private Company": (232, 118, 226), # rose 
                        "DAO": (237, 247, 94), # amber 
                        "Non-Profit Organization": (34, 197, 94), # green 
                        "Government": (245, 184, 122), # slate 
                        "Other": (250, 250, 250), # white
                        }

        _badge_map = {k: _badge_svg_uri(k, v, h=28) for k, v in _type_palette.items()}

        disp["Entity Type"] = disp["Entity Type"].map(lambda t: _badge_map.get(t, _badge_map["Other"]))

        # Compact formats
        disp["Planned USD"] = disp["Planned USD"].map(_pretty_usd)
        #disp["Invested/Cost USD"] = disp["Invested/Cost USD"].map(_pretty_usd)
        
        # --- Invested % for progress bar ---
        disp["Invested %"] = (
            (filt["Invested/Cost USD"].fillna(0) / filt["Planned USD"].replace(0, np.nan)) * 100
        ).clip(lower=0, upper=100)

        disp["Status"] = disp["Status"].map(lambda s: _status_map.get(str(s), str(s)))

        # Columns to show (note: no Country now)
        disp = disp[[
            "Entity","Ticker","Entity Type","Crypto Asset",
            "Planned USD", "Invested %","Status",
            "Funding Method","Timeline",
            "Data Source","Date Source","Comments"
        ]].sort_values(by='Date Source', ascending=False)

        disp = disp.reset_index(drop=True)
        disp.index = disp.index + 1
        disp.index.name = "#"

        rows = min(len(disp), len(disp))
        height = _df_auto_height(rows)

        st.dataframe(
            disp,
            height=height,
            width="stretch",
            hide_index=False,
            column_config={
                "#": st.column_config.TextColumn("#"),
                "Entity Type": st.column_config.ImageColumn("Entity Type", width="medium"),
                "Invested %": st.column_config.ProgressColumn("Progress", min_value=0, max_value=100, format="%.0f%%"),
                "Data Source": st.column_config.LinkColumn("Data Source", help="Open source", width="small"),
                "Date Source": st.column_config.DateColumn("Date", format="MMM DD, YYYY"),
                "Comments": st.column_config.TextColumn("Further Information", width="medium"),
            },
            key="planned_table",
        )


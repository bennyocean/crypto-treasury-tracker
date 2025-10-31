# sections/planned.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
    #st.title("Crypto Treasury Pipeline")

    df = st.session_state.get("planned_df")
    if df is None or df.empty:
        st.info("No planned/announced data available yet.")
        return

    st.markdown("")
    # ---------- Filters ----------
    with st.container(border=False):
        c1, c2, c3 = st.columns([2,1,1])

        asset_opts   = ["All"] + _opts(df["Crypto Asset"])
        country_opts = ["All"] + _opts(df["Country"])
        status_opts  = ["All"] + _opts(df["Status"])

        sel_asset = c1.pills("Select Asset", options=asset_opts, default= "All", key="pln_asset")
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

    st.markdown("")

    # ---- KPIs & Chart ----
    PROGRESS_RED = "#FF4B4B"

    total_announced = float(df["Planned USD"].fillna(0).sum())
    total_invested  = float(df["Invested/Cost USD"].fillna(0).sum())
    open_amount     = max(total_announced - total_invested, 0.0)

    # width shares in percent for the flex segments
    inv_pct  = (total_invested / total_announced * 100.0) if total_announced else 0.0
    inv_pct_formatted = f"{inv_pct:.1f}%"
    open_pct = max(100.0 - inv_pct, 0.0)

    #n_entities = int(df["Entity Name"].nunique())

    c1_2, c2_2, c3_2 = st.columns(3)

    with c1_2:
        with st.container(border=True):
            st.metric("Open Pipeline Value (USD)", _pretty_usd(open_amount))
            # USD based progress bar built like your working dominance bar
            st.markdown(
                f"""
                <div style='background-color:#1e1e1e;border-radius:8px;height:16px;width:100%;
                            display:flex;overflow:hidden;box-shadow:inset 0 0 0 1px rgba(255,255,255,0.06);'>
                    <div title='Invested { _pretty_usd(total_invested) }'
                        style='width:{inv_pct:.6f}%;background-color:{PROGRESS_RED};'></div>
                    <div title='Open { _pretty_usd(open_amount) }'
                        style='width:{open_pct:.6f}%;background-color:rgba(255,255,255,0.10);'></div>
                </div>
                <div style='margin-top:23px;font-size:13px;color:#aaa;text-align:right;'>
                    { _pretty_usd(total_invested) } ({inv_pct_formatted}) of { _pretty_usd(total_announced) } invested
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("")


    with c2_2:
        with st.container(border=True):
            st.text("Announced Treasury Strategies (USD)")

            df_time = df.copy()
            df_time["Date Source"] = pd.to_datetime(df_time["Date Source"], errors="coerce")
            df_time = df_time.dropna(subset=["Date Source"])
            df_time= df_time[df_time["Planned USD"] > 0]

            if not df_time.empty:
                weekly = (
                    df_time.groupby(pd.Grouper(key="Date Source", freq="W-MON"))["Planned USD"]
                    .sum()
                    .reset_index()
                    .rename(columns={"Date Source": "Week", "Planned USD": "Weekly USD"})
                )

                weekly["Label"] = weekly["Weekly USD"].apply(_pretty_usd)

                fig_w = px.bar(
                    weekly,
                    x="Week",
                    y="Weekly USD",
                    #text="Label",
                    color_discrete_sequence=["#FF4B4B"],
                )

                fig_w.update_layout(
                    height=110,
                    margin=dict(l=30, r=30, t=0, b=0),
                    yaxis_title=None,
                    xaxis_title=None,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(fixedrange=True),
                    xaxis=dict(fixedrange=True)
                )
                
                fig_w.update_traces(hoverinfo='skip', hovertemplate=None)

                st.plotly_chart(fig_w, use_container_width=True, config={"staticPlot": True})


    with c3_2:
        with st.container(border=True):
            st.text("Top 5 Assets in the Pipeline")

            by_asset = (
                df.assign(_open=np.clip(
                    df["Planned USD"].fillna(0) - df["Invested/Cost USD"].fillna(0),
                    0, None
                ))
                .groupby("Crypto Asset")["_open"]
                .sum()
                .sort_values(ascending=True)
                .tail(5)
                .reset_index()
            )

            if not by_asset.empty:

                by_asset["Label"] = by_asset["_open"].apply(_pretty_usd)

                fig = px.bar(
                    by_asset,
                    x="_open",
                    y="Crypto Asset",
                    orientation="h",
                    text="Label",
                    color_discrete_sequence=["#FF4B4B"],  # Streamlit red
                )

                fig.update_traces(
                    textposition="outside",
                    insidetextanchor="start",
                    textfont=dict(size=13, color="white"),
                    cliponaxis=False,
                )

                fig.update_layout(
                    height=110,
                    margin=dict(l=30, r=30, t=0, b=0),
                    xaxis_title=None,
                    yaxis_title=None,
                    xaxis=dict(showgrid=False, tickprefix="$", fixedrange=True),
                    yaxis=dict(fixedrange=True),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                
                fig.update_traces(hoverinfo='skip', hovertemplate=None)

                st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})

    st.markdown("")
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

    # Compact formats
    disp["Planned USD"] = disp["Planned USD"].map(_pretty_usd)
    disp = disp.rename(columns = {"Planned USD":"Pipeline (USD)"})
    #disp["Invested/Cost USD"] = disp["Invested/Cost USD"].map(_pretty_usd)
    
    # --- Invested % for progress bar ---
    disp["Invested %"] = (
        (filt["Invested/Cost USD"].fillna(0) / filt["Planned USD"].replace(0, np.nan)) * 100
    ).clip(lower=0, upper=100)

    disp["Status"] = disp["Status"].map(lambda s: _status_map.get(str(s), str(s)))

    # Columns to show (note: no Country now)
    disp = disp[[
        "Entity","Ticker","Entity Type","Crypto Asset",
        "Pipeline (USD)", "Invested %","Status",
        "Funding Method","Timeline",
        "Comments", "Date Source", "Data Source"
    ]].sort_values(by='Invested %', ascending=False)

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
            "Date Source": st.column_config.DateColumn("Date", format="MMM DD, YYYY"),
            "Entity Type": st.column_config.MultiselectColumn(
                "Holder Type",
                options=[
                    "Public Company",
                    "Private Company",
                    "Government",
                    "Non-Profit Organization",
                    "DAO",
                    "Other"
                ],
                color=["#7bc5ed", "#f759b0", "#f7c694", "#80d9b7",  "#eaf26f", "#ded9d9"],
            ),
            "Invested %": st.column_config.ProgressColumn("Target Progress", min_value=0, max_value=100, format="%.0f%%"),
            "Comments": st.column_config.TextColumn("Details", width="medium"),
            "Data Source": st.column_config.LinkColumn("Data Source", help="Open source", width="small"),
        },
        key="planned_table",
    )


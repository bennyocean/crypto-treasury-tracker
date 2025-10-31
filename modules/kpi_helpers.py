import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, base64
from datetime import date, timedelta

from modules.charts import render_rankings, _prepare_hist_with_snapshot, render_kpi_sparkline, build_monthly_flows_chart, render_flow_decomposition_chart, _prep_history, COLOR_MAP, build_top_five_jurisdictions_bar, build_supply_share_bar
from modules.ui import render_plotly
from modules.flow_utils import decompose_asset
from modules.data_loader import SUPPLY_CAPS

COLORS = {"BTC":"#f7931a","ETH":"#6F6F6F","XRP":"#00a5df","BNB":"#f0b90b","SOL":"#dc1fff", "SUI":"#C0E6FF", "LTC":"#345D9D", "HYPE":"#97fce4", "Other": "rgba(255,255,255,0.9)"}

_THIS = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.join(_THIS, "..", "assets")


def load_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

logo_b64 = load_base64_image("assets/ctt-symbol.svg")
btc_b64 = load_base64_image(os.path.join(_ASSETS, "bitcoin-logo.png"))
eth_b64 = load_base64_image(os.path.join(_ASSETS, "ethereum-logo.png"))
#sol_b64 = load_base64_image(os.path.join(_ASSETS, "solana-logo.png"))


def format_change(value):
    if value > 0:
        return f"↗ {value:.1f}%", "green"
    elif value < 0:
        return f"↘ {value:.1f}%", "red"
    else:
        return f"{value:.1f}%", "white"

def pretty_usd(x):
    if pd.isna(x):
        return "-"
    ax = abs(x)
    if ax >= 1e12:  return f"${x/1e12:.2f}T"
    if ax >= 1e9:  return f"${x/1e9:.2f}B"
    if ax >= 1e6:  return f"${x/1e6:.2f}M"
    if ax >= 1e3:  return f"${x/1e3:.2f}K"
    return f"${x:,.0f}"

def _latest_snapshot_value(snaps: pd.DataFrame, field: str):
    if snaps is None or snaps.empty or field not in snaps.columns:
        return None
    s = snaps[pd.notna(snaps[field])].sort_values(["date_utc","ts_utc"])
    return None if s.empty else float(s.iloc[-1][field])

def _snapshot_value_days_back(snaps: pd.DataFrame, field: str, days: int):
    if snaps is None or snaps.empty or field not in snaps.columns:
        return None
    target = date.today() - timedelta(days=days)
    s = snaps[pd.notna(snaps[field])].copy()
    s = s.sort_values(["date_utc","ts_utc"])
    # pick the last snapshot with date_utc <= target (nearest older or equal)
    older = s[s["date_utc"] <= target]
    row = older.iloc[-1] if not older.empty else s.iloc[0]  # fallback earliest
    return float(row[field])

def _pct_change(cur: float, base: float):
    try:
        cur = float(cur); base = float(base)
        if not np.isfinite(cur) or not np.isfinite(base) or base <= 0:
            return None
        return (cur - base) / base * 100.0
    except Exception:
        return None


# Summary KPIs
def render_kpis(df, snapshots_df=None):
    # ---- Two views: USD vs. holder counts ----
    # 1) USD math: take USD as-is (no filtering); pending rows already have USD=0 if you enforce that upstream.
    df_usd = df.copy()

    # 2) Holder counts: include
    #    - any row with Holdings > 0, OR
    #    - any row explicitly marked active, even if holdings are 0/unknown
    status_series = df.get("status", "").astype(str).str.lower()
    df_counts = df[(pd.to_numeric(df.get("Holdings (Unit)"), errors="coerce").fillna(0) > 0) | (status_series == "active")].copy()

    # ---- USD totals (BTC/ETH/SOL/Other) ----
    total_usd = pd.to_numeric(df_usd["USD Value"], errors="coerce").fillna(0).sum()

    btc_usd = pd.to_numeric(df_usd.loc[df_usd["Crypto Asset"] == "BTC", "USD Value"], errors="coerce").fillna(0).sum()
    eth_usd = pd.to_numeric(df_usd.loc[df_usd["Crypto Asset"] == "ETH", "USD Value"], errors="coerce").fillna(0).sum()
    #sol_usd = pd.to_numeric(df_usd.loc[df_usd["Crypto Asset"] == "SOL", "USD Value"], errors="coerce").fillna(0).sum()
    other_usd = max(total_usd - (btc_usd + eth_usd ), 0)

    usd_pct = {
        "BTC": (btc_usd / total_usd) if total_usd else 0.0,
        "ETH": (eth_usd / total_usd) if total_usd else 0.0,
       # "SOL": (sol_usd / total_usd) if total_usd else 0.0,
        "Other": (other_usd / total_usd) if total_usd else 0.0,
    }

    # ---- Holder counts (unique entities) with BTC/ETH/SOL disjoint split ----
    btc_df = df_counts[df_counts["Crypto Asset"] == "BTC"]
    eth_df = df_counts[df_counts["Crypto Asset"] == "ETH"]
    #sol_df = df_counts[df_counts["Crypto Asset"] == "SOL"]

    btc_entities = btc_df["Entity Name"].nunique()
    eth_entities = eth_df["Entity Name"].nunique()
    #sol_entities = sol_df["Entity Name"].nunique()

    total_entities = df_counts["Entity Name"].nunique()

    btc_set = set(btc_df["Entity Name"].dropna().unique())
    eth_set = set(eth_df["Entity Name"].dropna().unique())
    #sol_set = set(sol_df["Entity Name"].dropna().unique())
    #oth_excl = max(total_entities - len(btc_set | eth_set | sol_set), 0)
    oth_excl = max(total_entities - len(btc_set | eth_set), 0)

    pct_excl = {
        "BTC": (btc_entities / total_entities) if total_entities else 0.0,
        "ETH": (eth_entities / total_entities) if total_entities else 0.0,
        #"SOL": (sol_entities / total_entities) if total_entities else 0.0,
        "Other": (oth_excl / total_entities) if total_entities else 0.0,
    }

    #ENT_COUNTS = {"BTC": btc_entities, "ETH": eth_entities, "SOL": sol_entities, "Other": oth_excl}
    ENT_COUNTS = {"BTC": btc_entities, "ETH": eth_entities, "Other": oth_excl}

    # ---- Baselines / deltas (use the same definitions as the KPIs above) ----
    usd_base = _latest_snapshot_value(snapshots_df, "total_usd")
    ent_base = _snapshot_value_days_back(snapshots_df, "total_entities", 7)

    usd_pct_delta = ((float(total_usd) - float(usd_base)) / float(usd_base) * 100.0) if usd_base and usd_base > 0 else None
    ent_delta_units = ((int(total_entities) - int(ent_base))) if ent_base and ent_base > 0 else None

    usd_delta_label = f"{usd_pct_delta:+.1f}%" if usd_pct_delta is not None else "–"

    
    # ---- KPI layout ----
    col1, col2, col3, col4, col5 = st.columns(5)

    # --------- COL 1  Treasury Market Value -----------
    with col1:
        with st.container(border=True):
            st.metric(
                "Treasury Market Value",
                f"{pretty_usd(total_usd)}",
                delta=f"{usd_delta_label} (24H)",
                help="Aggregate treasury value in USD of all tracked crypto assets across entities, based on live market pricing (vs last day).",
                border=False
            )

            render_kpi_sparkline(snapshots_df, usd_pct_delta)


            st.markdown(
                f"""
                <div style='margin-top:0px;margin-bottom:10px;font-size:15px;color:#aaa;
                            display:flex;gap:12px;align-items:center;'>
                    <div style='display:flex;align-items:center;gap:6px;'>
                        <img src="data:image/png;base64,{btc_b64}" width="16" height="16"> {usd_pct["BTC"]*100:.1f}%
                    </div>
                    <div style='display:flex;align-items:center;gap:6px;'>
                        <img src="data:image/png;base64,{eth_b64}" width="16" height="16"> {usd_pct["ETH"]*100:.1f}%
                    </div>
                    <div style='display:flex;align-items:center;gap:6px;'>
                        <span style="display:inline-block;width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-bottom:10px solid rgba(255,255,255,0.9);vertical-align:middle"></span>
                        {usd_pct["Other"]*100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --------- COL 2  Monthly Net Flow -----------
    with col2:
        # ---- Monthly ΔUSD KPI ----
        df_hist = st.session_state.get("historic_df")
        df_curr = st.session_state.get("data_df")  # live snapshot for current month

        if isinstance(df_hist, pd.DataFrame) and not df_hist.empty:
            # include current month like in the dedicated decomposition section
            hist_merge, _ = _prepare_hist_with_snapshot(df_hist, df_curr)
            hist = _prep_history(hist_merge)

            # per-asset decomposition then aggregate to monthly totals
            decomp = (
                hist.groupby("Crypto Asset", group_keys=True)
                    .apply(decompose_asset)
                    .reset_index(drop=True)
            )
            decomp["Date"] = pd.to_datetime(decomp["Date"]).dt.to_period("M").dt.to_timestamp()

            monthly = (
                decomp.groupby("Date")[["d_usd", "price_effect", "units_effect"]]
                    .sum()
                    .reset_index()
                    .sort_values("Date")
            )

            # force a complete 6-month window ending at the latest month, fill gaps with 0
            last_month = pd.to_datetime(monthly["Date"].max()).to_period("M").to_timestamp()
            six_idx = pd.date_range(end=last_month, periods=6, freq="MS")

            monthly = (
                monthly.set_index("Date")
                    .reindex(six_idx, fill_value=0.0)
                    .rename_axis("Date")
                    .reset_index()
            )

            prev_month = pd.to_datetime(monthly.iloc[-2]["Date"])
            month_name = prev_month.strftime("%b")
            year = prev_month.strftime("%Y")

            if not monthly.empty:
                last = monthly.iloc[-1]
                prev = monthly.iloc[-2] if len(monthly) > 1 else None

                d_usd = float(last["d_usd"])
                pct_change = None
                if prev is not None and float(prev["d_usd"]) != 0:
                    pct_change = (d_usd - float(prev["d_usd"])) / abs(float(prev["d_usd"])) * 100.0

                delta_label = f"{pct_change:+.1f}%" if pct_change is not None else "–"

                with st.container(border=True):
                    st.metric(
                        label="Monthly Net Flow",
                        value=pretty_usd(d_usd),
                        delta=f"{delta_label} ({month_name} {year})",
                        help="Net inflow or outflow across all tracked entities in the current month, with percent change versus the previous month. Net equals price effect plus units effect"
                    )

                    fig = build_monthly_flows_chart(monthly)  # pass merged hist so chart matches KPI
                    if fig is not None:
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={
                                "displayModeBar": False,
                                "scrollZoom": False,
                                "doubleClick": False,
                                "editable": False
                            }
                        )
            else:
                with st.container(border=True):
                    st.metric("ΔUSD last month", "–", help="No sufficient monthly history available.")
        else:
            with st.container(border=True):
                st.metric("ΔUSD last month", "–", help="No historic data available for flow KPI.")

    # --- Entity type breakdown for holder distribution ---
    if "Entity Type" in df_counts.columns:
        type_counts = (
            df_counts.groupby("Entity Type", as_index=False)["Entity Name"]
                    .nunique()
                    .rename(columns={"Entity Name": "count"})
        )

        # Sort descending by number of entities
        type_counts = type_counts.sort_values("count", ascending=False)

        total_holders = int(type_counts["count"].sum())
        type_counts["share"] = type_counts["count"] / total_holders
    else:
        type_counts = pd.DataFrame(columns=["Entity Type", "count", "share"])
        total_holders = total_entities

    # --------- COL 3  Entity Number -----------
    with col3:
        with st.container(border=True):
            st.metric(
                "Total Holders",
                f"{total_entities}",
                delta=f"{ent_delta_units} (WoW)",
                help="Entities holding crypto assets or confirmed active holders (even if current holdings are not disclosed). Each entity counted once."
            )

            #st.markdown("")
            ENT_COLORS = {
                "BTC": COLORS["BTC"],
                "ETH": COLORS["ETH"],
                #"SOL": COLORS["SOL"],
                "Other": COLORS.get("Other", "#cccccc"),
            }

            st.markdown(
                f"""
                <div style='background-color:#1e1e1e;border-radius:8px;height:40px;width:100%;margin-top:13px;
                            display:flex;overflow:hidden;box-shadow:inset 0 0 0 1px rgba(255,255,255,0.06);'>
                    {''.join(
                        f"<div title='{k}: {ENT_COUNTS[k]} ({pct_excl[k]*100:.1f}%)' "
                        f"style='width:{pct_excl[k]*100:.4f}%;background-color:{ENT_COLORS[k]};'></div>"
                        for k in ["BTC","ETH","Other"]
                    )}
                </div>
                <div style='margin-top:20px;margin-bottom:10px;font-size:15px;color:#aaa;
                            display:flex;gap:12px;align-items:center;'>
                    <div style='display:flex;align-items:center;gap:6px;'>
                        <img src="data:image/png;base64,{btc_b64}" width="16" height="16"> {btc_entities}
                    </div>
                    <div style='display:flex;align-items:center;gap:6px;'>
                        <img src="data:image/png;base64,{eth_b64}" width="16" height="16"> {eth_entities}
                    </div>
                    <div style='display:flex;align-items:center;gap:6px;'>
                        <span style="display:inline-block;width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-bottom:10px solid rgba(255,255,255,0.9);vertical-align:middle"></span>
                        {oth_excl}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --------- COL 4  Top Jurisdictions mini bar -----------
    with col4:
        with st.container(border=True):
            st.metric(
                "Top Countries",
                value="",  # number not needed, chart carries info
                help="Top five countries by aggregate value in USD. Unknown, Decentralized and Other are excluded."
            )

            fig_geo = build_top_five_jurisdictions_bar(df_usd)
            if fig_geo is not None:
                st.plotly_chart(
                    fig_geo,
                    use_container_width=True,
                    config=dict(displayModeBar=False, scrollZoom=False, doubleClick=False, editable=False),
                )

    # --------- COL 5  Treasury Concentration mini bar ----------
    with col5:
        with st.container(border=True):
            st.metric(
                label="Share of Supply (Top 5)",
                value="",
                help="Top five crypto assets ranked by aggregated treasury holdings as a share of their circulating supply."
            )

            fig_supply = build_supply_share_bar(df_usd, sort_mode = "usd")
            if fig_supply is not None:
                st.plotly_chart(
                    fig_supply,
                    use_container_width=True,
                    config=dict(displayModeBar=False, scrollZoom=False, doubleClick=False, editable=False),
                )
    

# Top 5 Holders Chart
def top_5_holders(df, asset="BTC", key_prefix="top5"):
    with st.container(border=False):
        logo_b64 = {"BTC": btc_b64, "ETH": eth_b64, "SOL": sol_b64}.get(asset)
        st.markdown(
            f'''
            <img src="data:image/png;base64,{logo_b64}" style="height:26px; vertical-align: middle; margin: 0 4px 4px;"> Treasury Holders 
            ''',
            unsafe_allow_html=True,
            help=f"List of top 5 entities by {asset} treasury holdings shown in units or USD value."
        )

        mode = st.segmented_control(
            "Display mode",
            options=["USD Value", "Unit Count"],
            default="Unit Count",
            label_visibility="collapsed",
            key=f"{key_prefix}_{asset}_mode"
        )
        by = "units" if mode == "Unit Count" else "usd"
        
        st.markdown("")
        
        fig = render_rankings(df, asset=asset, by=by)
        render_plotly(
            fig,
            filename=f"top_5_{asset.lower()}_holders",
            #width="stretch",
            extra_config={
                "displaylogo": False,
                "displayModeBar": False,
                "staticPlot": True,
                "scrollZoom": False,
                "doubleClick": False,
                "showTips": False,
            }
        )

# Historic KPIs
def _latest_and_prev_dates(dates: pd.Series):
    """Return (latest_date, previous_available_date) from a series of pd.Timestamp."""
    uniq = sorted(dates.dropna().unique())
    if not uniq:
        return None, None
    if len(uniq) == 1:
        return uniq[-1], None
    return uniq[-1], uniq[-2]

def _year_end_total(df: pd.DataFrame, year: int, value_col: str):
    """Total at year-end (December of `year`) if present, else None."""
    if df.empty:
        return None
    ydf = df[(df["Year"] == year) & (df["Month"] == 12)]
    if ydf.empty:
        return None
    return float(ydf[value_col].sum())


def _pct_change(old, new):
    """Return float % change or None if invalid baseline."""
    if old is None or old <= 0:
        return None
    return (new - old) / old * 100.0


def _fmt_pct_value(x, na="N/A"):
    """Render a percentage or N/A if baseline missing."""
    return f"{x:.1f}%" if (x is not None and np.isfinite(x)) else na
    
def _compute_current_vs_last(df_current: pd.DataFrame, df_hist: pd.DataFrame, assets: list[str]):
    """
    Returns:
      current_usd, last_usd, usd_delta_pct,
      current_units (if single asset), last_units, units_delta_pct
    """
    # current snapshot (already priced via attach_usd_values)
    cur = df_current[df_current["Crypto Asset"].isin(assets)]
    current_usd = float(cur["USD Value"].sum())

    current_units = None
    if len(assets) == 1:
        a = assets[0]
        current_units = float(cur.loc[cur["Crypto Asset"] == a, "Holdings (Unit)"].sum())

    # last stored month in historic
    hist = df_hist[df_hist["Crypto Asset"].isin(assets)]
    if hist.empty:
        return current_usd, None, None, current_units, None, None

    last_date = pd.to_datetime(hist["Date"]).max()
    last_month = hist[pd.to_datetime(hist["Date"]) == last_date]

    last_usd = float(last_month["USD Value"].sum())

    last_units = None
    units_delta_pct = None
    if len(assets) == 1:
        a = assets[0]
        last_units = float(last_month.loc[last_month["Crypto Asset"] == a, "Holdings (Unit)"].sum())
        if last_units and last_units > 0 and current_units is not None:
            units_delta_pct = (current_units - last_units) / last_units * 100.0

    usd_delta_pct = (current_usd - last_usd) / last_usd * 100.0 if (last_usd is not None and last_usd > 0) else None

    return current_usd, last_usd, usd_delta_pct, current_units, last_units, units_delta_pct


def render_historic_kpis(df_filtered: pd.DataFrame):
    with st.container(border=False):
        col1, col2, col3 = st.columns(3)

        if df_filtered.empty:
            col1.metric("Monthly Change (USD)", "N/A")
            col2.metric("YTD Change (USD)", "N/A")
            col3.metric("CAGR (USD)", "N/A")
            #st.write("")  # spacer
            c1, c2, c3 = st.columns(3)
            c1.metric("Monthly Change (units)", "N/A")
            c2.metric("YTD Change (units)", "N/A")
            c3.metric("CAGR (units)", "N/A")
            return

        # Working frames
        dfw = df_filtered.copy()  # respects current UI filters
        df_full = st.session_state.get("historic_df", dfw)  # for prior Dec baselines
        
        # ensure datetime (no-op if already)
        dfw["Date"] = pd.to_datetime(dfw["Date"])
        df_full["Date"] = pd.to_datetime(df_full["Date"])

        # Dates
        latest_date, prev_date = _latest_and_prev_dates(dfw["Date"])
        assets_in_scope = sorted(dfw["Crypto Asset"].dropna().unique().tolist())

        # --- NEW: Current snapshot vs last stored month (USD), consistent across sections
        cur_usd, last_usd, cur_vs_last_usd_pct, cur_units, last_units, cur_vs_last_units_pct = _compute_current_vs_last(
            st.session_state["data_df"],   # current priced snapshot
            st.session_state["historic_df"],  # full historic
            assets_in_scope
        )

        # --- Aggregate USD KPIs (always shown) ---
        latest_total_usd = float(dfw.loc[dfw["Date"] == latest_date, "USD Value"].sum()) if latest_date is not None else 0.0
        prev_total_usd   = float(dfw.loc[dfw["Date"] == prev_date,   "USD Value"].sum()) if prev_date   is not None else None

        # Monthly % (USD)
        monthly_change_usd = _pct_change(prev_total_usd, latest_total_usd)

        # YTD % (USD) vs prior Dec (use full history but same asset scope)
        latest_year = int(pd.to_datetime(latest_date).year) if latest_date is not None else None
        ytd_base = df_full[df_full["Crypto Asset"].isin(assets_in_scope)] if assets_in_scope else df_full
        prior_dec_total_usd = _year_end_total(ytd_base, latest_year - 1, "USD Value") if latest_year else None
        ytd_change_usd = _pct_change(prior_dec_total_usd, latest_total_usd)


        # --- CAGR window from full history (not the UI-filtered window) ---
        df_cagr_base = df_full[df_full["Crypto Asset"].isin(assets_in_scope)] if assets_in_scope else df_full

        months = pd.to_datetime(df_cagr_base["Date"]).dt.to_period("M").sort_values().unique()
        if len(months) == 0:
            # ultra-guard: no months found
            start_period = end_period = None
            first_total_usd_cagr = latest_total_usd_cagr = 0.0
            n_months_cagr = 0
            cagr_usd = None
        else:
            if len(months) >= 12:
                start_period = months[-12]
                end_period   = months[-1]
            else:
                start_period = months[0]
                end_period   = months[-1]

            start_mask = df_cagr_base["Date"].dt.to_period("M") == start_period
            end_mask   = df_cagr_base["Date"].dt.to_period("M") == end_period

            first_total_usd_cagr  = float(df_cagr_base.loc[start_mask, "USD Value"].sum())
            latest_total_usd_cagr = float(df_cagr_base.loc[end_mask,   "USD Value"].sum())
            n_months_cagr = (end_period.year - start_period.year) * 12 + (end_period.month - start_period.month)
            cagr_usd = (((latest_total_usd_cagr / first_total_usd_cagr) ** (12 / n_months_cagr)) - 1) * 100.0 \
                if (first_total_usd_cagr is not None and first_total_usd_cagr > 0 and n_months_cagr > 0) else None

        with col1:
            with st.container(border=True):
                st.metric(
            "Current Value (USD) vs Last Month",
            value=f"${cur_usd:,.0f}",
            delta=(f"{cur_vs_last_usd_pct:+.1f}%" if cur_vs_last_usd_pct is not None else None),
            delta_color="normal",
            help="Current total USD value under the selected filters, with % change vs the last month. Note: Adjustment of −91,331 BTC applied for historical consistency with latest reported holdings due to data source inconsistencies."
        )

        with col2:
            with st.container(border=True):
                st.metric(
            "YTD Change (USD)",
            value=_fmt_pct_value(ytd_change_usd),
            delta_color="normal",
            help="Change since prior year‑end (Dec), based on USD value."
        )

        with col3:
            with st.container(border=True):
                st.metric(
            "CAGR (USD)",
            value=_fmt_pct_value(cagr_usd),
            help=("Compound annual growth rate based on USD value. "
                "Uses last 12 months if available, otherwise all available data.")
        )


        # --- Units KPIs only when exactly ONE asset is selected ---

        single_asset = assets_in_scope[0] if len(assets_in_scope) == 1 else None

        with st.expander("Unit KPIs (single asset)", expanded = (single_asset is not None)):

            if single_asset:
                dfw_asset = dfw[dfw["Crypto Asset"] == single_asset]

                latest_units = float(dfw_asset.loc[dfw_asset["Date"] == latest_date, "Holdings (Unit)"].sum()) if latest_date is not None else 0.0
                prev_units   = float(dfw_asset.loc[dfw_asset["Date"] == prev_date,   "Holdings (Unit)"].sum()) if prev_date   is not None else None

                monthly_change_units = _pct_change(prev_units, latest_units)

                ytd_base_asset = ytd_base[ytd_base["Crypto Asset"] == single_asset]
                prior_dec_units = _year_end_total(ytd_base_asset, latest_year - 1, "Holdings (Unit)") if latest_year else None
                ytd_change_units = _pct_change(prior_dec_units, latest_units)

                # Units CAGR uses SAME window as USD CAGR but restricted to the single asset
                if start_period is not None and n_months_cagr > 0:
                    df_cagr_asset = df_cagr_base[df_cagr_base["Crypto Asset"] == single_asset]
                    first_units_cagr  = float(df_cagr_asset.loc[df_cagr_asset["Date"].dt.to_period("M") == start_period, "Holdings (Unit)"].sum())
                    latest_units_cagr = float(df_cagr_asset.loc[df_cagr_asset["Date"].dt.to_period("M") == end_period,   "Holdings (Unit)"].sum())
                    cagr_units = (((latest_units_cagr / first_units_cagr) ** (12 / n_months_cagr)) - 1) * 100.0 \
                        if (first_units_cagr and first_units_cagr > 0) else None
                else:
                    cagr_units = None

                c1, c2, c3 = st.columns(3)

                with c1:
                    with st.container(border=True):
                        st.metric(
                            f"Current ({single_asset} units) vs Last Month",
                            value=(f"{int(cur_units):,}" if cur_units is not None and cur_units >= 1 else f"{(cur_units or 0):,.2f}"),
                            delta=(f"{cur_vs_last_units_pct:+.1f}%" if cur_vs_last_units_pct is not None else None),
                            delta_color="normal",
                            help=f"Current {single_asset} units under the selected filters, with % change vs the last month. Note: Adjustment of −91,331 BTC applied for historical consistency with latest reported holdings due to data source inconsistencies."
                        )

                with c2:
                    with st.container(border=True):
                        st.metric(                            f"YTD ({single_asset} units)",
                            value=_fmt_pct_value(ytd_change_units),
                            delta_color="normal",
                            help=f"Change since prior year‑end (Dec), in {single_asset} units. "
                        )
                with c3:
                    with st.container(border=True):
                        st.metric(                            f"CAGR ({single_asset} units)",
                            value=_fmt_pct_value(cagr_units),
                            help=(f"Compound annual growth rate in {single_asset} units. "
                                "Uses last 12 months if available, otherwise all available data.")
                        )

            else:
                st.caption("Select a single asset to view unit‑based KPIs.")


def _fmt_usd(x: float) -> str:
    try:
        if abs(x) >= 1e9:  return f"${x/1e9:,.1f}B"
        if abs(x) >= 1e6:  return f"${x/1e6:,.1f}M"
        if abs(x) >= 1e3:  return f"${x/1e3:,.1f}K"
        return f"${x:,.0f}"
    except Exception:
        return "$0"


def render_flow_decomposition(df_hist_filtered: pd.DataFrame, current_df: pd.DataFrame | None = None):
    if df_hist_filtered.empty:
        st.info("No historic data for the current filters.")
        return

    # 1) Merge historic with current snapshot month if provided
    hist_merge, _ = _prepare_hist_with_snapshot(df_hist_filtered, current_df)
    hist = _prep_history(hist_merge)

    # 2) Append current month aggregate when newer than last hist month
    if current_df is not None and not current_df.empty:
        snap_month = pd.Timestamp.today().to_period("M").to_timestamp()
        last_hist_m = pd.to_datetime(hist["Date"]).dt.to_period("M").dt.to_timestamp().max()

        if snap_month > last_hist_m:
            snap = current_df.copy()
            cur_month = (
                snap.groupby("Crypto Asset", as_index=False)[["Holdings (Unit)", "USD Value"]]
                    .sum()
            )
            cur_month["Price USD"] = np.where(
                pd.to_numeric(cur_month["Holdings (Unit)"], errors="coerce").gt(0),
                pd.to_numeric(cur_month["USD Value"], errors="coerce") / pd.to_numeric(cur_month["Holdings (Unit)"], errors="coerce"),
                np.nan
            )
            cur_month["Date"] = snap_month
            hist = pd.concat([hist, cur_month], ignore_index=True)
            hist = _prep_history(hist)

    # 3) Build a baseline month BEFORE the first month displayed
    start_m = pd.to_datetime(hist["Date"]).dt.to_period("M").dt.to_timestamp().min()
    prev_m  = start_m - pd.offsets.MonthBegin(1)
    assets_scope = hist["Crypto Asset"].dropna().unique().tolist()


    base_hist = st.session_state.get("historic_df")
    if isinstance(base_hist, pd.DataFrame) and not base_hist.empty:
        bh = base_hist.copy()
        bh["Date"] = pd.to_datetime(bh["Date"]).dt.to_period("M").dt.to_timestamp()
        base_rows = bh[(bh["Crypto Asset"].isin(assets_scope)) & (bh["Date"] == prev_m)]
        hist_decomp = pd.concat([base_rows, hist], ignore_index=True)
        hist_decomp = _prep_history(hist_decomp)
    else:
        hist_decomp = hist.copy()

    with st.container(border=False):
        st.markdown("### Flow & Decomposition (Price vs Accumulation)", help="Splits monthly ΔUSD into Price effect on prior units vs Units effect at current price.")
        st.markdown("")

        selected_assets = st.session_state.get("flt_assets", [])
        asset_pick = selected_assets[0] if isinstance(selected_assets, list) and len(selected_assets) == 1 else None

        # 4) Decompose per asset with the robust version
        decomp = (
            hist_decomp.groupby("Crypto Asset", group_keys=True)
                       .apply(decompose_asset)   # <- your local wrapper or import decompose_asset
                       .reset_index(drop=True)
        )

        # 5) Ensure the reindex includes the current month
        decomp["Date"] = pd.to_datetime(decomp["Date"]).dt.to_period("M").dt.to_timestamp()
        start_m = pd.to_datetime(df_hist_filtered["Date"]).dt.to_period("M").dt.to_timestamp().min()
        end_m   = max(
            pd.to_datetime(hist["Date"]).dt.to_period("M").dt.to_timestamp().max(),
            pd.Timestamp.today().to_period("M").to_timestamp()
        )
        full_months = pd.date_range(start=start_m, end=end_m, freq="MS")


        if asset_pick:
            view = decomp[decomp["Crypto Asset"] == asset_pick].copy()
        else:
            view = (decomp.groupby("Date")[["d_usd", "price_effect", "units_effect"]]
                          .sum()
                          .reset_index())

        view = (
            view.set_index("Date")
                .reindex(full_months, fill_value=0.0)
                .rename_axis("Date")
                .reset_index()
        )

        # 6) KPIs use the latest row which now is the current month
        last = view.sort_values("Date").tail(1)
        d_usd = float(last["d_usd"].iloc[0]) if not last.empty else 0.0
        pe    = float(last["price_effect"].iloc[0]) if not last.empty else 0.0
        ue    = float(last["units_effect"].iloc[0]) if not last.empty else 0.0

        k1, k2, k3 = st.columns(3)
        with k1:
            with st.container(border=True):
                st.metric("ΔUSD current month", _fmt_usd(d_usd))
        with k2:
            with st.container(border=True):
                st.metric("Price contribution", _fmt_usd(pe))
        with k3:
            with st.container(border=True):
                st.metric("Units contribution", _fmt_usd(ue))

        render_flow_decomposition_chart(view, asset_pick)
        st.caption("Note Current month is derived from live snapshot data when available.")

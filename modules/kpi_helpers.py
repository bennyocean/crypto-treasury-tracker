import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, base64
from datetime import date, timedelta

from modules.charts import render_rankings
from modules.ui import render_plotly


COLORS = {"BTC":"#f7931a","ETH":"#6F6F6F","XRP":"#00a5df","BNB":"#f0b90b","SOL":"#dc1fff", "SUI":"#C0E6FF", "LTC":"#345D9D", "HYPE":"#97fce4", "Other": "rgba(255,255,255,0.9)"}

_THIS = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.join(_THIS, "..", "assets")


def load_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

logo_b64 = load_base64_image("assets/ctt-symbol.svg")
btc_b64 = load_base64_image(os.path.join(_ASSETS, "bitcoin-logo.png"))
eth_b64 = load_base64_image(os.path.join(_ASSETS, "ethereum-logo.png"))
sol_b64 = load_base64_image(os.path.join(_ASSETS, "solana-logo.png"))


def format_change(value):
    if value > 0:
        return f"↗ {value:.1f}%", "green"
    elif value < 0:
        return f"↘ {value:.1f}%", "red"
    else:
        return f"{value:.1f}%", "white"

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
    # Compute values
    df = df[(df['USD Value'] > 0) | (df['Holdings (Unit)'] > 0)]

    total_usd = df["USD Value"].sum()

    btc_df = df[df["Crypto Asset"] == "BTC"]
    eth_df = df[df["Crypto Asset"] == "ETH"]
    sol_df = df[df["Crypto Asset"] == "SOL"]

    btc_usd = btc_df["USD Value"].sum()
    eth_usd = eth_df["USD Value"].sum()
    sol_usd = sol_df["USD Value"].sum()

    btc_entities = btc_df["Entity Name"].shape[0]#.nunique()
    eth_entities = eth_df["Entity Name"].shape[0]#.nunique()
    sol_entities = sol_df["Entity Name"].shape[0]#.nunique()
    total_entities_all = df["Entity Name"].shape[0]
    total_entities = df["Entity Name"].nunique()

    btc_units = btc_df["Holdings (Unit)"].sum()
    eth_units = eth_df["Holdings (Unit)"].sum()
    sol_units = sol_df["Holdings (Unit)"].sum()

    # baselines
    usd_base = _latest_snapshot_value(snapshots_df, "total_usd")            # 1d (latest stored)
    ent_base = _snapshot_value_days_back(snapshots_df, "total_entities", 7)  # 7d ago (or nearest older)
    print("usd base print: ", usd_base)

    usd_pct = ((float(total_usd) - float(usd_base)) / float(usd_base) * 100.0) if usd_base and usd_base > 0 else None
    ent_pct = ((float(total_entities) - float(ent_base)) / float(ent_base) * 100.0) if ent_base and ent_base > 0 else None
    ent_delta_units = ((int(total_entities) - int(ent_base))) if ent_base and ent_base > 0 else None

    print("usd pct print: ", usd_pct)

    usd_delta_label = f"{usd_pct:+.1f}%" if usd_pct is not None else "–"
    ent_delta_label = f"{ent_pct:+.1f}%" if ent_pct is not None else "–"

    
    # KPI layout
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.metric("Total USD Value",
                      f"${total_usd:,.0f}",
                      delta=f"{usd_delta_label} (24H)",
                      help="Aggregate USD value of all tracked crypto assets across entities, based on live market pricing (vs last day).")

            other_usd = max(total_usd - (btc_usd + eth_usd + sol_usd), 0)

            usd_pct = {
                "BTC": (btc_usd / total_usd) if total_usd else 0.0,
                "ETH": (eth_usd / total_usd) if total_usd else 0.0,
                "SOL": (sol_usd / total_usd) if total_usd else 0.0,
                "Other": (other_usd / total_usd) if total_usd else 0.0,
            }

            # Progress bar with hover tooltip
            st.markdown(
                f"""
                <div style='background-color:#1e1e1e;border-radius:8px;height:20px;width:100%;
                            display:flex;overflow:hidden;box-shadow:inset 0 0 0 1px rgba(255,255,255,0.06);'>
                    {''.join(
                        f"<div title='{a}: {usd_pct[a]*100:.1f}% (${val/1e9:.1f}B)' "
                        f"style='width:{usd_pct[a]*100:.4f}%;background-color:{COLORS[a]};'></div>"
                        for a, val in [("BTC", btc_usd), ("ETH", eth_usd), ("SOL", sol_usd), ("Other", other_usd)]
                    )}
                </div>
                <div style='margin-top:8px;margin-bottom:-5px;font-size:15px;color:#aaa;
                            display:flex;gap:12px;align-items:center;'>Dominance:
                    <div style='display:flex;align-items:center;gap:6px;'>
                        <img src="data:image/png;base64,{btc_b64}" width="16" height="16"> {usd_pct["BTC"]*100:.1f}%
                    </div>
                    <div style='display:flex;align-items:center;gap:6px;'>
                        <img src="data:image/png;base64,{eth_b64}" width="16" height="16"> {usd_pct["ETH"]*100:.1f}%
                    </div>
                    <div style='display:flex;align-items:center;gap:6px;'>
                        <img src="data:image/png;base64,{sol_b64}" width="16" height="16"> {usd_pct["SOL"]*100:.1f}%
                    </div>
                    <div style='display:flex;align-items:center;gap:6px;'>
                        <span style="display:inline-block;width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-bottom:10px solid rgba(255,255,255,0.9);vertical-align:middle"></span>
                        {usd_pct["Other"]*100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("")

    # ---- Adoption (entities) with OTH residual ----
    with col2:
        with st.container(border=True):
            st.metric(
                "Total Unique Holders",
                f"{total_entities}",
                delta=f"{ent_delta_units} (WoW)",
                help="Entities holding crypto assets (vs last week), excluding ETFs and other indirect vehicles. Note: some entities hold various crypto assets and are counted once."
            )

            # sets for exclusive partition (BTC first, then ETH not in BTC, then SOL not in BTC/ETH, then Other = none of these)
            btc_set = set(btc_df["Entity Name"].unique())
            eth_set = set(eth_df["Entity Name"].unique())
            sol_set = set(sol_df["Entity Name"].unique())
            union_bes = btc_set | eth_set | sol_set

            btc_excl = len(btc_set)
            eth_excl = len(eth_set - btc_set)
            sol_excl = len(sol_set - btc_set - eth_set)
            oth_excl = max(total_entities - len(union_bes), 0)
            other_excl_new = (total_entities_all - btc_entities - eth_entities - sol_entities)
            
            pct_excl = {
                "BTC": (btc_excl / total_entities) if total_entities else 0.0,
                "ETH": (eth_excl / total_entities) if total_entities else 0.0,
                "SOL": (sol_excl / total_entities) if total_entities else 0.0,
                "Other": (oth_excl / total_entities) if total_entities else 0.0,
            }

            # display counts (keep your original per-asset counts; Other uses the disjoint count)
            ENT_COUNTS = {"BTC": btc_entities, "ETH": eth_entities, "SOL": sol_entities, "Other": other_excl_new}

            ENT_COLORS = {
                "BTC": COLORS["BTC"],   # from your first KPI card
                "ETH": COLORS["ETH"],
                "SOL": COLORS["SOL"],
                "Other": COLORS["Other"],  # white
            }

            st.markdown(
                f"""
                <div style='background-color:#1e1e1e;border-radius:8px;height:20px;width:100%;
                            display:flex;overflow:hidden;box-shadow:inset 0 0 0 1px rgba(255,255,255,0.06);'>
                    {''.join(
                        f"<div title='{k}: {ENT_COUNTS[k]} ({pct_excl[k]*100:.1f}%)' "
                        f"style='width:{pct_excl[k]*100:.4f}%;background-color:{ENT_COLORS[k]};'></div>"
                        for k in ["BTC","ETH","SOL","Other"]
                    )}
                </div>
                <div style='margin-top:8px;margin-bottom:16px;font-size:15px;color:#aaa;
                            display:flex;gap:12px;align-items:center;'>Adoption:
                    <div style='display:flex;align-items:center;gap:6px;'>
                        <img src="data:image/png;base64,{btc_b64}" width="16" height="16"> {btc_entities}
                    </div>
                    <div style='display:flex;align-items:center;gap:6px;'>
                        <img src="data:image/png;base64,{eth_b64}" width="16" height="16"> {eth_entities}
                    </div>
                    <div style='display:flex;align-items:center;gap:6px;'>
                        <img src="data:image/png;base64,{sol_b64}" width="16" height="16"> {sol_entities}
                    </div>
                    <div style='display:flex;align-items:center;gap:6px;'>
                        <span style="display:inline-block;width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-bottom:10px solid rgba(255,255,255,0.9);vertical-align:middle"></span>
                        {other_excl_new}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            #st.markdown("")


    with col3:
        with st.container(border=True):
            st.metric("% of Supply", f"", help="Share of total circulating supply held by tracked entities (BTC ≈ 20M, ETH ≈ 120M, SOL ≈ 540M).")

            # compute percentages of supply
            btc_pct_supply = btc_units / 20_000_000
            eth_pct_supply = eth_units / 120_000_000
            sol_pct_supply = sol_units / 540_000_000

            # colors
            COL_HELD = {
                "BTC": "#f7931a",
                "ETH": "#6F6F6F",
                "SOL": "#dc1fff",
            }
            COL_REMAIN = "#2c2c2c"

            # figure and subplots
            fig = make_subplots(
                rows=1, cols=3,
                specs=[[{"type": "domain"}, {"type": "domain"}, {"type": "domain"}]],
                horizontal_spacing=0.08,
            )

            # pies
            fig.add_trace(go.Pie(
                labels=["Held", "Remaining"],
                values=[btc_pct_supply, 1 - btc_pct_supply],
                hole=0.7,
                marker_colors=[COL_HELD["BTC"], COL_REMAIN],
                textinfo="none",
                hoverinfo="skip",
                sort=False
            ), 1, 1)

            fig.add_trace(go.Pie(
                labels=["Held", "Remaining"],
                values=[eth_pct_supply, 1 - eth_pct_supply],
                hole=0.7,
                marker_colors=[COL_HELD["ETH"], COL_REMAIN],
                textinfo="none",
                hoverinfo="skip",
                sort=False
            ), 1, 2)

            fig.add_trace(go.Pie(
                labels=["Held", "Remaining"],
                values=[sol_pct_supply, 1 - sol_pct_supply],
                hole=0.7,
                marker_colors=[COL_HELD["SOL"], COL_REMAIN],
                textinfo="none",
                hoverinfo="skip",
                sort=False
            ), 1, 3)

            # helper to center a logo and its percent for a given pie trace
            def add_logo_and_pct(fig, trace_idx, b64, pct_text, img_scale=0.24, gap=0.25):
                """
                img_scale controls logo size relative to the pie domain
                gap controls vertical spacing between logo and text as a fraction of domain height
                """
                t = fig.data[trace_idx]
                try:
                    x0, x1 = t.domain.x
                    y0, y1 = t.domain.y
                except Exception:
                    x0, x1 = t.domain["x"]
                    y0, y1 = t.domain.get("y", [0, 1])

                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                dx = (x1 - x0)
                dy = (y1 - y0)

                # logo anchored to pie domain
                fig.add_layout_image(dict(
                    source=f"data:image/png;base64,{b64}",
                    xref="paper", yref="paper",
                    x=cx, y=cy + dy * (gap / 2),
                    sizex=dx * img_scale,
                    sizey=dy * img_scale,
                    xanchor="center", yanchor="middle",
                    sizing="contain",
                    layer="above",
                    opacity=1.0,
                ))

                # percent anchored to the same domain with symmetric offset
                fig.add_annotation(
                    text=pct_text,
                    x=cx, y=cy - dy * (gap / 2),
                    xref="paper", yref="paper",
                    xanchor="center", yanchor="middle",
                    showarrow=False,
                    font=dict(size=16, color="white"),
                    align="center",
                )

            # layout reset
            fig.update_layout(
                height=125,
                margin=dict(t=10, b=0, l=0, r=0),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                annotations=[],
            )

            # place logos and percents
            # assumes btc_b64 eth_b64 sol_b64 already loaded as base64 strings
            add_logo_and_pct(fig, 0, btc_b64, f"{btc_pct_supply:.2%}")
            add_logo_and_pct(fig, 1, eth_b64, f"{eth_pct_supply:.2%}")
            add_logo_and_pct(fig, 2, sol_b64, f"{sol_pct_supply:.2%}")

            st.plotly_chart(
                fig,
                config={"displayModeBar": False},
                use_container_width=True  # replaces width="stretch"
            )

            #st.markdown("")


            # {btc_units:,.0f}


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


def _coerce_num(s: pd.Series) -> pd.Series:
    """Parse numbers from both US and EU formats. Returns float Series."""
    s1 = pd.to_numeric(s, errors="coerce")
    if s1.isna().mean() > 0.5:
        s_alt = (s.astype(str)
                   .str.replace(r"\s", "", regex=True)
                   .str.replace(".", "", regex=False)
                   .str.replace(",", ".", regex=False))
        s1 = pd.to_numeric(s_alt, errors="coerce")
    return s1

def _fmt_usd(x: float) -> str:
    try:
        if abs(x) >= 1e9:  return f"${x/1e9:,.1f}B"
        if abs(x) >= 1e6:  return f"${x/1e6:,.1f}M"
        if abs(x) >= 1e3:  return f"${x/1e3:,.1f}K"
        return f"${x:,.0f}"
    except Exception:
        return "$0"

def _prep_history(hist: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns, ensure numerics, derive Price USD if missing, ffill/bfill per asset."""
    h = hist.copy()
    if "Date" not in h.columns:
        if "date" in h.columns:
            h["Date"] = pd.to_datetime(h["date"])
        else:
            h["Date"] = pd.to_datetime(dict(year=h["Year"], month=h["Month"], day=1), errors="coerce")

    h = h.rename(columns=lambda c: str(c).strip())
    alias = {
        "Price": "Price USD", "Price_USD": "Price USD",
        "USD": "USD Value",   "USD_Value": "USD Value",
        "Holdings": "Holdings (Unit)", "Units": "Holdings (Unit)",
    }
    for k, v in alias.items():
        if k in h.columns and v not in h.columns:
            h.rename(columns={k: v}, inplace=True)

    for col in ["Holdings (Unit)", "USD Value", "Price USD"]:
        if col not in h.columns:
            h[col] = np.nan

    h["Holdings (Unit)"] = _coerce_num(h["Holdings (Unit)"])
    h["USD Value"]       = _coerce_num(h["USD Value"])
    h["Price USD"]       = _coerce_num(h["Price USD"])

    need_p = h["Price USD"].isna()
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = h["USD Value"] / h["Holdings (Unit)"]
    h.loc[need_p, "Price USD"] = implied[need_p]

    need_usd = h["USD Value"].isna() & h["Price USD"].notna() & h["Holdings (Unit)"].notna()
    h.loc[need_usd, "USD Value"] = h.loc[need_usd, "Price USD"] * h.loc[need_usd, "Holdings (Unit)"]

    h = h.sort_values(["Crypto Asset", "Date"]).reset_index(drop=True)
    h["Price USD"] = h["Price USD"].groupby(h["Crypto Asset"], sort=False).transform(lambda s: s.ffill().bfill())
    return h

def _decompose_asset(g: pd.DataFrame) -> pd.DataFrame:
    """Exact ΔUSD decomposition per asset:
       ΔUSD = units_prev * Δprice  +  price_curr * Δunits
    """
    g = g.sort_values("Date").copy()
    g["units_prev"] = g["Holdings (Unit)"].shift()
    g["price_prev"] = g["Price USD"].shift()
    g["d_usd"] = g["USD Value"].diff()
    g["price_effect"] = (g["Price USD"] - g["price_prev"]) * g["units_prev"]
    g["units_effect"] = (g["Holdings (Unit)"] - g["units_prev"]) * g["Price USD"]
    g = g.dropna(subset=["d_usd", "price_effect", "units_effect"])
    return g

def render_flow_decomposition(df_hist_filtered: pd.DataFrame):
    """
    Render 'Flow & Decomposition (Price vs Accumulation)' using the ALREADY-filtered historic df.
    Expect df_hist_filtered to respect your apply_filters_historic (assets + time).
    """

    if df_hist_filtered.empty:
        st.info("No historic data for the current filters.")
        return

    hist = _prep_history(df_hist_filtered)

    # --- add prior month baseline so the first month (e.g. January) has deltas ---
    start_m = pd.to_datetime(hist["Date"].min()).to_period("M").to_timestamp()
    prev_m  = (start_m - pd.offsets.MonthBegin(1))

    assets_scope = hist["Crypto Asset"].dropna().unique().tolist()
    base_hist = st.session_state.get("historic_df")

    if isinstance(base_hist, pd.DataFrame) and not base_hist.empty:
        # ensure base_hist has month-start Date for comparison
        bh = base_hist.copy()
        bh["Date"] = pd.to_datetime(bh["Date"]).dt.to_period("M").dt.to_timestamp()
        base_rows = bh[(bh["Crypto Asset"].isin(assets_scope)) & (bh["Date"] == prev_m)]
        hist_decomp = pd.concat([base_rows, hist], ignore_index=True)
        hist_decomp = _prep_history(hist_decomp)  # <-- ensure base month has Price USD, numerics, etc.
    else:
        hist_decomp = hist.copy()

    with st.container(border=False):
        st.markdown("### Flow & Decomposition (Price vs Accumulation)", help="Shows whether growth came from new units or price beta by splitting monthly USD Delta into “Delta Price on prior units” vs “Delta Units at current price”.")
        st.markdown("")

        c1, c2 = st.columns([1, 1])

        # Single-asset toggle (aggregated vs one asset)
        view_mode = c1.segmented_control(
            "View",
            options=["Aggregated (selected assets)", "Single asset"],
            default="Aggregated (selected assets)",
            help="Aggregate sums across selected assets or inspect a single asset.",
            label_visibility="collapsed"
        )
        if view_mode == "Single asset":
            assets_in_scope = sorted(hist["Crypto Asset"].dropna().unique().tolist())
            asset_pick = c2.pills("Asset", assets_in_scope, default=["BTC"], label_visibility="collapsed")
        else:
            asset_pick = None

        decomp = (
            hist_decomp.groupby("Crypto Asset", group_keys=True)
                    .apply(_decompose_asset)
                    .reset_index(drop=True)
        )

        # keep only months within the filtered range (drop the injected prior month from the display)
        decomp["Date"] = pd.to_datetime(decomp["Date"]).dt.to_period("M").dt.to_timestamp()
        decomp = decomp[decomp["Date"] >= start_m]

        if decomp.empty:
            st.info("Not enough monthly history to compute flows for the current selection.")
            return

        if asset_pick:
            view = decomp[decomp["Crypto Asset"] == asset_pick].copy()
            bar_color_price = "#8892a6"
            bar_color_units = COLORS.get(asset_pick, "#43d1a0")
        else:
            view = (decomp.groupby("Date")[["d_usd", "price_effect", "units_effect"]]
                          .sum()
                          .reset_index())
            bar_color_price = "#8892a6"
            bar_color_units = "#43d1a0"

        view["Date"] = pd.to_datetime(view["Date"]).dt.to_period("M").dt.to_timestamp()

        start_m = pd.to_datetime(df_hist_filtered["Date"].min()).to_period("M").to_timestamp()
        end_m   = pd.to_datetime(df_hist_filtered["Date"].max()).to_period("M").to_timestamp()
        full_months = pd.date_range(start=start_m, end=end_m, freq="MS")  # Month Start

        view = (view.set_index("Date")
                    .reindex(full_months, fill_value=0.0)
                    .rename_axis("Date")
                    .reset_index())

        st.markdown("")

        # KPIs (last month)
        last = view.sort_values("Date").tail(1)
        d_usd = float(last["d_usd"].iloc[0]) if not last.empty else 0.0
        pe    = float(last["price_effect"].iloc[0]) if not last.empty else 0.0
        ue    = float(last["units_effect"].iloc[0]) if not last.empty else 0.0

        k1, k2, k3 = st.columns(3)
        with k1:
            with st.container(border=True):
                st.metric("ΔUSD (last month)", _fmt_usd(d_usd))
        with k2:
            with st.container(border=True):
                st.metric("Price contribution", _fmt_usd(pe),
                  help="Effect from price changing on prior units.")
        with k3:
            with st.container(border=True):
                st.metric("Units contribution", _fmt_usd(ue),
                  help="Effect from accumulation/reduction of units at current price.")

        # Chart
        fig = go.Figure()
        view["price_hover"] = view["price_effect"].apply(_fmt_usd)
        view["units_hover"] = view["units_effect"].apply(_fmt_usd)

        fig.add_bar(
            name="Price effect",
            x=view["Date"],
            y=view["price_effect"],
            marker_color=bar_color_price,
            hovertext=view["price_hover"],  # pass formatted strings
            hovertemplate="Date: %{x|%b %Y}<br>Price: %{hovertext}<extra></extra>",
        )

        fig.add_bar(
            name="Units effect",
            x=view["Date"],
            y=view["units_effect"],
            marker_color=bar_color_units,
            hovertext=view["units_hover"],
            hovertemplate="Date: %{x|%b %Y}<br>Units: %{hovertext}<extra></extra>",
        )

        fig.update_layout(
            barmode="relative",
            height=360,
            margin=dict(l=40, r=20, t=10, b=30),
            xaxis=dict(title=None, tickformat="%b %Y", fixedrange=True),
            yaxis=dict(title="ΔUSD", tickprefix="$", fixedrange=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            hoverlabel=dict(align="left"),
        )
        # Watermark
        fig.add_annotation(
            text="cryptotreasurytracker.xyz",
            x=0.5, y=0.9,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=30, color="white"),
            opacity=0.3,
            xanchor="center",
            yanchor="top",
        )
        render_plotly(fig, filename=f"flows_{asset_pick or 'agg'}".lower(), extra_config={"scrollZoom": False})

        st.caption("Note: Decomposition uses **asset-level monthly aggregates**; it respects the *filters (assets + time)*. "
                   "Holder Type and Country filters are not applied unless history exists at that granularity.")
        

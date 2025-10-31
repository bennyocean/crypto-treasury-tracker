import streamlit as st
import numpy as np
import pandas as pd
from streamlit.components.v1 import html as st_html

import plotly.graph_objects as go
from modules.ui import COLORS, render_plotly


WATERMARK_TEXT="cryptotreasurytracker.xyz"


def show_entity_dialog(
    *,
    # core identity
    name: str,
    ticker: str,
    country: str,
    flag: str,
    etype: str,
    etype_badge_uri: str,

    # clicked asset context
    asset: str,
    arank,            # int | None
    grank,            # int | None

    # single-asset metrics (used when it's not multi-asset)
    holdings_unit=None,
    pct_supply=None,
    usd_value=np.nan,
    market_cap=np.nan,
    mnav=np.nan,
    premium=np.nan,
    ttmcr=np.nan,

    # company meta
    sector: str = "-",
    industry: str = "-",
    is_datco: bool = False,
    about_text: str = "-",
    website: str = "-",

    # new pipeline fields
    status: str = "Active",
    target_usd: str = "–",
    target_units: str = "–",
    source_url: str = "",

    # cost/price (single-asset fallbacks)
    current_price=np.nan,
    avg_cost_per_unit=np.nan,

    # ---------- Multi-asset ----------
    rows_df: pd.DataFrame | None = None,      # all rows for this entity (across assets)
    token_logo_map: dict | None = None,       # {"BTC": "data:image/png;base64,...", ...}
    supply_caps: dict | None = None,          # {"BTC": 21_000_000, ...}

    events_enriched: pd.DataFrame | None = None,
    holdings_timeseries: pd.DataFrame | None = None,

    agg_global_rank: int | None = None,       # aggregated global rank if you compute it
    per_asset_ranks: dict | None = None,      # {"BTC": 4, "ETH": 7, ...}
    nav_list: list[tuple[str, str]] | None = None,   # [(entity, asset), ...]
    nav_index: int | None = None,                    # current index in nav_list
    nav_session_key: str | None = None,    
):
    # ---------- helpers ----------
    def _usd(x):
        try:
            v = float(x)
            if np.isnan(v): return "-"
            ax = abs(v)
            if ax >= 1e12:  return f"${v/1e12:.2f}T"
            if ax >= 1e9:   return f"${v/1e9:.2f}B"
            if ax >= 1e6:   return f"${v/1e6:.2f}M"
            if ax >= 1e3:   return f"${v/1e3:.2f}K"
            return f"${v:,.0f}"
        except Exception:
            return "-"

    def _usd_full(x):
        try:
            v = float(x)
            if np.isnan(v): 
                return "-"
            # no decimals if it’s an integer; else 2 decimals
            return f"${int(round(v)):,}" if abs(v - round(v)) < 1e-9 else f"${v:,.2f}"
        except Exception:
            return "-"

    def _num(x):
        try:
            v = float(x)
            if np.isnan(v): return "-"
            return f"{v:,.0f}".replace(",", " ")
        except Exception:
            return "-"

    def _x2(x):
        try:
            v = float(x)
            if np.isnan(v): return "-"
            return f"{v:.2f}×"
        except Exception:
            return "-"

    def _pct(x):
        try:
            v = float(x)
            if np.isnan(v): return "-"
            return f"{v:.2f}%"
        except Exception:
            return "-"

    def _pl_pct(curr, cost):
        try:
            curr = float(curr); cost = float(cost)
            if cost <= 0 or np.isnan(curr) or np.isnan(cost):
                return None
            return (curr - cost) / cost * 100.0
        except Exception:
            return None

    # --- Entity Type palette ---
    _type_palette = {
        "Public Company": (123, 197, 237),  # blue
        "Private Company": (247, 89, 176), # rose
        "DAO": (233, 242, 111),              # amber
        "Non-Profit Organization": (128, 217, 183),        # green
        "Government": (247, 198, 148),      # slate
        "Other": (222, 217, 217),           # white
    }

    def _best_text_on_rgb(rgb):
        r, g, b = [c/255.0 for c in rgb]
        def _lin(c): return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4
        L = 0.2126*_lin(r) + 0.7152*_lin(g) + 0.0722*_lin(b)
        # return white text if contrast vs bg is better
        return (255,255,255) if (1.05/(L+0.05) >= (L+0.05)/0.05) else (0,0,0)

    # Outlined chip (transparent bg)
    def _chip_outline(text: str) -> str:
        t = str(text)
        return (
            f"<span style='display:inline-block;padding:6px 12px;border-radius:999px;"
            f"border:1px solid rgba(255,255,255,0.6);color:rgba(255,255,255,0.95);"
            f"font-weight:600;font-size:0.90rem;margin-right:8px;white-space:nowrap;'>"
            f"{t}</span>"
        )

    # Solid chip (for Entity Type) using palette color
    def _chip_solid(text: str, rgb: tuple[int,int,int]) -> str:
        tr, tg, tb = _best_text_on_rgb(rgb)
        r, g, b = rgb
        return (
            f"<span style='display:inline-block;padding:6px 12px;border-radius:999px;"
            f"background: rgb({r},{g},{b}); border:1px solid rgb({r},{g},{b});"
            f"color: rgb({tr},{tg},{tb}); font-weight:700; font-size:0.90rem;"
            f"margin-right:8px; white-space:nowrap;'>{text}</span>"
        )

    def _render_rank_card(label: str, rank: int | None):
        # Build pill
        if isinstance(rank, int) and rank in (1, 2, 3):
            COLORS = {1: "#FFcc00", 2: "#d8d8d8", 3: "#b9722d"}  # gold, silver, bronze
            pill_html = (
                f'<span style="background:{COLORS[rank]};color:#000;'
                f'padding:2px 10px;border-radius:8px;display:inline-block;'
                f'min-width:56px;text-align:center;">#{rank}</span>'
            )
        elif isinstance(rank, int):
            # Plain number, no pill/border
            pill_html = (
                f'<span style="color:#fff;font-weight:700;'
                'font-size:1.6rem;font-family: sans-serif;">'
                f'#{rank}</span>'
            )
        else:
            pill_html = '<span style="opacity:.6">-</span>'

        # Card (transparent bg; same border as your other HTML “cards”)
        html_block = f"""
        <div style="border:1px solid rgba(255,255,255,0.15);
                    border-radius:12px;
                    padding:12px 14px;
                    background:transparent;
                    font-family: sans-serif;">
        <div style="font-size:0.85rem;color:#fff;margin-bottom:6px;">{label}</div>
        <div style="font-size:1.6rem;font-weight:700;line-height:1;">{pill_html}</div>
        </div>
        """
        st_html(html_block, height=90)


    # ---------- multi-asset detect + aggregates ----------
    is_multi = False
    per_asset_rows = pd.DataFrame()
    total_nav = usd_value
    mcap_use  = market_cap

    if rows_df is not None and isinstance(rows_df, pd.DataFrame) and not rows_df.empty:
        try:
            rows_df = rows_df[rows_df["Entity Name"] == name]
        except Exception:
            pass

        per_asset_rows = rows_df.copy()
        try:
            per_asset_rows = per_asset_rows[per_asset_rows["USD Value"] > 0]
        except Exception:
            pass

        if not per_asset_rows.empty:
            is_multi = per_asset_rows["Crypto Asset"].nunique() > 1

            # Use the company's market cap (same across assets). Fallback to first non-NaN.
            if per_asset_rows["Market Cap"].notna().any():
                mcap_use = float(per_asset_rows["Market Cap"].dropna().iloc[0])
            else:
                mcap_use = float("nan")

            # Aggregate NAV across assets
            if per_asset_rows["USD Value"].notna().any():
                total_nav = float(per_asset_rows["USD Value"].fillna(0).sum())
            else:
                total_nav = float("nan")

            # Build per-asset computed columns
            def _pct_supply_row(r):
                try:
                    caps = supply_caps or {}
                    cap = float(caps.get(str(r["Crypto Asset"]), 0))
                    if cap <= 0: return np.nan
                    return float(r["Holdings (Unit)"]) / cap * 100.0
                except Exception:
                    return np.nan

            per_asset_rows = per_asset_rows.assign(
                pct_supply_calc = per_asset_rows["% of Supply"]
                                    if "% of Supply" in per_asset_rows.columns
                                    else per_asset_rows.apply(_pct_supply_row, axis=1),
            )

    # Aggregate mNAV/TTMCR/Premium if multi-asset
    if is_multi:
        mnav_agg    = (mcap_use / total_nav) if (mcap_use and total_nav and total_nav > 0) else float("nan")
        premium_agg = ((mnav_agg - 1.0) * 100.0) if (mnav_agg and not np.isnan(mnav_agg)) else float("nan")
        ttmcr_agg   = (total_nav / mcap_use * 100.0) if (mcap_use and mcap_use > 0) else float("nan")
    else:
        mnav_agg, premium_agg, ttmcr_agg = mnav, premium, ttmcr

    pl_pct = _pl_pct(current_price, avg_cost_per_unit)

    # Transparent chips w/ white text & white border (looks good on dark backgrounds)
    CHIP_CSS = """
    <style>
      .row-inline { display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin-top:8px; }
      .chip-outline {
        display:inline-flex; align-items:center; padding:4px 10px; border-radius:9999px;
        font-size:12px; font-weight:700; border:1px solid rgba(255,255,255,0.55);
        color:#fff; background:transparent;
      }
      .logos-inline img { height:18px; width:18px; display:inline-block; margin-right:8px; vertical-align:middle; }
      .etype-line img.badge { vertical-align:middle; height:22px; }
      .etype-line .chip-outline { margin-left:8px; } /* DATCO chip right to entity type */
    </style>
    """

    @st.dialog("Entity details", width="large")
    def _details_dialog():
        st.markdown(CHIP_CSS, unsafe_allow_html=True)

        # ===== Header =====
        colL, colR = st.columns([0.8, 0.2])

        with colL:


            subcolL, subcolR = st.columns([0.8, 0.2])
            with subcolL:

                # 1) Name + Ticker (same line)
                st.markdown(f"### {flag} {name}  `{ticker}`")

                # 2) Metadata chips row: Entity type, Country, Sector, Industry, DATCO
                chips_html = ""
                # Entity Type as colored chip (palette)
                etype_key = str(etype) if etype else "Other"
                etype_rgb = _type_palette.get(etype_key, _type_palette["Other"])
                chips_html += _chip_solid(etype_key, etype_rgb)
                # country with flag inside chip to avoid duplicate flag above
                chips_html += _chip_outline(f"{country}" if country else "Country: -")
                # sector / industry
                chips_html += _chip_outline(f"Sector: {sector if sector else '–'}")
                chips_html += _chip_outline(f"Industry: {industry if industry else '–'}")
                # DATCO indicator (outlined)
                if is_datco:
                    chips_html += _chip_outline("DAT Wrapper")
                st.markdown(chips_html, unsafe_allow_html=True)

                # --- Pipeline status & targets row (right under the chips) ---
                _status_norm = str(status).strip().lower()
                _status_pill = {
                    "active": ("Active", "#2ecc71"),
                    "pending_funded": ("Funded", "#f1c40f"),
                    "pending_announced": ("Announced", "#c0eef1"),
                    "inactive": ("Inactive", "#f12e2e"),
                }.get(_status_norm, ("Active", "#2ecc71"))

                _status_lbl, _status_color = _status_pill



            with subcolR:

                # === Prev / Next navigation ===
                if isinstance(nav_list, list) and isinstance(nav_index, int) and nav_session_key:
                    nav_c1, nav_c3 = st.columns(2)
                    with nav_c1:
                        if st.button("‹ Prev", key="dlg_prev"):
                            if nav_index > 0:
                                st.session_state[nav_session_key]["idx"] = nav_index - 1
                            else:
                                # wrap to end (or disable; your choice)
                                st.session_state[nav_session_key]["idx"] = len(nav_list) - 1
                            st.rerun(scope="app")
                    #with nav_c2:
                     #   st.write("")  # spacer; keep centered title feel
                    with nav_c3:
                        if st.button("Next ›", key="dlg_next"):
                            if nav_index < len(nav_list) - 1:
                                st.session_state[nav_session_key]["idx"] = nav_index + 1
                            else:
                                # wrap to start (or disable; your choice)
                                st.session_state[nav_session_key]["idx"] = 0
                            st.rerun(scope="app")
                            
            # 3) Assets held row: logos + per-asset rank chips (best rank first)
            assets_row_html = ""
            held_assets = []
            if isinstance(rows_df, pd.DataFrame) and not rows_df.empty:
                held_assets = rows_df["Crypto Asset"].astype(str).unique().tolist()
            elif asset and isinstance(asset, str):
                held_assets = [asset]

            if held_assets:
                # Pair each asset with its rank and a sort key (None ranks go last)
                pairs = []
                for a in held_assets:
                    rk = per_asset_ranks.get(a) if isinstance(per_asset_ranks, dict) else None
                    sort_key = rk if isinstance(rk, (int, float)) else 10**9  # None -> bottom
                    pairs.append((a, rk, sort_key))

                assets_sorted = [ (a, rk) for (a, rk, _) in sorted(pairs, key=lambda t: t[2]) ]

                row_html = ""
                for a, rk in assets_sorted:
                    # logo first
                    if token_logo_map and token_logo_map.get(a):
                        row_html += (
                            f"<img src='{token_logo_map[a]}' alt='{a}' title='{a}' "
                            f"style='height:24px;margin-right:8px;border-radius:4px;vertical-align:middle;'>"
                        )
                    # then outlined rank chip
                    row_html += _chip_outline(f"{a} #{rk if rk else '-'}")

                st.markdown(row_html, unsafe_allow_html=True)


        with colR:
            if is_multi and agg_global_rank:
                _render_rank_card("Global Rank (Aggregate)", agg_global_rank)
            else:
                _render_rank_card("Global Rank", grank if isinstance(grank, int) else None)


        # ===== About (full width) =====
        about_clean = (about_text or "").strip()
        if not about_clean:
            about_clean = (
                "Company description coming soon. This section will summarize the business model, "
                "the role of crypto reserves, and relevant disclosures."
            )
        st.markdown(about_clean)

        # Website (if present)
        if website and isinstance(website, str) and website not in ("-", ""):
            st.markdown(f"[{website}]({website})")



        st.divider()

        # ===== KPIs =====
        st.markdown("#### KPIs")
        if is_multi and not per_asset_rows.empty:
            k1,k2,k3,k4 = st.columns(4)
            with k1: 
                with st.container(border=True):
                    st.metric("Total Crypto-NAV", _usd(total_nav))
            with k2: 
                with st.container(border=True):
                    st.metric("TTMCR (Aggregate)", _pct(ttmcr_agg))
            with k3: 
                with st.container(border=True):
                    st.metric("mNAV (Aggregate)", _x2(mnav_agg))
            with k4: 
                with st.container(border=True):
                    n_assets = int(per_asset_rows["Crypto Asset"].nunique())
                    st.metric("Assets Held", f"{n_assets}")

            # Per-asset breakdown
            st.markdown("##### Per-asset breakdown")
            tbl = per_asset_rows[[
                "Crypto Asset", "Holdings (Unit)", "pct_supply_calc", "USD Value"
            ]].rename(columns={
                "Crypto Asset": "Asset",
                "Holdings (Unit)": "Holdings",
                "pct_supply_calc": "% of Supply",
                "USD Value": "Crypto-NAV",
            }).copy()
            tbl["Holdings"]    = tbl["Holdings"].map(_num)
            tbl["% of Supply"] = tbl["% of Supply"].map(_pct)
            tbl["Crypto-NAV"]  = tbl["Crypto-NAV"].map(_usd)
            st.dataframe(tbl, width="stretch", hide_index=True)
        else:
            k1,k2,k3,k4 = st.columns(4)
            with k1:
                with st.container(border=True):
                    st.metric(f"{asset} Holdings", _num(holdings_unit))
            with k2: 
                with st.container(border=True):
                    st.metric(f"% of {asset} Supply", _pct(pct_supply))
            with k3: 
                with st.container(border=True):
                    st.metric(f"{asset}-NAV", _usd(usd_value))
            with k4: 
                with st.container(border=True):
                    st.metric("TTMCR", _pct(ttmcr))

        st.divider()

        # ===== Stock & Financials =====
        st.markdown("#### Stock & Financials")
        st.write(f"{name} is trading under the ticker `{ticker}`.")
        f1,f2,f3,f4,f5,f6 = st.columns(6)
        with f1:
            with st.container(border=True):
                st.metric("Market Cap — Basic", _usd(mcap_use))
        with f2:
            with st.container(border=True):
                st.metric("Market Cap — Diluted", "-")
        with f3:
            with st.container(border=True):
                st.metric("Enterprise Value", "-")
        with f4:
            with st.container(border=True):
                st.metric("Premium", _pct(premium_agg))
        with f5:
            with st.container(border=True):
                st.metric("mNAV — Basic", _x2(mnav_agg))
        with f6:
            with st.container(border=True):
                st.metric("mNAV — Diluted", "-")

        st.divider()

        # ===== Funding & Cost Basis =====
        st.markdown("#### Funding & Cost Basis")
        if is_multi and not per_asset_rows.empty:
            c1, c2, c3 = st.columns(3)
            with c1:
                with st.container(border=True):
                    st.metric(f"Avg Cost / {asset}", _usd(avg_cost_per_unit) if avg_cost_per_unit==avg_cost_per_unit else "-")
            with c2:
                with st.container(border=True):
                    pl = _pl_pct(current_price, avg_cost_per_unit)
                    st.metric("Unrealized P/L", (f"{pl:.2f}%" if pl is not None else "-"))
            with c3:
                with st.container(border=True):
                    st.metric(f"Current {asset} Price", f"${current_price}" if current_price==current_price else "-")

            st.markdown("##### Per-asset cost basis (coming soon)")
            _cost_tbl = per_asset_rows[["Crypto Asset"]].copy()
            _cost_tbl["Avg Cost"]      = "-"
            _cost_tbl["Current Price"] = "-"
            _cost_tbl["Unrealized P/L"]= "-"
            _cost_tbl = _cost_tbl.rename(columns={"Crypto Asset":"Asset"})
            st.dataframe(_cost_tbl, width="stretch")

            st.markdown("##### Funding Overview (coming soon)")
            st.info("Table coming soon")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                with st.container(border=True):
                    st.metric(f"Avg Cost / {asset}", _usd(avg_cost_per_unit) if avg_cost_per_unit==avg_cost_per_unit else "-")
            with c2:
                with st.container(border=True):
                    st.metric("Unrealized P/L", (f"{pl_pct:.2f}%" if pl_pct is not None else "-"))
            with c3:
                with st.container(border=True):
                    st.metric(f"Current {asset} Price", f"${current_price}" if current_price==current_price else "-")

            st.markdown("##### Funding Overview (coming soon)")
            st.info("Table coming soon")

        st.divider()

        # ===== Charts Holdings Over Time =====
        st.markdown(f"#### {name} — Holdings Over Time")

        if isinstance(holdings_timeseries, pd.DataFrame) and not holdings_timeseries.empty:
            ts = holdings_timeseries.copy()

            # One point per calendar day for the line
            ts["Date"] = pd.to_datetime(ts["Date"], errors="coerce").dt.normalize()
            ts = ts.dropna(subset=["Date", "Units"]).sort_values("Date")
            ts = ts.groupby("Date", as_index=False)["Units"].last()

            # Extend to the current month for a flat tail and cap x axis to this month only
            now_utc = pd.Timestamp.utcnow()

            # cap the x axis to today instead of next month
            today_curr = pd.Timestamp.utcnow().tz_localize(None).normalize()
            last_date  = ts["Date"].max()

            last_units = float(ts.loc[ts["Date"].idxmax(), "Units"])
            if last_date < today_curr:
                ts = pd.concat(
                    [ts, pd.DataFrame({"Date": [today_curr], "Units": [last_units]})],
                    ignore_index=True,
                )

            color = COLORS.get(asset, "#1f77b4")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts["Date"],
                y=ts["Units"],
                mode="lines",
                line=dict(color=color, width=3),
                #hovertemplate="skip",
                name="Balance",
            ))

            # Event bubbles aggregated per day with detailed hover
            if isinstance(events_enriched, pd.DataFrame) and not events_enriched.empty:
                ev = events_enriched.copy()
                ev["event_date"] = pd.to_datetime(ev["event_date"], errors="coerce")
                ev = ev.dropna(subset=["event_date"]).sort_values(["event_date", "event_id"])

                # Effective change and per row description
                ev["chg"] = pd.to_numeric(ev.get("units_effective"), errors="coerce")
                ev["chg"] = ev["chg"].fillna(pd.to_numeric(ev.get("units_delta"), errors="coerce")).fillna(0.0)
                ev["desc"] = [
                    f"{str(et).title()} {val:+,.0f} {asset}"
                    for et, val in zip(ev.get("event_type", ""), ev["chg"])
                ]

                # Day end balance for marker Y
                if "cum_units" in ev.columns and ev["cum_units"].notna().any():
                    day_balance = ev.groupby(ev["event_date"].dt.normalize())["cum_units"].last()
                else:
                    ev["_run"] = ev["chg"].cumsum()
                    day_balance = ev.groupby(ev["event_date"].dt.normalize())["_run"].last()

                # abbreviated net-change labels on bubbles
                def _abbr(n):
                    n = float(n)
                    s = "+" if n > 0 else ""
                    a = abs(n)
                    if a >= 1e9: return f"{s}{a/1e9:.1f}B"
                    if a >= 1e6: return f"{s}{a/1e6:.1f}M"
                    if a >= 1e3: return f"{s}{a/1e3:.0f}k"
                    return f"{s}{a:.0f}"
            
                # format per-event lines as 'Buy: +... ASSET' with bold numbers
                def _fmt_line(et, val):
                    return f"{str(et).title()}: <b>{val:+,.0f} {asset}</b>"

                ev["desc_fmt"] = [_fmt_line(et, v) for et, v in zip(ev.get("event_type", ""), ev["chg"])]

                g = ev.groupby(ev["event_date"].dt.normalize()).agg(
                    sum_chg=("chg", "sum"),
                    hover_list=("desc_fmt", list),
                ).reset_index(names="event_day")

                g["balance"] = g["event_day"].map(day_balance)
                g["size"] = (np.log10(g["sum_chg"].abs() + 1.0) * 2 + 2).clip(6, 18)
                g["hover_lines"] = g["hover_list"].map(lambda lst: "<br>".join(lst))
                g["label"] = g["sum_chg"].map(_abbr)  # keep your abbreviated marker labels
                g["textpos"] = np.where(np.arange(len(g)) % 2 == 0, "top center", "bottom center")

                # full hover text: Balance first, then per-event lines
                g["hover_text_full"] = [
                    f"Balance: <b>{bal:,.0f} {asset}</b><br>{lines}"
                    for bal, lines in zip(g["balance"], g["hover_lines"])
                ]

                # customdata: [balance, lines_html]
                customdata = np.column_stack([g["balance"].astype(float).values, g["hover_lines"].values])

                fig.add_trace(go.Scatter(
                    x=g["event_day"],
                    y=g["balance"],
                    mode="markers+text",
                    marker=dict(size=g["size"], color=color, opacity=0.85, line=dict(width=1, color=color)),
                    #text=g["label"],
                    textposition=g["textpos"],
                    textfont=dict(size=10),
                    customdata=customdata,
                    hovertemplate=(
                        "<b>%{x|%b %d, %Y}</b><br>"
                        f"Balance: <b>%{{customdata[0]:,.0f}} {asset}</b><br>"
                        "%{customdata[1]}"
                        "<extra></extra>"
                    ),
                    name="Events",
                    cliponaxis=False,   # <-- add this
                ))

            # Dynamic monthly tick spacing and clamp to the current month
            x_min = ts["Date"].min()
            x_max = today_curr
            x_min = x_min - pd.Timedelta(days=5)

            months_span = (x_max.year - x_min.year) * 12 + (x_max.month - x_min.month) + 1

            if months_span <= 12:
                dtick = "M1"   # every month
            elif months_span <= 36:
                dtick = "M2"   # every 2 months
            else:
                dtick = "M3"   # every 3 months

            # --- Y-axis padding for single/flat series ---
            y_series = ts["Units"].astype(float)
            if "g" in locals() and isinstance(g, pd.DataFrame) and not g.empty and "balance" in g:
                y_series = pd.concat([y_series, g["balance"].astype(float)], ignore_index=True)

            ymin = float(np.nanmin(y_series)) if len(y_series) else np.nan
            ymax = float(np.nanmax(y_series)) if len(y_series) else np.nan

            y_range = None
            if np.isfinite(ymin) and np.isfinite(ymax):
                if np.isclose(ymax, ymin):
                    pad = max(1.0, abs(ymax) * 0.02)   # ~2% or at least 1 unit
                else:
                    pad = max((ymax - ymin) * 0.10, 1.0)  # 10% padding or at least 1 unit
                y_range = [ymin - pad, ymax + pad]

            # apply computed y-range (only if available) and nicer tick format
            if y_range:
                fig.update_yaxes(range=y_range)

            fig.update_layout(
                margin=dict(l=30, r=20, t=20, b=20),
                height=350,
                hovermode="closest",
                template="plotly_white",
                xaxis=dict(
                    type="date",
                    tickformat="%b %Y",
                    hoverformat="%b %d, %Y",  # pretty date in unified hover header
                    ticklabelmode="period",
                    dtick=dtick,
                    range=[x_min, x_max],
                    title="",
                    showspikes=False,
                    fixedrange=True
                ),
                yaxis=dict(title="Units", showspikes=False, tickformat="~s", fixedrange=True), 
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            )

            fig.add_annotation(
                text=WATERMARK_TEXT,
                x=0.5, y=0.5,                      # Center of chart
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=30, color="white"),
                opacity=0.2,                       # Adjust for subtlety
                xanchor="center",
                yanchor="middle",
            )
            render_plotly(fig, filename=f"{name}_{asset}_holdings", extra_config={"scrollZoom": False})
        else:
            st.info("Chart coming soon")

        # ===== Historical Crypto Holdings =====
        st.markdown("#### Historical Crypto Holdings")

        if isinstance(events_enriched, pd.DataFrame) and not events_enriched.empty:
            ev = events_enriched.copy()

            ev["Date"] = pd.to_datetime(ev["event_date"], errors="coerce").dt.strftime("%Y-%m-%d")
            chg = pd.to_numeric(ev.get("units_effective"), errors="coerce")
            chg = chg.fillna(pd.to_numeric(ev.get("units_delta"), errors="coerce"))
            ev["Change_num"] = chg

            bal = pd.to_numeric(ev.get("cum_units"), errors="coerce")
            if bal.isna().all():
                bal = chg.fillna(0.0).cumsum()
            ev["Balance_num"] = bal

            def _fmt_signed(n):
                if pd.isna(n):
                    return "-"
                s = f"{n:,.0f}"
                return f"+{s}" if n > 0 else s

            ev["Balance"] = ev["Balance_num"].map(lambda x: "-" if pd.isna(x) else f"{x:,.0f}")
            ev["Change"] = ev["Change_num"].map(_fmt_signed)
            ev["Total Cost Basis"] = ev.get("usd_value_at_event").map(_usd)
            ev[f"Cost Basis per {asset}"] = ev.get("avg_usd_cost_per_asset").map(_usd_full)

            # Source fields
            ev["Source Type"] = ev.get("_ann_type", "")
            ev["Source Credibility"] = ev.get("_sconf", "")
            ev["Source Link"] = ev.get("_url", "")

            cols = [
                "Date",
                "Balance",
                "Change",
                "Total Cost Basis",
                f"Cost Basis per {asset}",
                "Source Type",
                "Source Credibility",
                "Source Link",
            ]
            out = ev[cols].copy().sort_values("Date", ascending=False)

            # Credibility pill helper
            def _cred_pill(text):
                t = str(text or "").strip().lower()
                bg = "#dedede"
                if t.startswith("high"):
                    bg = "#43d1a0"
                elif t.startswith("med"):
                    bg = "#ffe066"
                elif t.startswith("low"):
                    bg = "#f94144"
                return (
                    f"<span style='display:inline-block;padding:4px 10px;border-radius:9999px;"
                    f"background:{bg};color:#000;font-weight:700;font-size:12px;'>"
                    f"{text if text else '-'}"
                    f"</span>"
                )

            header_html = (
                "<tr>"
                "<th>Date</th>"
                f"<th style='text-align:right'>{asset} Balance</th>"
                "<th style='text-align:right'>Change</th>"
                "<th style='text-align:right'>Total Cost Basis</th>"
                f"<th style='text-align:right'>Cost Basis per {asset}</th>"
                "<th>Source Type</th>"
                "<th>Source Credibility</th>"
                "<th>Source Link</th>"
                "</tr>"
            )

            def _row_html(r):
                ch = r["Change"]
                color = "#43d1a0" if isinstance(ch, str) and ch.startswith("+") else ("#f94144" if isinstance(ch, str) and ch.startswith("-") else "inherit")
                link = r["Source Link"]
                link_html = f"<a href='{link}' target='_blank'>Open</a>" if isinstance(link, str) and link else "-"
                cred_html = _cred_pill(r["Source Credibility"])
                return (
                    "<tr>"
                    f"<td>{r['Date']}</td>"
                    f"<td style='text-align:right'>{r['Balance']}</td>"
                    f"<td style='text-align:right;color:{color};font-weight:600'>{ch}</td>"
                    f"<td style='text-align:right'>{r['Total Cost Basis']}</td>"
                    f"<td style='text-align:right'>{r[f'Cost Basis per {asset}']}</td>"
                    f"<td>{r['Source Type']}</td>"
                    f"<td>{cred_html}</td>"
                    f"<td style='min-width:140px'>{link_html}</td>"
                    "</tr>"
                )

            body_html = "\n".join(_row_html(r) for _, r in out.iterrows())

            st.markdown(
                f"""
                <div style="overflow-x:auto">
                <table style="width:100%; border-collapse:collapse">
                    <thead style="position:sticky; top:0; background:#fff; color:#000">{header_html}</thead>
                    <tbody>{body_html}</tbody>
                </table>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Coming soon")


        # ===== Close =====
        if st.button("Close"):
            st.session_state["active_dialog"] = None
            st.session_state["overview_editor_rev"] = st.session_state.get("overview_editor_rev", 0) + 1
            st.rerun(scope="app")

    _details_dialog()

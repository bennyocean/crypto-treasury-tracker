import streamlit as st
import numpy as np
import pandas as pd
from streamlit.components.v1 import html as st_html

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

    # cost/price (single-asset fallbacks)
    current_price=np.nan,
    avg_cost_per_unit=np.nan,

    # ---------- Multi-asset ----------
    rows_df: pd.DataFrame | None = None,      # all rows for this entity (across assets)
    token_logo_map: dict | None = None,       # {"BTC": "data:image/png;base64,...", ...}
    supply_caps: dict | None = None,          # {"BTC": 21_000_000, ...}
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
        "Private Company": (232, 118, 226), # rose
        "DAO": (237, 247, 94),              # amber
        "Foundation": (34, 197, 94),        # green
        "Government": (245, 184, 122),      # slate
        "Other": (250, 250, 250),           # white
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
                    chips_html += _chip_outline("DATCO")
                st.markdown(chips_html, unsafe_allow_html=True)

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

        # ===== Charts placeholders =====
        st.markdown(f"#### {name} — Holdings Over Time")
        st.info("Chart coming soon")
        cA, cB = st.columns(2)
        with cA:
            st.markdown("##### Market Cap History")
            st.info("Chart coming soon")
        with cB:
            st.markdown("##### Crypto-NAV History")
            st.info("Chart coming soon")

        st.divider()

        # ===== Historical crypto holdings =====
        st.markdown("#### Historical Crypto Holdings")
        st.info("Coming soon")

        # ===== Close =====
        if st.button("Close"):
            st.session_state["active_dialog"] = None
            st.session_state["overview_editor_rev"] = st.session_state.get("overview_editor_rev", 0) + 1
            st.rerun(scope="app")

    _details_dialog()

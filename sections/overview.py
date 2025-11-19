import streamlit as st
import pandas as pd
import numpy as np
import html
from zoneinfo import ZoneInfo
import pandas as pd

from urllib.parse import urlencode, quote_plus

from modules.kpi_helpers import render_kpis
from analytics import log_table_render
from modules.ui import render_ticker, btc_b64, eth_b64, sol_b64, sui_b64, ltc_b64, xrp_b64, hype_b64, bnb_b64, doge_b64, ada_b64, avax_b64, ath_b64, bera_b64, bonk_b64, link_b64, core_b64, cro_b64, trump_b64, pump_b64, ton_b64, trx_b64, wlfi_b64, zig_b64, vaulta_b64,fluid_b64,zec_b64
from modules.pdf_helper import _table_pdf_bytes
from modules.emojis import country_emoji_map
from modules.data_loader import read_central_prices_from_sheet, get_events_for_entity_id_asset
from modules.entity_dialog import show_entity_dialog


supply_caps = {
    "BTC": 19_908_153,
    "ETH": 120_707_840,
    "XRP": 59_418_500_720,
    "BNB": 139_287_622,
    "SOL": 540_069_892,
    "DOGE": 150_552_856_383,
    "TRX": 94_679_730_764,
    "ADA": 36_448_472_341,
    "SUI": 3_511_924_479,
    "LTC": 76_198_926,
    "HYPE": 270_772_999,
    "TON": 2_520_529_386,
    "WLFI": 27_255_958_920,
    "PUMP": 354_000_000_000,
    "ATH": 14_234_731_752,
    "BONK": 77_419_592_329_436,
    "AVAX": 426_584_745,
    "CRO": 36_069_453_408,
    "LINK": 696_849_970,
    "BERA": 129_478_858,
    "TRUMP": 199_999_973,
    "ZIG": 1_408_940_795,
    "CORE": 1_015_193_271,
    "VAULTA": 1_599_315_411,
    "FLUID": 77_753_292,
    "ZEC": 16_383_381
}

TYPE_PALETTE = {
    "Public Company": (123, 197, 237),  # blue
    "Private Company": (247, 89, 176), # rose
    "DAO": (233, 242, 111),              # amber
    "Non-Profit Organization": (128, 217, 183),        # green
    "Government": (247, 198, 148),      # slate
    "Other": (222, 217, 217),           # white
}

logo_map = {
    "BTC": f"data:image/png;base64,{btc_b64}",
    "ETH": f"data:image/png;base64,{eth_b64}",
    "XRP": f"data:image/png;base64,{xrp_b64}",
    "BNB": f"data:image/png;base64,{bnb_b64}",
    "SOL": f"data:image/png;base64,{sol_b64}",
    "DOGE": f"data:image/png;base64,{doge_b64}",
    "TRX": f"data:image/png;base64,{trx_b64}",
    "ADA": f"data:image/png;base64,{ada_b64}",
    "SUI": f"data:image/png;base64,{sui_b64}",
    "LTC": f"data:image/png;base64,{ltc_b64}",
    "HYPE": f"data:image/png;base64,{hype_b64}",
    "TON": f"data:image/png;base64,{ton_b64}",
    "WLFI": f"data:image/png;base64,{wlfi_b64}",
    "PUMP": f"data:image/png;base64,{pump_b64}",
    "ATH": f"data:image/png;base64,{ath_b64}",
    "BONK": f"data:image/png;base64,{bonk_b64}",
    "AVAX": f"data:image/png;base64,{avax_b64}",
    "CRO": f"data:image/png;base64,{cro_b64}",
    "LINK": f"data:image/png;base64,{link_b64}",
    "BERA": f"data:image/png;base64,{bera_b64}",
    "TRUMP": f"data:image/png;base64,{trump_b64}",
    "ZIG": f"data:image/png;base64,{zig_b64}",
    "CORE": f"data:image/png;base64,{core_b64}",
    "VAULTA": f"data:image/png;base64,{vaulta_b64}",
    "FLUID": f"data:image/png;base64,{fluid_b64}",
    "ZEC": f"data:image/png;base64,{zec_b64}",

}

def pretty_usd(x):
    if pd.isna(x):
        return "-"
    ax = abs(x)
    if ax >= 1e12:  return f"${x/1e12:.2f}T"
    if ax >= 1e9:  return f"${x/1e9:.2f}B"
    if ax >= 1e6:  return f"${x/1e6:.2f}M"
    if ax >= 1e3:  return f"${x/1e3:.2f}K"
    return f"${x:,.0f}"

def _df_auto_height(n_rows: int, row_px: int = 35) -> int:
    # header ≈ one row + thin borders
    return int((n_rows + 1) * row_px + 3)

# requires: pip install pillow
def _best_text_on(bg_rgb: tuple[int,int,int]) -> tuple[int,int,int]:
    r, g, b = [c/255.0 for c in bg_rgb]
    def _lin(c): return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4
    L = 0.2126*_lin(r) + 0.7152*_lin(g) + 0.0722*_lin(b)
    contrast_white = (1.0 + 0.05) / (L + 0.05)
    contrast_black = (L + 0.05) / 0.05
    return (255,255,255) if contrast_white >= contrast_black else (0,0,0)


def _badge_svg_uri(text: str,
                   bg_rgb: tuple[int,int,int],
                   h: int = 18,
                   pad_x: int = 5,
                   radius: int = 7) -> str:
    """Return a data:image/svg+xml;base64 URI for a rounded, vector 'pill'."""
    import base64, html
    font_size = 12
    est_tw = max(12, int(len(text) * font_size * 0.60))
    w = est_tw + 2 * pad_x

    def _best_text_on(bg_rgb):
        r, g, b = [c/255.0 for c in bg_rgb]
        def _lin(c): return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4
        L = 0.2126*_lin(r) + 0.7152*_lin(g) + 0.0722*_lin(b)
        return (255,255,255) if (1.05/(L+0.05) >= (L+0.05)/0.05) else (0,0,0)

    tr, tg, tb = _best_text_on(bg_rgb)
    r, g, b = bg_rgb
    txt = html.escape(text)

    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"
        f"<rect x='0' y='0' width='{w}' height='{h}' rx='{radius}' ry='{radius}' "
        f"fill='rgb({r},{g},{b})'/>"
        f"<text x='{w/2}' y='{h/2}' dominant-baseline='middle' text-anchor='middle' "
        f"fill='rgb({tr},{tg},{tb})' font-family='-apple-system,system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif' "
        f"font-size='{font_size}' font-weight='600'>{txt}</text>"
        f"</svg>"
    )

    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"



st.session_state.setdefault("active_dialog", None)

def render_overview():

    #st.title("Crypto Treasury Dashboard")

    #st.markdown("")
    render_ticker()
    
    df = st.session_state["data_df"]

    # KPIs
    render_kpis(df, st.session_state.get("kpi_snapshots"))

    st.markdown("")

    #with st.container(border=True):
    #st.markdown("#### Crypto Treasury Leaderboard", help="Ranked view of entities by digital asset treasury holdings.")

    table = df.copy()
    table = table.sort_values("USD Value", ascending=False).reset_index(drop=True)
    table.index = table.index + 1
    table.index.name = "Rank"

    table["Holdings (Unit)"] = table["Holdings (Unit)"].round(0)
    table["USD Value"] = table["USD Value"].round(0)

    table["% of Supply"] = table.apply(
        lambda row: row["Holdings (Unit)"] / supply_caps.get(row["Crypto Asset"], 1) * 100,
        axis=1
    ).round(2)

    cols = list(table.columns)
    if "Holdings (Unit)" in cols and "% of Supply" in cols:
        cols.remove("% of Supply")
        insert_pos = cols.index("Holdings (Unit)") + 1
        cols.insert(insert_pos, "% of Supply")
        table = table[cols]

    # --- all list options ---
    options = ["All Assets", "DAT Wrappers", "Pending"] + sorted(
        table["Crypto Asset"].dropna().unique().tolist()
    )

    list_choice = st.pills(
        "Select Asset List",
        options=options,
        selection_mode="single",
        default="All Assets",
        label_visibility="visible",
        key="tbl_asset_filter",
    )

    # --- normalize fallback ---
    list_choice = list_choice or "All Assets"

    # --- apply selection ---
    # --- apply selection ---
    et_global = st.session_state.get("flt_entity_type", "All")

    # Default
    asset_choice = list_choice

    # 1. DAT Wrappers pill clicked
    if list_choice == "DAT Wrappers":
        if "DAT" in table.columns:
            table = table[pd.to_numeric(table["DAT"], errors="coerce").fillna(0).astype(int) == 1]
        asset_choice = "All Assets"

    # 2. Pending pill clicked
    elif list_choice == "Pending":
        if "status" in table.columns:
            mask = table["status"].astype(str).str.lower().isin(
                ["pending", "pending_announced", "pending_funded"]
            ) | table["status"].astype(str).str.contains("pending", case=False, na=False)
            table = table[mask].copy()
        else:
            st.warning("No status column found in the dataset.")
            table = table.iloc[0:0]
        asset_choice = "Pending"

    # 3. Regular asset pill clicked
    elif list_choice not in ["All Assets"]:
        table = table[table["Crypto Asset"] == list_choice]

    # 4. Global filter active for DAT Wrappers → always enforce DAT == 1,
    #    even when clicking specific assets.
    if et_global == "DAT Wrappers" and "DAT" in table.columns:
        table = table[pd.to_numeric(table["DAT"], errors="coerce").fillna(0).astype(int) == 1]


    # safe downstream usage
    fname_asset = "all" if asset_choice == "All Assets" else asset_choice.lower()

    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)

    # --- search by entity name ---
    with c1:
        name_query = st.text_input(
            "Search Treasury Holder",
            value="",
            placeholder="Type a company name…",
            key="tbl_search",
            help="Filter the list by holder name."
        )

    if name_query:
        table_search = table[table["Entity Name"].astype(str).str.contains(name_query, case=False, na=False)]
    else:
        table_search = table

    len_table = int(table_search.shape[0])
    default_rows = min(50, len_table)      # max 50 oder weniger, wenn weniger vorhanden
    min_rows = 0 if len_table == 0 else 1
    max_rows = len_table

    # Prüfen, ob sich die Asset-Liste geändert hat
    last_asset = st.session_state.get("tbl_last_asset")
    if last_asset != list_choice:
        start_val = default_rows           # neu initialisieren
        st.session_state["tbl_last_asset"] = list_choice
    else:
        # User-Wert behalten, aber sicherstellen, dass er im gültigen Bereich bleibt
        prev = st.session_state.get("tbl_rows", default_rows)
        start_val = max(min_rows, min(int(prev), max_rows))


    with c2:
        if "ui_entity_type" not in st.session_state:
            st.session_state["ui_entity_type"] = st.session_state.get("flt_entity_type", "All")
        sel_et = st.selectbox(
            "Select Holder Type",
            options=st.session_state["opt_entity_types"],
            key="ui_entity_type",
        )
        st.session_state["flt_entity_type"] = sel_et

    with c3:
        if "ui_country" not in st.session_state:
            st.session_state["ui_country"] = st.session_state.get("flt_country", "All")
        sel_co = st.selectbox(
            "Select Country",
            options=st.session_state["opt_countries"],
            key="ui_country",
        )
        st.session_state["flt_country"] = sel_co

    with c4:
        row_count = st.number_input(
            f"Expand Table (Max. {max_rows})",
            min_value=min_rows,
            max_value=max_rows,
            value=start_val,     # safe starting value
            step=1,
            help=(
                "Select number of rows to display, sorted by USD value. "
                "Some treasuries hold more than one crypto asset and are shown separately."
            ),
            key="tbl_rows",
        )

    # apply global filters plus the local asset toggle
    filtered = table_search.copy()

    if asset_choice not in ["All Assets", "Pending"]:
        assets_sel = [asset_choice]
        filtered = filtered[filtered["Crypto Asset"].isin(assets_sel)]
    else:
        assets_sel = st.session_state.get("flt_assets", st.session_state["opt_assets"])

    et = st.session_state.get("flt_entity_type", "All")

    # Apply entity filter only when appropriate
    if et == "DAT Wrappers":
        # Always enforce DAT == 1 when global DAT filter is active
        if "DAT" in filtered.columns:
            filtered = filtered[pd.to_numeric(filtered["DAT"], errors="coerce").fillna(0).astype(int) == 1]
    elif et != "All":
        filtered = filtered[filtered["Entity Type"] == et]




    co = st.session_state.get("flt_country", "All")
    if co != "All":
        filtered = filtered[filtered["Country"] == co]


    # ensure uppercase tickers for matching
    filtered["Ticker"] = filtered["Ticker"].astype(str).str.upper()

    # --- KPI METRICS ---
    c1_kpi, c2_kpi, c3_kpi, c4_kpi, c5_kpi = st.columns(5)

    # Compute aggregates
    total_nav_all = float(df["USD Value"].sum())
    total_nav = filtered["USD Value"].sum(skipna=True)
    total_target = filtered["target_usd"].fillna(0).sum()
    pending_mask = filtered["status"].astype(str).str.contains("pending", case=False, na=False)
    pending_value = filtered.loc[pending_mask, "target_usd"].fillna(0).sum()

    # Adaptive denominator: if all pending, compare to total_target; else, compare to total_nav
    denominator = total_nav if total_nav > 0 else total_target
    pending_share = (pending_value / denominator * 100) if denominator > 0 else 0

    # % of Target covered by real NAV
    coverage = (total_nav / total_nav_all * 100) if total_nav_all > 0 else 0

    # --- Identify top asset by actual holdings, with graceful fallbacks ---
    if "USD Value" in filtered.columns and filtered["USD Value"].sum() > 0:
        metric_col = "USD Value"
        total_val = filtered["USD Value"].sum()
    elif "target_usd" in filtered.columns and filtered["target_usd"].sum() > 0:
        metric_col = "target_usd"
        total_val = filtered["target_usd"].sum()
    elif "target_units" in filtered.columns and filtered["target_units"].sum() > 0:
        metric_col = "target_units"
        total_val = filtered["target_units"].sum()
    else:
        metric_col, total_val = None, 0

    if metric_col and not filtered.empty:
        asset_sums = (
            filtered.groupby("Crypto Asset", as_index=False)[metric_col]
            .sum()
            .sort_values(metric_col, ascending=False)
        )
        if not asset_sums.empty:
            top_asset = asset_sums.iloc[0]["Crypto Asset"]
            top_asset_share = (
                asset_sums.iloc[0][metric_col] / total_val * 100 if total_val > 0 else 0
            )
        else:
            top_asset, top_asset_share = "-", 0
    elif not filtered.empty:
        top_asset, top_asset_share = filtered.iloc[0]["Crypto Asset"], 100
    else:
        top_asset, top_asset_share = "-", 0

    dat_count = (
        filtered.loc[pd.to_numeric(filtered["DAT"], errors="coerce").fillna(0).astype(int) == 1, "Entity Name"]
        .nunique()
    )

    def _fmt_usd(v): return pretty_usd(v)
    def _fmt_pct(v): return f"{v:.1f}%"

    with c1_kpi:
        with st.container(border=True):
            st.metric(
                "Selected Market Value",
                _fmt_usd(total_nav),
                help="Total realized value of selected treasuries in USD."
            )

    with c2_kpi:
        with st.container(border=True):
            st.metric(
                "Treasury Coverage",
                _fmt_pct(coverage),
                help="Share of total known crypto treasury value represented by this selection."
            )

    with c3_kpi:
        with st.container(border=True):
            logo_uri = logo_map.get(top_asset, "")
            if logo_uri:
                st.metric(
                    "Top Asset Dominance",
                    "",
                    help="Percentage of the largest crypto asset within the current selection."
                )
                st.markdown(
                    f"""
                    <div style="
                        display: flex;
                        align-items: left;
                        justify-content: left;
                        gap: 14px;
                        margin-top: -13px;
                        margin-bottom: 50px;
                        height: 10px;
                        line-height: 1;
                    ">
                        <img src="{logo_uri}" width="38" height="38" style="border-radius:0px; vertical-align:middle; margin-top:0px;" />
                        <span style="font-size:2.4rem; font-weight:400; vertical-align:middle;">
                            {_fmt_pct(top_asset_share)}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                #st.markdown("")
            else:
                st.metric(
                    "Top Asset Dominance",
                    f"{top_asset}: {_fmt_pct(top_asset_share)}",
                    help="Percentage of the largest crypto asset within the current selection."
                )
        
    with c4_kpi:
        with st.container(border=True):
            st.metric(
                "\# of DAT Wrappers",
                dat_count,
                help="Number of entities classified as Digital Asset Treasury Vehicles (DATs) included in the selected view."
            )

    with c5_kpi:
        with st.container(border=True):
            st.metric(
                "Treasury Pipeline",
                _fmt_usd(total_target),
                help="Total sum of known acquisition value targets across all holders."
            )

    # Optional 5th KPI (add if space allows)
    # with c5_kpi:
    #     with st.container(border=True):
    #         st.metric(
    #             "Entity Diversification Index",
    #             _fmt_pct(div_index * 100),
    #             help="Inverse of concentration across entities; higher = more diverse."
    #         )


    _status_order = {"active": 0, "pending": 1, "inactive": 2}
    filtered["__status_order"] = (
        filtered["status"].astype(str).str.lower().map(_status_order).fillna(1).astype(int)
    )

    # Ensure numeric for sort keys
    filtered["Holdings (Unit)"] = pd.to_numeric(filtered["Holdings (Unit)"], errors="coerce").fillna(0)
    filtered["USD Value"] = pd.to_numeric(filtered["USD Value"], errors="coerce").fillna(0)

    filtered = filtered.sort_values(
        by=["__status_order", "USD Value", "Holdings (Unit)", "Entity Name"],
        ascending=[True, False, False, False],
        kind="mergesort",
    ).drop(columns="__status_order", errors="ignore")

    sub = filtered.head(row_count).reset_index(drop=True)
    sub["__row_rank"] = np.arange(1, len(sub) + 1)

    sub["__row_rank"] = np.arange(1, len(sub) + 1)


    # --- RANKS on the full dataset (not filtered) ---
    df_all = st.session_state["data_df"].copy()

    _pending = {"pending_funded", "pending_announced"}
    df_all_active = df_all[~df_all["status"].isin(_pending)].copy()

    # Global rank by USD Value
    _global_sorted = df_all_active.sort_values("USD Value", ascending=False).copy()
    _global_sorted["__global_rank"] = np.arange(1, len(_global_sorted) + 1)

    # Per-asset rank by USD Value
    _asset_sorted = df_all_active.sort_values(["Crypto Asset", "USD Value"], ascending=[True, False]).copy()
    _asset_sorted["__asset_rank"] = _asset_sorted.groupby("Crypto Asset").cumcount() + 1


    # Lookup dicts by (Entity Name, Crypto Asset)
    _global_rank_map = {
        (r["Entity Name"], r["Crypto Asset"]): int(r["__global_rank"])
        for _, r in _global_sorted[["Entity Name", "Crypto Asset", "__global_rank"]].iterrows()
    }
    _asset_rank_map = {
        (r["Entity Name"], r["Crypto Asset"]): int(r["__asset_rank"])
        for _, r in _asset_sorted[["Entity Name", "Crypto Asset", "__asset_rank"]].iterrows()
    }
    # Determine display ranks per row, hiding ranks for pending rows
    is_pending_sub = sub["status"].isin(_pending)

    sub["Rank"] = np.where(is_pending_sub, "—", sub["__row_rank"])

    sub["Global Rank"] = [
        _global_rank_map.get((e, a)) if not p else "—"
        for e, a, p in zip(sub["Entity Name"], sub["Crypto Asset"], is_pending_sub)
    ]

    sub["Asset Rank"] = [
        _asset_rank_map.get((e, a)) if not p else "—"
        for e, a, p in zip(sub["Entity Name"], sub["Crypto Asset"], is_pending_sub)
    ]

    # --- Aggregate global rank by total Crypto-NAV across all assets ---
    _df_agg = (
        df_all_active.groupby("Entity Name", as_index=False)["USD Value"]
            .sum()
            .sort_values("USD Value", ascending=False)
            .reset_index(drop=True)
    )

    _df_agg["__agg_global_rank"] = np.arange(1, len(_df_agg) + 1)
    _agg_global_rank_map = dict(zip(_df_agg["Entity Name"], _df_agg["__agg_global_rank"]))

    display = sub.copy()

    def _details_url(row):
        qp = {"entity": row["Entity Name"], "asset": row["Crypto Asset"]}
        return "?" + urlencode(qp, quote_via=quote_plus)

    display["Open"] = display.apply(_details_url, axis=1)

    # Map country column to emoji
    flag_series = (
        display["Country"]
        .astype("string")
        .map(lambda c: country_emoji_map.get(c, "🏳️"))
    )

    # Prepend flag to Entity Name
    display["Entity"] = flag_series.fillna("🏳️") + " " + display["Entity Name"].astype("string")

    # show dashes for missing values
    display["Ticker"] = (display["Ticker"].replace({"": "-"}).astype("string").fillna("-"))

    # mark rows with no market cap or no ticker for display-only fallbacks
    _no_metrics = display["Market Cap"].isna() | display["Ticker"].isna()

    # display-only formatted metrics with dashes when not available
    display["mNAV_disp"]    = np.where(_no_metrics, "-", display["mNAV"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-"))
    display["Premium_disp"] = np.where(_no_metrics, "-", display["Premium"].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "-"))
    display["TTMCR_disp"]   = np.where(_no_metrics, "-", display["TTMCR"].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "-"))

    status_map = {
        "active": "Active",
        "pending_funded": "Pending",
        "pending_announced": "Pending",
        "inactive": "Inactive",
    }
    display["Status"] = display["status"].astype(str).str.lower().map(status_map).fillna("Active")

    display["Target USD"] = display["target_usd"].apply(
        lambda v: pretty_usd(float(v)) if pd.notna(v) and float(v) > 0 else "–"
    )

    # --- compute target vs. holdings progress ---
    holdings_units = pd.to_numeric(display["Holdings (Unit)"], errors="coerce")
    target_units   = pd.to_numeric(display["target_units"], errors="coerce")
    usd_value      = pd.to_numeric(display["USD Value"], errors="coerce")
    target_usd     = pd.to_numeric(display["target_usd"], errors="coerce")

    # base use units if available
    progress = pd.Series(np.nan, index=display.index, dtype="float64")
    has_units = target_units.notna() & (target_units > 0)
    progress.loc[has_units] = (holdings_units.loc[has_units] / target_units.loc[has_units] * 100).clip(0, 100)

    # fallback only when no valid unit target but USD target exists
    has_usd_target = (~has_units) & target_usd.notna() & (target_usd > 0)
    progress.loc[has_usd_target] = (usd_value.loc[has_usd_target] / target_usd.loc[has_usd_target] * 100).clip(0, 100)

    # do NOT fill missing here so cells without any target stay visually empty
    display["Target Units Progress"] = progress.round(0)


    display["Source"] = display["source_url"].fillna("").astype(str)

    display["Token"] = display["Crypto Asset"].map(lambda a: logo_map.get(a, ""))

    display["Market Cap"] = display["Market Cap"].map(pretty_usd)
    display["USD Value"] = display["USD Value"].map(pretty_usd)
    # Add a per-row Open button column

    display = display[[
        "Rank",
        "Status", "Entity", "Ticker", "Entity Type", "Open",
        "Token", "Crypto Asset", "Holdings (Unit)", "Target Units Progress", "% of Supply", "USD Value",
        "Market Cap", "mNAV_disp", "Premium_disp", "TTMCR_disp",
        "Source"
    ]]


    NEUTRAL_POS = "#43d1a0"
    NEUTRAL_NEG = "#f94144"

    def color_mnav(val):
        """Color mNAV based on threshold."""
        try:
            v = float(val)
        except (ValueError, TypeError):
            return ""
        if v > 1:
            return f"color: {NEUTRAL_POS}; font-weight: bold;"
        elif v < 1:
            return f"color: {NEUTRAL_NEG}; font-weight: bold;"

    # Apply styling to the *numeric* mNAV column
    # styled_display = display.style.map(color_mnav, subset=["mNAV_disp"])

    st.markdown(
        """
        <style>
        /* Right-align selected columns in st.dataframe */
        [data-testid="stDataFrame"] td:nth-child(8),
        [data-testid="stDataFrame"] td:nth-child(9),
        [data-testid="stDataFrame"] td:nth-child(10),
        [data-testid="stDataFrame"] td:nth-child(11),
        [data-testid="stDataFrame"] td:nth-child(12) {
            text-align: right !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- Data editor (momentary 'Open' checkbox) ---
    rows = min(row_count, len(display))
    height = _df_auto_height(rows)

    # Always create/overwrite the 'Open' column (so the checkbox never persists across runs)
    display = display.copy()
    display["Open"] = False

    # Editor key that we can rotate to clear sticky selections
    rev = st.session_state.get("overview_editor_rev", 0)

    # Entity type colored pills format
    options_ent_type=["Public Company","Private Company","Government","Non-Profit Organization","DAO","Other"]
    color_ent_type=["#7bc5ed", "#f759b0", "#f7c694", "#80d9b7",  "#eaf26f", "#ded9d9"]

    options_status=["Active","Pending","Inactive"]
    color_status=["#00B894",  "#F1C40F", "#ADB5BD"]

    # Render dashboard table
    edited = st.data_editor(
        display,
        width="stretch",
        height=height,
        hide_index=True,
        disabled=[c for c in display.columns if c != "Open"],  # read-only except checkbox
        column_config={
            "Rank": st.column_config.TextColumn("Rank"),
            "Status": st.column_config.MultiselectColumn("Status", width="small", options=options_status, color=color_status),
            "Token": st.column_config.ImageColumn("Token"),
            "Crypto Asset": st.column_config.TextColumn("Symbol"),
            "Entity Type": st.column_config.MultiselectColumn("Holder Type", options=options_ent_type, color=color_ent_type),
            "Open": st.column_config.CheckboxColumn("Info",help="Click to open details for this entity.",default=False),
            "Holdings (Unit)": st.column_config.NumberColumn("Holdings", format="%d"),
            "Target Units Progress": st.column_config.ProgressColumn("Progress",help="Progress toward unit target or USD target when units unknown",format="%d%%",min_value=0,max_value=100,width="small",),
            "% of Supply": st.column_config.NumberColumn("% of Supply",format="%.2f%%"),
            "Market Cap": st.column_config.TextColumn("Market Cap", width="small"),
            "USD Value": st.column_config.TextColumn("Crypto-NAV", width="small"),
            "mNAV_disp": st.column_config.TextColumn("mNAV", width="small"),
            "Premium_disp": st.column_config.TextColumn("Premium", width="small"),
            "TTMCR_disp": st.column_config.TextColumn("TTMCR", width="small"),
            #"Target USD": st.column_config.TextColumn("Target USD", width="small"),
            "Source": st.column_config.LinkColumn("Source", display_text="Open link", width="small"),
        },
        key=f"overview_editor_{rev}",
    )


    # --- Detect which row was opened (queue the dialog for the NEXT run) ---
    try:
        clicked_rows = [int(i) for i, v in edited["Open"].items() if bool(v)]
    except Exception:
        clicked_rows = []

    if clicked_rows:
        ridx = clicked_rows[-1]  # last-opened wins
        if ridx in sub.index:
            row = sub.loc[ridx]

            # Build ordered keys using entity_id not name
            ordered_keys = [(sub.loc[i, "entity_id"], sub.loc[i, "Crypto Asset"]) for i in sub.index]

            # Find the index of the clicked pair
            clicked_key = (row.get("entity_id", ""), row.get("Crypto Asset", "-"))
            try:
                clicked_idx = ordered_keys.index(clicked_key)
            except ValueError:
                clicked_idx = 0  # fallback

            # Queue dialog payload with nav state
            st.session_state["active_dialog"] = {
                "list_keys": ordered_keys,   # list of (entity_id, asset)
                "idx": clicked_idx,          # active index
            }

            # Rotate the editor key and rerun
            st.session_state["overview_editor_rev"] = st.session_state.get("overview_editor_rev", 0) + 1
            st.rerun(scope="app")


    # --- If a dialog is queued, render it now (checkboxes are already cleared) ---
    ad = st.session_state.get("active_dialog")
    if ad and isinstance(ad.get("list_keys"), list):
        lk  = ad["list_keys"]
        idx = int(ad.get("idx", 0))
        if 0 <= idx < len(lk):
            # list now holds (entity_id, asset)
            eid, asset_key = lk[idx]

            # Resolve the selected row preferring entity_id
            sel = sub[(sub["entity_id"] == eid) & (sub["Crypto Asset"] == asset_key)]
            if sel.empty:
                sel = sub[sub["entity_id"] == eid]

            if not sel.empty:
                row = sel.sort_values("USD Value", ascending=False).iloc[0]

                # Pull fields
                name    = row.get("Entity Name", "-")
                asset   = row.get("Crypto Asset", "-")
                ticker  = str(row.get("Ticker", "-") or "-")
                etype   = row.get("Entity Type", "-")
                country = row.get("Country", "-")
                flag    = country_emoji_map.get(str(country), "🏳️")

                # Build multi-asset rows using entity_id
                df_all = st.session_state["data_df"]
                rows_entity = df_all[df_all["entity_id"] == eid].copy()

                def _first_non_empty(series_like, default=""):
                    try:
                        ser = pd.Series(series_like)
                        ser = ser.replace([None, np.nan], "").astype(str).str.strip()
                        bad = {"", "-", "nan", "none", "null"}
                        for x in ser:
                            if x and x.lower() not in bad:
                                return x
                        return default
                    except Exception:
                        return default

                sector   = row.get("Sector", "")   or _first_non_empty(rows_entity.get("Sector",   pd.Series([])), "-")
                industry = row.get("Industry", "") or _first_non_empty(rows_entity.get("Industry", pd.Series([])), "-")
                about    = row.get("About", "")    or _first_non_empty(rows_entity.get("About",    pd.Series([])), "")
                website  = row.get("Website", "")  or _first_non_empty(rows_entity.get("Website",  pd.Series([])), "")

                # Ranks still keyed by name
                arank = _asset_rank_map.get((name, asset))
                grank = _global_rank_map.get((name, asset))

                # Percent of supply
                pct_supply = row.get("% of Supply", np.nan)
                if pd.isna(pct_supply):
                    try:
                        pct_supply = float(row["Holdings (Unit)"]) / float(supply_caps.get(asset, 1)) * 100.0
                    except Exception:
                        pct_supply = float("nan")

                usd_val = row.get("USD Value", float("nan"))
                mcap    = row.get("Market Cap", float("nan"))
                mnav    = row.get("mNAV", float("nan"))
                prem    = row.get("Premium", float("nan"))
                ttmcr   = row.get("TTMCR", float("nan"))

                status_label = {
                    "active": "Active",
                    "pending_funded": "Funded",
                    "pending_announced": "Announced",
                    "inactive": "Inactive",
                }.get(str(row.get("status", "active")).lower(), "Active")

                target_usd_disp = (
                    pretty_usd(float(row.get("target_usd"))) 
                    if pd.notna(row.get("target_usd")) and float(row.get("target_usd") or 0) > 0 
                    else "–"
                )
                target_units_disp = (
                    f"{float(row.get('target_units')):,.0f}"
                    if pd.notna(row.get("target_units")) and float(row.get("target_units") or 0) > 0 
                    else "–"
                )
                source_url = (row.get("source_url") or "").strip()

                etype_badge = _badge_svg_uri(str(etype), TYPE_PALETTE.get(str(etype), (250, 250, 250)), h=22)
                is_datco = bool(int(row.get("DAT", 0)))

                prices_cg, _ = read_central_prices_from_sheet() or ({}, None)
                current_price     = prices_cg.get(asset, float("nan"))
                avg_cost_per_unit = row.get("Avg Cost", float("nan"))

                token_logo_map = logo_map

                per_asset_ranks = {
                    a: _asset_rank_map.get((name, a))
                    for a in rows_entity["Crypto Asset"].dropna().astype(str).unique()
                }

                agg_grank = _agg_global_rank_map.get(name)
                
                with st.spinner(f"Loading {name} — {asset}…"):

                    # Load events by entity_id and asset
                    ev_df, ts_df = get_events_for_entity_id_asset(eid, asset_key)

                    show_entity_dialog(
                        name=name,
                        ticker=ticker,
                        country=country,
                        flag=flag,
                        etype=etype,
                        etype_badge_uri=etype_badge,
                        asset=asset,
                        arank=arank,
                        grank=grank,

                        holdings_unit=row.get("Holdings (Unit)", None),
                        pct_supply=pct_supply,
                        usd_value=usd_val,
                        market_cap=mcap,
                        mnav=mnav,
                        premium=prem,
                        ttmcr=ttmcr,

                        status=status_label,
                        target_usd=target_usd_disp,
                        target_units=target_units_disp,
                        source_url=source_url,

                        sector=sector,
                        industry=industry,
                        is_datco=is_datco,
                        about_text=about,
                        website=website,

                        current_price=current_price,
                        avg_cost_per_unit=avg_cost_per_unit,

                        rows_df=rows_entity,
                        token_logo_map=token_logo_map,
                        supply_caps=supply_caps,

                        events_enriched=ev_df,
                        holdings_timeseries=ts_df,

                        agg_global_rank=agg_grank,
                        per_asset_ranks=per_asset_ranks,
                        nav_list=lk,
                        nav_index=idx,
                        nav_session_key="active_dialog",
                    )
            else:
                st.session_state["active_dialog"] = None




    log_table_render("global_summary", "overview_table", len(display))


    # --- PDF download (defined unconditionally) ---
    fname_asset = "all" if asset_choice == "All" else asset_choice.lower()
    pdf_bytes = _table_pdf_bytes(
        sub, logo_map, title=f"Crypto Treasury Top {len(sub)} Ranking - {fname_asset.upper()}"
    )


    if st.download_button(
        "Download List as PDF",
        data=pdf_bytes,
        type="primary",
        file_name=f"crypto_treasury_list_{fname_asset}_top{len(sub)}.pdf",
        mime="application/pdf",
        key="dl_overview_table_pdf",
    ):
        from analytics import log_event
        log_event("download_click", {
            "target": "table_pdf",
            "file_name": f"crypto_treasury_list_{fname_asset}_top{len(sub)}.pdf",
            "rows_exported": int(len(sub)),
        })


    # Last update info
    prices_cg, ts_utc = read_central_prices_from_sheet()

    def _format_cet(ts):
        if ts is None or pd.isna(ts):
            return "n/a"
        berlin = ZoneInfo("Europe/Berlin")
        local = ts.astimezone(berlin)
        return local.strftime("%d/%m/%Y, %H:%M:%S ") + local.tzname()

    last_cg_str = _format_cet(ts_utc)

    st.divider()

    st.text("Data Sources")

    st.caption(f"**CoinGecko API**: Crypto prices are updated hourly. Last update: {last_cg_str}")
    st.caption("**Google Finance API**: Equity market caps are updated in real time (you may refresh the website to fetch latest data).")
    st.caption("**Treasury holdings**: Updated daily on a best-effort basis and may not always be fully accurate.")

import streamlit as st
import pandas as pd
import numpy as np
import html
from zoneinfo import ZoneInfo
import pandas as pd

from urllib.parse import urlencode, quote_plus

from modules.kpi_helpers import render_kpis
from analytics import log_table_render
from modules.ui import btc_b64, eth_b64, sol_b64, sui_b64, ltc_b64, xrp_b64, hype_b64
from modules.pdf_helper import _table_pdf_bytes
from modules.emojis import country_emoji_map
from modules.data_loader import read_central_prices_from_sheet
from modules.entity_dialog import show_entity_dialog


# Supply column row-wise
supply_caps = {
    "BTC": 20_000_000,  
    "ETH": 120_000_000,
    "XRP": 60_000_000_000,
    "BNB": 140_000_000,
    "SOL": 540_000_000,
    "SUI": 3_500_000_000,
    "LTC": 76_000_000,
    "HYPE": 270_000_000,
    }

TRUE_DAT_WHITELIST = {
    "BTC": {"Strategy Inc.", "Twenty One Capital (XXI)", "Bitcoin Standard Treasury Company", "Metaplanet Inc.", "ProCap Financial, Inc", "Capital B", "H100 Group", 
            "Bitcoin Treasury Corporation", "Treasury B.V.", "American Bitcoin Corp.", "Parataxis Holdings LLC", "Strive Asset Management", "ArcadiaB", "Cloud Ventures",
            "Stacking Sats, Inc.", "Melanion Digital", "Sequans Communications S.A.", "Africa Bitcoin Corporation", "Empery Digital Inc.", "B HODL", "OranjeBTC", "Sobtree",
            "kheAI Commerce"}, 
    "ETH": {"BitMine Immersion Technologies, Inc.", "SharpLink Gaming", "The Ether Machine", "ETHZilla Corporation", "FG Nexus", "GameSquare Holdings", "Centaurus Energy Inc.", "Ethero"},
    "SOL": {"Forward Industries, Inc.", "Upexi, Inc.", "DeFi Development Corp.", "Sharps Technology, Inc.", "Classover Holdings, Inc.", "Sol Strategies, Inc.", "Sol Treasury Corp.",
            "SOL Global Investments Corp.", "Helius Medical Technologies, Inc.", "Lion Group Holding Ltd."},
    "LTC": {"Lite Strategy, Inc."},
    "XRP": set(),
    "SUI": {"Lion Group Holding Ltd."},
    "HYPE": {"Hyperliquid Strategies Inc", "Hyperion DeFi, Inc.", "Lion Group Holding Ltd."},
}

TYPE_PALETTE = {
    "Public Company": (123, 197, 237),  # blue
    "Private Company": (247, 89, 176), # rose
    "DAO": (233, 242, 111),              # amber
    "Non-Profit Organization": (128, 217, 183),        # green
    "Government": (247, 198, 148),      # slate
    "Other": (222, 217, 217),           # white
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
    df = st.session_state["data_df"]

    # KPIs
    render_kpis(df, st.session_state.get("kpi_snapshots"))


    with st.container(border=True):
        st.markdown("#### Crypto Treasury Ranking", help="Ranked view of entities by digital asset treasury holdings.")

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

        c1_kpi, c2_kpi, c3_kpi, c4_kpi, c5_kpi = st.columns([1,0.5,0.5,0.5,0.5])

        # all list option
        options = ["All", "DATCOs"] + sorted(table["Crypto Asset"].dropna().unique().tolist())

        with c1_kpi:
            list_choice = st.pills(
                "Select Asset List",
                options=options,
                selection_mode="single",
                default="All",
                label_visibility="visible",
                key="tbl_asset_filter",
            )


        # apply selection
        if list_choice == "DATCOs":
            # union whitelist across assets present in the current table
            assets_present = sorted(table["Crypto Asset"].dropna().unique().tolist())
            whitelist_sets = [TRUE_DAT_WHITELIST.get(a, set()) for a in assets_present]
            active_whitelist = set().union(*whitelist_sets) if whitelist_sets else set()

            if "Entity Name" in table.columns:
                names_upper = table["Entity Name"].astype(str)
                table = table[names_upper.isin(active_whitelist)]  # <<< filter TABLE, not df!

            asset_choice = "All"

        else:
            asset_choice = list_choice
            if asset_choice != "All":
                table = table[table["Crypto Asset"] == asset_choice]


        c1, c2, c3, c4 = st.columns(4)

        # --- search by entity name ---
        with c1:
            name_query = st.text_input(
                "Search Entity",
                value="",
                placeholder="Type a company name…",
                key="tbl_search",
                help="Filter the list by entity name."
            )

        if name_query:
            table_search = table[table["Entity Name"].astype(str).str.contains(name_query, case=False, na=False)]
        else:
            table_search = table

        len_table = int(table_search.shape[0])
        default_rows = (len_table if list_choice == "DATCOs" else min(50, len_table))
        min_rows = 0 if len_table == 0 else 1
        max_rows = len_table

        # Determine a safe starting value WITHOUT writing session_state
        prev = st.session_state.get("tbl_rows", None)
        start_val = int(default_rows) if prev is None else max(min_rows, min(int(prev), max_rows))

        with c2:
            if "ui_entity_type" not in st.session_state:
                st.session_state["ui_entity_type"] = st.session_state.get("flt_entity_type", "All")
            sel_et = st.selectbox(
                "Select Entity Type",
                options=st.session_state["opt_entity_types"],
                key="ui_entity_type",
            )
            st.session_state["flt_entity_type"] = sel_et

        with c3:
            if "ui_country" not in st.session_state:
                st.session_state["ui_country"] = st.session_state.get("flt_country", "All")
            sel_co = st.selectbox(
                "Select Country/Region",
                options=st.session_state["opt_countries"],
                key="ui_country",
            )
            st.session_state["flt_country"] = sel_co

        with c4:
            row_count = st.number_input(
                f"Adjust Rows (Max. {max_rows})",
                min_value=min_rows,
                max_value=max_rows,
                value=start_val,     # safe starting value
                step=1,
                help=(
                    "Select number of rows to display, sorted by USD value. "
                    "Some entities hold more than one crypto asset and are shown separately."
                ),
                key="tbl_rows",
            )

        # apply global filters plus the local asset toggle
        filtered = table_search.copy()

        assets_sel = [asset_choice] if asset_choice != "All" else st.session_state.get("flt_assets", st.session_state["opt_assets"])
        filtered = filtered[filtered["Crypto Asset"].isin(assets_sel)]

        et = st.session_state.get("flt_entity_type", "All")
        if et != "All":
            filtered = filtered[filtered["Entity Type"] == et]

        co = st.session_state.get("flt_country", "All")
        if co != "All":
            filtered = filtered[filtered["Country"] == co]

        filtered = filtered[filtered["USD Value"] > 0]

        # ensure uppercase tickers for matching
        filtered["Ticker"] = filtered["Ticker"].astype(str).str.upper()

        # build the active whitelist based on current asset selection
        if asset_choice != "All":
            active_whitelist = set().union(*(TRUE_DAT_WHITELIST.get(a, set()) for a in [asset_choice]))
        else:
            active_assets = st.session_state.get("flt_assets", st.session_state["opt_assets"])
            active_whitelist = set().union(*(TRUE_DAT_WHITELIST.get(a, set()) for a in active_assets))

        # valid rows for averages
        valid = filtered.replace([np.inf, -np.inf], np.nan)
        valid = valid[(valid["mNAV"] > 0) & (valid["TTMCR"] > 0)]

        # Aggregate mNAV + TTMCR KPIs
        avg_mnav = valid["mNAV"].median()
        avg_ttmcr = valid["TTMCR"].median()
        valid_true = valid[valid["Entity Name"].isin(active_whitelist)]
        avg_mnav_true = valid_true["mNAV"].median()

        # DATCO
        sub_2 = filtered.head(int(row_count)).copy()
        
        total_entities = df["Entity Name"].nunique()

        filtered_count = sub_2.shape[0]
        # recompute DATCO mask on the sliced data
        names_sub = sub_2["Entity Name"].astype(str)
        datco_mask_sub = names_sub.isin(active_whitelist)

        # --- DATCO Adoption (count + % of Crypto-NAV) ---
        tickers_upper = filtered["Entity Name"].astype(str)
        datco_mask    = tickers_upper.isin(active_whitelist)

        # number of DATCO companies in current selection
        datco_count   = tickers_upper[datco_mask].nunique()

        nav_total = float(sub_2["USD Value"].sum())
        nav_datco = float(sub_2.loc[datco_mask_sub, "USD Value"].sum())

        def _fmt(x, pct=False):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "-"
            return f"{x:,.2f}%" if pct else f"{x:,.2f}"

        with c2_kpi:
            with st.container(border=True):
                st.metric(
                    "Total Crypto-NAV (selected)",
                    f"{pretty_usd(nav_total)}",
                    help=("Total USD value of selected crypto treasury entities (Crypto-NAV).")
                )

        with c3_kpi:
            with st.container(border=True):
                st.metric(
                    f"Number of Entities (selected)",
                    filtered_count,
                    help=("Current number view of selected rows (entities).")
                )

        with c4_kpi:
            with st.container(border=True):
                st.metric(
                    "DATCO mNAV (Median)",
                    f"{_fmt(avg_mnav_true)}×",
                    help="Median market to net asset value (mNAV) filtered for Digital Asset Treasury Companies (DATCO) only, excluding entities that use crypto assets for other strategic or operational purposes (e.g., mining activities). Current DATCOs include the following tickers (where data is publicly available): MSTR, NASDAQ:CEP, BSTR, MTPLF, CCCM, ALCPB, OTCMKTS:HOGPF, BTCT.V, MKBN, ABTC, 288330.KQ, BMNR, SBET, ETHM, ETHZ, FGNX, GAME, CTARF, UPXI, DFDV, STSS, KIDZ, STKE, and LITS."
                )

        with c5_kpi:
            with st.container(border=True):
                st.metric(
                    "TTMCR (Median)",
                    _fmt(avg_ttmcr, pct=True),
                    help="The Treasury-to-Market Cap Ratio (TTMCR) shows the share of a company's value represented by held crypto reserves (unweighted). It is calculated by dividing the crypto treasury (USD value) by the company's current market cap, shown as a percentage. For example, a TTMCR of 5% means that 5% of the company's market cap is backed by crypto assets."
                )

        sub = filtered.head(row_count)
        sub = sub.reset_index(drop=True)
        sub.index = sub.index + 1
        sub.index.name = "Rank"

        # --- RANKS on the full dataset (not filtered) ---
        df_all = st.session_state["data_df"].copy()

        # Global rank by USD Value (desc)
        _global_sorted = df_all.sort_values("USD Value", ascending=False).copy()
        _global_sorted["__global_rank"] = np.arange(1, len(_global_sorted) + 1)

        # Per-asset rank by USD Value (desc)
        _asset_sorted = df_all.sort_values(["Crypto Asset", "USD Value"], ascending=[True, False]).copy()
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

        # --- Aggregate global rank by total Crypto-NAV across all assets ---
        _df_agg = (
            df_all.groupby("Entity Name", as_index=False)["USD Value"]
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

        #Replace Country column for display with the emoji
        #display["Country"] = flag_series

        # show dashes for missing values
        display["Ticker"] = (display["Ticker"].replace({"": "-"}).astype("string").fillna("-"))

        # mark rows with no market cap or no ticker for display-only fallbacks
        _no_metrics = display["Market Cap"].isna() | display["Ticker"].isna()

        # display-only formatted metrics with dashes when not available
        display["mNAV_disp"]    = np.where(_no_metrics, "-", display["mNAV"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-"))
        display["Premium_disp"] = np.where(_no_metrics, "-", display["Premium"].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "-"))
        display["TTMCR_disp"]   = np.where(_no_metrics, "-", display["TTMCR"].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "-"))

        logo_map = {
            "BTC": f"data:image/png;base64,{btc_b64}",
            "ETH": f"data:image/png;base64,{eth_b64}",
            "SOL": f"data:image/png;base64,{sol_b64}",
            "XRP": f"data:image/png;base64,{xrp_b64}",
            "SUI": f"data:image/png;base64,{sui_b64}",
            "LTC": f"data:image/png;base64,{ltc_b64}",
            "HYPE": f"data:image/png;base64,{hype_b64}",
        }
        display["Token"] = display["Crypto Asset"].map(lambda a: logo_map.get(a, ""))

        display["Market Cap"] = display["Market Cap"].map(pretty_usd)
        display["USD Value"] = display["USD Value"].map(pretty_usd)
        # Add a per-row Open button column

        display = display[[
            "Entity", "Ticker", "Entity Type",                                          # Meta data
            "Token", "Crypto Asset", "Holdings (Unit)", "% of Supply", "USD Value",              # Crypto data
            "Market Cap", "mNAV_disp", "Premium_disp", "TTMCR_disp"               # Market data
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

        # Render dashboard table
        edited = st.data_editor(
            display,
            width="stretch",
            height=height,
            hide_index=False,
            disabled=[c for c in display.columns if c != "Open"],  # read-only except checkbox
            column_config={
                "Token": st.column_config.ImageColumn("Token"),
                "Crypto Asset": st.column_config.TextColumn("Symbol"),
                "Entity Type": st.column_config.MultiselectColumn("Type", options=options_ent_type, color=color_ent_type),
                "Holdings (Unit)": st.column_config.NumberColumn("Holdings", format="%d"),
                "% of Supply": st.column_config.ProgressColumn("% of Supply", min_value=0, max_value=100, format="%.2f%%"),
                "Market Cap": st.column_config.TextColumn("Market Cap", width="small"),
                "USD Value": st.column_config.TextColumn("Crypto-NAV", width="small"),
                "mNAV_disp": st.column_config.TextColumn("mNAV", width="small"),
                "Premium_disp": st.column_config.TextColumn("Premium", width="small"),
                "TTMCR_disp": st.column_config.TextColumn("TTMCR", width="small"),
                "Open": st.column_config.CheckboxColumn("Details",help="Click to open details for this entity.",default=False),
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
                # Queue the dialog payload (minimal identifiers are fine)
                # Build the ordered key list for the current visible table
                ordered_keys = [(sub.loc[i, "Entity Name"], sub.loc[i, "Crypto Asset"]) for i in sub.index]

                # Find the index of the clicked pair in that ordered list
                clicked_key = (row.get("Entity Name", "-"), row.get("Crypto Asset", "-"))
                try:
                    clicked_idx = ordered_keys.index(clicked_key)
                except ValueError:
                    clicked_idx = 0  # fallback

                # Queue dialog payload with nav state
                st.session_state["active_dialog"] = {
                    "list_keys": ordered_keys,   # list of (entity, asset) pairs in current view order
                    "idx": clicked_idx,          # which one is active
                }

                # Rotate the editor key (clears checkboxes) and rerun
                st.session_state["overview_editor_rev"] = st.session_state.get("overview_editor_rev", 0) + 1
                st.rerun(scope="app")


        # --- If a dialog is queued, render it now (checkboxes are already cleared) ---
        ad = st.session_state.get("active_dialog")
        if ad and isinstance(ad.get("list_keys"), list):
            lk  = ad["list_keys"]
            idx = int(ad.get("idx", 0))
            if 0 <= idx < len(lk):
                name_key, asset_key = lk[idx]
                sel = sub[(sub["Entity Name"] == name_key) & (sub["Crypto Asset"] == asset_key)]
                if sel.empty:
                    # if current filters hid the exact asset row, try by entity name only
                    sel = sub[sub["Entity Name"] == name_key]

                if not sel.empty:
                    row = sel.sort_values("USD Value", ascending=False).iloc[0]

                    # Pull fields (same as before)
                    name    = row.get("Entity Name", "-")
                    asset   = row.get("Crypto Asset", "-")
                    ticker  = str(row.get("Ticker", "-") or "-")
                    etype   = row.get("Entity Type", "-")
                    country = row.get("Country", "-")
                    flag    = country_emoji_map.get(str(country), "🏳️")

                    # Build multi-asset rows for this entity
                    df_all = st.session_state["data_df"]
                    rows_entity = df_all[df_all["Entity Name"] == name].copy()
                    rows_entity = rows_entity[rows_entity["USD Value"] > 0]  # optional noise filter

                    def _first_non_empty(series_like, default=""):
                        """Return first non-empty/meaningful string from a Series-like, else default."""
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

                    # Ranks (use your precomputed maps)
                    arank = _asset_rank_map.get((name, asset))
                    grank = _global_rank_map.get((name, asset))

                    # % of supply
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

                    etype_badge = _badge_svg_uri(str(etype), TYPE_PALETTE.get(str(etype), (250, 250, 250)), h=22)
                    is_datco = str(name).strip().lower() in {n.strip().lower() for n in TRUE_DAT_WHITELIST.get(asset, set())}

                    prices_cg, _ = read_central_prices_from_sheet() or ({}, None)
                    current_price     = prices_cg.get(asset, float("nan"))
                    avg_cost_per_unit = row.get("Avg Cost", float("nan"))

                    token_logo_map = logo_map 

                    # Per-asset ranks for all assets this entity holds
                    per_asset_ranks = {
                        a: _asset_rank_map.get((name, a))
                        for a in rows_entity["Crypto Asset"].dropna().astype(str).unique()
                    }

                    # Aggregate global rank (by total NAV across all assets)
                    agg_grank = _agg_global_rank_map.get(name)

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

                        # single-asset fallbacks
                        holdings_unit=row.get("Holdings (Unit)", None),
                        pct_supply=pct_supply,
                        usd_value=usd_val,
                        market_cap=mcap,
                        mnav=mnav,
                        premium=prem,
                        ttmcr=ttmcr,

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

                        agg_global_rank=agg_grank,
                        per_asset_ranks=per_asset_ranks,
                        nav_list=lk,
                        nav_index=idx,
                        nav_session_key="active_dialog",
                    )


                else:
                    # If filters changed and the entity is gone, clear the queued dialog
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

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import qualitative
from modules.emojis import country_emoji_map
from modules.flow_utils import decompose_asset


ASSETS_ORDER = [
    "BTC", "ETH", "SOL", "XRP", "BNB",
    "SUI", "LTC", "HYPE", "DOGE", "TRX",
    "ADA", "TON", "WLFI", "PUMP", "ATH",
    "BONK", "AVAX", "CRO", "LINK", "BERA",
    "TRUMP", "ZIG", "CORE", "VAULTA", "FLUID",
    "ZEC"
]
COLORS = {"BTC":"#f7931a",
          "ETH":"#6F6F6F",
          "XRP":"#00a5df",
          "BNB":"#f0b90b",
          "SOL":"#dc1fff", 
          "SUI":"#C0E6FF", 
          "LTC":"#345D9D",
          "HYPE":"#97fce4",
          "DOGE":  "#c2a633",
          "TRX":   "#ef0027",
          "ADA":   "#1b33ad",
          "TON":   "#0098ea",
          "WLFI":  "#FEED8B",
          "PUMP":  "#53d693",
          "ATH":   "#cff54c",
          "BONK":  "#f49317",
          "AVAX":  "#f4394b",
          "CRO":   "#112d74",
          "LINK":  "#45ace2",
          "BERA":  "#814626",
          "TRUMP": "#fdf7c4",
          "ZIG":   "#3db6b2",
          "CORE":  "#f79620",
          "VAULTA": "#170515",
          "FLUID": "#3a74ff",
          "ZEC": "#F5A800"
          }
TYPE_PALETTE = {
    "Public Company": (123, 197, 237),  # blue
    "Private Company": (247, 89, 176), # rose
    "DAO": (233, 242, 111),              # amber
    "Non-Profit Organization": (128, 217, 183),        # green
    "Government": (247, 198, 148),      # slate
    "Other": (222, 217, 217),           # white
}
COLOR_MAP = {k: f"rgb({r},{g},{b})" for k, (r, g, b) in TYPE_PALETTE.items()}
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
default_blue = "#66cded"

NEUTRAL_POS = "#43d1a0"
NEUTRAL_NEG = "#f94144"

WATERMARK_TEXT="cryptotreasurytracker.xyz"

def add_watermark(fig, main_text=WATERMARK_TEXT):
    # Main domain watermark
    fig.add_annotation(
        text=main_text,
        x=0.5, y=0.45, xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=20, color="white"),
        opacity=0.7,
        xanchor="center",
        yanchor="middle",
    )
    # Powered-by footer
    fig.add_annotation(
        text="<i>powered by <b>F5Crypto.com</b></i>",
        x=1, y=0,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=12, color="white"),
        #bgcolor="white",
        #bordercolor="white", borderwidth=4,
        xanchor="right", yanchor="bottom", opacity=0.7
    )
    return fig

def format_usd(value):
    sign = "-" if value < 0 else ""
    abs_val = abs(value)

    if abs_val >= 1_000_000_000_000:
        return f"{sign}${abs_val/1_000_000_000_000:.1f}T"
    elif abs_val >= 1_000_000_000:
        return f"{sign}${abs_val/1_000_000_000:.1f}B"
    elif abs_val >= 1_000_000:
        return f"{sign}${abs_val/1_000_000:.1f}M"
    elif abs_val >= 1_000:
        return f"{sign}${abs_val/1_000:.1f}K"
    else:
        return f"{sign}${abs_val:.0f}"

def pretty_usd(x):
    if pd.isna(x):
        return "-"
    ax = abs(x)
    if ax >= 1e12:  return f"${x/1e12:.2f}T"
    if ax >= 1e9:  return f"${x/1e9:.2f}B"
    if ax >= 1e6:  return f"${x/1e6:.2f}M"
    if ax >= 1e3:  return f"${x/1e3:.2f}K"
    return f"${x:,.0f}"

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


def render_kpi_sparkline(snapshots_df: pd.DataFrame, usd_pct_delta: float | None = None):

    if snapshots_df is None or snapshots_df.empty or "total_usd" not in snapshots_df.columns:
        st.info("No snapshot data available for KPI chart.")
        return

    chart_series = (
        snapshots_df.sort_values("date_utc").tail(30)[["date_utc", "total_usd"]]
    )
    chart_series["hover_text"] = [
        f"{d.strftime('%b.')} {d.day}: <b>{pretty_usd(v)}</b>"
        for d, v in zip(chart_series["date_utc"], chart_series["total_usd"])
    ]

    color_line = "#43d1a0" if usd_pct_delta is None or usd_pct_delta >= 0 else "#FF4B4B"  # Streamlit red

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_series["date_utc"],
        y=chart_series["total_usd"],
        mode="lines",
        line=dict(color=color_line, width=2),
        hovertext=chart_series["hover_text"],
        hovertemplate="%{hovertext}<extra></extra>"
    ))

    fig.update_layout(
        height=57,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, fixedrange=True, showgrid=False),
        yaxis=dict(visible=False, fixedrange=True, showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            font=dict(size=11, color="white"),
            bgcolor="rgba(30,30,30,0.85)",
            align="left"
        ),
        showlegend=False,
    )

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


def build_monthly_flows_chart(hist_like: pd.DataFrame) -> go.Figure | None:
    """
    Expects a history DataFrame already prepared via _prepare_hist_with_snapshot + _prep_history.
    Falls back to merging with st.session_state['data_df'] if raw historic is passed.
    """
    if hist_like is None or hist_like.empty:
        return None

    # Accept precomputed monthly totals directly, else compute from history
    if {"Date", "d_usd"}.issubset(hist_like.columns) and "Crypto Asset" not in hist_like.columns:
        # Caller already passed the 6-row monthly window used for the KPI
        monthly = (
            hist_like[["Date", "d_usd"]]
            .assign(Date=pd.to_datetime(hist_like["Date"]).dt.to_period("M").dt.to_timestamp())
            .sort_values("Date")
        )
    else:
        df_curr = st.session_state.get("data_df")
        hist_merge, _ = _prepare_hist_with_snapshot(hist_like, df_curr)
        hist = _prep_history(hist_merge)
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
        # Only in this compute path do we normalize to exactly 6 months
        last_month = pd.to_datetime(monthly["Date"].max()).to_period("M").to_timestamp()
        six_idx = pd.date_range(end=last_month, periods=6, freq="MS")
        monthly = (
            monthly.set_index("Date")
                .reindex(six_idx, fill_value=0.0)
                .rename_axis("Date")
                .reset_index()
    )


    if monthly.empty:
        return None

    # draw hairline bars for zeros so no empty slot appears
    y_raw = monthly["d_usd"].astype(float).values
    max_abs = np.nanmax(np.abs(y_raw)) if len(y_raw) else 0.0
    eps = max(1e-9, 0.005 * max_abs)   # 0.5 percent of range, min epsilon
    y_plot = np.where(y_raw == 0.0, eps, y_raw)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["Date"],
        y=y_plot,
        marker_color=["#43d1a0" if v >= 0 else "#FF4B4B" for v in y_raw],
        hovertext=[f"{d.strftime('%b %Y')}: <b>{pretty_usd(v)}</b>" for d, v in zip(monthly["Date"], y_raw)],
        hovertemplate="%{hovertext}<extra></extra>",
        marker_line_width=0,  # keep clean
    ))


    fig.update_layout(
        height=91,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, fixedrange=True, showgrid=False),
        yaxis=dict(visible=False, fixedrange=True, showgrid=False),
        bargap=0.25,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            font=dict(size=11, color="white"),
            bgcolor="rgba(30,30,30,0.85)",
            align="left"
        ),
        showlegend=False,
    )
    return fig


def build_top_five_jurisdictions_bar(df_usd: pd.DataFrame) -> go.Figure | None:
    if df_usd is None or df_usd.empty:
        return None
    d = df_usd.copy()
    d["Country"] = d["Country"].astype(str).str.strip()
    excl = {"", "Unknown", "Decentralized", "Other", "None", "nan"}
    d = d[~d["Country"].isin(excl)]
    if d.empty:
        return None

    topc = (
        d.groupby("Country", as_index=False)["USD Value"]
         .sum()
         .sort_values("USD Value", ascending=False)
         .head(5)
         .iloc[::-1]  # small at top for horizontal ordering
    )
    topc["label"] = topc["Country"].map(lambda c: f"{country_emoji_map.get(c, '🏳️')} {c}")
    topc["label_only"] = topc["Country"].map(lambda c: country_emoji_map.get(c, '🏳️'))

    fig = go.Figure(go.Bar(
        x=topc["USD Value"],
        y=topc["label_only"],  # only emojis on axis
        orientation="h",
        marker=dict(color="#43d1a0"),
        text=[pretty_usd(v) for v in topc["USD Value"]],
        textposition="outside",
        cliponaxis=False,
        # use topc["label"] (emoji + name) in hover
        hovertext=topc["label"],
        hovertemplate="%{hovertext}<br><b>%{text}</b><extra></extra>",
    ))

    fig.update_layout(
        height=164,
        margin=dict(l=40, r=60, t=0, b=0),
        xaxis=dict(visible=False, fixedrange=True, showgrid=False),
        yaxis=dict(
            visible=True,
            tickfont=dict(size=20),  # make emoji larger
            fixedrange=True,
            showgrid=False
        ),
        bargap=0.25,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )

    return fig


def build_supply_share_bar(df_usd: pd.DataFrame, 
                           sort_mode: str = "supply") -> go.Figure | None:
    """
    Build horizontal bar of treasury holdings as % of circulating supply.
    sort_mode: "supply" = top 5 by % of supply (default, detailed ranking)
               "usd"    = top 5 largest assets by USD value first, then ranked by % of supply
    """
    if df_usd is None or df_usd.empty:
        return None

    d = df_usd.copy()
    d["Holdings (Unit)"] = pd.to_numeric(d["Holdings (Unit)"], errors="coerce")
    d["USD Value"] = pd.to_numeric(d["USD Value"], errors="coerce").fillna(0.0)

    # --- compute per-asset aggregates ---
    per_asset_units = (
        d.groupby("Crypto Asset", as_index=False)["Holdings (Unit)"].sum()
    )
    per_asset_units = per_asset_units[per_asset_units["Crypto Asset"].isin(supply_caps.keys())]
    if per_asset_units.empty:
        return None

    # --- compute share of circulating supply ---
    per_asset_units["share"] = per_asset_units.apply(
        lambda r: (
            r["Holdings (Unit)"] / float(supply_caps.get(r["Crypto Asset"], np.nan))
            if float(supply_caps.get(r["Crypto Asset"], np.nan)) > 0 else np.nan
        ),
        axis=1
    ).astype(float)

    # --- selection logic ---
    if sort_mode == "usd":
        # Pick top 5 by total USD value first, then sort by share
        per_asset_usd = (
            d.groupby("Crypto Asset", as_index=False)["USD Value"].sum()
        )
        per_asset_usd = per_asset_usd[per_asset_usd["Crypto Asset"].isin(supply_caps.keys())]
        top_assets = (
            per_asset_usd.sort_values("USD Value", ascending=False)
                         .head(5)["Crypto Asset"].tolist()
        )
        per_asset_units = per_asset_units[per_asset_units["Crypto Asset"].isin(top_assets)]

    # Always order final display by share %
    top5 = (
        per_asset_units.dropna(subset=["share"])
                       .sort_values("share", ascending=False)
                       .head(5)
                       .iloc[::-1]
    )

    # --- plot ---
    fig = go.Figure(go.Bar(
        x=top5["share"] * 100.0,
        y=top5["Crypto Asset"],
        orientation="h",
        marker=dict(color=[COLORS.get(a, "#7f8c8d") for a in top5["Crypto Asset"]]),
        text=[f"{v:.1f}%" for v in (top5["share"] * 100.0)],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br><b>%{x:.1f}%%</b><extra></extra>",
    ))
    fig.update_layout(
        height=164,
        margin=dict(l=40, r=40, t=0, b=0),
        xaxis=dict(visible=False, fixedrange=True, showgrid=False),
        yaxis=dict(visible=True, tickfont=dict(size=11), fixedrange=True, showgrid=False),
        bargap=0.25,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def render_world_map(df, asset_filter, type_filter, value_range_filter):

    filtered = df.copy()

    # Asset filter accepts list or 'All'
    if asset_filter != "All":
        if isinstance(asset_filter, list):
            if len(asset_filter) == 0:
                # nothing selected, return an empty chart-friendly frame
                filtered = filtered.iloc[0:0]
            else:
                filtered = filtered[filtered["Crypto Asset"].isin(asset_filter)]
        else:
            filtered = filtered[filtered["Crypto Asset"] == asset_filter]

    if isinstance(asset_filter, list) and len(asset_filter) == 0:
        st.info("No data for the current filters.")
        return None

    
    # Entity type
    if type_filter != "All":
        filtered = filtered[filtered["Entity Type"] == type_filter]

    # Guard 1 empty after filtering
    if filtered.empty:
        st.info("No data for the current filters.")
        return None
    
    # Group first
    grouped = filtered.groupby("Country").agg(
        Total_USD=("USD Value", "sum"),
        Entity_Count=("Entity Name", "nunique"),
        Avg_Holdings=("USD Value", "mean")
    ).reset_index()

    # Guard 2 empty after grouping
    if grouped.empty:
        st.info("No data for the current filters.")
        return None

    # Value range filter (now on grouped data)
    if value_range_filter == "0–100M":
        grouped = grouped[grouped["Total_USD"] < 100_000_000]
    elif value_range_filter == "100M–1B":
        grouped = grouped[(grouped["Total_USD"] >= 100_000_000) & (grouped["Total_USD"] < 1_000_000_000)]
    elif value_range_filter == ">1B":
        grouped = grouped[grouped["Total_USD"] >= 1_000_000_000]

    # Guard 3 empty after value-bucket
    if grouped.empty:
        st.info("No data for the current filters.")
        return None
    
    # per-country per-asset breakdown + share of global 
    assets_in_scope = list(filtered["Crypto Asset"].dropna().unique())

    # Global USD across all selected assets/types (for % of global)
    total_global_usd = float(filtered["USD Value"].sum())

    # Per-country, per-asset aggregation
    per_country_asset = (
        filtered.groupby(["Country", "Crypto Asset"])
        .agg(Units=("Holdings (Unit)", "sum"), USD=("USD Value", "sum"))
        .reset_index()
    )

    # Map country -> HTML lines for hover
    def fmt_units(x: float) -> str:
        return f"{int(x):,}" if x >= 1 else f"{x:,.2f}"

    lines_by_country = {}
    for country in grouped["Country"]:
        rows = per_country_asset[per_country_asset["Country"] == country]
        lines = []
        # keep the order of the assets actually in scope
        for asset in assets_in_scope:
            r = rows[rows["Crypto Asset"] == asset]
            if not r.empty:
                u = float(r["Units"].iloc[0])
                usd = float(r["USD"].iloc[0])
                lines.append(f"{asset}: <b>{fmt_units(u)}</b> ({format_usd(usd)})")
        lines_by_country[country] = "<br>".join(lines) if lines else "—"

    grouped["PerAssetBreakdown"] = grouped["Country"].map(lines_by_country)
    grouped["Share_Global"] = grouped["Total_USD"] / total_global_usd if total_global_usd > 0 else 0.0
    grouped["Formatted_Share_Global"] = grouped["Share_Global"].apply(lambda x: f"{x:.1%}")
    
    # Format values for display
    grouped["Formatted_Total_USD"] = grouped["Total_USD"].apply(format_usd)
    grouped["Formatted_Avg_Holdings"] = grouped["Avg_Holdings"].apply(format_usd)

    # Prepare custom hover data columns
    grouped["Custom_Hover"] = (
        "Total Holdings: " + grouped["Formatted_Total_USD"] +
        "<br>Global Share: " + grouped["Formatted_Share_Global"] +
        "<br>Avg. USD per Holder: " + grouped["Formatted_Avg_Holdings"] +
        "<br># of Holders Reporting: " + grouped["Entity_Count"].astype(str) +
        "<br><br><b>Asset Breakdown:</b><br>" + grouped["PerAssetBreakdown"]
    )

    # Create choropleth
    fig = px.choropleth(
        grouped,
        locations="Country",
        locationmode="country names",
        color="Total_USD",
        hover_name="Country",
        custom_data=["Custom_Hover"],
        color_continuous_scale=px.colors.sequential.Agsunset, #https://plotly.com/python/builtin-colorscales/
        projection="natural earth",
        template="simple_white"
    )

    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>%{customdata[0]}<extra></extra>"
    )

    fig.add_annotation(
        text=WATERMARK_TEXT,
        x=0.5, y=0.45, xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=30, color="black"),
        opacity=0.7,
        xanchor="center",
        yanchor="middle",
    )
    # Powered-by footer
    fig.add_annotation(
        text="powered by <b>F5Crypto.com</b>",
        x=0.98, y=0.02,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=12, color="black"),
        bgcolor="white",
        bordercolor="white", borderwidth=4,
        xanchor="right", yanchor="bottom", opacity=0.7
    )

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth",
            center=dict(lat=20, lon=0),
            projection_scale=1  # optional zoom level
        ),
        uirevision="static-map",  # Prevents user interaction from updating layout
        margin=dict(l=20, r=20, t=10, b=10),
        height=600,
        coloraxis_colorbar=dict(title="Total USD"),
        font=dict(size=12),
    )


    return fig


def render_rankings(df, asset="BTC", by="units"):
    d = df[df["Crypto Asset"] == asset]

    top = (
        d.groupby("Entity Name", as_index=False)
        .agg(Holdings=("Holdings (Unit)", "sum"),
             USD_Value=("USD Value", "sum"))
        .sort_values("Holdings" if by == "units" else "USD_Value", ascending=False)
        .head(5)
    )

    values = top["Holdings"] if by == "units" else top["USD_Value"]
    value_labels = (
        top["Holdings"].apply(lambda x: f"{x:,.0f}")
        if by == "units"
        else top["USD_Value"].apply(lambda x: f"${x/1e9:.1f}B")
    )

    bar_color = COLORS.get(asset, "#A9A9A9")

    top["Custom Hover"] = top.apply(
        lambda row: f"<b>{row['Entity Name']}</b><br>"
                    + (f"Holdings {row['Holdings']:,.0f}" if by == "units"
                       else f"USD Value <b>${row['USD_Value']/1e9:.1f}B</b>"),
        axis=1
    )

    fig = go.Figure(go.Bar(
        x=values,
        y=top["Entity Name"],
        orientation="h",
        text=value_labels,
        textposition="auto",
        marker=dict(color=bar_color),
        customdata=top["Custom Hover"],
        hovertemplate="%{customdata}<extra></extra>"
    ))

    add_watermark(fig)


    fig.update_layout(
        title_text="",
        height=240,
        yaxis=dict(autorange="reversed", tickfont=dict(size=12), title_standoff=25),
        margin=dict(l=140, r=10, t=0, b=20),
        font=dict(size=12),
        hoverlabel=dict(align="left")
    )

    return fig


def historic_chart(df_historic: pd.DataFrame, current_df: pd.DataFrame | None = None, by="USD"):
    hist, selected_assets = _prepare_hist_with_snapshot(df_historic, current_df)
    df = hist.copy()

    # Ensure numeric
    df['USD Value'] = pd.to_numeric(df['USD Value'], errors='coerce')
    if 'Holdings (Unit)' in df.columns:
        df['Holdings (Unit)'] = pd.to_numeric(df['Holdings (Unit)'], errors='coerce')

    value_col = 'USD Value' if by == "USD" else 'Holdings (Unit)'

    # Aggregate monthly totals
    grouped = (
        df.groupby(['Date', 'Crypto Asset'])
        .agg({value_col: 'sum', 'USD Value': 'sum'})
        .reset_index()
    )

    # Build hover templates
    if by == "USD":
        breakdowns = (
            grouped.groupby('Date')
            .apply(lambda d: (
                f"<b>{d.name.strftime('%B %Y')}</b><br>" +
                "<br>".join([
                    f"{row['Crypto Asset']}: <b>{format_usd(row['USD Value'])}</b>"
                    for _, row in d.iterrows()
                ]) +
                f"<br>Total: <b>{format_usd(d['USD Value'].sum())}</b>"
            ))
            .to_dict()
        )
        grouped['Text'] = grouped[value_col].apply(format_usd)
    else:
        breakdowns = (
            grouped.groupby('Date')
            .apply(lambda d: (
                f"<b>{d.name.strftime('%B %Y')}</b><br>" +
                "<br>".join([
                    f"{row['Crypto Asset']}: <b>{int(row[value_col]):,}</b>"
                    for _, row in d.iterrows()
                ]) +
                f"<br>Total: <b>{int(d[value_col].sum()):,}</b>"
            ))
            .to_dict()
        )
        grouped['Text'] = grouped[value_col].apply(lambda x: f"{int(x):,}")

    grouped['Custom Hover'] = grouped['Date'].map(breakdowns)

    max_date = df['Date'].max()
    grouped = grouped[grouped['Date'] <= max_date]

    # Build figure
    fig = px.bar(
        grouped,
        x='Date',
        y=value_col,
        color='Crypto Asset',
        #text='Text' if by != "USD" else None,
        custom_data=['Custom Hover'],
        barmode='stack',
        color_discrete_map=COLORS
    )

    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        textposition='outside'
    )

    # Add annotations only for USD
    if by == "USD":
        totals = grouped.groupby('Date')['USD Value'].sum()
        for date, total in totals.items():

            fig.update_layout(
                yaxis=dict(
                    tickprefix="$",
                )
            )

    # Watermark
    add_watermark(fig)


    fig.update_layout(
        margin=dict(t=20, b=20, l=50, r=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
        hoverlabel=dict(align='left'),
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        xaxis_title=None,
        yaxis_title=None,
        legend_title_text=None,
    )

    fig.update_xaxes(
        dtick="M1",
        tickformat="%b %Y",
     #   ticklabelmode="period"  # ensures labels like "Jul 2025" represent the period
    )

    return fig


def historic_changes_chart(df_historic: pd.DataFrame,
                           current_df: pd.DataFrame | None = None,
                           by="Units",
                           start=None,
                           end=None):

    hist, selected_assets = _prepare_hist_with_snapshot(df_historic, current_df)
    d = hist.copy()

    col = "Holdings (Unit)" if by == "Units" else "USD Value"
    d[col] = pd.to_numeric(d[col], errors='coerce')

    # --- compute per asset always ---
    monthly = (
        d.groupby(['Date', 'Crypto Asset'])[col]
        .sum()
        .reset_index()
        .sort_values(['Crypto Asset', 'Date'])
    )
    monthly['Change'] = monthly.groupby('Crypto Asset')[col].diff().fillna(0)

    # --- keep 1 extra prior month for baseline ---
    if start is not None:
        start = pd.to_datetime(start)
        prev_month = start - pd.offsets.MonthBegin(1)
        monthly = monthly[monthly['Date'] >= prev_month]

    # Always include the merged latest date (so current snapshot shows up)
    merged_max = monthly['Date'].max()

    if end is None:
        end_dt = merged_max
    else:
        end_dt = max(pd.to_datetime(end), merged_max)  # <-- ensures current month is included

    monthly = monthly[monthly['Date'] <= end_dt]

    # Enforce start again (after end cap) if provided
    if start is not None:
        monthly = monthly[monthly['Date'] >= start]

    # --- build hover like historic_chart ---
    if by == "USD":
        breakdowns = (
            monthly.groupby('Date')
            .apply(lambda d: (
                f"<b>{d.name.strftime('%B %Y')}</b><br>" +
                "<br>".join(
                    f"{row['Crypto Asset']}: <b>{format_usd(row['Change'])}</b>"
                    for _, row in d.iterrows()
                ) +
                f"<br>Total: <b>{format_usd(d['Change'].sum())}</b>"
            ))
            .to_dict()
        )
    else:  # Units
        breakdowns = (
            monthly.groupby('Date')
            .apply(lambda d: (
                f"<b>{d.name.strftime('%B %Y')}</b><br>" +
                "<br>".join(
                    f"{row['Crypto Asset']}: <b>{row['Change']:+,.0f}</b>"
                    for _, row in d.iterrows()
                ) +
                f"<br>Total: <b>{d['Change'].sum():+,.0f}</b>"
            ))
            .to_dict()
        )

    monthly['Custom Hover'] = monthly['Date'].map(breakdowns)
    monthly['Label'] = monthly['Change'].apply(lambda x: f"{x:+,.0f}")

    # --- Plot bars STACKED (multi-asset if present) ---
    fig = px.bar(
        monthly,
        x='Date',
        y='Change',
        color='Crypto Asset',
        custom_data=['Custom Hover'],
        barmode='stack',
        color_discrete_map=COLORS
    )
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        textposition='outside',
        selector=dict(type='bar')
    )

    # --- Trend line over total ---
    window = max(2, min(6, len(monthly['Date'].unique()) // 2))
    trend = (
        monthly.groupby('Date')['Change'].sum()
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index()
    )
    trend['Hover'] = trend.apply(
        lambda r: f"<b>{r['Date'].strftime('%B %Y')}</b><br>"
                  f"{window}M trend: <b>{r['Change']:+,.0f}{'' if by=='Units' else ' $'}</b>",
        axis=1
    )
    fig.add_scatter(
        x=trend['Date'],
        y=trend['Change'],
        mode='lines',
        line=dict(color=default_blue, width=4, dash='dot'),
        name=f"Trend ({window}M avg)",
        customdata=trend['Hover'],
        hovertemplate="%{customdata}<extra></extra>",
        visible="legendonly"  # 👈 Hidden by default, toggle via legend
    )


    fig.update_layout(
        margin=dict(t=20, b=20, l=50, r=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
        hoverlabel=dict(align='left'),
        xaxis=dict(fixedrange=True),
        yaxis=dict(
            fixedrange=True,
            title="Net Change (Units)" if by == "Units" else "Net Change (USD)",
            tickprefix="" if by == "Units" else "$"
        ),
        xaxis_title=None,
        legend_title_text=None,
    )
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")

    add_watermark(fig)


    return fig


def _first_day_next_month(dt: pd.Timestamp) -> pd.Timestamp:
    dt = pd.Timestamp(dt).normalize()
    return (dt + pd.offsets.MonthBegin(1))


def _prepare_hist_with_snapshot(df_historic: pd.DataFrame, current_df: pd.DataFrame | None):
    dfh = df_historic.copy()
    dfh["USD Value"] = pd.to_numeric(dfh["USD Value"], errors="coerce").fillna(0.0)
    dfh["Holdings (Unit)"] = pd.to_numeric(dfh.get("Holdings (Unit)"), errors="coerce").fillna(0.0)
    dfh["Crypto Asset"] = dfh["Crypto Asset"].astype(str).str.upper()

    selected_assets = [a for a in ASSETS_ORDER if a in dfh["Crypto Asset"].unique()]
    if not selected_assets:
        return pd.DataFrame(), selected_assets

    hist = (
        dfh.groupby(["Date","Crypto Asset"], as_index=False)
           .agg({"USD Value":"sum","Holdings (Unit)":"sum"})
    )
    last_hist_month = hist["Date"].max() if not hist.empty else pd.Timestamp.today().normalize()

    if current_df is not None and not current_df.empty:
        snap = current_df.copy()
        snap["Crypto Asset"] = snap["Crypto Asset"].astype(str).str.upper()
        snap["USD Value"] = pd.to_numeric(snap["USD Value"], errors="coerce").fillna(0.0)
        snap["Holdings (Unit)"] = pd.to_numeric(snap.get("Holdings (Unit)"), errors="coerce").fillna(0.0)
        snap = snap.groupby("Crypto Asset", as_index=False).agg({"USD Value":"sum","Holdings (Unit)":"sum"})
        snap["Date"] = _first_day_next_month(last_hist_month)
        hist = pd.concat([hist, snap[["Date","Crypto Asset","USD Value","Holdings (Unit)"]]], ignore_index=True)

    # filter & stable order
    hist = hist[hist["Crypto Asset"].isin(selected_assets)]
    rank = {a:i for i,a in enumerate(ASSETS_ORDER)}
    hist = hist.sort_values(["Date","Crypto Asset"], key=lambda s: s.map(rank).fillna(999))
    return hist, selected_assets


# --- left: Cumulative Market Cap (USD) ---
def cumulative_market_cap_chart(df_historic: pd.DataFrame, current_df: pd.DataFrame | None = None):
    hist, selected_assets = _prepare_hist_with_snapshot(df_historic, current_df)
    fig = go.Figure()
    if hist.empty:
        return fig

    totals = hist.groupby("Date", as_index=True)["USD Value"].sum().reset_index().sort_values("Date")

    if len(selected_assets) == 1:
        a = selected_assets[0]
        s = (hist[hist["Crypto Asset"] == a].sort_values("Date")[["Date","USD Value","Holdings (Unit)"]])
        s["usd_fmt"] = s["USD Value"].apply(format_usd)

        # Units area (left axis)
        fig.add_trace(go.Scatter(
            x=s["Date"], y=s["Holdings (Unit)"],
            mode="lines",
            line=dict(width=0, color=COLORS.get(a, "#888")),
            fill="tozeroy",
            name=f"{a} Total Units",
            hovertemplate="<b>%{x|%b %Y}</b><br>Units: <b>%{y:,.0f}</b><extra></extra>",
            yaxis="y2"
        ))
        # USD line (right axis)
        fig.add_trace(go.Scatter(
            x=s["Date"], y=s["USD Value"],
            mode="lines",
            line=dict(width=3, color=COLORS.get(a, "#ff9393")),
            name=f"{a} Total USD Value",
            hovertext=s["usd_fmt"],
            hovertemplate="<b>%{x|%b %Y}</b><br>USD: <b>%{hovertext}</b><extra></extra>",
            yaxis="y"
        ))
        fig.update_layout(
            yaxis=dict(
                title="USD Value",
                rangemode="tozero",
                tickprefix="$",
                showgrid=True,           # left axis provides the grid
                zeroline=False, fixedrange=True
            ),
            yaxis2=dict(
                title="Total Units",
                overlaying="y",
                side="right",
                rangemode="tozero",
                showgrid=False,          # critical fix hide the second grid
                zeroline=False,
                showline=False,
                tickprefix="",
                linecolor="rgba(255,255,255,0.4)", fixedrange=True
            ),
        )

    else:
        # Multi-asset → one total USD line + per-asset USD lines
        series = totals.sort_values("Date")
        series["usd_fmt"] = series["USD Value"].apply(format_usd)
        fig.add_trace(go.Scatter(
            x=series["Date"], y=series["USD Value"],
            mode="lines",
            line=dict(width=5, dash="solid", color="#43d1a0"),
            name="Total",
            hovertext=series["usd_fmt"],
            hovertemplate="<b>%{x|%b %Y}</b><br>Total: <b>%{hovertext}</b><extra></extra>",
        ))

    fig.update_layout(
        margin=dict(t=20, b=20, l=50, r=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
        hoverlabel=dict(align='left'),
        xaxis_title=None, yaxis_title=None,
        legend_title_text=None,
        xaxis=dict(fixedrange=True),
        yaxis=dict(
            title=None,
            rangemode="tozero",
            tickprefix="$",   # only multi-asset mode gets a dollar prefix on the left axis
            showgrid=True,
            zeroline=False,
            fixedrange=True
        )
    )
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")

    add_watermark(fig)

    return fig


# --- right: Dominance (USD stacked area) ---
def dominance_area_chart_usd(df_historic: pd.DataFrame,
                             current_df: pd.DataFrame | None = None,
                             pct: bool = False):
    """
    pct=False  -> absolute USD stacked area (current behavior)
    pct=True   -> 100% stacked area (share per asset); y-axis = 0..100%
    """
    hist, selected_assets = _prepare_hist_with_snapshot(df_historic, current_df)
    fig = go.Figure()
    if hist.empty:
        return fig

    usds = (hist
            .pivot(index="Date", columns="Crypto Asset", values="USD Value")
            .fillna(0.0))
    usds = usds.reindex(columns=selected_assets)
    totals = usds.sum(axis=1).replace(0, np.nan)

    if pct:
        shares = usds.div(totals, axis=0).fillna(0.0)   # 0..1 fractions
        cum = None
        for i, a in enumerate([x for x in ASSETS_ORDER if x in selected_assets]):
            top = shares[a] if cum is None else (cum + shares[a])
            usd_fmt = usds[a].apply(format_usd)  # show USD in hover too
            # customdata[0] = share (fraction)
            fig.add_trace(go.Scatter(
                x=shares.index,
                y=top.values,
                mode="lines",
                line=dict(width=0.0, color=COLORS.get(a, "#888")),
                fill=("tozeroy" if i == 0 else "tonexty"),
                name=a,
                hovertext=usd_fmt,
                customdata=np.column_stack([shares[a].values]),
                hovertemplate=(
                    "<b>%{x|%b %Y}</b><br>"
                    f"{a}: <b>%{{customdata[0]:.1%}}</b><br>"
                    "USD: <b>%{hovertext}</b>"
                    "<extra></extra>"
                ),
            ))
            cum = top

        fig.update_yaxes(range=[0, 1], tickformat=".0%", fixedrange=True)
        fig.update_layout(yaxis_title=None)

    else:
        cum = None
        for i, a in enumerate([x for x in ASSETS_ORDER if x in selected_assets]):
            top = usds[a] if cum is None else (cum + usds[a])
            usd_fmt = usds[a].apply(format_usd)
            fig.add_trace(go.Scatter(
                x=usds.index,
                y=top.values,
                mode="lines",
                line=dict(width=0.0, color=COLORS.get(a, "#888")),
                fill=("tozeroy" if i == 0 else "tonexty"),
                name=a,
                hovertext=usd_fmt,
                hovertemplate="<b>%{x|%b %Y}</b><br>" + f"{a}: <b>%{{hovertext}}</b><extra></extra>",
            ))
            cum = top

        fig.update_yaxes(rangemode="tozero", tickprefix="$", fixedrange=True)

    fig.update_traces(opacity=0.95)
    fig.update_layout(
        margin=dict(t=20, b=20, l=50, r=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
        hoverlabel=dict(align='left'),
        xaxis_title=None, yaxis_title=None, legend_title_text=None,
    )
    fig.update_xaxes(dtick="M1", tickformat="%b %Y", fixedrange=True)
    add_watermark(fig)

    return fig


def render_flow_decomposition_chart(view: pd.DataFrame, asset_pick: str | None = None):

    if view.empty or "Date" not in view.columns:
        st.info("No valid flow data available for chart rendering.")
        return

    bar_color_price = "#8892a6"
    bar_color_units = "#43d1a0"

    view["price_hover"] = view["price_effect"].apply(pretty_usd)
    view["units_hover"] = view["units_effect"].apply(pretty_usd)

    # --- Build bar chart ---
    fig = go.Figure()
    fig.add_bar(
        name="Price effect",
        x=view["Date"],
        y=view["price_effect"],
        marker_color=bar_color_price,
        hovertext=view["price_hover"],
        hovertemplate="Date %{x|%b %Y}<br>Price %{hovertext}<extra></extra>",
    )

    fig.add_bar(
        name="Units effect",
        x=view["Date"],
        y=view["units_effect"],
        marker_color=bar_color_units,
        hovertext=view["units_hover"],
        hovertemplate="Date %{x|%b %Y}<br>Units %{hovertext}<extra></extra>",
    )

    fig.update_layout(
        barmode="relative",
        height=320,
        margin=dict(t=20, b=20, l=50, r=40),
        xaxis=dict(title=None, tickformat="%b %Y", fixedrange=True),
        yaxis=dict(title="ΔUSD", tickprefix="$", fixedrange=True),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
        ),
        hoverlabel=dict(align="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    # --- Watermark ---
    add_watermark(fig)


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


def holdings_by_entity_type_bar(df):
    # Step 1: Group by Entity Type and Crypto Asset
    grouped = (
        df.groupby(['Entity Type', 'Crypto Asset'])['USD Value']
        .sum()
        .reset_index()
    )

    # Step 2: Build custom hover text per Entity Type
    breakdowns = (
        grouped.groupby('Entity Type')
        .apply(lambda d: f"<b>{d.name}</b><br>" + "<br>".join(
            [f"{row['Crypto Asset']}: <b>{format_usd(row['USD Value'])}</b>" for _, row in d.sort_values('USD Value', ascending=False).iterrows()])
        ).to_dict()
    )
    grouped['Custom Hover'] = grouped['Entity Type'].map(breakdowns)

    # Step 3: Sort Entity Types by total USD Value descending
    totals = grouped.groupby('Entity Type')['USD Value'].sum().sort_values(ascending=False)
    sorted_types = totals.index.tolist()
    grouped['Entity Type'] = pd.Categorical(grouped['Entity Type'], categories=sorted_types, ordered=True)
    grouped = grouped.sort_values(['Entity Type', 'Crypto Asset'])

    # Step 4: Create chart with formatted labels
    fig = px.bar(
        grouped,
        x='Entity Type',
        y='USD Value',
        color='Crypto Asset',
        barmode='stack',
        custom_data=['Custom Hover'],
        color_discrete_map=COLORS,
        category_orders={'Entity Type': sorted_types}  # ✅ This line is key
    )

    add_watermark(fig)


    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>"
    )

    # Step 5: Layout updates
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        textposition='outside',
        textfont=dict(size=14)
    )

    fig.update_layout(
        margin=dict(t=20, b=20, l= 50, r= 50),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.1,
            xanchor='center',
            x=0.5
        ),
        xaxis=dict(fixedrange=True),
        yaxis=dict(
            tickprefix="$", fixedrange=True
        ),
        hoverlabel=dict(align='left'),
        xaxis_title=None,
        yaxis_title=None,
        legend_title_text=None
    )

    return fig


def _empty_pie(msg="No data"):
    fig = px.pie(pd.DataFrame({"Entity Type": [], "Value": []}),
                 values="Value", names="Entity Type", hole=0.65)
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False, font=dict(size=14))
    fig.update_traces(textinfo="none")
    fig.update_layout(showlegend=False)
    return fig


def entity_type_distribution_pie(df: pd.DataFrame, mode: str = "count"):
    required_base = {"Entity Name", "Entity Type"}
    if not required_base.issubset(df.columns):
        return _empty_pie("Missing Entity Type")

    ORDER = list(TYPE_PALETTE.keys())

    # sanitize source
    d0 = (df[list(required_base | {"USD Value"} if "USD Value" in df.columns else required_base)]
            .copy())
    d0["Entity Type"] = d0["Entity Type"].astype(str).str.strip()
    d0 = d0[d0["Entity Type"].ne("")]  # drop blank labels

    if mode == "count":
        # unique entities per type then value_counts with stable col names
        s = (d0[["Entity Name", "Entity Type"]]
                .drop_duplicates()["Entity Type"])
        data = (s.value_counts(dropna=False)
                  .rename_axis("Entity Type")
                  .reset_index(name="Value"))
    else:
        # aggregate USD per type, unique entity-type pairs
        if "USD Value" not in d0.columns:
            return _empty_pie("No USD values")
        dedup = d0[["Entity Name", "Entity Type", "USD Value"]].drop_duplicates(
            subset=["Entity Name", "Entity Type"]
        )
        data = (dedup.groupby("Entity Type", as_index=False, observed=False)["USD Value"]
                    .sum()
                    .rename(columns={"USD Value": "Value"}))

    # if nothing to plot
    if data.empty or "Entity Type" not in data.columns or "Value" not in data.columns:
        return _empty_pie()

    # use only types we know the palette for (optional)
    data["Entity Type"] = data["Entity Type"].astype(str)
    cat_order = [t for t in ORDER if t in set(data["Entity Type"])]

    fig = px.pie(
        data,
        values="Value",
        names="Entity Type",
        hole=0.6,
        color="Entity Type",
        color_discrete_map=COLOR_MAP,
        category_orders={"Entity Type": cat_order} if cat_order else None,
    )

    if mode == "usd":
        customdata = data["Value"].map(format_usd)
    else:
        customdata = None

    fig.update_traces(
        texttemplate="%{percent:.1%}",
        textfont=dict(size=18),
        hovertemplate="<b>%{label}</b><br>"
                    + ("Count: <b>%{value:,}</b>" if mode == "count"
                        else "USD Value: <b>%{customdata}</b>")
                    + "<extra></extra>",
        customdata=customdata
    )

    # Main domain watermark
    fig.add_annotation(
        text=WATERMARK_TEXT,
        x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=13, color="white"),
        opacity=0.7,
        xanchor="center",
        yanchor="middle",
    )
    # Powered-by footer
    fig.add_annotation(
        text="powered by <b>F5Crypto.com</b>",
        x=1, y=0,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=12, color="white"),
        #bgcolor="white",
        #bordercolor="white", borderwidth=4,
        xanchor="right", yanchor="bottom", opacity=0.7)

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.25, xanchor="center", x=0.5),
        hoverlabel=dict(align="left"),
        height=400,
        margin=dict(t=40, b=20, l= 50, r= 50),
    )

    return fig


def top_countries_by_entity_count(df):
    # Step 1: Group by Country and Entity Type to count unique entities
    grouped = (
        df.groupby(['Country', 'Entity Type'])['Entity Name']
        .nunique()
        .reset_index(name='Entity Count')
    )

    # Step 2: Get top 5 countries by total entity count
    top_countries = (
        grouped.groupby('Country')['Entity Count']
        .sum()
        .nlargest(10)
        .index.tolist()
    )

    filtered = grouped[grouped['Country'].isin(top_countries)]

    # Step 3: Prepare custom hover text with aggregated breakdown per country
    country_breakdowns = (
        filtered.groupby('Country')
        .apply(lambda d: f"<b>{d.name}</b><br>" + "<br>".join(
            [f"{row['Entity Type']}: <b>{int(row['Entity Count'])}</b>" for _, row in d.sort_values('Entity Count', ascending=False).iterrows()])
        ).to_dict()
    )

    #filtered['Custom Hover'] = filtered['Country'].map(country_breakdowns)
    filtered = filtered.copy()
    filtered.loc[:, 'Custom Hover'] = filtered['Country'].map(country_breakdowns)
    filtered['Entity Type'] = filtered['Entity Type'].astype(str).str.strip()

    # Map country column to emoji
    flag_series = (
        filtered["Country"]
        .astype("string")
        .map(lambda c: country_emoji_map.get(c, "🏳️"))
    )

    # Prepend flag to Entity Name
    filtered["Country"] = flag_series.fillna("🏳️") + " " + filtered["Country"].astype("string")

    # Step 4: Create stacked bar chart
    fig = px.bar(
        filtered,
        x='Entity Count',
        y='Country',
        color='Entity Type',
        color_discrete_map=COLOR_MAP,
        category_orders={'Entity Type': list(TYPE_PALETTE.keys())},  # consistent legend/stack
        orientation='h',
        labels={'Entity Count': 'Entities'},
        custom_data=['Custom Hover'],
        text=None  # Remove individual labels
    )

    # Step 5: Add total text at end of each full bar (sum by country)
    totals = (
        filtered.groupby('Country')['Entity Count']
        .sum()
        .sort_values(ascending=True)  # Match y-axis order
    )

    for country, total in totals.items():
        fig.add_annotation(
            x=total,
            y=country,
            text=str(int(total)),
            showarrow=False,
            font=dict(size=16, color="white"),
            xanchor='left',
            yanchor='middle'
        )

    add_watermark(fig)


    # Step 6: Final layout adjustments
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        textfont=dict(size=12),
        hoverlabel=dict(align="left")
    )

    fig.update_layout(
        #height=400,
        margin=dict(t=40, b=20, l=40, r= 30),
        yaxis=dict(categoryorder='total ascending', title=None, tickfont=dict(size=14), fixedrange=True),
        xaxis=dict(title=None, tickprefix = "$", fixedrange=True),
        legend_title_text=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.25, xanchor="center", x=0.5),
    )

    return fig


def top_countries_by_usd_value(df):
    # Step 1: Group by Country and Entity Type to get USD sums
    grouped = (
        df.groupby(['Country', 'Entity Type'])['USD Value']
        .sum()
        .reset_index()
    )

    # Step 2: Get top 5 countries by total USD value
    top_countries = (
        grouped.groupby('Country')['USD Value']
        .sum()
        .nlargest(10)
        .index.tolist()
    )

    filtered = grouped[grouped['Country'].isin(top_countries)]

    # Step 3: Prepare custom hover text with aggregated breakdown per country
    country_breakdowns = (
        filtered.groupby('Country')
        .apply(lambda d: f"<b>{d.name}</b><br>" + "<br>".join(
            [f"{row['Entity Type']}: <b>{format_usd(row['USD Value'])}</b>" for _, row in d.sort_values('USD Value', ascending=False).iterrows()])
        ).to_dict()
    )

    filtered['Custom Hover'] = filtered['Country'].map(country_breakdowns)
    filtered['Entity Type'] = filtered['Entity Type'].astype(str).str.strip()

    # Map country column to emoji
    flag_series = (
        filtered["Country"]
        .astype("string")
        .map(lambda c: country_emoji_map.get(c, "🏳️"))
    )

    # Prepend flag to Entity Name
    filtered["Country"] = flag_series.fillna("🏳️") + " " + filtered["Country"].astype("string")

    # Step 4: Create stacked bar chart
    fig = px.bar(
        filtered,
        x='USD Value',
        y='Country',
        color='Entity Type',
        color_discrete_map=COLOR_MAP,
        category_orders={'Entity Type': list(TYPE_PALETTE.keys())},  # consistent legend/stack
        orientation='h',
        labels={'USD Value': 'USD'},
        custom_data=['Custom Hover'],
        text=None
    )

    # Step 5: Add total value at end of bar
    totals = (
        filtered.groupby('Country')['USD Value']
        .sum()
        .sort_values(ascending=True)
    )

    for country, total in totals.items():
        fig.add_annotation(
            x=total,
            y=country,
            text=format_usd(total),
            showarrow=False,
            font=dict(size=14, color='white'),
            xanchor='left',
            yanchor='middle'
        )

    add_watermark(fig)


    # Step 6: Final layout adjustments
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        textfont=dict(size=12),
        hoverlabel=dict(align="left")
    )

    fig.update_layout(
        #height=400,
        margin=dict(t=20, b=20, l=40, r= 30),
        yaxis=dict(categoryorder='total ascending', title=None, tickfont=dict(size=14), fixedrange=True),
        xaxis=dict(title=None, tickprefix = "$", fixedrange=True),
        legend_title_text=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    return fig


def entity_ranking(df, by="USD", top_n: int = 5, show_totals=True):
    value_col = "USD Value" if by == "USD" else "Holdings (Unit)"

    grouped = (
        df.groupby(["Entity Name", "Crypto Asset"], as_index=False)[value_col]
          .sum()
    )

    usd_totals = (
        df.groupby("Entity Name")["USD Value"]
          .sum()
          .sort_values(ascending=False)
    )

    top_entities = usd_totals.head(top_n).index.tolist()
    grouped = grouped[grouped["Entity Name"].isin(top_entities)].copy()
    grouped["USD Total"] = grouped["Entity Name"].map(usd_totals)

    if by == "USD":
        grouped["Text"] = grouped[value_col].apply(format_usd)
        hover_map = (
            grouped.groupby("Entity Name")
                   .apply(lambda d: f"<b>{d.name}</b><br>" + "<br>".join(
                       [f"{r['Crypto Asset']}  <b>{format_usd(r[value_col])}</b>" for _, r in d.iterrows()]
                   ))
                   .to_dict()
        )
    else:
        grouped["Text"] = grouped[value_col].apply(lambda x: f"{int(x):,}")
        hover_map = (
            grouped.groupby("Entity Name")
                   .apply(lambda d: f"<b>{d.name}</b><br>" + "<br>".join(
                       [f"{r['Crypto Asset']}  <b>{int(r[value_col]):,}</b>" for _, r in d.iterrows()]
                   ))
                   .to_dict()
        )
    grouped["Custom Hover"] = grouped["Entity Name"].map(hover_map)

    sorted_entities = (
        grouped.groupby("Entity Name")["USD Total"]
               .max()
               .sort_values(ascending=False)
               .index.tolist()
    )
    grouped["Entity Name"] = pd.Categorical(grouped["Entity Name"], categories=sorted_entities, ordered=True)
    grouped = grouped.sort_values(["Entity Name", "Crypto Asset"])

    fig = px.bar(
        grouped,
        y="Entity Name",
        x=value_col,
        color="Crypto Asset",
        orientation="h",
        barmode="stack",
        custom_data=["Custom Hover"],
        color_discrete_map=COLORS,
        category_orders={"Entity Name": sorted_entities},
    )

    fig.update_traces(
        textposition="none",
        hovertemplate="%{customdata[0]}<extra></extra>"
    )

    if show_totals:
        totals = (
            grouped.groupby("Entity Name")[value_col]
                   .sum()
                   .reindex(sorted_entities)
        )
        if by == "USD":
            total_text = [format_usd(v) for v in totals.tolist()]
        else:
            total_text = [f"{int(v):,}" for v in totals.tolist()]

        fig.add_trace(go.Scatter(
            x=totals.values,                   # nudge outside the stack
            y=sorted_entities,
            mode="text",
            text=total_text,
            textposition="middle right",
            textfont=dict(size=12),
            hoverinfo="skip",
            showlegend=False
        ))

    # watermark
    add_watermark(fig)

    max_total = float(totals.max() or 0.0)

    fig.update_layout(
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(
            title=None,
            tickvals=fig.layout.xaxis.tickvals,
            range=[0, max_total * 1.15],
            ticktext=[format_usd(v) for v in fig.layout.xaxis.tickvals] if fig.layout.xaxis.tickvals else None,
            fixedrange=True,
            showgrid=True,
        ),
        yaxis=dict(
            title=None,
            categoryorder="array",
            categoryarray=sorted_entities,
            autorange="reversed",      # top entity on top
            fixedrange=True
        ),
        bargap=0.25,
        bargroupgap=0.05,
        font=dict(size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            title=""
        ),
        hoverlabel=dict(align="left")
    )

    # pending ribbon when relevant
    _pending_ctx = st.session_state.get("pending_context", {})
    if _pending_ctx.get("active"):
        fig.add_annotation(
            text="PENDING TREASURIES",
            x=0.99, y=0.05, xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=12, color="black"),
            bgcolor="white",
            bordercolor="white", borderwidth=4,
            xanchor="right", yanchor="bottom", opacity=0.9
        )

    return fig


def entity_supply_share_ranking(df_filtered: pd.DataFrame, top_n: int = 5) -> go.Figure:

    d = df_filtered.copy()
    d["Holdings (Unit)"] = pd.to_numeric(d["Holdings (Unit)"], errors="coerce").fillna(0.0)
    d = d[d["Holdings (Unit)"] > 0]

    grp = (
        d.groupby(["Entity Name", "Crypto Asset"], as_index=False)["Holdings (Unit)"]
         .sum()
         .rename(columns={"Holdings (Unit)": "units"})
    )

    grp["supply"] = grp["Crypto Asset"].map(supply_caps).astype(float)
    grp = grp[np.isfinite(grp["supply"]) & (grp["supply"] > 0)]
    grp["share"] = grp["units"] / grp["supply"]

    if grp.empty:
        return go.Figure()

    totals = (
        grp.groupby("Entity Name")["share"]
           .sum()
           .sort_values(ascending=True)
    )
    top_entities = totals.tail(int(top_n)).index.tolist()

    grp = grp[grp["Entity Name"].isin(top_entities)].copy()
    grp["Entity Name"] = pd.Categorical(grp["Entity Name"], categories=top_entities, ordered=True)
    grp = grp.sort_values(["Entity Name", "Crypto Asset"])

    fig = px.bar(
        grp,
        y="Entity Name",
        x="share",
        color="Crypto Asset",
        orientation="h",
        barmode="stack",
        color_discrete_map=COLORS,
        category_orders={"Entity Name": top_entities},
        custom_data=["Entity Name", "Crypto Asset", "units", "share"]
    )
    # watermark
    add_watermark(fig)

    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}: <b>%{customdata[2]:,.0f}</b> units<br>share <b>%{customdata[3]:.2%}</b><extra></extra>",
        selector=dict(type="bar")
    )

    # assign the correct text per trace
    for tr in fig.data:
        mask = (grp["Crypto Asset"] == tr.name)
        tr.text = [f"{v:.2%}" for v in grp.loc[mask, "share"]]
        tr.textposition = "outside"

    tot_share = grp.groupby("Entity Name")["share"].sum()
    max_share = float(tot_share.max() or 0.0)
    fig.update_layout(
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(visible=True, tickformat=".0%",range=[0, max_share * 1.15], fixedrange=True, title=None, showgrid=True),
        yaxis=dict(title=None, fixedrange=True, showgrid=False, autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, title=""),
        hoverlabel=dict(align="left")
    )
    return fig


def asset_totals_usd_bar(df_filtered: pd.DataFrame, top_n: int = 5) -> go.Figure:
    d = df_filtered.copy()
    d["USD Value"] = pd.to_numeric(d["USD Value"], errors="coerce").fillna(0.0)
    agg = (
        d.groupby("Crypto Asset", as_index=False)["USD Value"]
         .sum()
         .sort_values("USD Value", ascending=True)
         .tail(top_n)
    )

    fig = go.Figure(go.Bar(
        x=agg["USD Value"],
        y=agg["Crypto Asset"],
        orientation="h",
        marker=dict(color=[COLORS.get(a, "#7f8c8d") for a in agg["Crypto Asset"]]),
        text=[format_usd(v) for v in agg["USD Value"]],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br><b>%{text}</b><extra></extra>",
    ))

    fig.update_layout(
        margin=dict(l=80, r=40, t=50, b=40),
        xaxis=dict(
            title=None,
            tickvals=fig.layout.xaxis.tickvals,
            ticktext=[format_usd(v) for v in fig.layout.xaxis.tickvals] if fig.layout.xaxis.tickvals else None,
            fixedrange=True,
            showgrid=True,
        ),
        yaxis=dict(fixedrange=True, showgrid=False, title=None, tickfont=dict(size=11)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.25,
        showlegend=False,
    )

    # watermark
    add_watermark(fig)

    return fig


def asset_totals_supply_share_bar(df_filtered: pd.DataFrame, top_n: int = 5) -> go.Figure:
    d = df_filtered.copy()
    d["Holdings (Unit)"] = pd.to_numeric(d["Holdings (Unit)"], errors="coerce").fillna(0.0)
    agg = (
        d.groupby("Crypto Asset", as_index=False)["Holdings (Unit)"]
         .sum()
         .rename(columns={"Holdings (Unit)": "units"})
    )

    _supply = pd.Series(supply_caps)
    agg["supply"] = agg["Crypto Asset"].map(_supply)
    agg["supply"] = pd.to_numeric(agg["supply"], errors="coerce")
    agg = agg[agg["supply"].gt(0) & agg["units"].ge(0)]
    agg["share"] = agg["units"] / agg["supply"]

    agg = agg.sort_values("share", ascending=True).tail(top_n)

    fig = go.Figure(go.Bar(
        x=agg["share"],
        y=agg["Crypto Asset"],
        orientation="h",
        marker=dict(color=[COLORS.get(a, "#7f8c8d") for a in agg["Crypto Asset"]]),
        text=[f"{v:.1%}" for v in agg["share"]],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br><b>%{text}</b><extra></extra>",
    ))

    fig.update_layout(
        margin=dict(l=80, r=40, t=50, b=40),
        xaxis=dict(fixedrange=True, tickformat=".0%", showgrid=True),
        yaxis=dict(fixedrange=True, showgrid=False, tickfont=dict(size=11)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.25,
        showlegend=False,
    )
    fig.update_traces(
        hovertemplate="%{y}<br><b>%{text}</b><extra></extra>",
        text=[f"{v:.2%}" for v in agg["share"]],
        textposition="outside",
        cliponaxis=False
    )
    # watermark
    add_watermark(fig)

    return fig

def diversification_bar(df_filtered):

    df = df_filtered.copy()
    counts = (
        df.groupby("Entity Name")["Crypto Asset"]
          .nunique()
          .reset_index(name="Asset Count")
          .sort_values("Asset Count", ascending=True)
    )
    counts = counts.tail(10)
    fig = px.bar(
        counts,
        y="Entity Name",
        x="Asset Count",
        orientation="h",
        text="Asset Count",
        #height=520,
        color="Asset Count",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        title="",
        xaxis_title="Number of Assets Held",
        yaxis_title="",
        margin=dict(l=50, r=40, t=20, b=20),
        coloraxis_showscale=False,
        yaxis=dict(fixedrange=True),
        xaxis=dict(fixedrange=True),
    )
    fig.update_traces(textposition="outside")
    add_watermark(fig)

    return fig


def target_progress(df_filtered):

    df = df_filtered.copy()
    #df = df[df["DAT"] == 1]
    for c in ["USD Value", "target_usd"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df = df[df["target_usd"] > 0]
    df["Progress %"] = (df["USD Value"] / df["target_usd"]) * 100
    df["Progress %"] = df["Progress %"].clip(0, 200)

    prog = (
        df.groupby("Entity Name", as_index=False)
          .agg(progress=("Progress %", "mean"),
               status=("status", lambda x: x.mode().iat[0] if len(x) else ""))
          .sort_values("progress", ascending=True)
          .tail(20)
    )

    fig = px.bar(
        prog,
        y="Entity Name",
        x="progress",
        color="status",
        orientation="h",
        text=prog["progress"].map(lambda x: f"{x:.0f}%"),
        color_discrete_sequence=px.colors.qualitative.Plotly,
        #height=520,
    )
    fig.update_layout(
        title="",
        xaxis_title="Progress (%)",
        yaxis_title="",
        margin=dict(l=50, r=40, t=20, b=20),
        yaxis=dict(fixedrange=True),
        xaxis=dict(fixedrange=True),
        showlegend=False
    )
    fig.update_traces(textposition="outside")
    add_watermark(fig)

    return fig


def premium_vs_size(df_filtered):

    df = df_filtered.copy()
    df = df[df["DAT"] == 1]  # DAT Wrappers only
    for c in ["Premium", "USD Value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Premium", "USD Value"])
    df = df[df["USD Value"] > 0]
    if df.empty:
        return px.scatter(title="No DAT Wrapper data for Premium-vs-Size map")

    df["log_usd"] = np.log10(df["USD Value"])
    df["Hover"] = (
        "<b>" + df["Entity Name"] + "</b><br>"
        "Asset " + df["Crypto Asset"] + "<br>"
        "Premium " + df["Premium"].map("{:.1f}%".format) + "<br>"
        "USD Value " + df["USD Value"].map(lambda v: f"${v/1e6:.1f} M")
    )

    y_range = [-200, 1600]
    size_min, size_max = 10, 100
    s_scaled = np.interp(
        np.log10(df["USD Value"]),
        (df["log_usd"].min(), df["log_usd"].max()),
        (size_min, size_max),
    )

    fig = px.scatter(
        df,
        x="log_usd",
        y="Premium",
        color="Crypto Asset",
        size=s_scaled,
        color_discrete_map=COLORS,
        custom_data=["Hover"],
        opacity=0.9,
        height=520,
    )
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")

    fig.update_layout(
        title="",
        xaxis_title="log₁₀ (USD Value)",
        yaxis_title="Premium (%)",
        yaxis_range=y_range,
        yaxis=dict(fixedrange=True),
        xaxis=dict(fixedrange=True),
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
        hoverlabel=dict(align="left")
    )
    fig.update_xaxes(
        tickvals=[4,5,6,7,8,9,10,11],
        ticktext=["10k","100k","1M","10M","100M","1B","10B","100B"]
    )
    add_watermark(fig)

    return fig



def _clip_name(s: str, n: int = 20) -> str:
    s = str(s).strip()
    return (s[: n - 1] + "…") if len(s) > n else s


def treemap_composition(df, mode: str = "country_type"):
    """
    mode:
      - "country_type": Country → Entity Type (area = USD)
      - "type_entity":  Entity Type → Entity Name (area = USD)
    """
    d = df.copy()
    d = d[d["USD Value"] > 0]  # guard: only positive areas
    d["Country"] = d["Country"].fillna("Decentralized").astype(str).str.strip()

    order = list(TYPE_PALETTE.keys())
    d["Entity Type"] = (
        d["Entity Type"].fillna("Other").astype(str).str.strip().replace({"": "Other"})
    )
    d.loc[~d["Entity Type"].isin(order), "Entity Type"] = "Other"

    # palette → CSS colors
    color_map = {k: f"rgb({r},{g},{b})" for k, (r, g, b) in TYPE_PALETTE.items()}
    parent_color = "rgb(22,24,28)"  # dark tile for parents (Streamlit dark)

    # ---------- helper to create per-leaf UNITS lines ----------
    def _format_units_lines(grouped_units_rows, key_cols):
        grp = grouped_units_rows.sort_values("Holdings (Unit)", ascending=False)
        out = {}
        for keys, sub in grp.groupby(key_cols, observed=True):
            lines = [f"{a}: {int(u):,}".replace(",", " ") for a, u in zip(sub["Crypto Asset"], sub["Holdings (Unit)"])]
            out[tuple(keys if isinstance(keys, tuple) else (keys,))] = "<br>".join(lines)
        return out

    if mode == "type_entity":
        # Entity Type → Entity Name
        units_rows = (
            d.groupby(["Entity Type", "Entity Name", "Crypto Asset"], as_index=False, observed=True)
             .agg(**{"Holdings (Unit)": ("Holdings (Unit)", "sum")})
        )
        units_text_map = _format_units_lines(units_rows, ["Entity Type", "Entity Name"])

        grouped = (
            d.groupby(["Entity Type", "Entity Name"], as_index=False, observed=True)
             .agg(USD_Value=("USD Value", "sum"),
                  Country=("Country", lambda x: x.mode().iat[0] if len(x) else ""))
        )

        fig = px.treemap(
            grouped,
            path=["Entity Type", "Entity Name"],
            values="USD_Value",
            color="Entity Type",
            color_discrete_map=color_map,
            custom_data=["Country", "USD_Value"],
        )

        # Post-style: dark parents, white centered text, units lines in leaves
        tr = fig.data[0]
        labels, parents = list(tr.labels), list(tr.parents)
        customs = tr.customdata if tr.customdata is not None else [[None, None]] * len(labels)

        colors = list(getattr(tr.marker, "colors", [None] * len(labels)))
        if not colors or len(colors) != len(labels):
            colors = [""] * len(labels)

        text_labels, hovertext = [], []
        for i, (lab, par) in enumerate(zip(labels, parents)):
            if par == "":  # top-level type node
                colors[i] = parent_color
                total_usd = grouped.loc[grouped["Entity Type"] == lab, "USD_Value"].sum()
                hovertext.append(f"<b>{lab}</b><br>USD: ${total_usd:,.0f}")
                #text_labels.append(lab)
                text_labels.append(f"<b>{_clip_name(lab)}</b>")

            else:          # leaf = entity
                et = par
                units_lines = units_text_map.get((et, lab), "")
                #text_labels.append(f"{lab}<br>{units_lines}" if units_lines else lab)
                text_labels.append(
                    f"<b>{_clip_name(lab)}</b><br>{units_lines}" if units_lines else f"<b>{_clip_name(lab)}</b>"
                )
                country = customs[i][0]; usd = customs[i][1]
                hovertext.append(f"<b>{lab}</b><br>Country: {country}<br>USD: ${usd:,.0f}")

        tr.marker.colors   = colors
        tr.text            = text_labels
        tr.texttemplate    = "%{text}"
        tr.textfont.color  = "white"
        tr.textposition    = "middle center"
        tr.hovertext       = hovertext
        tr.hovertemplate   = "%{hovertext}<extra></extra>"

    else:
        # Country → Entity Type
        units_rows = (
            d.groupby(["Country", "Entity Type", "Crypto Asset"], as_index=False, observed=True)
             .agg(**{"Holdings (Unit)": ("Holdings (Unit)", "sum")})
        )
        units_text_map = _format_units_lines(units_rows, ["Country", "Entity Type"])

        grouped = (
            d.groupby(["Country", "Entity Type"], as_index=False, observed=True)
             .agg(USD_Value=("USD Value", "sum"),
                  Entities=("Entity Name", "nunique"))
        )
        # Map country column to emoji
        flag_series = (
            grouped["Country"]
            .astype("string")
            .map(lambda c: country_emoji_map.get(c, "🏳️"))
        )

        # Prepend flag to Entity Name
        grouped["Country"] = flag_series.fillna("🏳️") + " " + grouped["Country"].astype("string")

        fig = px.treemap(
            grouped,
            path=["Country", "Entity Type"],
            values="USD_Value",
            color="Entity Type",
            color_discrete_map=color_map,
            custom_data=["Entities", "USD_Value"],
        )

        tr = fig.data[0]
        labels, parents = list(tr.labels), list(tr.parents)
        customs = tr.customdata if tr.customdata is not None else [[None, None]] * len(labels)

        colors = list(getattr(tr.marker, "colors", [None] * len(labels)))
        if not colors or len(colors) != len(labels):
            colors = [""] * len(labels)

        text_labels, hovertext = [], []
        for i, (lab, par) in enumerate(zip(labels, parents)):
            if par == "":  # country node
                colors[i] = parent_color
                csum = grouped.loc[grouped["Country"] == lab, "USD_Value"].sum()
                ent  = int(grouped.loc[grouped["Country"] == lab, "Entities"].sum())
                hovertext.append(f"<b>{lab}</b><br>Entities: {ent}<br>USD: ${csum:,.0f}")
                #text_labels.append(lab)
                text_labels.append(f"<b>{_clip_name(lab)}</b>")

            else:          # leaf = entity type inside country
                units_lines = units_text_map.get((par, lab), "")
                #text_labels.append(f"{lab}<br>{units_lines}" if units_lines else lab)
                text_labels.append(
                    f"<b>{_clip_name(lab)}</b><br>{units_lines}" if units_lines else f"<b>{_clip_name(lab)}</b>"
                )
                ent = customs[i][0]; usd = customs[i][1]
                hovertext.append(f"<b>{par}</b> · {lab}<br>Entities: {ent}<br>USD: ${usd:,.0f}")

        tr.marker.colors   = colors
        tr.text            = text_labels
        tr.texttemplate    = "%{text}"
        tr.textfont.color  = "white"
        tr.textposition    = "middle center"
        tr.hovertext       = hovertext
        tr.hovertemplate   = "%{hovertext}<extra></extra>"

    # Global layout polish
    fig.update_traces(
        root_color="rgba(0,0,0,0)",
        tiling=dict(pad=3),
        marker=dict(line=dict(width=0)),
        pathbar={"visible": False},
    )
    fig.update_layout(
        height=520,
        margin=dict(l=8, r=8, t=8, b=8),
        font=dict(size=14, color="white"),
        uniformtext=dict(minsize=11, mode="hide"),   
        hoverlabel=dict(align="left"),
        legend_traceorder="normal",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    # subtle watermark
    fig.add_annotation(
        text=WATERMARK_TEXT,
        x=0.5, y=0.45, xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=30, color="black"),
        opacity=0.7,
        xanchor="center",
        yanchor="middle",
    )
    # Powered-by footer
    fig.add_annotation(
        text="powered by <b>F5Crypto.com</b>",
        x=1, y=-0.02,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=12, color="white"),
        #bgcolor="white",
        #bordercolor="white", borderwidth=4,
        xanchor="right", yanchor="bottom", opacity=0.7
    )
    return fig


def lorenz_curve_chart(p, L, asset: str | None = None):
    """Lorenz curve; if a single asset is passed, color the line with its color."""

    line_color = COLORS.get(asset, default_blue) if asset else default_blue

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=p, y=L, mode="lines+markers",
        name="Lorenz",
        line=dict(color=line_color, width=3),
        marker=dict(color=line_color, size=6),
        hovertemplate="Population share: %{x:.1%}<br>Weight share: %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode="lines",
        name="Equality", line=dict(dash="dash", width=1), hoverinfo="skip"
    ))
    fig.update_layout(
        height=350,
        margin=dict(l=50, r=50, t=20, b=20),
        xaxis=dict(title="Cumulative share of groups", tickformat=".0%", range=[0,1], fixedrange=True),
        yaxis=dict(title="Cumulative share of weight", tickformat=".0%", range=[0,1], fixedrange=True),
        showlegend=False,
        hoverlabel=dict(align="left")
    )
    add_watermark(fig)
    return fig


def _entity_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to one row per entity with:
      - MarketCap  (expects column 'Market Cap')
      - CryptoNAV  (sum of USD Value across assets)
      - Exposure%  (CryptoNAV / MarketCap)
      - Premium%   (uses 'Premium %' if present, else (MarketCap - MNAV)/MNAV if MNAV present)
      - Entity Type, Country (mode)
    """
    g = (df.groupby("Entity Name", as_index=False)
           .agg(**{
               "CryptoNAV": ("USD Value", "sum"),
               "MarketCap": ("Market Cap", "max"),
               "Entity Type": ("Entity Type", lambda s: s.mode().iat[0] if len(s) else None),
               "Country": ("Country", lambda s: s.mode().iat[0] if len(s) else None),
               "MNAV": ("mNAV", lambda s: s.dropna().iloc[0] if "mNAV" in df.columns and s.dropna().size else np.nan),
               "PremiumCol": ("Premium", lambda s: s.dropna().iloc[0] if "Premium" in df.columns and s.dropna().size else np.nan),
           }))
    # Exposure
    g["Exposure %"] = np.where(g["MarketCap"] > 0, g["CryptoNAV"] / g["MarketCap"] * 100.0, np.nan)

    # Premium
    prem = g["PremiumCol"].copy()
    if prem.isna().all() and "MNAV" in g.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            prem = (g["MarketCap"] - g["MNAV"]) / g["MNAV"] * 100.0
    g["Premium %"] = prem

    # Core proxy (for decomposition)
    g["Core Proxy"] = np.maximum(g["MarketCap"] - g["CryptoNAV"], 0.0)

    return g[["Entity Name","Entity Type","Country","MarketCap","CryptoNAV","Core Proxy","Exposure %","Premium %"]]


def exposure_ladder_bar(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    snap = _entity_snapshot(df).dropna(subset=["MarketCap"])
    snap = snap[snap["MarketCap"] > 0].copy()
    snap = snap.sort_values("Exposure %", ascending=True).tail(top_n)

    DEFAULT_BAR = "#66cded"
    unique_assets = sorted(pd.Series(df.get("Crypto Asset", [])).dropna().unique().tolist())
    if len(unique_assets) == 1:
        asset_color = COLORS.get(unique_assets[0], DEFAULT_BAR)
        bar_colors = [asset_color] * len(snap)
    else:
        bar_colors = [DEFAULT_BAR] * len(snap)

    fig = go.Figure(go.Bar(
        x=snap["Exposure %"],
        y=snap["Entity Name"],
        orientation="h",
        text=[f"{v:.1f}%" if np.isfinite(v) else "—" for v in snap["Exposure %"]],
        textposition="outside",
        marker_color=bar_colors,#[COLOR_MAP.get(t, "rgba(200,200,200,1)") for t in snap["Entity Type"]],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Exposure: %{x:.1f}%<br>"
            "Crypto-NAV: %{customdata[0]}<br>"
            "Market Cap: %{customdata[1]}<extra></extra>"
        ),
        customdata=np.c_[snap["CryptoNAV"].apply(lambda x: format_usd(x)), snap["MarketCap"].apply(lambda x: format_usd(x))],
    ))
    fig.update_layout(
        height=max(400, 25 * len(snap) + 40),
        margin=dict(l=50, r=20, t=20, b=20),
        xaxis=dict(title="Crypto Treasury as % of Market Cap", ticksuffix="%", fixedrange=True, showgrid=True),
        yaxis=dict(title=None, fixedrange=True),
        showlegend=False,
        hoverlabel=dict(align="left"),
    )

    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(r=40))

    add_watermark(fig)


    return fig


def mcap_decomposition_bar(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    snap = _entity_snapshot(df).dropna(subset=["MarketCap"])
    snap["Core Proxy"] = np.maximum(snap["MarketCap"] - snap["CryptoNAV"], 0.0)

    snap = snap[snap["MarketCap"] > 0].copy()

    snap = snap.sort_values("MarketCap", ascending=True).tail(top_n)
    total_labels = snap["MarketCap"].map(lambda x: format_usd(x))

    fig = go.Figure()
    fig.add_bar(
        name="CryptoNAV",
        x=snap["CryptoNAV"],
        y=snap["Entity Name"],
        orientation="h",
        marker_color="#43d1a0",
        customdata=np.c_[snap["MarketCap"].apply(lambda x: format_usd(x)), snap["CryptoNAV"].apply(lambda x: format_usd(x)), snap["Core Proxy"].apply(lambda x: format_usd(x))],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Market Cap: %{customdata[0]}<br>"
            "Crypto-NAV: %{customdata[1]}<br>"
            "Core Proxy: %{customdata[2]}<extra></extra>"
        ),
    )
    fig.add_bar(
        name="Core Proxy",
        x=snap["Core Proxy"],
        y=snap["Entity Name"],
        orientation="h",
        text=total_labels,
        textposition="outside",
        cliponaxis=False,
        marker_color="#8892a6",
        customdata=np.c_[snap["MarketCap"].apply(lambda x: format_usd(x)), snap["CryptoNAV"].apply(lambda x: format_usd(x)), snap["Core Proxy"].apply(lambda x: format_usd(x))],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Market Cap: %{customdata[0]}<br>"
            "Crypto-NAV: %{customdata[1]}<br>"
            "Core Proxy: %{customdata[2]}<extra></extra>"
        ),
    )
    fig.update_layout(
        barmode="stack",
        height=max(400, 25 * len(snap) + 40),
        margin=dict(l=50, r=20, t=20, b=20),
        xaxis=dict(title="Market Cap (stacked)", fixedrange=True, showgrid=True),
        yaxis=dict(title=None, fixedrange=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hoverlabel=dict(align="left"),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(r=40))
    fig.update_layout(
        barmode="stack",
        xaxis=dict(title="Market Cap in USD (stacked)", tickprefix="$", separatethousands=True),
    )
    add_watermark(fig)

    return fig


def _entity_snapshot_by_asset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-entity, per-asset snapshot:
      - MarketCap (one per entity; we take max)
      - AssetNAV = USD Value per asset
    """
    g = (df.groupby(["Entity Name", "Crypto Asset"], as_index=False)
            .agg(AssetNAV=("USD Value", "sum"),
                 MarketCap=("Market Cap", "max"),
                 EntityType=("Entity Type", lambda s: s.mode().iat[0] if len(s) else None),
                 Country=("Country", lambda s: s.mode().iat[0] if len(s) else None)))
    g = g.dropna(subset=["MarketCap"])
    g = g[g["MarketCap"] > 0]
    return g


def mnav_comparison_bar(df: pd.DataFrame, top_n: int = 20, max_mnav: float | None = None) -> go.Figure:

    snap = _entity_snapshot(df).dropna(subset=["MarketCap"])
    snap = snap[snap["MarketCap"] > 0].copy()

    with np.errstate(divide="ignore", invalid="ignore"):
        snap["mNAV"] = np.where(snap["CryptoNAV"] > 0, snap["MarketCap"] / snap["CryptoNAV"], np.nan)

    # Optional outlier cap BEFORE picking Top-N
    if max_mnav is not None:
        snap = snap[snap["mNAV"] <= float(max_mnav)]

    # 1) choose Top-N BY CryptoNAV
    d = snap.dropna(subset=["mNAV"]).sort_values("CryptoNAV", ascending=False).head(top_n).copy()
    if d.empty:
        return go.Figure()

    # 2) DISPLAY order: mNAV descending → left = largest, right = lowest
    d = d.sort_values("mNAV", ascending=False)

    colors   = np.where(d["mNAV"] >= 1.0, "#43d1a0", "#f04438")  # premium green / discount red
    text_lbl = [f"{v:.2f}×" if np.isfinite(v) else "—" for v in d["mNAV"]]

    ymax = float(np.nanmax(d["mNAV"]))
    y_range = [0, ymax * 1.15] if np.isfinite(ymax) and ymax > 0 else None

    fig = go.Figure(go.Bar(
        x=d["Entity Name"],
        y=d["mNAV"],
        marker_color=colors,
        text=text_lbl,
        textposition="outside",
        cliponaxis=False,
        customdata=np.c_[d["MarketCap"], d["CryptoNAV"]],
        hovertemplate="<b>%{x}</b><br>mNAV: %{y:.2f}×<br>Market Cap: %{customdata[0]:$,.0f}<br>Crypto-NAV: %{customdata[1]:$,.0f}<extra></extra>",
    ))

    fig.add_hline(y=1.0, line_width=1, line_dash="dash", line_color="#cbd5e1")

    fig.update_layout(
        height=500,
        margin=dict(l=50, r=30, t=20, b=20),
        xaxis=dict(title=None, tickangle=45, automargin=True, fixedrange=True),
        yaxis=dict(title="mNAV (×)", range=y_range, fixedrange=True),
        showlegend=False,
        hoverlabel=dict(align="left"),
    )

    # subtle watermark
    fig.add_annotation(
        text=WATERMARK_TEXT,
        x=0.5, y=0.45, xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=20, color="white"),
        opacity=0.7,
        xanchor="center",
        yanchor="middle",
    )
    # Powered-by footer
    fig.add_annotation(
        text="<i>powered by <b>F5Crypto.com</b></i>",
        x=1, y=0.9,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=12, color="white"),
        #bgcolor="white",
        #bordercolor="white", borderwidth=4,
        xanchor="right", yanchor="bottom", opacity=0.7
    )
    return fig


def corporate_sensitivity_bar(
    df: pd.DataFrame,
    shock_pct: float | None = None,              # e.g. +0.10 for +10% (uniform)
    per_asset_shocks: dict[str, float] | None = None,  # {"BTC":0.1,"ETH":-0.1,...}
    top_n: int = 25
) -> go.Figure:
    """
    Computes ΔMarketCap implied by crypto price shocks:
      ΔMC (USD) = Σ_asset (AssetNAV_asset * shock_asset)
    Implied equity % move ~ ΔMC / MarketCap  (shares assumed constant).

    If per_asset_shocks is provided, it takes precedence over shock_pct.
    """
    snap = _entity_snapshot_by_asset(df)
    if snap.empty:
        return go.Figure()

    # Pick shocks
    if per_asset_shocks:
        shocks = per_asset_shocks
        # ensure missing assets get 0 shock
        assets_present = snap["Crypto Asset"].unique().tolist()
        shocks = {a: float(shocks.get(a, 0.0)) for a in assets_present}
        # compute per-entity delta via sum over assets
        deltas = (snap.assign(Shock=snap["Crypto Asset"].map(shocks).fillna(0.0))
                       .assign(Delta=lambda x: x["AssetNAV"] * x["Shock"])
                       .groupby("Entity Name", as_index=False)
                       .agg(Delta_USD=("Delta", "sum"),
                            MarketCap=("MarketCap", "max")))
    else:
        s = float(shock_pct or 0.0)
        deltas = (snap.groupby("Entity Name", as_index=False)
                       .agg(Delta_USD=("AssetNAV", lambda v: v.sum() * s),
                            MarketCap=("MarketCap", "max")))

    deltas["Impact %"] = np.where(deltas["MarketCap"] > 0,
                                  deltas["Delta_USD"] / deltas["MarketCap"] * 100.0,
                                  np.nan)

    d = deltas.sort_values("Impact %", key=lambda s: s.abs()).tail(top_n)

    colors = [NEUTRAL_POS if x >= 0 else NEUTRAL_NEG for x in d["Delta_USD"]]

    fig = go.Figure(go.Bar(
        x=d["Impact %"],
        y=d["Entity Name"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}%" if np.isfinite(v) else "—" for v in d["Impact %"]],
        textposition="outside",
        cliponaxis=False,
        customdata=np.c_[d["Delta_USD"], d["MarketCap"]],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Implied Equity Move: %{x:+.2f}%<br>"
            "ΔMC: %{customdata[0]:$,.0f}<br>"
            "Baseline MC: %{customdata[1]:$,.0f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        height=max(420, 24 * len(d) + 60),
        margin=dict(l=50, r=50, t=20, b=20),
        xaxis=dict(title="Implied Equity Move (%)", ticksuffix="%", zeroline=True, fixedrange=True, showgrid=True),
        yaxis=dict(title=None, fixedrange=True),
        showlegend=False,
        hoverlabel=dict(align="left"),
    )

    fig.add_annotation(
        text=WATERMARK_TEXT,
        x=0.5, y=0.45, xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=20, color="white"),
        opacity=0.7,
        xanchor="center",
        yanchor="middle",
    )
    # Powered-by footer
    fig.add_annotation(
        text="<i>powered by <b>F5Crypto.com</b></i>",
        x=0.5, y=0.05,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=12, color="white"),
        #bgcolor="white",
        #bordercolor="white", borderwidth=4,
        xanchor="center", yanchor="middle", opacity=0.7
    )

    return fig

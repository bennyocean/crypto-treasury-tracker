import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import qualitative


ASSETS_ORDER = ["BTC","ETH","SOL","XRP","BNB","SUI","LTC", "HYPE"]  # stable order for stacking/colors
COLORS = {"BTC":"#f7931a","ETH":"#6F6F6F","XRP":"#00a5df","BNB":"#f0b90b","SOL":"#dc1fff", "SUI":"#C0E6FF", "LTC":"#345D9D", "HYPE":"#97fce4"}
TYPE_PALETTE = {
    "Public Company": (123, 197, 237),  # blue
    "Private Company": (247, 89, 176), # rose
    "DAO": (233, 242, 111),              # amber
    "Non-Profit Organization": (128, 217, 183),        # green
    "Government": (247, 198, 148),      # slate
    "Other": (222, 217, 217),           # white
}
COLOR_MAP = {k: f"rgb({r},{g},{b})" for k, (r, g, b) in TYPE_PALETTE.items()}

WATERMARK_TEXT="cryptotreasurytracker.xyz"

def format_usd(value):
    if value >= 1_000_000_000_000:
        return f"${value/1_000_000_000_000:.1f}T"
    if value >= 1_000_000_000:
        return f"${value/1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:.0f}"


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
        "Total Reserves: " + grouped["Formatted_Total_USD"] +
        "<br>Share of Global: " + grouped["Formatted_Share_Global"] +
        "<br>Avg. Reserve per Entity: " + grouped["Formatted_Avg_Holdings"] +
        "<br>Entities Reporting: " + grouped["Entity_Count"].astype(str) +
        "<br><br><b>By Asset</b><br>" + grouped["PerAssetBreakdown"]
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
        x=0.5, y=0.5,                      # Center of chart
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=30, color="black"),
        opacity=0.3,
        xanchor="center",
        yanchor="middle",
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
        margin=dict(l=0, r=0, t=0, b=0),
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

    fig.add_annotation(
        text=WATERMARK_TEXT,
        x=0.95, y=0.05,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=15, color="white"),
        opacity=0.3,
        xanchor="right",
        yanchor="middle",
    )

    fig.update_layout(
        title_text="",
        height=240,
        yaxis=dict(autorange="reversed", tickfont=dict(size=12), title_standoff=25),
        margin=dict(l=140, r=10, t=0, b=20),
        font=dict(size=12),
        hoverlabel=dict(align="left")
    )

    return fig


def historic_chart(df, by="USD"):
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
        text='Text' if by != "USD" else None,
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
            fig.add_annotation(
                x=date,
                y=total,
                text=(f"${total/1_000_000_000:.1f}B" if total >= 1_000_000_000
                      else f"${total/1_000_000:.1f}M"),
                showarrow=False,
                font=dict(size=14, color='white'),
                yanchor='bottom'
            )
            fig.update_layout(
                yaxis=dict(
                    tickprefix="$",
                )
            )

    # Watermark
    fig.add_annotation(
        text=WATERMARK_TEXT,
        x=0.5, y=0.95,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=30, color="white"),
        opacity=0.3,
        xanchor="center",
        yanchor="top",
    )

    fig.update_layout(
        margin=dict(t=50, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
        hoverlabel=dict(align='left'),
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
            hovertemplate="<b>%{x|%b %Y}</b><br>USD: <b>%{y:$,.0f}</b><extra></extra>",
            yaxis="y"
        ))
        fig.update_layout(
            yaxis=dict(
                title="USD Value",
                rangemode="tozero",
                tickprefix="$",
                showgrid=True,           # left axis provides the grid
                zeroline=False
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
                linecolor="rgba(255,255,255,0.4)"
            ),
        )

    else:
        # Multi-asset → one total USD line + per-asset USD lines
        series = totals.reset_index().sort_values("Date")
        # total: solid, thicker
        fig.add_trace(go.Scatter(
            x=series["Date"], y=series["USD Value"],
            mode="lines",
            line=dict(width=3, dash="solid", color="#ffffff"),
            name="Total",
            hovertemplate="<b>%{x|%b %Y}</b><br>Total: <b>%{y:$,.0f}</b><extra></extra>",
        ))
        # per-asset lines: thinner, dashed, asset colors
        for a in selected_assets:
            s = (hist[hist["Crypto Asset"] == a]
                .groupby("Date", as_index=False)["USD Value"].sum()
                .sort_values("Date"))
            fig.add_trace(go.Scatter(
                x=s["Date"], y=s["USD Value"],
                mode="lines",
                line=dict(width=1.8, dash="dot", color=COLORS.get(a, "#888")),
                name=f"{a}",
                hovertemplate="<b>%{x|%b %Y}</b><br>"+f"{a}: <b>%{{y:$,.0f}}</b><extra></extra>",
            ))


    fig.update_layout(
        margin=dict(t=50, b=20, l=40, r=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
        hoverlabel=dict(align='left'),
        xaxis_title=None, yaxis_title=None,
        legend_title_text=None,
        yaxis=dict(
            title=None,
            rangemode="tozero",
            tickprefix="$",   # only multi-asset mode gets a dollar prefix on the left axis
            showgrid=True,
            zeroline=False
        )
    )
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")

    fig.add_annotation(
        text=WATERMARK_TEXT, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=30, color="white"), opacity=0.3,
        xanchor="center", yanchor="middle"
    )
    return fig


# --- right: Dominance (USD stacked area) ---
def dominance_area_chart_usd(df_historic: pd.DataFrame, current_df: pd.DataFrame | None = None):
    hist, selected_assets = _prepare_hist_with_snapshot(df_historic, current_df)
    fig = go.Figure()
    if hist.empty:
        return fig

    usds = hist.pivot(index="Date", columns="Crypto Asset", values="USD Value").fillna(0.0)
    usds = usds.reindex(columns=selected_assets)
    totals_usd = usds.sum(axis=1)

    cum = None
    for i, a in enumerate([x for x in ASSETS_ORDER if x in selected_assets]):
        top = (usds[a] if cum is None else (cum + usds[a]))
        cd = list(zip(usds[a].values, totals_usd.values))
        fig.add_trace(go.Scatter(
            x=usds.index, y=top.values,
            mode="lines",
            line=dict(width=0.0, color=COLORS.get(a, "#888")),
            fill=("tozeroy" if i == 0 else "tonexty"),
            name=a,
            hovertemplate=(
                "<b>%{x|%b %Y}</b><br>"
                f"{a}: <b>%{{customdata[0]:$,.0f}}</b>"
                "<extra></extra>"
            ),
            customdata=cd
        ))
        cum = top


    fig.update_traces(opacity=0.95)
    fig.update_layout(
        margin=dict(t=50, b=20, l=40, r=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
        hoverlabel=dict(align='left'),
        xaxis_title=None, yaxis_title=None, legend_title_text=None,
    )
    fig.update_yaxes(rangemode="tozero", tickprefix="$")
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")
    fig.add_annotation(
        text=WATERMARK_TEXT, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=30, color="white"), opacity=0.3,
        xanchor="center", yanchor="middle"
    )

    return fig


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


    # Add total USD value as annotation above each full bar
    totals = grouped.groupby('Entity Type', observed=False)['USD Value'].sum()
    for i, entity_type in enumerate(totals.index):
        fig.add_annotation(
            x=entity_type,
            y=totals[entity_type],
            text=(
                f"${totals[entity_type]/1_000_000_000:.1f}B" if totals[entity_type] >= 1_000_000_000
                else f"${totals[entity_type]/1_000_000:.1f}M"
            ),
            showarrow=False,
            font=dict(size=14, color='white'),
            yanchor='bottom'
        )

    fig.add_annotation(
        text=WATERMARK_TEXT,
        x=0.5, y=0.95,                      # Center of chart
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=15, color="white"),
        opacity=0.3,                       # Adjust for subtlety
        xanchor="center",
        yanchor="top",
    )

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
        margin=dict(t=50, b=20),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.1,
            xanchor='center',
            x=0.5
        ),
        yaxis=dict(
            tickprefix="$"
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

    fig.add_annotation(
        text=WATERMARK_TEXT,
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=15, color="white"),
        opacity=0.3,
        xanchor="center",
        yanchor="middle",
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5),
        hoverlabel=dict(align="left"),
        height=400,
        margin=dict(t=0, b=10),  # ↓ reduce top and bottom margin
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
        .nlargest(5)
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

    fig.add_annotation(
        text=WATERMARK_TEXT,
        x=0.5, y=0.05,                      # Center of chart
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=15, color="white"),
        opacity=0.3,                       # Adjust for subtlety
        xanchor="center",
        yanchor="middle",
    )

    # Step 6: Final layout adjustments
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        textfont=dict(size=12),
        hoverlabel=dict(align="left")
    )

    fig.update_layout(
        height=400,
        margin=dict(t=10, b=20),  # ↓ reduce top and bottom margin
        yaxis=dict(categoryorder='total ascending', title=None, tickfont=dict(size=14)),
        xaxis=dict(tickformat=',d', title=None),
        showlegend=False
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
        .nlargest(5)
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
            font=dict(size=16, color='white'),
            xanchor='left',
            yanchor='middle'
        )

    fig.add_annotation(
        text=WATERMARK_TEXT,
        x=0.5, y=0.05,                      # Center of chart
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=15, color="white"),
        opacity=0.3,                       # Adjust for subtlety
        xanchor="center",
        yanchor="middle",
    )

    # Step 6: Final layout adjustments
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>",
        textfont=dict(size=12),
        hoverlabel=dict(align="left")
    )

    fig.update_layout(
        height=400,
        margin=dict(t=10, b=20),  # ↓ reduce top and bottom margin
        yaxis=dict(categoryorder='total ascending', title=None),
        xaxis=dict(title=None, tickprefix = "$"),
        showlegend=False,
    )

    return fig


def entity_ranking(df, by="USD", top_n=10):
    value_col = 'USD Value' if by == "USD" else 'Holdings (Unit)'

    # Step 1: Aggregate values for plotting
    grouped = (
        df.groupby(['Entity Name', 'Crypto Asset'])[value_col]
        .sum()
        .reset_index()
    )

    # Step 2: USD total ranking
    usd_totals = (
        df.groupby('Entity Name')['USD Value']
        .sum()
        .sort_values(ascending=False)
    )

    # Limit to top N by USD value
    top_entities = usd_totals.head(top_n).index.tolist()
    grouped = grouped[grouped['Entity Name'].isin(top_entities)]
    grouped["USD Total"] = grouped["Entity Name"].map(usd_totals)

    # Step 3: Hover & label formatting
    if by == "USD":
        grouped["Text"] = grouped[value_col].apply(format_usd)
        hover = grouped.groupby('Entity Name').apply(
            lambda d: f"<b>{d.name}</b><br>" + "<br>".join(
                [f"{row['Crypto Asset']}: <b>{format_usd(row[value_col])}</b>" for _, row in d.iterrows()])
        ).to_dict()
    else:
        grouped["Text"] = grouped[value_col].apply(lambda x: f"{int(x):,}")
        hover = grouped.groupby('Entity Name').apply(
            lambda d: f"<b>{d.name}</b><br>" + "<br>".join(
                [f"{row['Crypto Asset']}: <b>{int(row[value_col]):,}</b>" for _, row in d.iterrows()])
        ).to_dict()

    grouped['Custom Hover'] = grouped['Entity Name'].map(hover)

    # Step 4: Enforce x-axis sort
    sorted_entities = (
        grouped.groupby('Entity Name')['USD Total']
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )

    grouped['Entity Name'] = pd.Categorical(grouped['Entity Name'], categories=sorted_entities, ordered=True)
    grouped = grouped.sort_values(['Entity Name', 'Crypto Asset'])

    # Step 5: Plot
    fig = px.bar(
        grouped,
        x='Entity Name',
        y=value_col,
        color='Crypto Asset',
        barmode='stack',
        custom_data=['Custom Hover'],
        color_discrete_map=COLORS,
        category_orders={'Entity Name': sorted_entities}
    )

    totals = grouped.groupby('Entity Name', observed=False)[value_col].sum()
    for entity in totals.index:
        label = format_usd(totals[entity]) if by == "USD" else f"{int(totals[entity]):,}"
        fig.add_annotation(
            x=entity,
            y=totals[entity],
            text=label,
            showarrow=False,
            font=dict(size=14, color='white'),
            xanchor='center',
            yanchor='bottom'
        )

    fig.add_annotation(
        text=WATERMARK_TEXT,
        x=0.5, y=0.5,                      # Center of chart
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=30, color="white"),
        opacity=0.3,                       # Adjust for subtlety
        xanchor="center",
        yanchor="middle",
    )

    fig.update_traces(
        textposition="none",
        hovertemplate="%{customdata[0]}<extra></extra>"
    )

    fig.update_layout(
        height=500,
        xaxis=dict(title=None, tickfont=dict(size=12)),
        yaxis=dict(title=None if by == "USD" else None,
                   tickprefix="$" if by == "USD" else ""
                   ),
        margin=dict(l=40, r=40, t=50, b=60),
        font=dict(size=12),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.1,
            xanchor='center',
            x=0.5
        ),
        legend_title_text='',
        hoverlabel=dict(align='left')
    )

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
        x=0.5, y=0.15, xref="paper", yref="paper",
        showarrow=False, font=dict(size=35, color="black"), opacity=0.30,
        xanchor="center", yanchor="middle",
    )
    return fig


def lorenz_curve_chart(p, L, asset: str | None = None):
    """Lorenz curve; if a single asset is passed, color the line with its color."""
    default_blue = "#66cded"

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
        height=320,
        margin=dict(l=40, r=20, t=10, b=30),
        xaxis=dict(title="Cumulative share of groups", tickformat=".0%", range=[0,1]),
        yaxis=dict(title="Cumulative share of weight", tickformat=".0%", range=[0,1]),
        showlegend=False,
        hoverlabel=dict(align="left")
    )
    fig.add_annotation(
    text=WATERMARK_TEXT,
    x=0.5, y=0.45, xref="paper", yref="paper",
    showarrow=False, font=dict(size=35, color="white"), opacity=0.30,
    xanchor="center", yanchor="middle",
    )
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
        margin=dict(l=100, r=20, t=60, b=10),
        xaxis=dict(title="Crypto Treasury as % of Market Cap", ticksuffix="%",),
        yaxis=dict(title=None),
        showlegend=False,
        hoverlabel=dict(align="left"),
    )

    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(r=40))

    fig.add_annotation(
    text=WATERMARK_TEXT,
    x=0.5, y=0.45, xref="paper", yref="paper",
    showarrow=False, font=dict(size=20, color="white"), opacity=0.30,
    xanchor="center", yanchor="middle",
    )

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
        margin=dict(l=100, r=20, t=60, b=10),
        xaxis=dict(title="Market Cap (stacked)"),
        yaxis=dict(title=None),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hoverlabel=dict(align="left"),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(r=40))
    fig.update_layout(
        barmode="stack",
        xaxis=dict(title="Market Cap in USD (stacked)", tickprefix="$", separatethousands=True),
    )
    fig.add_annotation(
    text=WATERMARK_TEXT,
    x=0.5, y=0.45, xref="paper", yref="paper",
    showarrow=False, font=dict(size=20, color="white"), opacity=0.30,
    xanchor="center", yanchor="middle",
    )
    return fig


NEUTRAL_POS = "#43d1a0"
NEUTRAL_NEG = "#f94144"

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
        margin=dict(l=120, r=50, t=50, b=10),
        xaxis=dict(title="Implied Equity Move (%)", ticksuffix="%", zeroline=True),
        yaxis=dict(title=None),
        showlegend=False,
        hoverlabel=dict(align="left"),
    )

    fig.add_annotation(
    text=WATERMARK_TEXT,
    x=0.5, y=0.45, xref="paper", yref="paper",
    showarrow=False, font=dict(size=35, color="white"), opacity=0.30,
    xanchor="center", yanchor="middle",
    )

    return fig


def mnav_comparison_bar(df: pd.DataFrame, top_n: int = 20, max_mnav: float | None = None) -> go.Figure:
    """
    Pick Top-N by CryptoNAV (largest treasuries), then DISPLAY sorted by mNAV (desc)
    with a 1× guideline. Optional mNAV cap to drop outliers.
    """
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
        margin=dict(l=40, r=30, t=20, b=120),
        xaxis=dict(title=None, tickangle=-35, automargin=True),
        yaxis=dict(title="mNAV (×)", range=y_range),
        showlegend=False,
        hoverlabel=dict(align="left"),
    )

    # subtle watermark
    fig.add_annotation(
        text=WATERMARK_TEXT,
        x=0.5, y=0.45, xref="paper", yref="paper",
        showarrow=False, font=dict(size=35, color="white"), opacity=0.3,
        xanchor="center", yanchor="bottom",
    )
    return fig


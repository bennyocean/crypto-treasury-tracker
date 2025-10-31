import streamlit as st
import numpy as np
import pandas as pd

from modules.filters import apply_filters
from modules.ui import render_plotly
from modules import charts

def render_valuation_insights():
    #st.title("Crypto Treasury Valuation Insights")

    df = st.session_state["data_df"]
    df_filtered = apply_filters(df)

    if df_filtered.empty:
        st.info("No data for current filters.")
        return

    # Snapshot KPI block
    snap = (df_filtered.groupby("Entity Name", as_index=False)
              .agg(CryptoNAV=("USD Value","sum"),
                   MarketCap=("Market Cap","max")))
    snap = snap.dropna(subset=["MarketCap"])
    snap = snap[snap["MarketCap"] > 0]
    total_mcap = float(snap["MarketCap"].sum())
    total_nav  = float(snap["CryptoNAV"].sum())

    # Premium (portfolio-weighted if available)
    prem = st.session_state.get("has_premium", False)
    if ("Premium %" in df_filtered.columns) or ("MNAV" in df_filtered.columns):
        # compute from charts helper for consistency
        _snap_full = charts._entity_snapshot(df_filtered)
        w = np.where(_snap_full["MarketCap"] > 0, _snap_full["MarketCap"], 0.0)
        pw_prem = np.nansum((_snap_full["Premium %"] * w)) / np.sum(w) if np.sum(w) > 0 else np.nan
    else:
        pw_prem = np.nan


    st.markdown("")

    # --- mNAV comparison ---
    with st.container(border=True):
        st.markdown("#### mNAV Benchmarking", help="Premium/discount vs the crypto treasury holding value. 1× is parity; above 1× = premium, below 1× = discount.")
        # Controls row
        c0, c1, c2, c3 = st.columns([1,1,1,1])
        cap_outliers = c2.checkbox("Cap mNAV (exclude outliers)", value=False, help="Drop extreme mNAV to make comparisons readable.")
        max_cap = c3.number_input("Max mNAV (×)", min_value=1, max_value=200, value=25, step=1,
                                help="Applied only when 'Cap mNAV' is on.", disabled=not cap_outliers)

        top_n_mnav = c0.number_input("Top N (by Crypto-NAV)", 5, 50, 20, help="Pick companies by largest crypto-NAV, then sort by mNAV for display.")
        datco_only_mnav = c1.checkbox("DAT Wrappers (mNAV)", value=True, help="Limit to verified Direct Asset Treasury Companies.")

        # Filter dataset for chart/KPIs
        df_mnav = df_filtered.copy()
        if datco_only_mnav:
            df_mnav = df_mnav[df_mnav["DAT"].astype(int) == 1]

        # Compute mNAV series for KPIs (respect the same cap rule)
        snap_for_kpi = charts._entity_snapshot(df_mnav).dropna(subset=["MarketCap"])
        snap_for_kpi = snap_for_kpi[snap_for_kpi["MarketCap"] > 0].copy()

        with np.errstate(divide="ignore", invalid="ignore"):
            snap_for_kpi["mNAV"] = np.where(
                snap_for_kpi["CryptoNAV"] > 0,
                snap_for_kpi["MarketCap"] / snap_for_kpi["CryptoNAV"],
                np.nan,
            )

        # apply mNAV cap BEFORE picking Top-N (same as chart)
        if cap_outliers:
            snap_for_kpi = snap_for_kpi[snap_for_kpi["mNAV"] <= float(max_cap)]

        # pick Top-N by CryptoNAV (selection), display order can later sort by mNAV
        d_kpi = (snap_for_kpi.dropna(subset=["mNAV"])
                            .sort_values("CryptoNAV", ascending=False)
                            .head(int(top_n_mnav)))

        sel = d_kpi["mNAV"].replace([np.inf, -np.inf], np.nan).dropna()
        st.markdown("")

        # KPIs (Median / Mean / Share <1× / Max)
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            with st.container(border=True):
                st.metric("Median mNAV (selected)", f"{(sel.median() if not sel.empty else np.nan):.2f}×")
        with k2:
            with st.container(border=True):
                st.metric("Mean mNAV (selected)",   f"{(sel.mean()   if not sel.empty else np.nan):.2f}×")
        if len(sel) > 0:
            under = int((sel < 1.0).sum())
            share_under = under / len(sel) * 100.0
            with k3:
                with st.container(border=True):
                    st.metric("mNAV < 1× (selected)", f"{under} ({share_under:.0f}%)")
            with k4:
                with st.container(border=True):
                    st.metric("Max mNAV (selected)",  f"{sel.max():.2f}×")
        else:
            with k3:
                with st.container(border=True):
                    st.metric("mNAV < 1× (selected)", "—")
            with k4:
                with st.container(border=True):
                    st.metric("Max mNAV (selected)",  "—")

        # Chart
        if df_mnav.empty:
            st.info("No data for the current selection.")
        else:
            fig_mnav = charts.mnav_comparison_bar(
                df_mnav,
                top_n=int(top_n_mnav),
                max_mnav=(float(max_cap) if cap_outliers else None),
            )
            render_plotly(fig_mnav, "mnav_comparison")

    with st.container(border=True):

        st.markdown("#### Premium vs Scale of DAT Wrappers", help="This chart visualizes how market size relates to valuation deviation (size effect using log USD Value)")

        render_plotly(charts.premium_vs_size(df_filtered), "premium_vs_size_chart")

    # Row of charts
    c1, c2 = st.columns([1,1])
    
    with c1:
        with st.container(border=True):
            st.markdown("#### Exposure Ladder", help="Treasury as a % of Market Cap; higher value means more sensitive to sudden changes in crypto prices.")
            sub_c1, sub_c2 = st.columns([2,1])

            top_n = sub_c1.number_input(
                "Top N (by Crypto Exposure)", 5, 50, 20, key="exp_topn",
                help="Number of entities to display, sorted by Exposure (%)."
            )
            datco_only_exp = sub_c2.checkbox(
                "DAT Wrappers", value=True, key="ladder_datco_only",
                help="Restrict to Digital Asset Treasury Wrappers only."
            )

            # build dataframe AFTER controls, then render regardless of checkbox
            df_ladder = df_filtered.copy()
            if datco_only_exp:
                df_ladder = df_ladder[df_ladder["DAT"].astype(int) == 1]

            if df_ladder.empty:
                st.info("No data for the current selection.")
            else:
                fig1 = charts.exposure_ladder_bar(df_ladder, top_n=int(top_n))
                render_plotly(fig1, "exposure_ladder")


    with c2:
        with st.container(border=True):
            st.markdown("#### Market Cap Decomposition", help="Stacked Market Cap split into Crypto-NAV and a residual Core Proxy.")
            
            sub_c3, sub_c4 = st.columns([2,1])

            top_n2 = sub_c3.number_input("Top N (by Market Cap)", 5, 50, 20, key="mcapdec_topn",
                                    help="Number of entities to display, sorted by Market Cap (USD).")

            datco_only_dec = sub_c4.checkbox("DAT Wrappers", value=True, key="dec_datco_only",
                                        help="Restrict to Digital Asset Treasury Companies only.")

            df_dec = df_filtered.copy()
            if datco_only_dec:
                df_dec = df_dec[df_dec["DAT"].astype(int) == 1]

            if df_dec.empty:
                st.info("No data for the current selection.")
            else:
                fig3 = charts.mcap_decomposition_bar(df_dec, top_n=top_n2)
                render_plotly(fig3, "mcap_decomposition")


    # Optional weighted premium info
    if np.isfinite(pw_prem):
        st.caption(f"Weighted Premium to MNAV (by Market Cap): **{pw_prem:.1f}%**")


    def _shock_controls(df_filtered: pd.DataFrame):
        assets = sorted(df_filtered["Crypto Asset"].dropna().unique().tolist())
        st.markdown("")

        c1, c2, c3, c4 = st.columns(4)

        datco_only = c4.checkbox(
            "DAT Wrappers",
            value=True,
            help="Restrict to verified Direct-Asset Treasury Companies."
        )        

        mode = c1.segmented_control(
            "Shock mode",
            options=["Uniform (all selected assets)", "Per-asset"],
            default="Uniform (all selected assets)",
            help="Apply one %-change to all selected assets, or specify a separate %-change per crypto asset.",
        )

        top_n = c3.number_input("Top N (by Crypto Exposure)", 5, 50, 20,  help="Number of entities to display, sorted by Exposure (%).")

        if mode.startswith("Uniform"):
            pct = c2.slider("Adjust crypto price shock (%)", -50, 50, value=-5, step=1,format="%d%%",
                            help="Apply price shock as %-change to all selected crypto assets.")
            return {"uniform": pct / 100.0, "overrides": None}, top_n, datco_only
        else:
            cols = st.columns(min(3, max(1, len(assets))))
            shocks = {}
            for i, a in enumerate(assets):
                shocks[a] = cols[i % len(cols)].slider(
                    f"{a} shock", -50, 50, value=0, step=1,
                    help=f"{a} price change in %."
                ) / 100.0
            return {"uniform": None, "overrides": shocks}, top_n, datco_only

    with st.container(border=True):
        st.markdown("#### DAT Leverage Simulation (NAV-Implied β)", help="Computes the Δ Market Cap (Equity) implied by crypto price shocks. Assumes 1:1 pass-through of Crypto-NAV changes; core business unchanged; not a historical beta.")

        cfg, top_n, datco_only = _shock_controls(df_filtered)

        df_sens = df_filtered.copy()
        if datco_only:
            df_sens = df_sens[df_sens["DAT"].astype(int) == 1]

        if df_sens.empty:
            st.info("No DAT Wrappers in the current selection.")
        else:
            if cfg["overrides"] is not None:
                fig = charts.corporate_sensitivity_bar(df_sens, per_asset_shocks=cfg["overrides"], top_n=top_n)
            else:
                fig = charts.corporate_sensitivity_bar(df_sens, shock_pct=cfg["uniform"], top_n=top_n)
            render_plotly(fig, "corporate_price_sensitivity")

        st.caption(
            "Method: ΔEquity ≈ Σ(Crypto-NAV × price shock) / Market Cap. "
            "Assumes share count is constant and market cap adjusts one-for-one to Crypto-NAV changes."
        )

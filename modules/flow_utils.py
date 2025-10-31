import pandas as pd
import numpy as np

def decompose_asset(g: pd.DataFrame) -> pd.DataFrame:
    """
    Exact ΔUSD decomposition per asset
      ΔUSD = units_prev * Δprice + price_curr * Δunits
    Robust to first rows and zero/NaN holdings.
    """
    g = g.sort_values("Date").copy()

    # safe numerics
    for col in ["Holdings (Unit)", "USD Value", "Price USD"]:
        if col not in g.columns:
            g[col] = np.nan
    g["Holdings (Unit)"] = pd.to_numeric(g["Holdings (Unit)"], errors="coerce")
    g["USD Value"]       = pd.to_numeric(g["USD Value"], errors="coerce")
    g["Price USD"]       = pd.to_numeric(g["Price USD"], errors="coerce")

    # derive price if missing and units > 0
    need_p = g["Price USD"].isna() & g["Holdings (Unit)"].gt(0)
    g.loc[need_p, "Price USD"] = g.loc[need_p, "USD Value"] / g.loc[need_p, "Holdings (Unit)"]

    g["units_prev"] = g["Holdings (Unit)"].shift()
    g["price_prev"] = g["Price USD"].shift()

    # safe fill for prev values to avoid NaNs exploding the first computable diff
    g["units_prev"] = g["units_prev"].fillna(0.0)
    g["price_prev"] = g["price_prev"].fillna(method="ffill").fillna(method="bfill")

    g["d_usd"] = g["USD Value"].diff()

    # price and units effects
    g["price_effect"] = (g["Price USD"] - g["price_prev"]) * g["units_prev"]
    g["units_effect"] = (g["Holdings (Unit)"] - g["units_prev"]) * g["Price USD"]

    # drop rows that truly have no delta information
    g = g.dropna(subset=["d_usd"])
    g[["price_effect", "units_effect"]] = g[["price_effect", "units_effect"]].fillna(0.0)
    return g

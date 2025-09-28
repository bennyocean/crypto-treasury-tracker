import streamlit as st
import numpy as np
import json, os, time, requests, gspread, pandas as pd
from google.oauth2.service_account import Credentials


CENTRAL_FILE = "data/prices.json"
LOCAL_FALLBACK_FILE = "data/last_prices.json"

ASSETS = [
    "BTC",
    "ETH",
    "SOL",
    "SUI",
    "LTC",
    "XRP",
    "HYPE",
    #"BNB",
    ]

COINGECKO_IDS = {
    "BTC":"bitcoin", 
    "ETH":"ethereum",
    "XRP":"ripple",
    # "BNB":"binancecoin",
    "SOL":"solana",
    "SUI": "sui",
    "LTC":"litecoin",
    "HYPE":"hyperliquid",
    }

DEFAULT_PRICES = {
    "BTC":120_000,
    "ETH":4_000,
    "XRP":3.50,
    #"BNB":700.00,
    "SOL":150.00, 
    "HYPE":40.00, 
    "SUI":3.50, 
    "LTC":110.00
    }

# Supply column row-wise (retrieved from Coingecko)
SUPPLY_CAPS = {
    "BTC": 20_000_000,  
    "ETH": 120_000_000,
    "XRP": 60_000_000_000,
    #"BNB": 140_000_000,
    "SOL": 540_000_000,
    "SUI": 3_500_000_000,
    "LTC": 76_000_000,
    "HYPE": 270_000_000,
    }


def _batch_get_tables(sheet, ranges):
    """Return a list of tables (each as a list-of-rows) for the given A1 ranges.
    Tries batch_get (one API call). Falls back to values_batch_get. As a last resort,
    returns [] so caller can decide to skip or do per-sheet reads."""
    # Newer gspread
    try:
        return sheet.batch_get(ranges, value_render_option="FORMATTED_VALUE")
    except Exception:
        pass
    # Older gspread: values_batch_get
    try:
        resp = sheet.values_batch_get(ranges, params={"valueRenderOption": "FORMATTED_VALUE"})
        return [vr.get("values", []) for vr in resp.get("valueRanges", [])]
    except Exception:
        return []

def _df_from_table(rows):
    if not rows:
        return None
    header, data = rows[0], rows[1:]
    width = len(header)
    # Right-pad or trim each row to match header width
    data_fixed = [(r + [""] * (width - len(r)))[:width] for r in data]
    return pd.DataFrame(data_fixed, columns=header)

def ensure_dir_for(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def load_last_prices(filename=None):
    filename = filename or LOCAL_FALLBACK_FILE
    ensure_dir_for(filename)
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
            return {k.upper(): float(v) for k, v in data.items()}
    return DEFAULT_PRICES.copy()

def save_last_prices(prices: dict, filename=None):
    filename = filename or LOCAL_FALLBACK_FILE
    ensure_dir_for(filename)
    out = {k.lower(): float(v) for k, v in prices.items() if k.upper() in ASSETS}
    with open(filename, "w") as f:
        json.dump(out, f)


@st.cache_data(ttl=300, show_spinner=False)
def read_central_prices_from_sheet() -> tuple[dict | None, pd.Timestamp | None]:
    """
    Returns a tuple  prices dict  latest timestamp as tz aware UTC pandas Timestamp
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
        client = gspread.authorize(creds)
        ws = client.open("master_table_v01").worksheet("prices")
        rows = ws.get_all_records(value_render_option="UNFORMATTED_VALUE")
        if not rows:
            return None, None
        df = pd.DataFrame(rows)
        if df.empty or "asset" not in df or "usd" not in df or "timestamp" not in df:
            return None, None

        # cast timestamp column to int, interpret as epoch seconds
        df["ts"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)

        df = df.sort_values("ts")
        latest_prices = df.groupby("asset")["usd"].last().to_dict()
        last_ts = df["ts"].max()

        return ({k.upper(): float(str(v).replace(",", ".")) for k, v in latest_prices.items()},
                last_ts)
    except Exception:
        return None, None


@st.cache_data(ttl=3600, show_spinner=False)
def get_prices():
    # 1) central sheet
    price_map, _ts = read_central_prices_from_sheet()  # (dict|None, ts|None)
    if isinstance(price_map, dict) and all(a in price_map for a in ASSETS):
        return tuple(float(price_map[a]) for a in ASSETS)

    # 2) CoinGecko API then persist to local
    try:
        ids = ",".join(COINGECKO_IDS[a] for a in ASSETS)
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": ids, "vs_currencies": "usd"},
            timeout=8,
        )
        r.raise_for_status()
        js = r.json()
        prices = {a: float(js[COINGECKO_IDS[a]]["usd"]) for a in ASSETS}
        save_last_prices(prices)
        return tuple(prices[a] for a in ASSETS)
    except Exception:
        pass

    # 3) local fallback
    last = load_last_prices()
    return tuple(float(last.get(a, 0.0)) for a in ASSETS)


# Function to get raw treasury data from master sheets
def load_units():
    scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
    service_account_info = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open("master_table_v01")

    ranges = [f"aggregated_{a.lower()}_data!A:Z" for a in ASSETS]  # e.g., aggregated_btc_data!A:Z
    tables = _batch_get_tables(sheet, ranges)  # one API call

    dfs = []
    for rows in tables:
        if not rows:
            continue
        header, data = rows[0], rows[1:]
        if not header or not data:
            continue
        df_a = _df_from_table(rows)
        if df_a is not None and not df_a.empty:
            dfs.append(df_a)

    if not dfs:
        return pd.DataFrame(columns=["Entity Name", "Ticker", "Market Cap", "Entity Type","Country","Crypto Asset","Holdings (Unit)", "Sector", "Industry", "About", "Website"]) #

    df = pd.concat(dfs, ignore_index=True)
    df = df[["Entity Name", "Ticker", "Market Cap", "Entity Type","Country","Crypto Asset","Holdings (Unit)", "Sector", "Industry", "About", "Website"]] # 
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    #df["Ticker"] = df["Ticker"].replace({"None": np.nan, "": np.nan}).astype("string")

    df["Market Cap"] = pd.to_numeric(df["Market Cap"], errors="coerce")  # NaN for missing

    df["Holdings (Unit)"] = (
        df["Holdings (Unit)"].astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
    df["Holdings (Unit)"] = pd.to_numeric(df["Holdings (Unit)"], errors="coerce").fillna(0.0)

    return df


def attach_usd_values(df_units: pd.DataFrame, prices_input):
    # accept either tuple in ASSETS order or dict keyed by symbols
    if isinstance(prices_input, tuple) or isinstance(prices_input, list):
        price_map = dict(zip(ASSETS, map(float, prices_input)))
    else:
        price_map = {k.upper(): float(v) for k, v in prices_input.items()}

    df = df_units.copy()
    # 1) Calculation of total crypto treasury value in USD
    df["USD Value"] = df["Crypto Asset"].map(price_map).fillna(0.0) * df["Holdings (Unit)"]

    # 2)  mNAV multiple  -> Market Cap over crypto NAV
    df["mNAV"] = df["Market Cap"] / df["USD Value"]
    df.loc[df["Market Cap"].isna() | (df["Market Cap"] <= 0) | (df["USD Value"] <= 0), "mNAV"] = np.nan
    df["mNAV"] = df["mNAV"].round(2)

    # 3) Premium or Discount percent -> equals mNAV minus 1
    df["Premium"] = ((df["Market Cap"] / df["USD Value"]) - 1) * 100
    df.loc[df["Market Cap"].isna() | (df["Market Cap"] <= 0) | (df["USD Value"] <= 0), "Premium"] = np.nan
    df["Premium"] = df["Premium"].round(2)

    # 4) Treasury to Market Cap ratio percent -> share of company value in crypto
    df["TTMCR"] = (df["USD Value"] / df["Market Cap"]) * 100
    df.loc[df["Market Cap"].isna() | (df["Market Cap"] <= 0), "TTMCR"] = np.nan
    df["TTMCR"] = df["TTMCR"].round(2)
    
    return df


def load_kpi_snapshots():
    try:
        info = st.secrets["gcp_service_account"]
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(info, scopes=scope)
        ss = gspread.authorize(creds).open("master_table_v01")
        ws = ss.worksheet("kpi_snapshots")
    except Exception:
        return pd.DataFrame(columns=["date_utc","ts_utc","total_usd","total_entities"])

    recs = ws.get_all_records()
    df = pd.DataFrame(recs)
    if df.empty:
        return df

    # types
    df["ts_utc"] = pd.to_numeric(df["ts_utc"], errors="coerce")
    df["total_usd"] = pd.to_numeric(df["total_usd"], errors="coerce")
    df["total_entities"] = pd.to_numeric(df["total_entities"], errors="coerce")
    # parse date
    df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce").dt.date
    df = df.sort_values(["date_utc", "ts_utc"])
    return df.reset_index(drop=True)


# Function to get historic treasury data from master sheets
@st.cache_data(ttl=900, show_spinner=False)
def load_historic_data():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    service_account_info = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open("master_table_v01")

    ranges = [f"historic_{a.lower()}!A:Z" for a in ASSETS]  # e.g., historic_btc!A:Z
    tables = _batch_get_tables(sheet, ranges)  # one API call

    dfs = []
    for rows in tables:
        if not rows:
            continue
        header, data = rows[0], rows[1:]
        if not header or not data:
            continue
        df_a = _df_from_table(rows)

        dfs.append(df_a)

    if not dfs:
        return pd.DataFrame(columns=["Year","Month","Crypto Asset","Holdings (Unit)","USD Value","Date"])

    df = pd.concat(dfs, ignore_index=True)
    df = df[["Year","Month","Crypto Asset","Holdings (Unit)","USD Value"]]

    df["Crypto Asset"] = df["Crypto Asset"].astype(str).str.upper()
    df["Year"]  = pd.to_numeric(df["Year"], errors="coerce")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df = df[df["Year"] > 2023].dropna(subset=["Year","Month"])

    df["Holdings (Unit)"] = pd.to_numeric(
        df["Holdings (Unit)"].astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce"
    ).fillna(0.0)
    df["USD Value"] = pd.to_numeric(df["USD Value"], errors="coerce").fillna(0.0)

    df["Date"] = pd.to_datetime(
        {"year": df["Year"].astype(int), "month": df["Month"].astype(int), "day": 1},
        errors="coerce"
    )
    df = df.dropna(subset=["Date"])
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_planned_data() -> pd.DataFrame:
    """
    Reads planned data and returns a normalized DataFrame.
    """
    scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
    service_account_info = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open("master_table_v01")

    # One-range batch read (keeps same pattern as your other loaders)
    tables = _batch_get_tables(sheet, ["planned_data!A:Z"])
    if not tables or not tables[0]:
        return pd.DataFrame(columns=[
            "Entity Name","Ticker","Entity Type","Country","Crypto Asset",
            "Planned USD","Invested/Cost USD",
            "Funding Method","Status","Timeline",
            "Data Source","Date Source","Comments"
        ])

    df = _df_from_table(tables[0])
    if df is None or df.empty:
        return df

    # Normalize columns you care about (create if missing)
    needed = [
        "Entity Name","Ticker","Entity Type","Country","Crypto Asset",
        "Planned USD","Invested/Cost USD",
        "Funding Method","Status","Timeline",
        "Data Source","Date Source","Comments"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    df = df[needed].copy()

    # Clean numerics (allow "N/A" etc.)
    def _to_float(s):
        s = (str(s) if pd.notna(s) else "").strip()
        if s == "" or s.lower() in {"n/a","na","none","-"}:
            return np.nan
        # Handle simple thousand/decimal variations
        s = s.replace(",", "").replace(" ", "")
        try:
            return float(s)
        except Exception:
            return np.nan

    df["Planned USD"] = df["Planned USD"].map(_to_float)
    df["Invested/Cost USD"] = df["Invested/Cost USD"].map(_to_float)

    # Tidy strings
    for c in ["Entity Name","Ticker","Entity Type","Country","Crypto Asset",
              "Funding Method","Status","Timeline",
              "Data Source","Date Source","Comments"]:
        df[c] = df[c].astype(str).str.strip()

    # Keep as-is; we’ll filter in the section
    return df

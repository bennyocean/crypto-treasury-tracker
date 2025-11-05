import streamlit as st
import numpy as np
import json, os, time, requests, gspread, pandas as pd
from google.oauth2.service_account import Credentials


CENTRAL_FILE = "data/prices.json"
LOCAL_FALLBACK_FILE = "data/last_prices.json"

ASSETS = [
    "BTC",
    "ETH",
    "XRP",
    "BNB",
    "SOL",
    "DOGE",
    "TRX",
    "ADA",
    "SUI",
    "LTC",
    "HYPE",
    "TON",
    "WLFI",
    "PUMP",
    "ATH",
    "BONK",
    "AVAX",
    "CRO",
    "LINK",
    "BERA",
    "TRUMP",
    "ZIG",
    "CORE",
    "VAULTA",
    "FLUID",
]

COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "XRP": "ripple",
    "BNB": "binancecoin",
    "SOL": "solana",
    "DOGE": "dogecoin",
    "TRX": "tron",
    "ADA": "cardano",
    "SUI": "sui",
    "LTC": "litecoin",
    "HYPE": "hyperliquid",
    "TON": "the-open-network",
    "WLFI": "world-liberty-financial",
    "PUMP": "pump-fun",
    "ATH": "aethir",
    "BONK": "bonk",
    "AVAX": "avalanche-2",
    "CRO": "crypto-com-chain",
    "LINK": "chainlink",
    "BERA": "berachain-bera",
    "TRUMP": "official-trump",
    "ZIG": "zignaly",
    "CORE": "coredaoorg",
    "VAULTA": "vaulta",
    "FLUID": "instadapp",
}

DEFAULT_PRICES = {
    "BTC": 110_000.0,
    "ETH": 4_000.0,
    "XRP": 2.50,
    "BNB": 1_100.0,
    "SOL": 200.0,
    "DOGE": 0.20,
    "TRX": 0.30,
    "ADA": 0.80,
    "SUI": 3.50,
    "LTC": 110.0,
    "HYPE": 48.0,
    "TON": 2.00,
    "WLFI": 0.25,
    "PUMP": 0.003,
    "ATH": 0.04,
    "BONK": 0.00002,
    "AVAX": 20.0,
    "CRO": 0.12,
    "LINK": 20.00,
    "BERA": 2.00,
    "TRUMP": 6.00,
    "ZIG": 0.12,
    "CORE": 0.25,
    "VAULTA": 0.35,
    "FLUID": 5.0,
}

SUPPLY_CAPS = {
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


@st.cache_data(ttl=3600, show_spinner=False)
def read_central_prices_from_sheet() -> tuple[dict | None, pd.Timestamp | None]:
    """Read live prices from Google Sheet; robust against comma/float timestamps."""
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
        if df.empty or not {"asset","usd","timestamp"}.issubset(df.columns):
            return None, None

        # --- Robust parsing ---
        df["asset"] = df["asset"].astype(str).str.upper().str.strip()
        df["usd"] = (
            df["usd"].astype(str)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )
        # allow float / string timestamps
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(int)
        df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

        latest = (
            df.sort_values("ts")
              .groupby("asset", as_index=False)["usd"].last()
        )
        price_map = dict(zip(latest["asset"], latest["usd"]))
        last_ts = df["ts"].max()
        return price_map, last_ts
    except Exception as e:
        print("❌ read_central_prices_from_sheet error:", e)
        return None, None



@st.cache_data(ttl=3600, show_spinner=False)
def get_prices():
    # 1) central sheet
    time.sleep(0.2) 
    price_map, _ts = read_central_prices_from_sheet()
    if isinstance(price_map, dict) and len(price_map) > 0:
        merged = {a: float(price_map.get(a, DEFAULT_PRICES[a])) for a in ASSETS}
        return tuple(merged[a] for a in ASSETS)

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
    print("Loaded price sample:", list(price_map.items())[:5])

    return tuple(float(last.get(a, 0.0)) for a in ASSETS)


# Function to get raw treasury data from master sheets

ALLOWED_STATUS = {"active", "pending_funded", "pending_announced", "inactive"}

def load_units():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    service_account_info = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open("master_table_v01")

    rows = sheet.worksheet("aggregated_data").get_all_values()
    if not rows:
        return pd.DataFrame()

    header, data = rows[0], rows[1:]
    if not header or not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=header)
    df = df[[c for c in [
        "entity_id", "Entity Name", "Ticker", "Market Cap", "Entity Type", "DAT",
        "Country", "Crypto Asset", "Holdings (Unit)", "Sector", "Industry", "About",
        "Website", "asset_id", "target_usd", "target_units", "status",
        "source_url", "date_source_utc"
    ] if c in df.columns]]

    # Ensure presence of new columns with safe defaults
    defaults = {
        "DAT": "no",
        "asset_id": "",
        "target_usd": np.nan,
        "target_units": np.nan,
        "status": "active",
        "source_url": "",
        "date_source_utc": np.nan,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    # Normalize keys and types
    if "entity_id" in df.columns:
        df["entity_id"] = df["entity_id"].astype(str).str.strip()
    else:
        df["entity_id"] = ""

    df["Entity Name"] = df["Entity Name"].astype(str).str.strip()
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["Country"] = df["Country"].astype(str).str.strip()

    # Crypto Asset as uppercase ticker label
    df["Crypto Asset"] = df["Crypto Asset"].astype(str).str.strip().str.upper()

    # Market Cap numeric
    df["Market Cap"] = pd.to_numeric(df.get("Market Cap"), errors="coerce")

    # Holdings numeric with your locale cleanup
    df["Holdings (Unit)"] = (
        df["Holdings (Unit)"].astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df["Holdings (Unit)"] = pd.to_numeric(df["Holdings (Unit)"], errors="coerce").fillna(0.0)

    # DAT yes or no normalized
    df["DAT"] = df["DAT"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)

    # Targets numeric
    df["target_usd"] = pd.to_numeric(df["target_usd"], errors="coerce")
    df["target_units"] = pd.to_numeric(df["target_units"], errors="coerce")

    # Status normalized to allowed values
    def _norm_status(s):
        s = str(s).strip().lower()
        return s if s in ALLOWED_STATUS else "active"
    df["status"] = df["status"].apply(_norm_status)

    # Source URL as string
    df["source_url"] = df["source_url"].astype(str).str.strip()

    # Date of source as datetime
    df["date_source_utc"] = pd.to_datetime(df["date_source_utc"], errors="coerce")

    # Keep consistent column order and pass through existing descriptive fields
    keep_cols = [
        "entity_id",
        "Entity Name", "Ticker", "Market Cap", "Entity Type", "Country",
        "Crypto Asset", "Holdings (Unit)", "Sector", "Industry", "About", "Website",
        "DAT", "asset_id", "target_usd", "target_units", "status", "source_url", "date_source_utc"
    ]
    # Add any extra columns that may exist but are not in the standard list to avoid dropping data
    df = df[keep_cols]

    return df

def attach_usd_values(df_units: pd.DataFrame, prices_input=None, use_default_prices: bool = False):
    """
    Compute USD Value and derived metrics.
    - Normal mode (use_default_prices=False): use prices_input exactly as before.
    - Local test mode (use_default_prices=True): ignore prices_input and use DEFAULT_PRICES.
    """
    # Build price map
    if use_default_prices or prices_input is None:
        price_map = {sym: float(DEFAULT_PRICES.get(sym, 0.0)) for sym in ASSETS}
    else:
        if isinstance(prices_input, (tuple, list)):
            price_map = dict(zip(ASSETS, map(float, prices_input)))
        else:
            price_map = {str(k).upper(): float(v) for k, v in prices_input.items()}

    df = df_units.copy()

    # 1) Calculate total crypto treasury value in USD (works for holdings == 0)
    df["USD Value"] = df["Crypto Asset"].map(price_map).fillna(0.0) * df["Holdings (Unit)"]

    # 2) Market Cap over crypto NAV (mNAV multiple)
    df["mNAV"] = df["Market Cap"] / df["USD Value"]
    df.loc[df["Market Cap"].isna() | (df["Market Cap"] <= 0) | (df["USD Value"] <= 0), "mNAV"] = np.nan
    df["mNAV"] = df["mNAV"].round(2)

    # 3) Premium or Discount percent (mNAV - 1)
    df["Premium"] = ((df["Market Cap"] / df["USD Value"]) - 1) * 100
    df.loc[df["Market Cap"].isna() | (df["Market Cap"] <= 0) | (df["USD Value"] <= 0), "Premium"] = np.nan
    df["Premium"] = df["Premium"].round(2)

    # 4) Treasury to Market Cap ratio (share of company value in crypto)
    df["TTMCR"] = (df["USD Value"] / df["Market Cap"]) * 100
    df.loc[df["Market Cap"].isna() | (df["Market Cap"] <= 0), "TTMCR"] = np.nan
    df["TTMCR"] = df["TTMCR"].round(2)

    # Optional: enforce 0 value for pending pipeline rows
    df.loc[df["status"].isin({"pending_funded", "pending_announced"}), ["USD Value"]] = 0.0

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
@st.cache_data(ttl=3600, show_spinner=False)
def load_historic_data():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    service_account_info = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open("master_table_v01")

    # Read from the new single tab
    rows = sheet.worksheet("historic_data").get_all_values()

    if not rows or len(rows) < 2:
        return pd.DataFrame(columns=["Year","Month","Crypto Asset","Holdings (Unit)","USD Value","Date"])

    header, data = rows[0], rows[1:]
    if not header or not data:
        return pd.DataFrame(columns=["Year","Month","Crypto Asset","Holdings (Unit)","USD Value","Date"])

    df = pd.DataFrame(data, columns=header)

    # Keep only relevant columns (ignore extra ones)
    df = df[["Year","Month","Crypto Asset","Holdings (Unit)","USD Value"]]

    # Clean and normalize
    df["Crypto Asset"] = df["Crypto Asset"].astype(str).str.upper()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df = df[df["Year"] > 2023].dropna(subset=["Year","Month"])

    # Convert EU number formatting
    df["Holdings (Unit)"] = pd.to_numeric(
        df["Holdings (Unit)"].astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce"
    ).fillna(0.0)

    df["USD Value"] = pd.to_numeric(
        df["USD Value"].astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce"
    ).fillna(0.0)

    # Create Date column for plotting/sorting (day = 1)
    df["Date"] = pd.to_datetime(
        {"year": df["Year"].astype(int), "month": df["Month"].astype(int), "day": 1},
        errors="coerce"
    )

    df = df.dropna(subset=["Date"])
    return df



"""@st.cache_data(ttl=3600, show_spinner=False)
def load_planned_data() -> pd.DataFrame:
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
    return df"""


CTT_DB_NAME = "ctt_database_v01"

def _open_book(book_name: str):
    scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
    return gspread.authorize(creds).open(book_name)

@st.cache_data(ttl=3600, show_spinner=False)
def load_entities_reference_ctt() -> pd.DataFrame:
    ss = _open_book(CTT_DB_NAME)
    tables = _batch_get_tables(ss, ["entities_reference!A:Q"])
    df = _df_from_table(tables[0]) if tables and tables[0] else pd.DataFrame()
    if df.empty:
        return df
    for c in ["entity_id","ent_name","ent_ticker","ent_type","ent_country","ent_sector","ent_industry","ent_sub_industry"]:
        if c in df:
            df[c] = df[c].astype(str).str.strip()
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_assets_reference_ctt() -> pd.DataFrame:
    ss = _open_book(CTT_DB_NAME)
    tables = _batch_get_tables(ss, ["assets_reference!A:I"])
    df = _df_from_table(tables[0]) if tables and tables[0] else pd.DataFrame()
    if df.empty:
        return df
    for c in ["asset_id","ticker","name","cg_id"]:
        if c in df:
            df[c] = df[c].astype(str).str.strip()
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_sources_reference_ctt() -> pd.DataFrame:
    ss = _open_book(CTT_DB_NAME)
    tables = _batch_get_tables(ss, ["sources_reference!A:F"])
    df = _df_from_table(tables[0]) if tables and tables[0] else pd.DataFrame()
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

    
def _to_float_eu(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"n/a", "na", "none", "null", "-"}:
        return np.nan
    # remove thin spaces & normal spaces
    s = s.replace("\u202f", "").replace(" ", "")
    # if both '.' and ',' exist, assume '.' thousands sep and ',' decimal
    if "," in s and "." in s:
        s = s.replace(".", "")
    # convert decimal comma to dot
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return pd.to_numeric(s, errors="coerce")


@st.cache_data(ttl=3600, show_spinner=False)
def load_holdings_events_ctt() -> pd.DataFrame:
    ss = _open_book(CTT_DB_NAME)
    tables = _batch_get_tables(ss, ["holdings_events!A:U"])
    df = _df_from_table(tables[0]) if tables and tables[0] else pd.DataFrame()
    if df.empty:
        return df

    # parse dates (keep old logic)
    if "event_date" in df:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

    if "avg_usd_cost_per_asset" in df:
        df["avg_usd_cost_per_asset"] = pd.to_numeric(
            df["avg_usd_cost_per_asset"]
            .astype(str)
            .str.replace(".", "", regex=False)      # remove thousand separators
            .str.replace(",", ".", regex=False),    # convert decimal comma to dot
            errors="coerce"
        )


    # keep your other numeric conversions if needed
    for c in ["units_delta", "usd_value_at_event", "units_effective"]:
        if c in df:
            df[c] = df[c].map(_to_float_eu)

    return df


# ---------- convenient joiner ----------

@st.cache_data(ttl=3600, show_spinner=False)
def get_events_for_entity_id_asset(entity_id: str, asset_symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ents   = load_entities_reference_ctt()
    assets = load_assets_reference_ctt()
    srcs   = load_sources_reference_ctt()
    evts   = load_holdings_events_ctt()

    if evts.empty:
        return pd.DataFrame(), pd.DataFrame()

    # map ticker to asset_id
    asset_map = {str(t).upper(): aid for aid, t in zip(assets["asset_id"], assets["ticker"])}
    aid = asset_map.get(str(asset_symbol).upper())
    if not aid:
        return pd.DataFrame(), pd.DataFrame()

    d = evts[(evts["entity_id"] == str(entity_id)) & (evts["asset_id"] == aid)].copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()

    d["event_date"] = pd.to_datetime(d["event_date"], errors="coerce")
    d = d.sort_values(["event_date","event_id"]).reset_index(drop=True)

    if "source_id" in d.columns and not srcs.empty and "source_id" in srcs:
        d = d.merge(
            srcs.rename(columns={
                "source_id":"_sid",
                "source_type":"_stype",
                "source_announcement_type":"_ann_type",
                "source_url":"_url",
                "source_date":"_sdate",
                "source_confidence":"_sconf"
            }),
            left_on="source_id", right_on="_sid", how="left"
        )

    delta = d["units_effective"].fillna(d["units_delta"]).fillna(0.0)
    d["cum_units"] = delta.cumsum()

    events_enriched = d[[
        "event_id","event_type","event_date","units_delta","units_effective",
        "usd_value_at_event","avg_usd_cost_per_asset","method_of_holding",
        "funding_category","funding_method","funding_vehicle","funding_debt_mat_date",
        "confidence_level","precision","notes","source_id",
        "_stype","_ann_type","_url","_sdate","_sconf","cum_units"
    ]].copy()

    ts = d[["event_date","cum_units"]].rename(columns={"event_date":"Date","cum_units":"Units"})
    ts = ts.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    return events_enriched, ts

import os, time, json, math
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials


SHEET_NAME = "master_table_v01"
SNAP_WS     = "kpi_snapshots"
PRICES_WS   = "prices"

ASSETS = ["BTC", "ETH", "SOL", "XRP", "SUI", "LTC", "HYPE"]

SUPPLY_CAPS = {
    "BTC": 20_000_000,  
    "ETH": 120_000_000,
    "XRP": 60_000_000_000,
    "BNB": 140_000_000,
    "SOL": 540_000_000,
    "SUI": 3_500_000_000,
    "LTC": 76_000_000,
    "HYPE": 270_000_000,
    }

def _get_creds():
    with open("service_account.json", "r") as f:
        info = json.load(f)
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    return Credentials.from_service_account_info(info, scopes=scope)

def _open_spreadsheet():
    client = gspread.authorize(_get_creds())
    return client.open(SHEET_NAME)

def parse_units_like_app(x) -> float:
    """Mirror app parsing for EU/US formats."""
    if x is None:
        return 0.0
    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        return float(x)
    s = str(x).strip().replace(" ", "")
    if s == "" or s.lower() in {"nan", "none"}:
        return 0.0
    # If both separators exist, decide by the rightmost one
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            # EU: '.' thousands, ',' decimal
            s = s.replace(".", "").replace(",", ".")
        else:
            # US: ',' thousands, '.' decimal
            s = s.replace(",", "")
    elif "," in s:
        # Likely EU decimal
        s = s.replace(",", ".")
    # else: only '.' or plain digits -> leave as-is
    try:
        return float(s)
    except Exception:
        return 0.0

def read_latest_prices(ss) -> dict:
    ws = ss.worksheet(PRICES_WS)
    rows = ws.get_all_records(value_render_option="UNFORMATTED_VALUE")
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    if df.empty or not {"asset","usd","timestamp"}.issubset(df.columns):
        return {}
    df["ts"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df = df.sort_values("ts")
    latest = df.groupby("asset")["usd"].last().to_dict()
    # normalize to {SYM: float}
    return {str(k).upper(): float(str(v).replace(",", ".")) for k, v in latest.items()}

def load_holdings_aggregated(ss) -> pd.DataFrame:
    dfs = []
    for a in ASSETS:
        try:
            ws = ss.worksheet(f"aggregated_{a.lower()}_data")
        except gspread.WorksheetNotFound:
            continue
        values = ws.get_all_values()
        if not values or len(values) < 2:
            continue
        header, data = values[0], values[1:]
        df_a = pd.DataFrame(data, columns=header)
        # Ensure required columns exist
        for col in ["Entity Name","Crypto Asset","Holdings (Unit)"]:
            if col not in df_a.columns:
                df_a[col] = ""
        # Standardize asset column in case it's missing/blank
        df_a["Crypto Asset"] = df_a["Crypto Asset"].replace("", a)
        dfs.append(df_a)

    if not dfs:
        return pd.DataFrame(columns=["Entity Name","Crypto Asset","Holdings (Unit)"])

    df = pd.concat(dfs, ignore_index=True)

    df["Holdings (Unit)"] = (
        df["Holdings (Unit)"]
        .astype(str)
        .map(parse_units_like_app)
        .fillna(0.0)
    )

    # Keep only valid/positive
    df = df[df["Holdings (Unit)"] > 0]

    # De-duplicate by (entity, asset) just in case
    df = (
        df.groupby(["Entity Name","Crypto Asset"], as_index=False)["Holdings (Unit)"]
          .sum()
    )

    return df

# ---- Upsert snapshot (one row per UTC date) ----
def upsert_snapshot(ss, total_usd: float, total_entities: int):
    try:
        ws = ss.worksheet(SNAP_WS)
    except gspread.WorksheetNotFound:
        ws = ss.add_worksheet(title=SNAP_WS, rows=2000, cols=6)
        ws.update("A1:D1", [["date_utc","ts_utc","total_usd","total_entities"]])

    # ensure header
    first = ws.get_values("A1:D1")
    if not first or first[0] != ["date_utc","ts_utc","total_usd","total_entities"]:
        ws.update("A1:D1", [["date_utc","ts_utc","total_usd","total_entities"]])

    now = int(time.time())
    date_utc = time.strftime("%Y-%m-%d", time.gmtime(now))

    recs = ws.get_all_records()
    idx_by_date = {str(r.get("date_utc")): i for i, r in enumerate(recs, start=2)}  # 2..n
    row = [date_utc, now, float(total_usd), int(total_entities)]
    if date_utc in idx_by_date:
        r = idx_by_date[date_utc]
        ws.update(values=[row], range_name=f"A{r}:D{r}")
    else:
        ws.append_row(row, value_input_option="RAW")

def main():
    ss = _open_spreadsheet()

    # 1) Load holdings (units) from aggregated tabs
    df = load_holdings_aggregated(ss)

    # 2) Load latest prices
    price_map = read_latest_prices(ss)
    if not price_map:
        print("No prices found in 'prices' tab.")
        return

    # 3) Compute USD Value exactly
    df["asset_px"] = df["Crypto Asset"].str.upper().map(price_map).astype(float)
    df = df[df["asset_px"].notna()]
    df["USD Value"] = df["Holdings (Unit)"] * df["asset_px"]

    # 4) Totals
    total_usd = float(df["USD Value"].sum())
    total_entities = int(df["Entity Name"].nunique())

    # Sanity prints: per-asset units and USD (helps catch a rogue parser)
    by_asset_units = df.groupby("Crypto Asset")["Holdings (Unit)"].sum().sort_values(ascending=False)
    by_asset_usd   = df.groupby("Crypto Asset")["USD Value"].sum().sort_values(ascending=False)
    print("Units by asset:\n", by_asset_units)
    print("USD by asset:\n", by_asset_usd)
    print("Snapshot:", f"Total USD: {total_usd}", f"Entities: {total_entities}")

    # 5) Upsert snapshot
    upsert_snapshot(ss, total_usd, total_entities)
    print("Upserted KPI snapshot for today.")

if __name__ == "__main__":
    main()

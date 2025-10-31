# scripts/update_prices_to_sheet.py
import os, time, json, requests
import gspread
from google.oauth2.service_account import Credentials

# === Configure assets and CoinGecko mapping ===
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



# === Google Sheet target ===
SHEET_NAME = "master_table_v01"
PRICES_WS = "prices"
HEADERS = ["asset", "usd", "timestamp"]

def _get_creds():
    # In GitHub Action we wrote the secret to service_account.json
    with open("service_account.json", "r") as f:
        info = json.load(f)
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    return Credentials.from_service_account_info(info, scopes=scope)

def _open_ws():
    creds = _get_creds()
    client = gspread.authorize(creds)
    ss = client.open(SHEET_NAME)
    try:
        ws = ss.worksheet(PRICES_WS)
    except gspread.WorksheetNotFound:
        ws = ss.add_worksheet(title=PRICES_WS, rows=10, cols=5)
        ws.update("A1:C1", [HEADERS])
    return ws

def _ensure_headers(ws):
    first = ws.get_values("A1:C1")
    if not first or first[0] != HEADERS:
        ws.update("A1:C1", [HEADERS])

def fetch_prices(symbols):
    ids = ",".join(COINGECKO_IDS[s] for s in symbols)
    params = {"ids": ids, "vs_currencies": "usd"}
    ua = {"User-Agent": "CTT/cron"}
    demo_key = os.getenv("COINGECKO_API_KEY", "").strip()

    # If a demo key is set, try public API with demo header
    if demo_key:
        try:
            headers = {**ua, "x-cg-demo-api-key": demo_key}
            r = requests.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params=params, headers=headers, timeout=15
            )
            r.raise_for_status()
            js = r.json()
            return {sym: float(js[COINGECKO_IDS[sym]]["usd"]) for sym in symbols}
        except requests.HTTPError:
            # fall back to public without key
            pass

    # Fallback: public API without any key
    r = requests.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params=params, headers=ua, timeout=15
    )
    r.raise_for_status()
    js = r.json()
    return {sym: float(js[COINGECKO_IDS[sym]]["usd"]) for sym in symbols}

def upsert_prices(ws, price_map):
    """
    Upsert by 'asset' key:
      - If asset exists in column A, overwrite its row.
      - Else append a new row.
    Rows: asset | usd | timestamp (epoch seconds)
    """
    _ensure_headers(ws)
    existing = ws.get_all_records()  # list of dicts with headers
    index_by_asset = {row["asset"]: idx for idx, row in enumerate(existing, start=2)}  # row 2..n

    now = int(time.time())
    for asset, usd in price_map.items():
        row = [asset, usd, now]
        if asset in index_by_asset:
            r = index_by_asset[asset]
            ws.update(values=[row], range_name=f"A{r}:C{r}")
        else:
            ws.append_row(row, value_input_option="RAW")

def main():
    ws = _open_ws()
    prices = fetch_prices(ASSETS)
    upsert_prices(ws, prices)
    print("Updated prices:", prices)

if __name__ == "__main__":
    main()

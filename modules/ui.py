import os, base64
import streamlit as st


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
          "ZEC": "#F5A800",
          }

def load_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# preload logos once
_THIS = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.join(_THIS, "..", "assets")
btc_b64 = load_base64_image(os.path.join(_ASSETS, "bitcoin-logo.png"))
eth_b64 = load_base64_image(os.path.join(_ASSETS, "ethereum-logo.png"))
sol_b64 = load_base64_image(os.path.join(_ASSETS, "solana-logo.png"))
sui_b64 = load_base64_image(os.path.join(_ASSETS, "sui-logo.png"))
ltc_b64 = load_base64_image(os.path.join(_ASSETS, "litecoin-logo.png"))
xrp_b64 = load_base64_image(os.path.join(_ASSETS, "xrp-logo.png"))
hype_b64 = load_base64_image(os.path.join(_ASSETS, "hyperliquid-logo.png"))
bnb_b64 = load_base64_image(os.path.join(_ASSETS, "bnb-logo.png"))
doge_b64 = load_base64_image(os.path.join(_ASSETS, "dogecoin-logo.png"))
ada_b64 = load_base64_image(os.path.join(_ASSETS, "cardano-logo.png"))
avax_b64 = load_base64_image(os.path.join(_ASSETS, "avalanche-logo.png"))
ath_b64 = load_base64_image(os.path.join(_ASSETS, "aethir-logo.png"))
bera_b64 = load_base64_image(os.path.join(_ASSETS, "berachain-bera-logo.png"))
bonk_b64 = load_base64_image(os.path.join(_ASSETS, "bonk-logo.png"))
link_b64 = load_base64_image(os.path.join(_ASSETS, "chainlink-logo.png"))
core_b64 = load_base64_image(os.path.join(_ASSETS, "core-dao-logo.png"))
cro_b64 = load_base64_image(os.path.join(_ASSETS, "cronos-logo.png"))
trump_b64 = load_base64_image(os.path.join(_ASSETS, "official-trump-logo.png"))
pump_b64 = load_base64_image(os.path.join(_ASSETS, "pump-fun-logo.png"))
ton_b64 = load_base64_image(os.path.join(_ASSETS, "toncoin-logo.png"))
trx_b64 = load_base64_image(os.path.join(_ASSETS, "tron-logo.png"))
wlfi_b64 = load_base64_image(os.path.join(_ASSETS, "world-liberty-financial-logo.png"))
zig_b64 = load_base64_image(os.path.join(_ASSETS, "zigchain-logo.png"))
vaulta_b64 = load_base64_image(os.path.join(_ASSETS, "vaulta-logo.png"))
fluid_b64 = load_base64_image(os.path.join(_ASSETS, "fluid-logo.png"))
zec_b64 = load_base64_image(os.path.join(_ASSETS, "zec-logo.png"))

cg_b64  = load_base64_image(os.path.join(_ASSETS, "coingecko-logo.png"))
logo_b64 = load_base64_image(os.path.join(_ASSETS, "ctt-symbol.svg"))
logo_loading = load_base64_image(os.path.join(_ASSETS, "ctt-logo.svg"))
f5_logo_64 = load_base64_image(os.path.join(_ASSETS, "f5-logo.jpg"))
telegram_qr_64 = load_base64_image(os.path.join(_ASSETS, "t_me-DATNewsAndAlerts.jpg"))

CTA_URL = "https://digitalfinancebriefing.substack.com/?utm_source=ctt_app&utm_medium=sidebar_cta&utm_campaign=subscribe"
SUPPORT_URL = "https://buymeacoffee.com/cryptotreasurytracker"
TELEGRAM_BOT = "https://t.me/DATNewsAndAlerts"

def render_header():
    btc = st.session_state["prices"][0]
    eth = st.session_state["prices"][1]
    xrp = st.session_state["prices"][2]
    bnb = st.session_state["prices"][3]
    sol = st.session_state["prices"][4]
    doge = st.session_state["prices"][5]
    trx = st.session_state["prices"][6]
    ada = st.session_state["prices"][7]
    sui = st.session_state["prices"][8]
    ltc = st.session_state["prices"][9]
    hype = st.session_state["prices"][10]
    ton = st.session_state["prices"][11]
    wlfi = st.session_state["prices"][12]
    pump = st.session_state["prices"][13]
    ath = st.session_state["prices"][14]
    bonk = st.session_state["prices"][15]
    avax = st.session_state["prices"][16]
    cro = st.session_state["prices"][17]
    link = st.session_state["prices"][18]
    bera = st.session_state["prices"][19]
    trump = st.session_state["prices"][20]
    zig = st.session_state["prices"][21]
    core = st.session_state["prices"][22]
    a = st.session_state["prices"][23]
    fluid = st.session_state["prices"][24]
    zec = st.session_state["prices"][25]

    st.markdown(
        """
        """,
        unsafe_allow_html=True
    )
    html = f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding:0.5rem 1rem;background-color:#f8f9fa;border-radius:0.5rem;
                font-size:1.2rem;color:#333;">
      <div>
        <img src="data:image/png;base64,{btc_b64}" style="height:20px;vertical-align:middle;margin-top:-3px;margin-right:2px;">
        <b>${btc:,.0f}</b>
        &nbsp;&nbsp;
        <img src="data:image/png;base64,{eth_b64}" style="height:20px;vertical-align:middle;margin-top:-3px;margin-right:2px;">
        <b>${eth:,.0f}</b>
        &nbsp;&nbsp;
        <img src="data:image/png;base64,{sol_b64}" style="height:20px;vertical-align:middle;margin-top:-3px;margin-right:2px;">
        <b>${sol:,.2f}</b>
        &nbsp;&nbsp;
        <img src="data:image/png;base64,{bnb_b64}" style="height:20px;vertical-align:middle;margin-top:-3px;margin-right:2px;">
        <b>${bnb:,.0f}</b>
        &nbsp;&nbsp;
        <img src="data:image/png;base64,{xrp_b64}" style="height:20px;vertical-align:middle;margin-top:-3px;margin-right:2px;">
        <b>${xrp:,.2f}</b>
        &nbsp;&nbsp;
        <img src="data:image/png;base64,{hype_b64}" style="height:20px;vertical-align:middle;margin-top:-3px;margin-right:2px;">
        <b>${hype:,.2f}</b>     
        &nbsp;&nbsp;
        <img src="data:image/png;base64,{doge_b64}" style="height:20px;vertical-align:middle;margin-top:-3px;margin-right:2px;">
        <b>${doge:,.3f}</b>   
        &nbsp;&nbsp;
        <img src="data:image/png;base64,{ltc_b64}" style="height:20px;vertical-align:middle;margin-top:-3px;margin-right:2px;">
        <b>${ltc:,.2f}</b>
        &nbsp;&nbsp;
        | Powered by
        <img src="data:image/png;base64,{cg_b64}" style="height:20px;vertical-align:middle;margin-top:-3px;margin-left:4px;margin-right:0px;">
        <a href="https://www.coingecko.com/" target="_blank" style="text-decoration:none;color:inherit;">CoinGecko</a>
      </div>
      <div>
        <img src="data:image/svg+xml;base64,{logo_b64}" style="height:35px;vertical-align:middle;">
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_subscribe_cta():

    st.sidebar.markdown(
        f"""
        <div style="
            text-align:center;
            background-color:rgba(0,136,204,0.08);
            padding:16px 8px;
            border-radius:12px;
            margin-top:30px;
            margin-bottom:30px;
        ">
            <p style="
                margin-bottom:6px;
                font-size:20px;
                font-weight:600;
                color:#7ad4fa;
            ">
                Get Latest DAT News
            </p>
            <p style="
                margin-top:0;
                margin-bottom:14px;
                font-size:13px;
                color:#CCCCCC;
            ">
                Scan the QR to join the community
            </p>
            <a href="{TELEGRAM_BOT}" target="_blank">
                <img src="data:image/jpeg;base64,{telegram_qr_64}"
                    width="160" style="border-radius:8px;">
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_support():
    #st.sidebar.markdown("---")
    st.sidebar.write(" ")
    st.sidebar.write(" ")
    st.sidebar.write(" ")
    st.sidebar.write(" ")
    st.sidebar.write(" ")

    st.sidebar.link_button(
        "Support the CTT ❤️",
        SUPPORT_URL,
        type="secondary",
        width="stretch",
        help="Click here to help keeping the CTT running & updated."
    )

    st.sidebar.write("")
    st.sidebar.write("")
  

def show_global_loader(msg="Loading data"):

    placeholder = st.empty()
    placeholder.markdown(
        f"""
        <div id="ctt-loader"
             style="position:fixed; inset:0; z-index:9999;
                    display:flex; flex-direction:column; align-items:center; justify-content:center;
                    gap:14px;
                    background:rgba(0,0,0,0.55); backdrop-filter:saturate(140%) blur(2px);">

          <div style="width:220px; height:120px; 
                      border-radius:14px; background-color:#111;
                      background-image:url('data:image/svg+xml;base64,{logo_loading}');
                      background-repeat:no-repeat; background-position:center; background-size:contain;
                      box-shadow:0 6px 20px rgba(0,0,0,0.35);">
          </div>

          <div class="spinner" style="width:34px; height:34px; border-radius:50%;
                                      border:3px solid #444; border-top-color:#fff;
                                      animation:spin 0.9s linear infinite;"></div>

          <div style="font-size:0.95rem; color:#fff; opacity:0.9;">{msg}</div>
        </div>

        <style>
          @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return placeholder


def render_plotly(fig, filename: str, scale: int = 3, fmt: str = "png", extra_config: dict | None = None):
    config = {
        "displaylogo": False,
        "modeBarButtonsToAdd": ["toImage"],
        "toImageButtonOptions": {"format": fmt, "filename": filename, "scale": scale},
    }
    if extra_config:
        config.update(extra_config)

    st.plotly_chart(fig, config=config)  # ← no width

def render_ticker():
    st.markdown(
        """
        <div style="width: 100%; margin-bottom: 10px; height: 30px; overflow: hidden; border: none; border-radius: 0;">
            <iframe 
                src="https://rss.app/embed/v1/ticker/liTK5CfH7VtOsY0q" 
                frameborder="0" 
                style="width: 100%; height: 100%; border: 0; overflow: hidden;">
            </iframe>
        </div>
        """,
        unsafe_allow_html=True
    )

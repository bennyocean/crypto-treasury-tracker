import os, base64
import streamlit as st


COLORS = {"BTC":"#f7931a","ETH":"#6F6F6F","XRP":"#00a5df","BNB":"#f0b90b","SOL":"#dc1fff", "SUI":"#C0E6FF", "LTC":"#345D9D", "HYPE":"#97fce4"}


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

cg_b64  = load_base64_image(os.path.join(_ASSETS, "coingecko-logo.png"))
logo_b64 = load_base64_image(os.path.join(_ASSETS, "ctt-symbol.svg"))
logo_loading = load_base64_image(os.path.join(_ASSETS, "ctt-logo.svg"))

CTA_URL = "https://digitalfinancebriefing.substack.com/?utm_source=ctt_app&utm_medium=sidebar_cta&utm_campaign=subscribe"
SUPPORT_URL = "https://buymeacoffee.com/cryptotreasurytracker"

def render_header():
    btc = st.session_state["prices"][0]
    eth = st.session_state["prices"][1]
    sol = st.session_state["prices"][2]
    sui = st.session_state["prices"][3]
    ltc = st.session_state["prices"][4]
    xrp = st.session_state["prices"][5]
    hype = st.session_state["prices"][6]

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
        <img src="data:image/png;base64,{xrp_b64}" style="height:20px;vertical-align:middle;margin-top:-3px;margin-right:2px;">
        <b>${xrp:,.2f}</b>
        &nbsp;&nbsp;
        <img src="data:image/png;base64,{hype_b64}" style="height:20px;vertical-align:middle;margin-top:-3px;margin-right:2px;">
        <b>${hype:,.2f}</b>        
        &nbsp;&nbsp;
        <img src="data:image/png;base64,{sui_b64}" style="height:20px;vertical-align:middle;margin-top:-3px;margin-right:2px;">
        <b>${sui:,.2f}</b>
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
    st.sidebar.write(" ")
    st.sidebar.write("")
    st.sidebar.link_button(
        "📥  Sign Up for the Weekly Crypto Treasury Newsletter",
        CTA_URL,
        type="secondary",
        width="stretch",
        help="Click here to sign up for the weekly crypto treasury newsletter."
    )
    st.sidebar.write(" ")
    st.sidebar.write(" ")


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


import streamlit as st
import os, base64

from modules.ui import render_ticker


_THIS = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.join(_THIS, "..", "assets")

def load_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
    
telegram_qr = load_base64_image(os.path.join(_ASSETS, "t_me-DATNewsAndAlerts.jpg"))
telegram_bot = "https://t.me/DATNewsAndAlerts"

def render_about():
    render_ticker()

    st.markdown(
        """
        <style>
        .about-card {
            padding: 18px;
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.35);
            background: #f8fafc;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.06);
        }
        @media (prefers-color-scheme: dark) {
            .about-card {
                background: #111827;
                border-color: rgba(148, 163, 184, 0.35);
                box-shadow: 0 14px 30px rgba(0, 0, 0, 0.45);
            }
        }
        .about-pill {
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(99, 102, 241, 0.16);
            color: #4338ca;
            font-weight: 600;
            font-size: 12px;
            letter-spacing: 0.01em;
        }
        @media (prefers-color-scheme: dark) {
            .about-pill {
                background: rgba(129, 140, 248, 0.24);
                color: #c7d2fe;
            }
        }
        .about-card h3 {
            margin: 6px 0 8px 0;
            font-size: 1.15rem;
        }
        .about-card p, .about-card ul {
            font-size: 0.95rem;
            line-height: 1.6;
        }
        .about-card ul { padding-left: 18px; margin: 6px 0 0 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=False):
        st.markdown(
            """
            <div class="about-card">
                <span class="about-pill">About</span>
                <h3>The most complete view of crypto treasuries</h3>
                <p>Crypto treasury positions now shape market structure, liquidity, and sentiment. The <strong>Crypto Treasury Tracker (CTT)</strong> unifies asset-level and entity-level holdings into one analytics layer across public and private companies, DAOs, non-profits, and sovereigns.</p>
                <p>Run cross-sectional, regional, and time-series views with live filters, interactive charts, and ready-to-export tables — built for investors, finance teams, researchers, and policy analysts.</p>
                <p>CTT is featured within Sentora’s DeFi Risk Analytics <a href="https://sentora.com/research/dashboards/crypto-treasury-tracker" target="_blank">dashboard</a> and referenced in Sentora’s research report on <a href="https://sentora.com/research/reports/bitcoin-treasury-strategies" target="_blank">Bitcoin Treasury Strategies</a>. The project has also been formerly sponsored by <a href="https://f5crypto.com/" target="_blank">F5 Crypto</a>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    with st.container(border=False):
        st.markdown(
            """
            <div class="about-card">
                <span class="about-pill">Data & Methodology</span>
                <h3>Trusted sources, live signals</h3>
                <ul style="margin-bottom: 0;">
                    <li>Crypto prices via <a href="https://docs.coingecko.com/reference/simple-price" target="_blank">CoinGecko</a>, refreshed hourly</li>
                    <li>Regulatory filings, press releases, and on-chain data for treasury positions</li>
                    <li>Equity market caps from Google Finance for supported tickers</li>
                    <li>Dynamic metrics: <strong>mNAV</strong> (Market Cap ÷ Crypto NAV), <strong>Premium/Discount</strong> ((mNAV − 1) × 100%), <strong>TTMCR</strong> (Crypto NAV ÷ Market Cap)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    with st.container(border=False):
        st.markdown(
            f"""
            <div class="about-card">
                <span class="about-pill">News</span>
                <h3>DAT News & Alerts Telegram Group</h3>
                <p style="margin-bottom: 8px;">Get real-time treasury moves, filings, and funding updates. Join the DAT community on Telegram.</p>
                <div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap;">
                    <a href="{telegram_bot}" target="_blank" style="
                        padding:10px 14px;
                        border-radius:10px;
                        background: rgba(99, 102, 241, 0.16);
                        color: inherit;
                        text-decoration: none;
                        font-weight: 600;
                        border: 1px solid rgba(148, 163, 184, 0.35);
                    ">
                        Join DAT News Channel
                    </a>
                    <img src="data:image/jpeg;base64,{telegram_qr}" alt="Join on Telegram" width="140" style="border-radius:10px;">
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    with st.container(border=False):
        st.markdown(
            """
            <div class="about-card">
                <span class="about-pill">Support & Attribution</span>
                <h3>Keep the tracker running</h3>
                <p>Your support funds infrastructure and continuous dataset updates.</p>
                <ul style="margin-bottom: 10px;">
                  <li>BTC: bc1pujcv929agye4w7fmppkt94rnxf6zfv3c7zpc25lurv7rdtupprrsxzs5g6</li>
                  <li>ETH: 0xe1b0Ae7b8496450ea09e60b110C2665ba0CB888f</li>
                  <li>Prefer fiat? <a href="https://buymeacoffee.com/cryptotreasurytracker" target="_blank">Buy Me a Coffee</a></li>
                </ul>
                <p style="margin-bottom: 0;">Please cite as: <em>Crypto Treasury Tracker by Benjamin Schellinger, PhD (2025), https://cryptotreasurytracker.xyz</em></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    with st.container(border=False):
        st.markdown(
            """
            <div class="about-card">
                <span class="about-pill">Contact</span>
                <h3>Feedback or collaboration?</h3>
                <p style="margin-bottom: 0;">Connect on <a href="https://www.linkedin.com/in/benjaminschellinger/" target="_blank">LinkedIn</a> or <a href="https://x.com/CTTbyBen" target="_blank">X</a>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    with st.container(border=False):
        st.markdown(
            """
            <div class="about-card">
                <span class="about-pill">Disclaimer</span>
                <p style="margin-bottom: 6px;; margin-top: 10px">All information is for informational purposes only and does not constitute financial, investment, or trading advice. Always conduct your own research and consult with a qualified financial professional before making any investment decisions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

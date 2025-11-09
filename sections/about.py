import streamlit as st

from modules.ui import render_ticker

def render_about():
    
    render_ticker()
    #st.title("About")
    
    # Box 1: Project Overview
    with st.container(border=False):

        st.subheader("The Most Comprehensive Crypto Treasury Platform")

        st.markdown(
            """            
            <p>Crypto assets held by corporate and institutional organizations are increasingly shaping digital asset market structure and narratives.</p>

            <p>The <strong>Crypto Treasury Tracker (CTT)</strong> brings transparency to digital asset treasury holdings. It benchmarks <strong>all</strong> crypto treasuries established by public & private companies, DAOs, non-profit organizations, and sovereigns.</p>

            <p>Instead of data silos, the <strong>CTT</strong> merges asset‑level and entity‑level crypto treasury data into a unified analytics layer. Moreover, it allows cross‑sectional, regional, and historical analysis of crypto treasury holdings using dynamic filters and interactive charts, delivering actionable insights and signals for institutional investors, finance professionals, researchers, and strategic observers.</p>

            </div>
            """,
            unsafe_allow_html=True
        )

    # Box 2: Data Sources & Update Logic
    with st.container(border=False):

        st.subheader("Data Sources")

        st.markdown(
            """
            <ul>
                <li>Live crypto price feeds via <a href="https://docs.coingecko.com/reference/simple-price" target="_blank">CoinGecko API</a>, automatically refreshed every hour</li>
                <li>Treasury data is updated regularly and based on regulatory filings, company press releases, and on-chain data</li>
                <li>Stock market data is retrieved real-time via Google Finance for supported tickers</li>
                <li>Key metrics such as <strong>mNAV</strong> (Market Cap ÷ Crypto NAV), <strong>Premium/Discount</strong> ((mNAV − 1) × 100%), and <strong>TTMCR</strong> (Crypto NAV ÷ Market Cap) are calculated dynamically from the latest data</li>
            </ul>
            """,
            unsafe_allow_html=True
        )

    # Box 4: Support, Attribution & Contact
    with st.container(border=False):

        st.subheader("Support & Attribution")

        st.markdown(
            """
            <p>Your support helps keep the Tracker running, cover server costs, and fund ongoing updates.</p>
            <ul style="margin-top: 0; font-size: 0.9rem;">
              <li>BTC: bc1pujcv929agye4w7fmppkt94rnxf6zfv3c7zpc25lurv7rdtupprrsxzs5g6</li>
              <li>ETH: 0xe1b0Ae7b8496450ea09e60b110C2665ba0CB888f</li>
              <li>SOL: 3JWdqYuy2cvMVdRbvXrQnZvtXJStBV5ooQ3QdqemtScQ</li>
              <li>Prefer fiat? <a href="https://buymeacoffee.com/cryptotreasurytracker" target="_blank">Buy Me a Coffee</a></li>
            </ul>

            When using data, charts, or signals from the <strong>Crypto Treasury Tracker</strong>, please cite as follows:<br>
            >Crypto Treasury Tracker by Benjamin Schellinger, PhD (2025), url: https://cryptotreasurytracker.xyz</br>
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("Feedback or collaboration?")
    st.markdown(
        """
        <p>Connect via <a href="https://www.linkedin.com/in/benjaminschellinger/" target="_blank">LinkedIn</a> or <a href="https://x.com/CTTbyBen" target="_blank">X</a>.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Box 5: Blog
    with st.container(border=False):

        st.subheader("Crypto Treasury Newsletter & Articles")

        st.markdown(
            """
            <p>For deeper insights on digital asset markets, view the <a href="https://digitalfinancebriefing.substack.com/ target="_blank">Digital Finance Briefing</a>.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Box 6: Disclaimer
    st.divider()

    st.text("Disclaimer")

    st.caption("All information is for informational purposes only and does not constitute financial, investment, or trading advice. Always conduct your own research and consult with a qualified financial professional before making any investment decisions.")
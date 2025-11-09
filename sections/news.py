import streamlit as st


def render_news():
    st.markdown("### Crypto Treasury News Feed")
    st.caption("Curated updates and insights on digital asset treasury companies from market sources.")

    st.markdown(
        """
        <div style="position: relative; width: 100%; height: 1300px; overflow: hidden; border: 0;">
            <iframe 
                src="https://rss.app/embed/v1/wall/liTK5CfH7VtOsY0q" 
                frameborder="0" 
                style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: 0; overflow: hidden;">
            </iframe>
        </div>
        """,
        unsafe_allow_html=True
    )
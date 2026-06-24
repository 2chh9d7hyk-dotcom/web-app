"""Shared top navigation bar for all pages.

Uses st.page_link() which:
- Always navigates in the same tab (Streamlit built-in behaviour)
- Auto-highlights the current page's link
- Requires no iframe, no JS, no CSP workarounds
"""
import streamlit as st


def render_top_nav(active_key: str = "home") -> None:
    """Render horizontal nav bar with st.page_link() links."""
    brand_col, c_home, c_m1, c_m2, c_m3 = st.columns([3, 2, 3, 3, 3])
    with brand_col:
        st.markdown('<p class="nav-brand-inline">🧠 AI LAB</p>', unsafe_allow_html=True)
    with c_home:
        st.page_link("main_app.py",            label="🏠 Home",          use_container_width=True)
    with c_m1:
        st.page_link("pages/1_vision.py",      label="👁️ M01: AIの目",  use_container_width=True)
    with c_m2:
        st.page_link("pages/2_adversarial.py", label="🎭 M02: AI騙し",  use_container_width=True)
    with c_m3:
        st.page_link("pages/3_training.py",    label="🧬 M03: AI育成",  use_container_width=True)
    st.markdown('<div class="nav-bottom-bar"></div>', unsafe_allow_html=True)

"""Shared hamburger navigation component for all pages.

Uses st.components.v1.html() (srcdoc iframe) so:
- onclick handlers work regardless of Streamlit Cloud's CSP
- window.parent.location.href navigates the parent tab (same-tab)
- window.frameElement.style.height resizes the iframe on open/close
"""
import streamlit.components.v1 as components

_H_CLOSED = 52
_H_OPEN   = 242

_PAGES = [
    ("home",        "/",             "🏠 Home"),
    ("vision",      "/vision",       "👁️ M01: AIの目"),
    ("adversarial", "/adversarial",  "🎭 M02: AI騙し"),
    ("training",    "/training",     "🧬 M03: AI育成"),
]


def render_top_nav(active_key: str = "home") -> None:
    """Render the hamburger nav bar in a same-origin srcdoc iframe."""
    links = "\n    ".join(
        '<a href="javascript:void(0)" onclick="navTo(\'{path}\')"'
        ' class="nav-link{active}">{label}</a>'.format(
            path=path,
            active=" nav-active" if key == active_key else "",
            label=label,
        )
        for key, path, label in _PAGES
    )

    html = """<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0d1117; overflow:hidden; }}
.top-nav {{
  display:flex; align-items:center; flex-wrap:wrap;
  background:#0d1117; border-bottom:2px solid #00d4ff;
  padding:5px 10px;
}}
.nav-brand {{
  font-family:'Courier New',monospace; font-size:0.58rem;
  color:#00d4ff; white-space:nowrap; flex:1;
  text-shadow:2px 2px 0 #003355;
}}
#nav-toggle {{ display:none; }}
.hamburger-btn {{
  display:block; cursor:pointer; font-size:1rem; color:#ffd700;
  background:#1e2a3a; border:2px solid #00d4ff; padding:3px 12px;
  margin-left:auto; font-weight:bold; user-select:none; line-height:1.4;
  box-shadow:2px 2px 0 #002244;
}}
#nav-toggle:checked ~ .hamburger-btn {{ color:#ff71ce; border-color:#ff71ce; }}
.nav-links {{
  display:none; width:100%; flex-direction:column; gap:3px; padding:6px 0 4px 0;
}}
#nav-toggle:checked ~ .nav-links {{ display:flex; }}
.nav-link {{
  color:#8b9ab0; text-decoration:none; padding:7px 14px;
  border:2px solid #2d3f5a; border-left:3px solid #2d3f5a;
  background:#1e2a3a; font-family:'Courier New',monospace; font-size:0.82rem;
  box-shadow:2px 2px 0 #000; display:block; width:100%; text-align:left;
}}
.nav-link:hover {{
  color:#00d4ff; border-color:#00d4ff; border-left-color:#00d4ff;
  background:#263548;
}}
.nav-link.nav-active {{
  color:#00d4ff; background:#263548;
  border-left:3px solid #00d4ff; font-weight:bold;
}}
</style>
</head><body>
<nav class="top-nav">
  <span class="nav-brand">🧠 AI LAB</span>
  <input type="checkbox" id="nav-toggle">
  <label for="nav-toggle" class="hamburger-btn">☰ Menu</label>
  <div class="nav-links">
    {links}
  </div>
</nav>
<script>
function navTo(p) {{ window.parent.location.href = p; }}
document.getElementById('nav-toggle').addEventListener('change', function() {{
  if (window.frameElement) {{
    window.frameElement.style.height = (this.checked ? {h_open} : {h_closed}) + 'px';
  }}
}});
if (window.frameElement) window.frameElement.style.height = '{h_closed}px';
</script>
</body></html>""".format(links=links, h_open=_H_OPEN, h_closed=_H_CLOSED)

    components.html(html, height=_H_CLOSED, scrolling=False)

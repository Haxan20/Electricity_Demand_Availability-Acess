"""
Shared visual theme: custom Google Fonts + CSS polish layered on top of the
dark base theme in .streamlit/config.toml. Call inject_theme() once near the
top of every page (app.py and each pages/*.py).

This gets Streamlit reasonably close to an "editorial tech report" look
(monospace label tags, tighter type scale, card-style metrics, accent
underlines) -- it will NOT produce the fully custom scrollytelling layout
of a hand-built static site (sticky side-nav, scroll-snap sections, bespoke
chart styling) -- Streamlit's component model doesn't allow that level of
layout control. For that look, see the standalone static report in
/report.
"""
import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}

/* Kicker / eyebrow label style, used via st.markdown('<div class="kicker">...</div>') */
.kicker {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #f5a623;
    margin-bottom: 0.2rem;
}

/* Metric cards get a subtle border + rounded corners instead of flat text */
div[data-testid="stMetric"] {
    background: #151a24;
    border: 1px solid #262d3a;
    border-radius: 10px;
    padding: 0.9rem 1rem 0.6rem 1rem;
}
div[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    opacity: 0.75;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif;
    color: #f5a623;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    word-break: break-word;
}

/* Section divider with accent tick, used via st.markdown(section_header(...)) */
.section-header {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    margin-top: 1.2rem;
    margin-bottom: 0.4rem;
}
.section-header .tick {
    width: 22px;
    height: 3px;
    background: #f5a623;
    display: inline-block;
}
.section-header .num {
    font-family: 'JetBrains Mono', monospace;
    color: #f5a623;
    font-size: 0.85rem;
}

/* Sidebar nav: monospace title tag like the reference site's "FIELD_LOG" */
[data-testid="stSidebarNav"]::before {
    content: "ENERGY_FORECAST · NG";
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    color: #f5a623;
    display: block;
    padding: 1rem 1rem 0.5rem 1rem;
}

/* Tighter dataframe/table headers */
[data-testid="stDataFrame"] thead th {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
</style>
"""


def inject_theme():
    st.markdown(_CSS, unsafe_allow_html=True)


def kicker(text: str):
    st.markdown(f'<div class="kicker">{text}</div>', unsafe_allow_html=True)


def section_header(number: str, title: str):
    st.markdown(
        f'<div class="section-header"><span class="tick"></span>'
        f'<span class="num">{number}</span><h3 style="margin:0">{title}</h3></div>',
        unsafe_allow_html=True,
    )

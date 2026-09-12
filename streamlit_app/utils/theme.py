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
/* Custom metric cards (see metric_card() below) -- replaces st.metric()
   everywhere, since Streamlit 1.63's built-in metric widget truncates long
   values with a JS-measured ellipsis that CSS text-overflow overrides
   cannot undo (confirmed: short values like "0.5 kW" rendered fine, long
   ones like "1.509 kWh" or "\u20a6528,000" got cut to "1.509..." regardless
   of CSS). Full custom HTML sidesteps that entirely.  */
.metric-card {
    background: #151a24;
    border: 1px solid #262d3a;
    border-radius: 10px;
    padding: 0.9rem 1rem 0.7rem 1rem;
    min-width: 0;
}
.metric-card .metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    opacity: 0.75;
    white-space: normal;
    overflow-wrap: break-word;
    margin-bottom: 0.3rem;
}
.metric-card .metric-value {
    font-family: 'Space Grotesk', sans-serif;
    color: #f5a623;
    font-size: 1.75rem;
    line-height: 1.2;
    white-space: normal;
    overflow-wrap: break-word;
    word-break: break-word;
}
.metric-card .metric-delta {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    color: #9aa4b2;
    margin-top: 0.25rem;
    white-space: normal;
    overflow-wrap: break-word;
}

div[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    opacity: 0.75;
}
div[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif;
    color: #f5a623;
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


def metric_card(label: str, value: str, delta: str = None):
    """Drop-in replacement for st.metric() that never truncates long values.
    Call inside a st.columns() cell same as you would st.metric():
        col1.metric(...) -> with col1: metric_card(...)
    or just: metric_card(..., container=col1)
    """
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'{delta_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

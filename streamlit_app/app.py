"""
Home page. Run with:  streamlit run app.py
(from inside streamlit_app/, with data/ and models/ populated one level up
via src/01_build_pipeline.py and src/02_train_demand_model.py)
"""
import streamlit as st
import pandas as pd

from utils.data_loader import load_engineered_data, address_feeder_lookup, feeder_reliability_summary
from utils.search import search_addresses, search_feeders
from utils.theme import inject_theme, kicker

st.set_page_config(
    page_title="Electricity Availability Forecast",
    page_icon="\u26a1",
    layout="wide",
)
inject_theme()
kicker("DISCO FORECASTING · LAGOS, NG")

st.title("\u26a1 Electricity Availability Forecast")
st.caption("Check your area's expected electricity supply and get personalized backup power recommendations.")

df = load_engineered_data()
lookup = address_feeder_lookup(df)
reliability = feeder_reliability_summary(df)

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
st.subheader("Find your address or feeder")
tab_address, tab_feeder = st.tabs(["Search by address", "Search by feeder"])

selected_feeder = None
selected_address = None

with tab_address:
    query = st.text_input("Start typing your street or area name", placeholder="e.g. Cele Street")
    if query:
        matches = search_addresses(lookup, query, limit=10)
        if matches.empty:
            st.info("No matching addresses found. Try a shorter or differently spelled query.")
        else:
            options = [f"{row.ADDRESS} — {row.FEEDER_NAME}" for row in matches.itertuples()]
            choice = st.selectbox("Select your address", options)
            if choice:
                idx = options.index(choice)
                selected_address = matches.iloc[idx]["ADDRESS"]
                selected_feeder = matches.iloc[idx]["FEEDER_NAME"]

with tab_feeder:
    feeders = search_feeders(lookup)
    chosen_feeder = st.selectbox("Select a feeder", ["-- choose --"] + feeders)
    if chosen_feeder != "-- choose --":
        selected_feeder = chosen_feeder
        addresses_for_feeder = lookup[lookup["FEEDER_NAME"] == chosen_feeder]["ADDRESS"].tolist()
        chosen_address = st.selectbox("Select an address on this feeder", addresses_for_feeder)
        selected_address = chosen_address

if selected_feeder and selected_address:
    st.session_state["selected_feeder"] = selected_feeder
    st.session_state["selected_address"] = selected_address
    st.success(f"Selected: **{selected_address}** (Feeder: {selected_feeder})")
    st.page_link("pages/1_Forecast.py", label="\u2192 View 7-day forecast", icon="📈")
    st.page_link("pages/2_Recommendations.py", label="\u2192 Get backup power recommendation", icon="🔌")

st.divider()

# ---------------------------------------------------------------------------
# Quick insights
# ---------------------------------------------------------------------------
st.subheader("Network overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Feeders tracked", f"{reliability['FEEDER_NAME'].nunique():,}")
col2.metric("Addresses tracked", f"{lookup.shape[0]:,}")
col3.metric("Avg. daily hours of supply", f"{reliability['avg_actual_hours'].mean():.1f}h")
col4.metric("Avg. daily shortfall", f"{reliability['avg_shortfall'].mean():.1f}h")

st.markdown("**Top 5 areas needing backup power most** (lowest average hours of supply)")
worst = reliability.nsmallest(5, "avg_actual_hours")[
    ["FEEDER_NAME", "avg_actual_hours", "band", "avg_availability_pct", "n_addresses"]
].rename(columns={
    "FEEDER_NAME": "Feeder", "avg_actual_hours": "Avg hours/day", "band": "Band (h)",
    "avg_availability_pct": "Availability vs band %", "n_addresses": "Addresses",
})
worst["Avg hours/day"] = worst["Avg hours/day"].round(1)
worst["Availability vs band %"] = worst["Availability vs band %"].round(1)
st.dataframe(worst, use_container_width=True, hide_index=True)

st.markdown("**Top 5 most reliable areas** (highest average hours of supply)")
best = reliability.nlargest(5, "avg_actual_hours")[
    ["FEEDER_NAME", "avg_actual_hours", "band", "avg_availability_pct", "n_addresses"]
].rename(columns={
    "FEEDER_NAME": "Feeder", "avg_actual_hours": "Avg hours/day", "band": "Band (h)",
    "avg_availability_pct": "Availability vs band %", "n_addresses": "Addresses",
})
best["Avg hours/day"] = best["Avg hours/day"].round(1)
best["Availability vs band %"] = best["Availability vs band %"].round(1)
st.dataframe(best, use_container_width=True, hide_index=True)

st.caption(
    "Historical data: Jan-Aug 2026. Weather inputs used for forecasting are "
    "currently synthetic (no live weather feed connected) -- see README."
)

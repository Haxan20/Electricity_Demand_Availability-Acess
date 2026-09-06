import streamlit as st
import pandas as pd

from utils.data_loader import load_engineered_data, load_weather_cache, load_model
from utils.forecast import forecast_n_days
from utils.recommend import build_recommendation, appliance_list_to_kwh, hours_to_kwh, APPLIANCE_WATTAGE
from utils.theme import inject_theme, kicker

st.set_page_config(page_title="Recommendations", page_icon="🔌", layout="wide")
inject_theme()
kicker("02 · BACKUP POWER RECOMMENDATION")
st.title("🔌 Backup Power Recommendation")

feeder = st.session_state.get("selected_feeder")
address = st.session_state.get("selected_address")

if not feeder or not address:
    st.warning("No address selected yet. Go to the Home page and search for your address first.")
    st.page_link("app.py", label="\u2190 Back to Home")
    st.stop()

st.markdown(f"**{address}** — Feeder: *{feeder}*")

# Make sure we have a shortfall forecast even if the user skipped the Forecast page
if "avg_shortfall_hours" not in st.session_state:
    df = load_engineered_data()
    weather = load_weather_cache()
    model, features = load_model()
    with st.spinner("Computing forecast..."):
        fc = forecast_n_days(df, weather, model, features, feeder, address, n_days=7)
    st.session_state["avg_shortfall_hours"] = fc["PREDICTED_SHORTFALL"].mean()
    st.session_state["shortfall_frequency"] = (fc["PREDICTED_SHORTFALL"] > 2).mean()
    st.session_state["band"] = fc["BAND"].iloc[0]

avg_shortfall_hours = st.session_state["avg_shortfall_hours"]
shortfall_frequency = st.session_state["shortfall_frequency"]
band = st.session_state["band"]

st.info(f"Based on your forecast: average shortfall of **{avg_shortfall_hours:.1f}h/day** "
        f"against a {band:.0f}h band, with a shortfall on **{shortfall_frequency*100:.0f}%** of forecast days.")

st.divider()

# ---------------------------------------------------------------------------
# Energy need input
# ---------------------------------------------------------------------------
st.subheader("Tell us your daily electricity need")
input_method = st.radio(
    "How would you like to enter it?",
    ["kWh per day", "Hours per day", "Appliance list"],
    horizontal=True,
)

daily_need_kwh = None

if input_method == "kWh per day":
    daily_need_kwh = st.number_input("Daily need (kWh)", min_value=0.5, value=10.0, step=0.5)

elif input_method == "Hours per day":
    hrs = st.number_input("Hours of electricity needed per day", min_value=1.0, max_value=24.0, value=15.0, step=0.5)
    assumed_load = st.number_input(
        "Assumed average load while running (kW)", min_value=0.2, value=1.5, step=0.1,
        help="Rough total wattage of what you'd typically have running at once, divided by 1000.",
    )
    daily_need_kwh = hours_to_kwh(hrs, assumed_load)
    st.caption(f"\u2192 approximately {daily_need_kwh} kWh/day")

else:  # Appliance list
    st.caption("Select appliances and how many hours per day you'd run each.")
    selections = {}
    cols = st.columns(2)
    for i, (name, watts) in enumerate(APPLIANCE_WATTAGE.items()):
        with cols[i % 2]:
            hrs = st.slider(f"{name} ({watts}W)", 0.0, 24.0, 0.0, 0.5, key=f"appliance_{name}")
            if hrs > 0:
                selections[name] = hrs
    daily_need_kwh = appliance_list_to_kwh(selections) if selections else 0.0
    st.caption(f"\u2192 approximately {daily_need_kwh} kWh/day")

if not daily_need_kwh:
    st.stop()

# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------
rec = build_recommendation(daily_need_kwh, avg_shortfall_hours, shortfall_frequency)

st.divider()
st.subheader("Recommendation")

PRIMARY_LABELS = {
    "none": ("\u2705 You're mostly covered", "Grid supply meets your stated need on most days."),
    "generator": ("🛢️ Generator recommended", "Your shortfall is occasional -- a generator for backup is the more cost-effective option."),
    "solar": ("\u2600️ Solar recommended", "Your shortfall is significant enough that solar panels are worth considering."),
    "solar_battery": ("🔋 Solar + Battery recommended", "Frequent, significant shortfall -- solar with battery storage gives the most reliable coverage."),
}
label, desc = PRIMARY_LABELS[rec.primary]
st.markdown(f"### {label}")
st.write(desc)

if rec.primary == "none":
    st.stop()

cols = st.columns(3)

if rec.solar_kw:
    with cols[0]:
        st.markdown("**\u2600️ Solar system**")
        st.metric("Recommended size", f"{rec.solar_kw} kW")
        st.metric("Estimated cost", f"\u20a6{rec.solar_cost_naira:,.0f}")

if rec.battery_kwh:
    with cols[1]:
        st.markdown("**🔋 Battery storage**")
        st.metric("Recommended capacity", f"{rec.battery_kwh} kWh")
        st.metric("Estimated cost", f"\u20a6{rec.battery_cost_naira:,.0f}")

if rec.generator_kva:
    with cols[0]:
        st.markdown("**🛢️ Generator**")
        st.metric("Recommended size", f"{rec.generator_kva} kVA")
        st.metric("Estimated cost", f"\u20a6{rec.generator_cost_naira:,.0f}")
        st.metric("Fuel cost", f"\u20a6{rec.generator_fuel_naira_per_day:,.0f}/day")

if rec.payback_years:
    with cols[2]:
        st.markdown("**💰 Payback**")
        st.metric("Estimated payback", f"{rec.payback_years} years")
        st.metric("Annual value of gap (grid tariff)", f"\u20a6{rec.annual_grid_gap_cost_naira:,.0f}")

st.divider()
st.markdown("**Energy efficiency (always worth doing regardless of backup choice)**")
st.markdown(
    "- Switch to an inverter AC (30-50% less energy than a standard split unit)\n"
    "- LED bulbs instead of incandescent/CFL (up to 80% less energy for the same light)\n"
    "- Unplug idle chargers and standby electronics\n"
    "- A well-insulated fridge/freezer with a good door seal cuts compressor runtime"
)

with st.expander("Assumptions and caveats"):
    for note in rec.notes:
        st.caption(f"- {note}")
    st.caption(
        "Cost figures use placeholder market rates defined in `utils/recommend.py` "
        "(COST_ASSUMPTIONS) -- update them to current prices, or better, treat "
        "these as a starting point and get 2-3 real installer quotes."
    )

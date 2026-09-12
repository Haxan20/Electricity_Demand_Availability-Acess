import streamlit as st
import pandas as pd

from utils.data_loader import load_engineered_data, load_weather_cache, load_model, address_feeder_lookup
from utils.forecast import forecast_n_days
from utils.recommend import build_recommendation, appliance_list_to_kwh, hours_to_kwh
from utils.reports import build_excel_report, build_pdf_report, build_comparison_excel
from utils.search import search_addresses
from utils.theme import inject_theme, kicker

st.set_page_config(page_title="Reports", page_icon="📄", layout="wide")
inject_theme()
kicker("04 · EXPORTS")
st.title("📄 Reports")

df = load_engineered_data()
weather = load_weather_cache()
model, features = load_model()
lookup = address_feeder_lookup(df)

tab_single, tab_compare = st.tabs(["Download report for my address", "Compare multiple addresses"])

# ---------------------------------------------------------------------------
# Single-address report
# ---------------------------------------------------------------------------
with tab_single:
    feeder = st.session_state.get("selected_feeder")
    address = st.session_state.get("selected_address")

    if not feeder or not address:
        st.info("No address selected yet -- pick one below, or go to Home to search.")
        query = st.text_input("Search address", key="reports_search")
        if query:
            matches = search_addresses(lookup, query, limit=10)
            if not matches.empty:
                options = [f"{r.ADDRESS} — {r.FEEDER_NAME}" for r in matches.itertuples()]
                choice = st.selectbox("Select", options, key="reports_select")
                idx = options.index(choice)
                feeder = matches.iloc[idx]["FEEDER_NAME"]
                address = matches.iloc[idx]["ADDRESS"]

    if feeder and address:
        st.markdown(f"**{address}** — Feeder: *{feeder}*")
        n_days = st.slider("Report horizon (days)", 1, 14, 7, key="report_horizon")

        with st.spinner("Generating forecast..."):
            fc = forecast_n_days(df, weather, model, features, feeder, address, n_days=n_days)

        include_rec = st.checkbox("Include backup power recommendation", value=True)
        rec = None
        if include_rec:
            need_kwh = st.number_input("Daily need for recommendation (kWh)", min_value=0.5, value=hours_to_kwh(15), step=0.5)
            avg_shortfall = fc["PREDICTED_SHORTFALL"].mean()
            shortfall_freq = (fc["PREDICTED_SHORTFALL"] > 2).mean()
            rec = build_recommendation(need_kwh, avg_shortfall, shortfall_freq)

        st.dataframe(fc, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            excel_bytes = build_excel_report(feeder, address, fc, rec)
            st.download_button(
                "\u2b07️ Download Excel report", data=excel_bytes,
                file_name=f"forecast_{feeder}_{address[:20]}.xlsx".replace(" ", "_").replace("/", "-"),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with col2:
            pdf_bytes = build_pdf_report(feeder, address, fc, rec)
            st.download_button(
                "\u2b07️ Download PDF report", data=pdf_bytes,
                file_name=f"forecast_{feeder}_{address[:20]}.pdf".replace(" ", "_").replace("/", "-"),
                mime="application/pdf",
            )

# ---------------------------------------------------------------------------
# Multi-address comparison
# ---------------------------------------------------------------------------
with tab_compare:
    st.caption("Compare forecasted availability across up to 5 addresses.")
    all_labels = (lookup["ADDRESS"].astype(str) + " — " + lookup["FEEDER_NAME"].astype(str)).tolist()
    chosen = st.multiselect("Select addresses to compare", all_labels, max_selections=5)

    if chosen:
        rows = []
        with st.spinner("Generating forecasts..."):
            for label in chosen:
                addr, feed = label.rsplit(" — ", 1)
                fc = forecast_n_days(df, weather, model, features, feed, addr, n_days=7)
                rows.append({
                    "Feeder": feed, "Address": addr,
                    "Avg predicted hours/day": round(fc["PREDICTED_ACTUAL_HOURS"].mean(), 2),
                    "Avg predicted shortfall (h)": round(fc["PREDICTED_SHORTFALL"].mean(), 2),
                    "Avg availability %": round(fc["PREDICTED_AVAILABILITY_PCT"].mean(), 1),
                    "Band (h)": fc["BAND"].iloc[0],
                })
        comp_df = pd.DataFrame(rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        comp_bytes = build_comparison_excel(comp_df)
        st.download_button(
            "\u2b07️ Download comparison as Excel", data=comp_bytes,
            file_name="address_comparison.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

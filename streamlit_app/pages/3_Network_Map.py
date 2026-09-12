import streamlit as st
import pandas as pd

from utils.data_loader import load_engineered_data, feeder_reliability_summary, reliability_color
from utils.theme import inject_theme, kicker, metric_card

st.set_page_config(page_title="Network Map", page_icon="🗺️", layout="wide")
inject_theme()
kicker("03 · NETWORK RELIABILITY")
st.title("🗺️ Network Map")
st.caption("All feeders, color-coded by average hours of supply per day (out of 24h). Click a marker for details.")

df = load_engineered_data()
reliability = feeder_reliability_summary(df)
reliability["color"] = reliability["avg_actual_hours"].apply(reliability_color)

col_filter, col_legend = st.columns([3, 1])
with col_filter:
    band_filter = st.multiselect(
        "Filter by reliability", ["green", "yellow", "orange", "red"],
        default=["green", "yellow", "orange", "red"],
        format_func=lambda c: {"green": "Good (>=18h/day)", "yellow": "Fair (12-18h/day)",
                                "orange": "Poor (6-12h/day)", "red": "Critical (<6h/day)"}[c],
    )
with col_legend:
    st.markdown(
        "🟢 Good &nbsp; 🟡 Fair &nbsp; 🟠 Poor &nbsp; 🔴 Critical",
        unsafe_allow_html=True,
    )

filtered = reliability[reliability["color"].isin(band_filter)]

try:
    import folium
    from streamlit_folium import st_folium

    center_lat = filtered["latitude"].mean()
    center_lon = filtered["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="cartodbpositron")

    for row in filtered.itertuples():
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=6,
            color=row.color,
            fill=True,
            fill_color=row.color,
            fill_opacity=0.8,
            popup=folium.Popup(
                f"<b>{row.FEEDER_NAME}</b><br>"
                f"Avg hours/day: {row.avg_actual_hours:.1f}h (of 24h)<br>"
                f"Band: {row.band:.0f}h max<br>"
                f"Availability vs band: {row.avg_availability_pct:.1f}%<br>"
                f"Avg shortfall: {row.avg_shortfall:.1f}h<br>"
                f"Addresses: {row.n_addresses}",
                max_width=250,
            ),
            tooltip=row.FEEDER_NAME,
        ).add_to(m)

    map_data = st_folium(m, use_container_width=True, height=600)

    if map_data and map_data.get("last_object_clicked_tooltip"):
        clicked_feeder = map_data["last_object_clicked_tooltip"]
        details = reliability[reliability["FEEDER_NAME"] == clicked_feeder]
        if not details.empty:
            d = details.iloc[0]
            st.subheader(f"Feeder: {clicked_feeder}")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                metric_card("Avg hours/day", f"{d['avg_actual_hours']:.1f}h")
            with c2:
                metric_card("Band (max hours)", f"{d['band']:.0f}h")
            with c3:
                metric_card("Availability vs band", f"{d['avg_availability_pct']:.1f}%")
            with c4:
                metric_card("Avg shortfall", f"{d['avg_shortfall']:.1f}h")
            with c5:
                metric_card("Addresses", f"{d['n_addresses']:.0f}")

except ImportError:
    st.warning(
        "`folium` / `streamlit-folium` aren't installed, so the interactive map can't "
        "render here. Falling back to a plain point map. Run "
        "`pip install folium streamlit-folium` for the full version with popups."
    )
    st.map(filtered.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])

st.divider()
st.subheader("All feeders")
table = filtered[["FEEDER_NAME", "avg_actual_hours", "band", "avg_availability_pct", "avg_shortfall", "n_addresses"]].rename(
    columns={
        "FEEDER_NAME": "Feeder", "avg_actual_hours": "Avg hours/day",
        "band": "Band (h)", "avg_availability_pct": "Avg availability %",
        "avg_shortfall": "Avg shortfall (h)", "n_addresses": "Addresses",
    }
).sort_values("Avg hours/day")
st.dataframe(table.round(1), use_container_width=True, hide_index=True)

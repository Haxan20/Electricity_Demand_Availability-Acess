"""
Recommendation engine: turns a shortfall forecast + a customer's stated
energy need into a solar / generator / battery / efficiency recommendation
with indicative sizing and cost estimates.

IMPORTANT: all Naira figures below (COST_ASSUMPTIONS) are placeholder
defaults, not live market prices. They will drift out of date and vary a
lot by installer, brand, and region. Treat every cost/ROI number this
module returns as a rough planning estimate, not a quote -- the UI should
say so explicitly, and a real installer quote is the only reliable number
for an actual purchase decision.
"""
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Cost assumptions -- EDIT THESE to match current market prices before using
# this for anything customer-facing. Sourced as rough 2024-era Nigerian
# solar/generator market ballparks, not verified against a live price feed.
# ---------------------------------------------------------------------------
COST_ASSUMPTIONS = {
    "solar_naira_per_kw": 700_000,       # installed cost, panels+inverter+labour
    "battery_naira_per_kwh": 350_000,    # lithium battery storage
    "install_flat_fee_naira": 150_000,
    "generator_naira_per_kva": 120_000,
    "fuel_naira_per_liter": 1_200,
    "generator_fuel_l_per_hour_per_kva": 0.30,
    "grid_tariff_naira_per_kwh": 68,     # what NOT running a generator "saves" per kWh, roughly
    "peak_sun_hours": 5.0,               # Lagos average usable sunlight hours/day
    "solar_system_efficiency": 0.75,
    "battery_depth_of_discharge": 0.80,
    "assumed_baseline_load_kw": 1.5,     # used only if user gives hours/day, not kWh/day
}

APPLIANCE_WATTAGE = {
    "LED Bulb (x5)": 50,
    "TV": 120,
    "Refrigerator": 150,
    "Fan": 75,
    "Freezer": 200,
    "Air Conditioner (1HP)": 900,
    "Air Conditioner (1.5HP)": 1300,
    "Washing Machine": 500,
    "Microwave": 1000,
    "Water Pump": 750,
    "Laptop/Charging": 65,
    "Iron": 1000,
}


@dataclass
class Recommendation:
    primary: str  # "none" | "solar" | "solar_battery"  (generator removed as a recommendable option)
    daily_need_kwh: float
    shortfall_energy_kwh_per_day: float
    avg_shortfall_hours: float
    shortfall_frequency: float  # fraction of forecast days with meaningful shortfall
    solar_kw: float = None
    solar_cost_naira: float = None
    battery_kwh: float = None
    battery_cost_naira: float = None
    generator_kva: float = None
    generator_cost_naira: float = None
    generator_fuel_naira_per_day: float = None
    annual_grid_gap_cost_naira: float = None
    payback_years: float = None
    notes: list = field(default_factory=list)


def appliance_list_to_kwh(selections: dict) -> float:
    """selections: {appliance_name: hours_per_day}"""
    total_wh = sum(APPLIANCE_WATTAGE.get(name, 0) * float(hrs) for name, hrs in selections.items())
    return round(total_wh / 1000, 2)


def hours_to_kwh(hours_needed: float, assumed_load_kw: float = None) -> float:
    load = float(assumed_load_kw) if assumed_load_kw else COST_ASSUMPTIONS["assumed_baseline_load_kw"]
    return round(float(hours_needed) * load, 2)


def build_recommendation(daily_need_kwh: float, avg_shortfall_hours: float,
                          shortfall_frequency: float, cost_assumptions: dict = None) -> Recommendation:
    """
    daily_need_kwh: customer's total daily electricity need, in kWh
    avg_shortfall_hours: average predicted SHORTFALL (hours/day) over the forecast window
    shortfall_frequency: fraction of forecast days (0-1) where shortfall exceeds a
        "meaningful" threshold (e.g. >2h) -- used to distinguish "occasional" from "frequent"
    """
    # Cast to native Python float immediately. Inputs are typically numpy
    # float32 (they come from a pandas Series with float32 dtype elsewhere
    # in the app). round() on a float32 returns a float32, and formatting a
    # bare float32 in an f-string or str() prints its full double-precision
    # expansion instead of the rounded value (e.g. round(float32(1.51), 2)
    # displays as "1.5099999904632568", not "1.51"). Casting to float here
    # means every round() call below operates on and returns a genuine
    # Python float, fixing this at the source instead of patching every
    # display site individually.
    daily_need_kwh = float(daily_need_kwh)
    avg_shortfall_hours = float(avg_shortfall_hours)
    shortfall_frequency = float(shortfall_frequency)

    c = {**COST_ASSUMPTIONS, **(cost_assumptions or {})}
    notes = []

    # Energy gap: proportion of the day the grid can't cover, applied to total need.
    # Simplifying assumption: demand is roughly uniform through the day. Real load
    # curves (e.g. AC load concentrated in daytime heat) will shift this -- flagged
    # as a known limitation, not corrected for in this MVP.
    shortfall_energy_kwh = round(daily_need_kwh * (avg_shortfall_hours / 24), 2)
    notes.append(
        "Energy gap assumes roughly even demand through the day; if your biggest "
        "loads run at specific times, actual gap may differ."
    )

    rec = Recommendation(
        primary="none",
        daily_need_kwh=daily_need_kwh,
        shortfall_energy_kwh_per_day=shortfall_energy_kwh,
        avg_shortfall_hours=round(avg_shortfall_hours, 2),
        shortfall_frequency=round(shortfall_frequency, 2),
        notes=notes,
    )

    if avg_shortfall_hours <= 1 and shortfall_frequency < 0.2:
        rec.primary = "none"
        rec.notes.append("Grid supply covers your stated need most days -- backup is optional.")
        return rec

    # Generator removed as a recommendable option (per request) -- every
    # shortfall level above "none" now gets solar, or solar+battery when
    # the shortfall is frequent, rather than a generator for the
    # in-between/occasional case.
    rec.primary = "solar_battery" if shortfall_frequency >= 0.5 else "solar"

    # --- Solar sizing (used for "solar" and "solar_battery") ---
    if rec.primary in ("solar", "solar_battery"):
        solar_kw = shortfall_energy_kwh / (c["peak_sun_hours"] * c["solar_system_efficiency"])
        rec.solar_kw = round(max(solar_kw, 0.5), 2)
        rec.solar_cost_naira = round(rec.solar_kw * c["solar_naira_per_kw"] + c["install_flat_fee_naira"])

    # --- Battery sizing (only when shortfall is frequent, per the brief) ---
    if rec.primary == "solar_battery":
        battery_kwh = shortfall_energy_kwh / c["battery_depth_of_discharge"]
        rec.battery_kwh = round(max(battery_kwh, 1.0), 2)
        rec.battery_cost_naira = round(rec.battery_kwh * c["battery_naira_per_kwh"])

    # --- ROI: value of the energy gap if bought at grid tariff, vs solar cost ---
    annual_gap_kwh = shortfall_energy_kwh * 365
    rec.annual_grid_gap_cost_naira = round(annual_gap_kwh * c["grid_tariff_naira_per_kwh"])

    if rec.solar_cost_naira:
        total_cost = rec.solar_cost_naira + (rec.battery_cost_naira or 0)
        # Compare against the cost of running a generator instead, purely as
        # a payback-period benchmark -- this is a cost comparison, not a
        # generator recommendation (generator is never rec.primary).
        equiv_generator_kva = max((daily_need_kwh / 24) / 0.8, 1.0)
        equiv_fuel_l_per_day = c["generator_fuel_l_per_hour_per_kva"] * equiv_generator_kva * avg_shortfall_hours
        equiv_annual_fuel_cost = equiv_fuel_l_per_day * c["fuel_naira_per_liter"] * 365
        rec.payback_years = round(total_cost / max(equiv_annual_fuel_cost, 1), 1)

    rec.notes.append(
        "All costs are indicative planning estimates using placeholder market "
        "rates (see COST_ASSUMPTIONS in recommend.py) -- get real installer "
        "quotes before making a purchase decision."
    )
    return rec

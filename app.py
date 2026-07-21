"""Streamlit UI for the perfect-foresight battery dispatch solver.

Run:  pixi run streamlit run app.py

Upload a CSV with a price signal (LMP) and a carbon signal (CO2 MOER, or a
pre-combined MOER+MBER), pick the columns, set battery parameters, and compare a
price-only "baseline" dispatch against a carbon-aware co-optimized dispatch. The
headline output is the realized marginal abatement cost ($/tonne CO2).
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dispatch_core import MASS_PER_TONNE, infer_dt_hours, run_comparison

# WattTime brand palette (see BRAND.md).
BG = "#f6f6f6"
BASELINE_COLOR = "#434343"   # neutral grey for the price-only run
CARBON_COLOR = "#83c341"     # green = clean-energy aggregate
PRICE_COLOR = "#000000"
MAX_INTERVALS = 200_000

st.set_page_config(page_title="Battery Dispatch Solver", page_icon="🔋", layout="wide")


@st.cache_data
def load_sample():
    return pd.read_csv("sample_signals_week.csv")


@st.cache_data(show_spinner=False)
def cached_run(price, carbon, dt, power_mw, power_discharge_mw, energy_mwh, rte,
               carbon_price, carbon_units, soc_init, soc_min, cycle_cost, terminal_soc):
    """Cache solves keyed on the signal arrays + all parameters."""
    return run_comparison(
        price, carbon, dt,
        power_mw=power_mw, power_discharge_mw=power_discharge_mw,
        energy_mwh=energy_mwh, rte=rte,
        carbon_price_per_tonne=carbon_price, carbon_units=carbon_units,
        soc_init=soc_init, soc_min=soc_min, cycle_cost=cycle_cost,
        terminal_soc=terminal_soc,
    )


def guess_col(cols, *keywords):
    """Return the index of the first column whose name contains any keyword."""
    lower = [c.lower() for c in cols]
    for kw in keywords:
        for i, c in enumerate(lower):
            if kw in c:
                return i
    return 0


st.title("🔋 Battery Dispatch Solver")
st.caption(
    "Perfect-foresight, price-taking battery. Compares a price-only baseline "
    "against a carbon-aware co-optimized dispatch. Open-source lightweight "
    "alternative to StorageVET for the single-service (energy-arbitrage) case."
)
with st.expander("About this MVP — assumptions & roadmap"):
    st.markdown(
        "- **Perfect foresight.** The optimizer sees the entire uploaded signal and "
        "dispatches optimally against it, so every result is an **upper bound** (ceiling) "
        "on what a real controller with imperfect forecasts would capture.\n"
        "- **Intended inputs:** historical actuals — real-time LMP for price, historical "
        "CO2 MOER (or a pre-combined MOER+MBER) for carbon. The tool is signal-agnostic, "
        "so any $/MWh price and any mass/MWh carbon column will work.\n"
        "- **Energy arbitrage only** — no ancillary services, no capacity.\n"
        "- **Coming next:** a performance *floor* (day-ahead self-schedule / limited "
        "lookahead) and a *realistic middle* (dispatch on forecast, settle on actuals) to "
        "bracket the ceiling shown here."
    )

# --------------------------------------------------------------------------- #
# Data input
# --------------------------------------------------------------------------- #
st.subheader("1. Signal data")
col_a, col_b = st.columns([3, 1])
with col_a:
    uploaded = st.file_uploader("Upload a CSV (timestamp + price + carbon columns)", type="csv")
with col_b:
    use_demo = st.button("Load synthetic demo", width="stretch")

df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_demo or st.session_state.get("_demo_loaded"):
    df = load_sample()
    st.session_state["_demo_loaded"] = True
    st.info("Loaded the synthetic demo week (168 hourly intervals).")

if df is None:
    st.stop()

if len(df) > MAX_INTERVALS:
    st.error(f"{len(df):,} intervals exceeds the {MAX_INTERVALS:,} cap for this hosted MVP.")
    st.stop()

cols = list(df.columns)
c1, c2, c3 = st.columns(3)
with c1:
    ts_choice = st.selectbox("Timestamp column", ["(none / set interval manually)"] + cols,
                             index=(cols.index("timestamp") + 1) if "timestamp" in cols else 0)
with c2:
    price_col = st.selectbox("Price column ($/MWh)", cols, index=guess_col(cols, "lmp", "price"))
with c3:
    carbon_col = st.selectbox("Carbon column", cols, index=guess_col(cols, "moer", "carbon", "co2"))

# Interval length
irregular = False
if ts_choice != "(none / set interval manually)":
    ts = pd.to_datetime(df[ts_choice])
    dt_hours, irregular = infer_dt_hours(ts.values)
    st.caption(f"Inferred interval: **{dt_hours * 60:.1f} min** ({dt_hours:.4f} h) "
               f"from the median timestamp gap.")
else:
    ts = pd.RangeIndex(len(df))
    dt_hours = st.number_input("Interval length (hours)", value=1.0, min_value=1e-3, step=0.25)

if irregular:
    st.warning("Irregular timestamp gaps detected (>1% from median). The solver assumes a "
               "uniform interval; results may be off if gaps are real.")

# Coerce to numeric (stray text -> NaN) and handle gaps: real LMP/MOER exports
# routinely have missing intervals, and the solver needs a gap-free finite series.
price_raw = pd.to_numeric(df[price_col], errors="coerce").to_numpy(dtype=float)
carbon_raw = pd.to_numeric(df[carbon_col], errors="coerce").to_numpy(dtype=float)
n_bad_p = int(np.sum(~np.isfinite(price_raw)))
n_bad_c = int(np.sum(~np.isfinite(carbon_raw)))

if n_bad_p or n_bad_c:
    st.warning(f"Missing/non-numeric values found: **{n_bad_p}** in '{price_col}', "
               f"**{n_bad_c}** in '{carbon_col}'. The solver needs a gap-free series.")
    fix = st.radio("Handle them by:",
                   ["Linear-interpolate gaps", "Stop (I'll fix the file)"], horizontal=True)
    if fix.startswith("Stop"):
        st.stop()
    price = pd.Series(price_raw).interpolate(limit_direction="both").bfill().ffill().to_numpy()
    carbon = pd.Series(carbon_raw).interpolate(limit_direction="both").bfill().ffill().to_numpy()
    if not (np.all(np.isfinite(price)) and np.all(np.isfinite(carbon))):
        st.error("A selected column is entirely empty after coercion — pick a numeric column.")
        st.stop()
    st.caption(f"Filled {n_bad_p + n_bad_c} value(s) by linear interpolation.")
else:
    price, carbon = price_raw, carbon_raw

# --------------------------------------------------------------------------- #
# Parameters (sidebar)
# --------------------------------------------------------------------------- #
st.sidebar.header("Battery")
asymmetric = st.sidebar.checkbox("Different charge vs discharge power", value=False)
if asymmetric:
    power_mw = st.sidebar.number_input("Max charge power (MW)", value=10.0, min_value=0.01)
    power_discharge_mw = st.sidebar.number_input("Max discharge power (MW)", value=10.0, min_value=0.01)
else:
    power_mw = st.sidebar.number_input("Power (MW)", value=10.0, min_value=0.01)
    power_discharge_mw = None
energy_mwh = st.sidebar.number_input("Usable energy (MWh)", value=40.0, min_value=0.01)
rte_pct = st.sidebar.slider("Round-trip efficiency (%)", 50, 100, 85)
soc_init_pct = st.sidebar.slider("Initial SOC (% of energy)", 0, 100, 0)
soc_min_pct = st.sidebar.slider("Minimum SOC (% of energy)", 0, 100, 0)
terminal_soc = st.sidebar.checkbox("Force final SOC = initial SOC", value=True,
                                   help="Prevents value inflation from draining the battery over the horizon.")
cycle_cost = st.sidebar.number_input("Cycle cost ($/MWh discharged)", value=0.0, min_value=0.0, step=1.0,
                                     help="Throughput/degradation penalty. 0 = off (may over-cycle).")

st.sidebar.header("Carbon")
carbon_units = st.sidebar.selectbox("Carbon signal units", list(MASS_PER_TONNE.keys()), index=0)
carbon_price = st.sidebar.number_input("Carbon price ($/tonne CO2)", value=50.0, min_value=0.0, step=5.0,
                                       help="How aggressively the carbon-aware run trades $ for emissions. "
                                            "The input value is arbitrary; the realized abatement cost is the "
                                            "meaningful output.")

soc_init = energy_mwh * soc_init_pct / 100.0
soc_min = energy_mwh * soc_min_pct / 100.0
if soc_init < soc_min:
    st.sidebar.error("Initial SOC is below minimum SOC.")
    st.stop()

# --------------------------------------------------------------------------- #
# Solve
# --------------------------------------------------------------------------- #
run = st.button("Run dispatch", type="primary")
if not run:
    st.stop()

with st.spinner("Solving baseline and carbon-aware dispatch..."):
    comp = cached_run(
        price, carbon, dt_hours,
        power_mw, power_discharge_mw, energy_mwh, rte_pct / 100.0,
        carbon_price, carbon_units, soc_init, soc_min, cycle_cost, terminal_soc,
    )

if not (comp.baseline.success and comp.carbon_aware.success):
    st.error(f"Solver failed. Baseline: {comp.baseline.status}. "
             f"Carbon-aware: {comp.carbon_aware.status}.")
    st.stop()

# --------------------------------------------------------------------------- #
# Headline
# --------------------------------------------------------------------------- #
st.subheader("2. Result")
h1, h2, h3 = st.columns(3)
ac = comp.abatement_cost_per_tonne
h1.metric("Realized abatement cost",
          "N/A" if np.isnan(ac) else f"${ac:,.0f} / tonne",
          help="Revenue foregone divided by tonnes abated. The number that actually "
               "informs a decision (the carbon-price input is arbitrary).")
h2.metric("CO2 abated", f"{comp.tonnes_abated:,.1f} tonnes",
          help="Baseline net emissions minus carbon-aware net emissions.")
h3.metric("Revenue foregone", f"${comp.revenue_foregone:,.0f}",
          help="Baseline revenue minus carbon-aware revenue.")

# --------------------------------------------------------------------------- #
# Comparison table
# --------------------------------------------------------------------------- #
bm, cm = comp.baseline_metrics, comp.carbon_aware_metrics
table = pd.DataFrame(
    {
        "Baseline (price only)": [bm.revenue, bm.net_emissions_tonnes, bm.equiv_cycles,
                                  bm.mwh_discharged, bm.simultaneous_intervals],
        "Carbon-aware": [cm.revenue, cm.net_emissions_tonnes, cm.equiv_cycles,
                         cm.mwh_discharged, cm.simultaneous_intervals],
    },
    index=["Revenue ($)", "Net emissions (tonnes CO2)", "Equivalent full cycles",
           "MWh discharged", "Simultaneous charge+discharge intervals"],
)
st.dataframe(table.style.format({"Baseline (price only)": "{:,.2f}", "Carbon-aware": "{:,.2f}"}),
             width="stretch")

# Data-quality tripwires
neg = int(np.sum(price < 0))
sim = bm.simultaneous_intervals + cm.simultaneous_intervals
notes = [f"{neg} negative-price interval(s)"]
if sim:
    notes.append(f":red[{sim} simultaneous charge+discharge interval(s) — check the guard]")
if irregular:
    notes.append(":red[irregular timestamp gaps]")
st.caption("Data-quality: " + " · ".join(notes))

# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
x = ts if ts_choice != "(none / set interval manually)" else np.arange(len(df))
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    subplot_titles=("Signal ($/MWh): price vs carbon-aware effective signal",
                    "Net dispatch (MW, + = discharge)", "State of charge (MWh)"),
)
# Panel 1: price and the effective carbon-aware signal (both $/MWh)
fig.add_trace(go.Scatter(x=x, y=comp.price, name="Price (LMP)", line=dict(color=PRICE_COLOR)), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=comp.carbon_aware_signal, name="Carbon-aware signal",
                         line=dict(color=CARBON_COLOR)), row=1, col=1)
# Panel 2: net dispatch
fig.add_trace(go.Scatter(x=x, y=comp.baseline.discharge_mw - comp.baseline.charge_mw,
                         name="Baseline dispatch", line=dict(color=BASELINE_COLOR)), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=comp.carbon_aware.discharge_mw - comp.carbon_aware.charge_mw,
                         name="Carbon-aware dispatch", line=dict(color=CARBON_COLOR)), row=2, col=1)
# Panel 3: SOC
fig.add_trace(go.Scatter(x=x, y=comp.baseline.soc_mwh, name="Baseline SOC",
                         line=dict(color=BASELINE_COLOR)), row=3, col=1)
fig.add_trace(go.Scatter(x=x, y=comp.carbon_aware.soc_mwh, name="Carbon-aware SOC",
                         line=dict(color=CARBON_COLOR)), row=3, col=1)
fig.update_layout(height=750, plot_bgcolor=BG, paper_bgcolor=BG,
                  legend=dict(orientation="h", y=-0.08), margin=dict(t=40, b=40))
st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
out = pd.DataFrame({
    "timestamp": ts if ts_choice != "(none / set interval manually)" else np.arange(len(df)),
    "price": comp.price,
    "carbon_tonnes_per_mwh": comp.carbon_tonnes_per_mwh,
    "baseline_charge_mw": comp.baseline.charge_mw,
    "baseline_discharge_mw": comp.baseline.discharge_mw,
    "baseline_soc_mwh": comp.baseline.soc_mwh,
    "carbon_aware_charge_mw": comp.carbon_aware.charge_mw,
    "carbon_aware_discharge_mw": comp.carbon_aware.discharge_mw,
    "carbon_aware_soc_mwh": comp.carbon_aware.soc_mwh,
})
buf = io.StringIO()
out.to_csv(buf, index=False)
st.download_button("Download dispatch CSV", buf.getvalue(), file_name="dispatch_results.csv",
                   mime="text/csv")

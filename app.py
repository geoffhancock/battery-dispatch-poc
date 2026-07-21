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

from dispatch_core import (
    MASS_PER_TONNE,
    abatement_frontier,
    infer_dt_hours,
    run_comparison,
    signal_alignment,
)

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


@st.cache_data(show_spinner=False)
def cached_frontier(price, carbon, dt, power_mw, power_discharge_mw, energy_mwh, rte,
                    carbon_units, n_points, soc_init, soc_min, cycle_cost, terminal_soc):
    """Cache the carbon-price sweep (one baseline + n_points carbon-aware solves)."""
    return abatement_frontier(
        price, carbon, dt,
        power_mw=power_mw, power_discharge_mw=power_discharge_mw,
        energy_mwh=energy_mwh, rte=rte, carbon_units=carbon_units, n_points=n_points,
        soc_init=soc_init, soc_min=soc_min, cycle_cost=cycle_cost, terminal_soc=terminal_soc,
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

st.sidebar.header("Analysis")
n_points = st.sidebar.slider("Abatement-curve points", 4, 20, 10,
                             help="Carbon prices swept for the abatement curve. "
                                  "More points = smoother curve but more solves.")

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
# Signal alignment
# --------------------------------------------------------------------------- #
with st.spinner("Computing abatement curve..."):
    fr = cached_frontier(price, carbon, dt_hours, power_mw, power_discharge_mw,
                         energy_mwh, rte_pct / 100.0, carbon_units, n_points,
                         soc_init, soc_min, cycle_cost, terminal_soc)

align = signal_alignment(comp.price, comp.carbon_tonnes_per_mwh, dispatch=comp.baseline)

st.subheader("3. Signal alignment")
g1, g2, g3 = st.columns(3)
g1.metric("LMP vs MOER alignment (Spearman)", f"{align['spearman']:+.2f}",
          help="Rank correlation of price vs carbon. Positive = cheap hours are also "
               "clean hours, so price-only dispatch incidentally abates carbon. Rank "
               "(not Pearson) because dispatch is about ordering and it resists price spikes.")
g2.metric("Alignment in used hours", "n/a" if np.isnan(align["spearman_active"])
          else f"{align['spearman_active']:+.2f}",
          help="Spearman restricted to the intervals the battery actually charges/discharges "
               "-- the alignment it experiences. Tails matter more than the whole distribution.")
g3.metric("CO2 price-only captures",
          "n/a" if np.isnan(fr.capture_fraction) else f"{fr.capture_fraction * 100:,.0f}%",
          help="Share of the maximum avoidable CO2 that price-only dispatch already gets "
               "for free (zero revenue cost).")

# Scatter: price vs carbon, colored by hour-of-day if available
has_ts = ts_choice != "(none / set interval manually)"
hod = pd.to_datetime(df[ts_choice]).dt.hour if has_ts else np.zeros(len(price))
sc = go.Figure(go.Scatter(
    x=price, y=carbon, mode="markers",
    marker=dict(size=5, color=hod, colorscale="Twilight", showscale=has_ts,
                colorbar=dict(title="hour") if has_ts else None, opacity=0.6),
))
sc.update_layout(height=360, plot_bgcolor=BG, paper_bgcolor=BG,
                 xaxis_title=f"Price ({price_col})", yaxis_title=f"Carbon ({carbon_col})",
                 margin=dict(t=30, b=40),
                 title=f"Each point = one interval  ·  Spearman {align['spearman']:+.2f}")
st.plotly_chart(sc, width="stretch")

# --------------------------------------------------------------------------- #
# Abatement cost curve
# --------------------------------------------------------------------------- #
st.subheader("4. Abatement cost curve")
tonnes = fr.tonnes_abated
mask = tonnes > 1e-9
avg_cost = np.full_like(tonnes, np.nan)
avg_cost[mask] = fr.revenue_foregone[mask] / tonnes[mask]

fig_mac = go.Figure()
fig_mac.add_trace(go.Scatter(x=tonnes, y=avg_cost, mode="lines+markers",
                             name="Average $/tonne", line=dict(color=CARBON_COLOR)))
fig_mac.add_trace(go.Scatter(x=tonnes[1:], y=fr.marginal_cost, mode="lines",
                             name="Marginal $/tonne (MAC)",
                             line=dict(color=BASELINE_COLOR, dash="dash")))
if not np.isnan(comp.abatement_cost_per_tonne):
    fig_mac.add_trace(go.Scatter(
        x=[comp.tonnes_abated], y=[comp.abatement_cost_per_tonne], mode="markers",
        name=f"Current (@ ${carbon_price:,.0f}/t)",
        marker=dict(size=13, color=PRICE_COLOR, symbol="x")))
fig_mac.update_layout(height=380, plot_bgcolor=BG, paper_bgcolor=BG,
                      xaxis_title="Additional CO2 abated beyond price-only (tonnes)",
                      yaxis_title="Cost ($/tonne)", legend=dict(orientation="h", y=-0.2),
                      margin=dict(t=30, b=40))
st.plotly_chart(fig_mac, width="stretch")
if tonnes.max() > 1e-9:
    st.caption(
        f"Beyond the price-only dispatch, up to **{tonnes.max():,.0f} more tonnes** of CO2 "
        f"can be abated; the final increment costs ~**${np.nanmax(fr.marginal_cost):,.0f}/tonne**. "
        "The curve steepens as you push from price-optimal toward carbon-optimal."
    )

# --------------------------------------------------------------------------- #
# Carpet plots (day x hour-of-day)
# --------------------------------------------------------------------------- #
st.subheader("5. Dispatch & signal patterns")
if not has_ts:
    st.caption("Upload data with a timestamp column to see day x hour-of-day carpet plots.")
else:
    tsi = pd.to_datetime(df[ts_choice])

    def carpet(values):
        d = pd.DataFrame({"date": tsi.dt.normalize(), "hour": tsi.dt.hour, "v": values})
        return d.pivot_table(index="hour", columns="date", values="v", aggfunc="mean")

    base_net = comp.baseline.discharge_mw - comp.baseline.charge_mw
    ca_net = comp.carbon_aware.discharge_mw - comp.carbon_aware.charge_mw

    cc1, cc2 = st.columns(2)
    disp_choice = cc1.radio("Dispatch (MW, + = discharge)",
                            ["Carbon-aware", "Baseline", "Difference (CA - Base)"])
    sig_choice = cc2.radio("Signal", ["MOER", "LMP", "Carbon-aware signal"])

    disp_vals = {"Carbon-aware": ca_net, "Baseline": base_net,
                 "Difference (CA - Base)": ca_net - base_net}[disp_choice]
    sig_vals = {"MOER": carbon, "LMP": price,
                "Carbon-aware signal": comp.carbon_aware_signal}[sig_choice]

    # All colors below are from the WattTime palette; MOER uses a deliberate
    # clean->dirty green/yellow/coal-red ramp (all documented palette colors).
    disp_scale = [[0.0, "#aadee8"], [0.5, "#f6f6f6"], [1.0, "#83c341"]]  # charge->discharge
    moer_scale = [[0.0, "#83c341"], [0.5, "#fbd20b"], [1.0, "#cc4125"]]  # clean->dirty
    money_scale = [[0.0, "#dbf0f2"], [1.0, "#434343"]]
    sig_scale = moer_scale if sig_choice == "MOER" else money_scale

    disp_pv, sig_pv = carpet(disp_vals), carpet(sig_vals)
    cfig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                         subplot_titles=(f"Dispatch — {disp_choice}", f"Signal — {sig_choice}"))
    cfig.add_trace(go.Heatmap(z=disp_pv.values, x=disp_pv.columns.astype(str),
                              y=disp_pv.index, colorscale=disp_scale, zmid=0,
                              colorbar=dict(title="MW", len=0.45, y=0.79)), row=1, col=1)
    cfig.add_trace(go.Heatmap(z=sig_pv.values, x=sig_pv.columns.astype(str),
                              y=sig_pv.index, colorscale=sig_scale,
                              colorbar=dict(len=0.45, y=0.21)), row=2, col=1)
    cfig.update_layout(height=620, plot_bgcolor=BG, paper_bgcolor=BG,
                       margin=dict(t=40, b=30))
    cfig.update_yaxes(title="hour of day", autorange="reversed")
    st.plotly_chart(cfig, width="stretch")
    st.caption("Discharge (green) should line up with high-price / high-MOER cells where the "
               "signals align. The **Difference** view isolates the hours carbon-awareness "
               "changes behavior. Sub-hourly data is averaged into hourly cells for display.")

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

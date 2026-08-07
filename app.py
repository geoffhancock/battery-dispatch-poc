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
    evaluate_actual,
    infer_dt_hours,
    parse_timestamps,
    run_comparison,
    signal_alignment,
    solve_dispatch,
)

# WattTime brand palette (see BRAND.md).
BG = "#ffffff"               # chart bg matches the white main page (sidebar is gray)
ZERO_COLOR = "#f6f6f6"       # diverging midpoint for dispatch carpets
BASELINE_COLOR = "#434343"   # grey = Model A (price-optimized)
CARBON_COLOR = "#83c341"     # green = Model B (price + CO2)
MOER_COLOR = "#e08b66"       # gas orange for the MOER line (deliberate non-fuel choice)
ACTUAL_COLOR = "#2b6cb0"     # blue = Actual (metered) dispatch (readable blue; not in brand palette)
PRICE_COLOR = "#000000"
MAX_INTERVALS = 200_000
# Solver bounds: the negative-price guard adds one binary per negative interval, so a
# full year of 5-min data can blow up to a hang. A 1% gap + time limit returns a
# near-optimal dispatch fast (proving exact optimality is what explodes).
MIP_GAP = 0.01
SOLVE_TIME_LIMIT = 120  # seconds per solve

# Scenario naming (see the "Model A / Model B" key under Results).
A_BRIEF, B_BRIEF, CMAX_BRIEF, ACT_BRIEF = (
    "A: Price Optimized", "B: Price+CO2", "C: CO2 only", "Actual Dispatch")
A_SIG, B_SIG, C_SIG = "A: LMP", "B: LMP+CO2", "C: CO2 MOER"

st.set_page_config(page_title="Battery Dispatch Solver", page_icon="🔋", layout="wide")


@st.cache_data
def load_sample():
    return pd.read_csv("sample_signals_week.csv")


@st.cache_data(show_spinner=False)
def cached_solve(signal, **kw):
    """Cache one MILP solve keyed on the signal array + params. run_comparison /
    abatement_frontier are called UNCACHED (their assembly is cheap) with this as
    solve_fn, so the progress bar can be driven from the main thread while the
    heavy solves stay cached across reruns."""
    return solve_dispatch(signal, **kw)


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

actual_col = st.selectbox(
    "Actual dispatch column — net MW, + = discharge (optional)",
    ["(none)"] + cols,
    index=next((i + 1 for i, c in enumerate(cols)
                if any(k in c.lower() for k in ("dispatch", "actual", "metered"))), 0),
    help="A metered/real net-dispatch series. Adds real revenue/CO2 to the comparison "
         "table and a real pattern to the carpet plots.",
)

# Interval length
irregular_concern = False  # only the "likely missing data" case feeds the data-quality note
if ts_choice != "(none / set interval manually)":
    ts, is_utc = parse_timestamps(df[ts_choice])
    dt_hours, n_irr, max_gap_h = infer_dt_hours(ts.values)
    st.caption(f"Inferred interval: **{dt_hours * 60:.1f} min** ({dt_hours:.4f} h) "
               f"from the median timestamp gap.")
    if is_utc:
        st.warning("This column is in **UTC**, so carpets and the x-axis will be in UTC hours — "
                   "rarely what you want for hour-of-day patterns. Upload local (or offset) "
                   "timestamps to get local-time plots.")
    if n_irr:
        # DST transitions in naive local time produce ~1-2 ~1-hour gaps/year -- benign.
        if n_irr <= 4 and max_gap_h <= 3.0:
            st.caption(f"{n_irr} irregular interval(s) (largest {max_gap_h * 60:.0f} min) — "
                       "consistent with daylight-saving transitions. The solver treats rows in "
                       "order at a uniform step, so the effect is negligible.")
        else:
            irregular_concern = True
            st.warning(f"{n_irr} irregular interval(s) ({100 * n_irr / (len(ts) - 1):.1f}% of rows; "
                       f"largest {max_gap_h * 60:.0f} min). The solver assumes a uniform step and "
                       "treats rows in order — this many gaps may indicate missing data.")
else:
    ts = pd.RangeIndex(len(df))
    dt_hours = st.number_input("Interval length (hours)", value=1.0, min_value=1e-3, step=0.25)

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

# Optional metered dispatch series (net MW, + = discharge).
actual_net = None
if actual_col != "(none)":
    actual_net = pd.to_numeric(df[actual_col], errors="coerce").to_numpy(dtype=float)
    n_bad_a = int(np.sum(~np.isfinite(actual_net)))
    if n_bad_a:
        actual_net = np.nan_to_num(actual_net, nan=0.0)
        st.caption(f"Actual dispatch: filled {n_bad_a} missing value(s) with 0 MW.")

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
compute_curve = st.sidebar.checkbox(
    "Compute abatement curve", value=True,
    help="Section 4 runs extra solves (one per curve point) to trace revenue vs CO2. "
         "Uncheck to skip it for much faster runs on large datasets.")
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
# Latch the run state: st.button is True only on the click's rerun, so any later
# widget interaction (e.g. a carpet radio) would otherwise re-hide the results.
if st.button("Run dispatch", type="primary"):
    st.session_state["has_run"] = True
if not st.session_state.get("has_run"):
    st.info("Set parameters in the sidebar and click **Run dispatch**.")
    st.stop()

_run_prog = st.progress(0.0, text="Solving dispatch…")
comp = run_comparison(
    price, carbon, dt_hours,
    power_mw=power_mw, power_discharge_mw=power_discharge_mw,
    energy_mwh=energy_mwh, rte=rte_pct / 100.0,
    carbon_price_per_tonne=carbon_price, carbon_units=carbon_units,
    soc_init=soc_init, soc_min=soc_min, cycle_cost=cycle_cost, terminal_soc=terminal_soc,
    mip_gap=MIP_GAP, time_limit=SOLVE_TIME_LIMIT, solve_fn=cached_solve,
    _progress=lambda f, m: _run_prog.progress(min(f, 1.0), text=m),
)
_run_prog.empty()

if not (comp.baseline.success and comp.carbon_aware.success):
    st.error(f"Solver failed. Baseline: {comp.baseline.status}. "
             f"Carbon-aware: {comp.carbon_aware.status}.")
    st.stop()

# --------------------------------------------------------------------------- #
# Headline
# --------------------------------------------------------------------------- #
st.subheader("2. Result")
st.caption(
    "**Model A (Baseline):** revenue-optimized on the LMP signal  ·  "
    "**Model B (Intervention):** carbon-aware, optimized on LMP + CO2  ·  "
    "**Model C (CO2 only):** optimal carbon outcome, for reference  ·  "
    "**Actual Dispatch:** metered series (if provided)"
)
# --------------------------------------------------------------------------- #
# Comparison table (scenarios as columns, metrics as rows)
# --------------------------------------------------------------------------- #
bm, cm, cmx = comp.baseline_metrics, comp.carbon_aware_metrics, comp.carbon_max_metrics
REVCAP_ROW = "% of max revenue capture"   # denominator = Model A (revenue-optimal)
CO2CAP_ROW = "% of max CO2 capture"       # denominator = Model C (carbon-optimal)
# (metric label, attribute, format) -- format gives 0 dp except cycles at 1 dp.
# &#36; = literal "$" (avoids Streamlit rendering $...$ as LaTeX inside the table HTML).
metric_specs = [
    ("Revenue from arbitrage (&#36;)", "revenue", "&#36;{:,.0f}"),
    ("Emissions change, negative = avoided (tonnes CO2)", "net_emissions_tonnes", "{:,.0f}"),
    ("Equivalent full cycles", "equiv_cycles", "{:,.1f}"),
    ("MWh discharged", "mwh_discharged", "{:,.0f}"),
    ("Simultaneous charge+discharge intervals", "simultaneous_intervals", "{:,.0f}"),
]
scenarios = [(A_BRIEF, bm), (B_BRIEF, cm), (CMAX_BRIEF, cmx)]
if actual_net is not None:
    am = evaluate_actual(actual_net, price, carbon, dt_hours, energy_mwh, carbon_units)
    scenarios.append((ACT_BRIEF, am))

max_av = comp.max_avoided_tonnes            # Model C avoids the most CO2
max_rev = comp.baseline_metrics.revenue     # Model A earns the most revenue


def _co2_cap(m):
    return "n/a" if abs(max_av) < 1e-9 else f"{100 * (-m.net_emissions_tonnes) / max_av:,.0f}%"


def _rev_cap(m):
    return "n/a" if abs(max_rev) < 1e-9 else f"{100 * m.revenue / max_rev:,.0f}%"


names = [name for (name, _, _) in metric_specs]
# Pair each headline metric with its capture %: revenue -> % of max revenue,
# emissions -> % of max CO2.
row_index = [names[0], REVCAP_ROW, names[1], CO2CAP_ROW] + names[2:]
data = {}
for label, m in scenarios:
    col = {name: fmt.format(getattr(m, attr)) for (name, attr, fmt) in metric_specs}
    col[REVCAP_ROW] = _rev_cap(m)
    col[CO2CAP_ROW] = _co2_cap(m)
    data[label] = col
table = pd.DataFrame(data).reindex(row_index)
table.columns.name = "Dispatch Scenario"
green_cols = [B_BRIEF]                          # highlight the intervention (Model B)
unbold = names[2:]                              # cycles, MWh, simultaneous -> normal-weight labels
styler = (
    table.style
    .set_properties(subset=green_cols, **{"background-color": "#d1e49f"})
    .map_index(lambda v: "background-color: #d1e49f" if v in green_cols else "", axis="columns")
    .map_index(lambda v: "font-weight: normal" if v in unbold else "", axis="index")
    .set_table_styles([
        {"selector": "thead th", "props": [("text-align", "center"), ("padding", "6px 12px")]},
        {"selector": "tbody th", "props": [("text-align", "left"), ("padding", "6px 12px")]},
        {"selector": "td", "props": [("text-align", "center"), ("padding", "6px 12px")]},
    ])
)
st.markdown(styler.to_html(), unsafe_allow_html=True)

if abs(max_av) < 1e-9:
    st.caption("Model C avoids ~0 CO2 (carbon price is 0, or MOER varies too little to beat "
               "round-trip losses), so '% of max CO2 capture' is undefined.")

# B-vs-A highlights box (custom HTML so the light-green wash renders reliably; &#36; = literal
# "$" to avoid LaTeX; title= gives hover tooltips like st.metric's help).
ac = comp.abatement_cost_per_tonne
base_rev = comp.baseline_metrics.revenue
rev_pct = (100 * comp.revenue_foregone / base_rev) if abs(base_rev) > 1e-9 else float("nan")
ac_str = "N/A" if np.isnan(ac) else f"&#36;{ac:,.0f} / tonne"
co2_str = f"{comp.tonnes_abated:,.1f} tonnes"
rev_str = (f"&#36;{comp.revenue_foregone:,.0f}"
           + ("" if np.isnan(rev_pct) else f" ({rev_pct:.1f}%)"))


def _stat(label, value, tip):
    return (f"<div title='{tip}' style='flex:1; min-width:150px;'>"
            f"<div style='color:#434343; font-size:0.8rem;'>{label}</div>"
            f"<div style='font-size:1.7rem; font-weight:600; color:#000;'>{value}</div></div>")


st.markdown(
    "<div style='background-color:#d1e49f; border:1px solid #83c341; border-radius:8px; "
    "padding:14px 18px; margin:10px 0;'>"
    "<div style='font-weight:700; margin-bottom:10px;'>Model B (Price+CO2) vs Model A "
    "(Price-optimized) — modeled, perfect foresight</div>"
    "<div style='display:flex; gap:28px; flex-wrap:wrap;'>"
    + _stat("Realized abatement cost", ac_str,
            "Revenue foregone divided by tonnes abated (the carbon-price input is arbitrary).")
    + _stat("CO2 abated", co2_str, "Model A net emissions minus Model B net emissions.")
    + _stat("Revenue foregone (%)", rev_str,
            "Model A revenue minus Model B revenue; percent is of the max (Model A) revenue.")
    + "</div></div>",
    unsafe_allow_html=True,
)

# Data-quality tripwires
neg = int(np.sum(price < 0))
sim = bm.simultaneous_intervals + cm.simultaneous_intervals
notes = [f"{neg} negative-price interval(s) ({100 * neg / len(price):.1f}%)"]
if sim:
    notes.append(f":red[{sim} simultaneous charge+discharge interval(s) — check the guard]")
if irregular_concern:
    notes.append(":red[irregular timestamp gaps]")
st.caption("Data-quality: " + " · ".join(notes))

# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
x = ts if ts_choice != "(none / set interval manually)" else np.arange(len(df))

with st.expander("Signal chart y-axis limits (optional)"):
    st.caption("Tip: you can also drag on either y-axis in the chart to zoom, and double-click "
               "to reset. MOER (right) and LMP (left) auto-scale independently by default.")
    manual_y = st.checkbox("Set y-axis limits manually")
    if manual_y:
        yc1, yc2 = st.columns(2)
        l_lo = yc1.number_input("$/MWh axis min", value=float(np.floor(np.nanmin(price))))
        l_hi = yc1.number_input("$/MWh axis max", value=float(np.ceil(np.nanmax(price))))
        r_lo = yc2.number_input("MOER axis min", value=float(np.floor(np.nanmin(carbon))))
        r_hi = yc2.number_input("MOER axis max", value=float(np.ceil(np.nanmax(carbon))))

fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
    specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]],
    subplot_titles=("Signal — A: LMP & B: LMP+CO2 ($/MWh, left) · MOER (lbs/MWh, right)",
                    "Net dispatch (MW, + = discharge)", "State of charge (%)"),
)
# Panel 1: A/B effective signal ($/MWh, left); MOER (lbs/MWh, right).
# One legend entry per series (legendgroup) so clicking toggles it across all panels.
fig.add_trace(go.Scatter(x=x, y=comp.price, name=A_SIG, legendgroup="A",
                         line=dict(color=BASELINE_COLOR)), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=comp.carbon_aware_signal, name=B_SIG, legendgroup="B",
                         line=dict(color=CARBON_COLOR)), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=carbon, name="MOER", legendgroup="MOER",
                         line=dict(color=MOER_COLOR, dash="dot")), row=1, col=1, secondary_y=True)
# Panel 2: net dispatch (A, B, and Actual if provided)
fig.add_trace(go.Scatter(x=x, y=comp.baseline.discharge_mw - comp.baseline.charge_mw,
                         name=A_SIG, legendgroup="A", showlegend=False,
                         line=dict(color=BASELINE_COLOR)), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=comp.carbon_aware.discharge_mw - comp.carbon_aware.charge_mw,
                         name=B_SIG, legendgroup="B", showlegend=False,
                         line=dict(color=CARBON_COLOR)), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=comp.carbon_max.discharge_mw - comp.carbon_max.charge_mw,
                         name=CMAX_BRIEF, legendgroup="C",
                         line=dict(color=MOER_COLOR)), row=2, col=1)
if actual_net is not None:
    fig.add_trace(go.Scatter(x=x, y=actual_net, name=ACT_BRIEF, legendgroup="Actual",
                             line=dict(color=ACTUAL_COLOR, dash="dash", width=2)), row=2, col=1)
# Panel 3: SOC as percent of usable energy
fig.add_trace(go.Scatter(x=x, y=comp.baseline.soc_mwh / energy_mwh * 100,
                         name=A_SIG, legendgroup="A", showlegend=False,
                         line=dict(color=BASELINE_COLOR)), row=3, col=1)
fig.add_trace(go.Scatter(x=x, y=comp.carbon_aware.soc_mwh / energy_mwh * 100,
                         name=B_SIG, legendgroup="B", showlegend=False,
                         line=dict(color=CARBON_COLOR)), row=3, col=1)
fig.add_trace(go.Scatter(x=x, y=comp.carbon_max.soc_mwh / energy_mwh * 100,
                         name=CMAX_BRIEF, legendgroup="C", showlegend=False,
                         line=dict(color=MOER_COLOR)), row=3, col=1)
fig.update_yaxes(title_text="$/MWh", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="MOER (lbs/MWh)", row=1, col=1, secondary_y=True,
                 color=MOER_COLOR, showgrid=False)
fig.update_yaxes(title_text="MW", row=2, col=1)
fig.update_yaxes(title_text="SOC (%)", range=[0, 100], row=3, col=1)
if manual_y:
    fig.update_yaxes(range=[l_lo, l_hi], row=1, col=1, secondary_y=False)
    fig.update_yaxes(range=[r_lo, r_hi], row=1, col=1, secondary_y=True)
fig.update_layout(height=790, plot_bgcolor=BG, paper_bgcolor=BG,
                  legend=dict(orientation="h", yanchor="bottom", y=1.10,
                              xanchor="center", x=0.5),
                  margin=dict(t=100, b=40))
st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- #
# Signal alignment
# --------------------------------------------------------------------------- #
align = signal_alignment(comp.price, comp.carbon_tonnes_per_mwh, dispatch=comp.baseline)

st.subheader("3. Signal alignment")
g1, g2 = st.columns(2)
g1.metric("LMP vs MOER alignment (Spearman)", f"{align['spearman']:+.2f}",
          help="Rank correlation of price vs carbon. Positive = cheap hours are also "
               "clean hours, so price-only dispatch incidentally abates carbon. Rank "
               "(not Pearson) because dispatch is about ordering and it resists price spikes.")
g2.metric("Alignment in used hours", "n/a" if np.isnan(align["spearman_active"])
          else f"{align['spearman_active']:+.2f}",
          help="Spearman restricted to the intervals the battery actually charges/discharges "
               "-- the alignment it experiences. Tails matter more than the whole distribution.")

has_ts = ts_choice != "(none / set interval manually)"

# MOER bifurcates (~0 during renewable curtailment vs high on fossil margin), so a raw
# scatter just shows horizontal bands. These three views instead show whether
# low/negative prices coincide with MOER~0.
low_thresh = 0.1 * np.nanmax(carbon)
low_moer = carbon <= low_thresh
t_dens, t_bar, t_dist, t_scat, t_high = st.tabs(
    ["2D density", "Curtailment by price bucket", "MOER by price sign", "Scatter",
     "High-MOER regime"])

with t_scat:
    b0 = comp.baseline
    is_c, is_d = b0.charge_mw > 1e-6, b0.discharge_mw > 1e-6
    is_i = ~(is_c | is_d)
    sc = go.Figure()
    sc.add_trace(go.Scatter(x=price[is_i], y=carbon[is_i], mode="markers", name="Idle",
                            marker=dict(size=4, color="#cccccc", opacity=0.25)))
    sc.add_trace(go.Scatter(x=price[is_c], y=carbon[is_c], mode="markers",
                            name="A charges (cheap hrs)",
                            marker=dict(size=7, color="#aadee8", opacity=0.45)))
    sc.add_trace(go.Scatter(x=price[is_d], y=carbon[is_d], mode="markers",
                            name="A discharges (expensive hrs)",
                            marker=dict(size=7, color="#83c341", opacity=0.45)))
    sc.update_layout(height=400, plot_bgcolor=BG, paper_bgcolor=BG,
                     xaxis_title="Price ($/MWh)", yaxis_title=f"MOER ({carbon_units})",
                     legend=dict(orientation="h", y=-0.2), margin=dict(t=30, b=40),
                     title=f"Colored by what Model A does · Spearman {align['spearman']:+.2f}")
    st.plotly_chart(sc, width="stretch")
    st.caption("Each interval, colored by what Model A (price-only) does. Strong alignment puts "
               "charge points (cheap) low on the MOER axis and discharge points (expensive) high.")

with t_bar:
    edges = [-np.inf, 0, 20, 40, 60, 80, 100, np.inf]
    labels = ["<0", "0-20", "20-40", "40-60", "60-80", "80-100", ">100"]
    buckets = pd.cut(price, bins=edges, labels=labels)
    grp = pd.Series(low_moer).groupby(buckets, observed=False)
    pct = (grp.mean() * 100).reindex(labels)
    counts = grp.size().reindex(labels)
    fb = go.Figure(go.Bar(x=labels, y=pct.values, marker_color=CARBON_COLOR,
                          customdata=counts.values,
                          hovertemplate="%{x} $/MWh<br>%{y:.0f}% curtailment"
                                        "<br>%{customdata} intervals<extra></extra>"))
    fb.update_layout(height=380, plot_bgcolor=BG, paper_bgcolor=BG, margin=dict(t=30, b=40),
                     xaxis_title="Price bucket ($/MWh)",
                     yaxis_title=f"% of intervals with MOER <= {low_thresh:,.0f} {carbon_units}")
    st.plotly_chart(fb, width="stretch")
    st.caption(f"Share of intervals in each price bucket where MOER is near zero "
               f"(<= {low_thresh:,.0f} {carbon_units}, ~10% of max = renewable curtailment). "
               "A tall left bar means cheap/negative prices are when the grid is cleanest.")

with t_dens:
    ybin = float(np.nanmax(carbon)) / 25 if np.nanmax(carbon) > 0 else 50.0
    fd = go.Figure(go.Histogram2d(x=price, y=carbon, colorscale="Blues",
                                  xbins=dict(start=-20, end=100, size=5),
                                  ybins=dict(size=ybin), colorbar=dict(title="intervals")))
    fd.update_layout(height=400, plot_bgcolor=BG, paper_bgcolor=BG, margin=dict(t=30, b=40),
                     xaxis=dict(title="Price ($/MWh)", range=[-20, 100]),
                     yaxis_title=f"MOER ({carbon_units})")
    st.plotly_chart(fd, width="stretch")
    st.caption("Density of intervals in the -\\$20 to \\$100 window (where most hours sit), "
               "5 \\$/MWh bins. Look for a dense cell at low/negative price + near-zero MOER "
               "(curtailment).")

with t_dist:
    neg_mask = price < 0
    cmax = float(np.nanmax(carbon))
    xb = dict(start=0.0, end=(np.floor(cmax / 20) + 1) * 20, size=20)  # aligned 20-unit bins
    fh = go.Figure()
    fh.add_trace(go.Histogram(x=carbon[neg_mask], histnorm="probability", name="price < 0",
                              marker_color=CARBON_COLOR, opacity=0.6, xbins=xb))
    fh.add_trace(go.Histogram(x=carbon[~neg_mask], histnorm="probability", name="price >= 0",
                              marker_color=BASELINE_COLOR, opacity=0.6, xbins=xb))
    fh.update_layout(height=400, plot_bgcolor=BG, paper_bgcolor=BG, barmode="overlay",
                     margin=dict(t=30, b=40), legend=dict(orientation="h", y=1.05),
                     xaxis_title=f"MOER ({carbon_units})", yaxis_title="probability")
    st.plotly_chart(fh, width="stretch")
    st.caption("MOER distribution split by price sign (aligned 20-unit bins). A tall near-zero "
               "spike for the **price < 0** series means negative prices coincide with "
               "curtailment (MOER~0).")

with t_high:
    hi_thresh = 0.5 * np.nanmax(carbon)
    hi = carbon >= hi_thresh
    if int(hi.sum()) > 3:
        a_hi = signal_alignment(price[hi], carbon[hi])
        sh = go.Figure(go.Scatter(x=price[hi], y=carbon[hi], mode="markers",
                                  marker=dict(size=6, color=MOER_COLOR, opacity=0.4)))
        sh.update_layout(height=400, plot_bgcolor=BG, paper_bgcolor=BG, margin=dict(t=40, b=40),
                         xaxis_title="Price ($/MWh)", yaxis_title=f"MOER ({carbon_units})",
                         title=f"MOER >= {hi_thresh:,.0f} {carbon_units} only · "
                               f"Spearman {a_hi['spearman']:+.2f}, Pearson {a_hi['pearson']:+.2f}")
        st.plotly_chart(sh, width="stretch")
        st.caption("Within the fossil-margin regime (high MOER) only, does price track MOER? A "
                   "positive correlation here means that when the grid has fossil on the margin, "
                   "more expensive hours are also dirtier, in that case price-following would "
                   "avoid carbon.")
    else:
        st.caption(f"Not enough intervals with MOER >= {hi_thresh:,.0f} {carbon_units} "
                   "to show a correlation.")

# --------------------------------------------------------------------------- #
# Abatement cost curve — placeholder here; filled at the END of the script so the
# rest of the page paints first (the curve runs one extra solve per point).
# --------------------------------------------------------------------------- #
st.subheader("4. Revenue vs CO2 tradeoff")
curve_slot = st.empty()
if compute_curve:
    curve_slot.caption("Computing the abatement curve — the rest of the page loads first…")
else:
    curve_slot.info("Abatement curve disabled — enable **Compute abatement curve** in the sidebar.")

# --------------------------------------------------------------------------- #
# Carpet plots (day x hour-of-day)
# --------------------------------------------------------------------------- #
st.subheader("5. Dispatch & signal patterns")
if not has_ts:
    st.caption("Upload data with a timestamp column to see day x hour-of-day carpet plots.")
else:
    tsi = ts  # already parsed robustly above (reuse; don't re-parse and risk a crash)

    def carpet(values):
        d = pd.DataFrame({"date": tsi.dt.normalize(), "hour": tsi.dt.hour, "v": values})
        return d.pivot_table(index="hour", columns="date", values="v", aggfunc="mean")

    base_net = comp.baseline.discharge_mw - comp.baseline.charge_mw
    ca_net = comp.carbon_aware.discharge_mw - comp.carbon_aware.charge_mw
    cmax_net = comp.carbon_max.discharge_mw - comp.carbon_max.charge_mw

    disp_opts = [A_BRIEF, B_BRIEF, CMAX_BRIEF, "Difference (B - A)"]
    disp_map = {A_BRIEF: base_net, B_BRIEF: ca_net, CMAX_BRIEF: cmax_net,
                "Difference (B - A)": ca_net - base_net}
    if actual_net is not None:
        disp_opts.insert(3, ACT_BRIEF)
        disp_map[ACT_BRIEF] = actual_net
        disp_opts += ["Difference (Actual - A)", "Difference (Actual - B)"]
        disp_map["Difference (Actual - A)"] = actual_net - base_net
        disp_map["Difference (Actual - B)"] = actual_net - ca_net

    cc1, cc2 = st.columns(2)
    disp_choice = cc1.radio("Dispatch (MW, + = discharge)", disp_opts)
    sig_choice = cc2.radio("Signal", [A_SIG, B_SIG, C_SIG])

    disp_vals = disp_map[disp_choice]
    sig_vals = {A_SIG: price, B_SIG: comp.carbon_aware_signal, C_SIG: carbon}[sig_choice]

    # WattTime palette; MOER uses a deliberate clean->dirty green/yellow/coal ramp.
    disp_scale = [[0.0, "#aadee8"], [0.5, ZERO_COLOR], [1.0, "#83c341"]]  # charge->discharge
    moer_scale = [[0.0, "#83c341"], [0.5, "#fbd20b"], [1.0, "#cc4125"]]   # clean->dirty
    money_scale = [[0.0, "#dbf0f2"], [1.0, "#434343"]]

    disp_pv, sig_pv = carpet(disp_vals), carpet(sig_vals)

    # Dispatch: normalize each side to its own max so charging isn't washed out when
    # charge/discharge ranges are asymmetric (e.g. 17 vs 58 MW). Ticks/hover show true MW.
    z = disp_pv.values
    pos_max = np.nanmax(z) if np.nanmax(z) > 0 else 1e-9
    neg_max = -np.nanmin(z) if np.nanmin(z) < 0 else 1e-9
    znorm = np.where(z >= 0, z / pos_max, z / neg_max)
    disp_ticktext = [f"{-neg_max:,.0f}", f"{-neg_max / 2:,.0f}", "0",
                     f"{pos_max / 2:,.0f}", f"{pos_max:,.0f}"]

    # Signal: money color scale spans the 5th-95th percentile (same rule for both A
    # and B) so a few extreme prices don't drown the rest; MOER keeps its full ramp.
    if sig_choice == C_SIG:
        sig_scale, sig_unit, sig_zmin, sig_zmax = moer_scale, carbon_units, None, None
    else:
        sig_scale, sig_unit = money_scale, "$/MWh"
        sig_zmin = float(np.nanpercentile(sig_pv.values, 5))
        sig_zmax = float(np.nanpercentile(sig_pv.values, 95))

    cfig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                         subplot_titles=(f"Dispatch — {disp_choice}", f"Signal — {sig_choice}"))
    cfig.add_trace(go.Heatmap(z=znorm, x=disp_pv.columns.astype(str), y=disp_pv.index,
                              colorscale=disp_scale, zmin=-1, zmax=1, customdata=z,
                              colorbar=dict(title="MW", len=0.45, y=0.79,
                                            tickvals=[-1, -0.5, 0, 0.5, 1], ticktext=disp_ticktext),
                              hovertemplate="hour %{y}<br>%{x}<br>%{customdata:.1f} MW<extra></extra>"),
                   row=1, col=1)
    cfig.add_trace(go.Heatmap(z=sig_pv.values, x=sig_pv.columns.astype(str), y=sig_pv.index,
                              colorscale=sig_scale, zmin=sig_zmin, zmax=sig_zmax,
                              colorbar=dict(title=sig_unit, len=0.45, y=0.21)), row=2, col=1)
    cfig.update_layout(height=620, plot_bgcolor=BG, paper_bgcolor=BG, margin=dict(t=40, b=30))
    cfig.update_yaxes(title="hour of day", autorange="reversed")
    st.plotly_chart(cfig, width="stretch")
    st.caption("Discharge (green) vs charge (blue). The dispatch scale is normalized **per side**, "
               "so charge and discharge color intensity are not on the same MW scale — read the "
               "colorbar/hover for magnitudes. Price color spans the 5th-95th percentile so extreme "
               "intervals don't wash out the rest. Sub-hourly data is averaged into hourly cells.")

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

# --------------------------------------------------------------------------- #
# Deferred: compute the abatement curve LAST and fill the section-4 placeholder,
# so everything above renders first. Skipped entirely if the checkbox is off.
# --------------------------------------------------------------------------- #
if compute_curve:
    with curve_slot.container():
        _cur_prog = st.progress(0.0, text="Computing abatement curve…")
        fr = abatement_frontier(
            price, carbon, dt_hours,
            power_mw=power_mw, power_discharge_mw=power_discharge_mw,
            energy_mwh=energy_mwh, rte=rte_pct / 100.0, carbon_units=carbon_units,
            n_points=n_points, soc_init=soc_init, soc_min=soc_min, cycle_cost=cycle_cost,
            terminal_soc=terminal_soc, mip_gap=MIP_GAP, time_limit=SOLVE_TIME_LIMIT,
            solve_fn=cached_solve,
            _progress=lambda f, m: _cur_prog.progress(min(f, 1.0), text=m))
        _cur_prog.empty()
        if not (abs(fr.baseline_revenue) > 1e-9 and abs(fr.max_avoided_tonnes) > 1e-9):
            st.caption("Tradeoff curve unavailable: this scenario has ~zero max revenue or ~zero "
                       "avoidable CO2, so the percentages are undefined.")
        else:
            # x = REALIZED abatement cost = revenue foregone / tonnes abated (same as the
            # headline metric). x=0 is the Model A (price-only) endpoint.
            with np.errstate(divide="ignore", invalid="ignore"):
                realized_cost = np.where(fr.tonnes_abated > 1e-9,
                                         fr.revenue_foregone / fr.tonnes_abated, 0.0)
            y_rev = 100 * fr.revenue / fr.baseline_revenue
            y_co2 = 100 * fr.avoided_tonnes / fr.max_avoided_tonnes
            order = np.argsort(realized_cost)
            xr, y_rev, y_co2 = realized_cost[order], y_rev[order], y_co2[order]

            fig_mac = make_subplots(specs=[[{"secondary_y": True}]])
            fig_mac.add_trace(go.Scatter(x=xr, y=y_rev, name="% of max revenue",
                                         mode="lines+markers", line=dict(color=BASELINE_COLOR),
                                         hovertemplate="revenue %{y:.1f}%<br>$%{x:.1f}/t<extra></extra>"),
                              secondary_y=False)
            fig_mac.add_trace(go.Scatter(x=xr, y=y_co2, name="% of optimal CO2",
                                         mode="lines+markers", line=dict(color=CARBON_COLOR),
                                         hovertemplate="CO2 %{y:.1f}%<br>$%{x:.1f}/t<extra></extra>"),
                              secondary_y=True)
            fig_mac.add_vline(x=0, line=dict(color=BASELINE_COLOR, dash="dot"),
                              annotation_text="A: LMP ($0/t)", annotation_position="top left")
            cur = comp.abatement_cost_per_tonne
            if not np.isnan(cur):
                fig_mac.add_vline(x=cur, line=dict(color=PRICE_COLOR, dash="dot"),
                                  annotation_text=f"B: LMP+CO2 (${cur:,.0f}/t)", annotation_position="top")

            allv = np.concatenate([y_rev, y_co2])
            lo = min(0.0, np.floor(np.nanmin(allv) / 25) * 25)
            hi = max(100.0, np.ceil(np.nanmax(allv) / 25) * 25)
            fig_mac.update_xaxes(title_text="Realized abatement cost ($/tonne CO2)")
            fig_mac.update_yaxes(title_text="% of max revenue", secondary_y=False, range=[lo, hi],
                                 dtick=25, color=BASELINE_COLOR,
                                 tickfont=dict(color=BASELINE_COLOR), title_font=dict(color=BASELINE_COLOR))
            fig_mac.update_yaxes(title_text="% of optimal CO2", secondary_y=True, range=[lo, hi],
                                 dtick=25, showgrid=False, color=CARBON_COLOR,
                                 tickfont=dict(color=CARBON_COLOR), title_font=dict(color=CARBON_COLOR))
            fig_mac.update_layout(height=420, plot_bgcolor=BG, paper_bgcolor=BG,
                                  legend=dict(orientation="h", y=-0.2), margin=dict(t=40, b=40))
            st.plotly_chart(fig_mac, width="stretch")
            st.caption(
                "Realized abatement cost is revenue foregone / tonnes abated — the actual \\$/tonne "
                "(different than the carbon-price input). Dotted lines: A: price optimized (\\$0/t) "
                "and price+CO2 co-optimized operating points."
            )

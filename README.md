# Battery Dispatch Solver

A lightweight, open-source, perfect-foresight battery dispatch tool — a simple
alternative to EPRI [DER-VET / StorageVET](https://www.der-vet.com/) for the
single-service arbitrage case.

You give it battery parameters (MW, MWh, round-trip efficiency) and a per-MWh
signal timeseries; it returns the optimal dispatch over that horizon. Its
distinguishing feature is **two scenarios in one run**, so you can quantify the
carbon-vs-cost tradeoff of a battery:

1. **Baseline** — dispatch against price only (usually LMP, $/MWh).
2. **Carbon-aware** — co-optimize price *and* carbon (usually CO2 MOER, lbs/MWh;
   or a MOER+MBER combined signal prepared upstream).

The headline output is the **realized marginal abatement cost** — dollars of
arbitrage revenue foregone per tonne of CO2 abated. That number, not the carbon
price you dial in, is what tells you whether carbon-aware operation is worth it.

## Quick start

```bash
# With pixi (recommended on Windows):
pixi install
pixi run test        # sanity tests
pixi run app         # launch the Streamlit UI

# Or with a plain venv:
pip install -r requirements.txt
pytest test_solver.py
streamlit run app.py
```

In the app: upload a CSV (or click **Load synthetic demo**), pick the timestamp /
price / carbon columns, set battery parameters and a carbon price in the sidebar,
and click **Run dispatch**.

### Input CSV format

One row per interval, with a price column and a carbon column. A timestamp column
is optional (used to infer the interval length; otherwise set it manually).
`sample_signals_week.csv` is a synthetic hourly week you can upload as a template:

```
timestamp,lmp,moer
2026-01-05 00:00:00,29.14,1041.2
2026-01-05 01:00:00,24.87,1008.6
...
```

## Formulation

MILP solved with HiGHS via `scipy.optimize.milp`. Per interval `t` of length `dt`
hours, with `eta = sqrt(rte)`:

```
variables : charge c_t (MW), discharge d_t (MW), soc_t (MWh)
0 <= c_t, d_t <= power_mw
soc_t = soc_{t-1} + eta*c_t*dt - d_t*dt/eta       (soc_{-1} := soc_init)
soc_min <= soc_t <= energy_mwh
optional  : soc_last = soc_init                    (terminal-SOC lock)
objective : max  sum(signal_t*(d_t - c_t)*dt)  -  cycle_cost*sum(d_t*dt)
```

### Price + carbon co-optimization

Co-optimization is just an **effective signal** fed to the same solver:

```
baseline      : signal_t = price_t
carbon-aware  : signal_t = price_t + lambda * carbon_t
                lambda [$/tonne] = carbon_price / mass_per_tonne
                (2204.62 lb/tonne for lbs/MWh MOER; 1000 for kg/MWh)
```

Discharging displaces grid emissions (`+carbon_t*d_t`); charging causes them
(`-carbon_t*c_t`), so net avoided emissions `= carbon_t*(d_t - c_t)*dt`, symmetric
with the price term. Round-trip losses are respected automatically — the optimizer
only charges when the price/MOER differential pays for the loss.

Because the input carbon price is somewhat arbitrary (it does not map 1:1 to
revenue foregone), the tool reports the **realized** abatement cost from the two
dispatches:

```
abatement_cost = (baseline_revenue - carbon_aware_revenue)
                 / (baseline_emissions - carbon_aware_emissions)
```

### Simultaneous charge+discharge guard

A pure LP will run charge and discharge together on any interval where the
*effective* signal is negative, to burn energy through round-trip losses and get
paid/credited for the net import. The solver adds a binary only on those intervals
(`c_t <= P*z_t`, `d_t <= P*(1-z_t)`), keyed on the scenario's own effective signal.
A nonzero `cycle_cost` also suppresses this independently.

## Module layout

| File | Purpose |
|------|---------|
| `dispatch_core.py` | Solver + accounting. No I/O or Streamlit — importable and testable on its own. Key functions: `solve_dispatch`, `evaluate`, `run_comparison`, `infer_dt_hours`. |
| `app.py` | Streamlit UI. |
| `test_solver.py` | Sanity tests (`pixi run test`, or `python test_solver.py` for a standalone run + performance smoke test). |
| `sample_signals_week.csv` | Synthetic hourly week for the demo/upload. |

`run_comparison` is the seam for a future Pareto sweep — loop it over `lambda`.

## Known simplifications (vs StorageVET)

Perfect foresight; price-taker (dispatch does not move the price); single service
(no ancillary/capacity stacking); symmetric efficiency split (`eta = sqrt(rte)` on
both charge and discharge); throughput penalty only (no SOC-dependent degradation,
calendar aging, or auxiliary load); no separate import/export tariffs.

### Validating against StorageVET

Run one matched scenario through both tools — same MW/MWh/RTE, price signal only
(`carbon_price = 0`), no extra services, terminal SOC locked. Dispatch and total
value should agree closely; any divergence points to a StorageVET default this
model omits.

## Deploying

Free options: [Streamlit Community Cloud](https://share.streamlit.io) or Hugging
Face Spaces (push the repo, point at `app.py`). Before publishing publicly:

- Pin exact versions (`==`) in `requirements.txt` so rebuilds cannot break.
- Keep the input-size cap in `app.py` (`MAX_INTERVALS`) to protect free-tier memory.
- Add caching keyed on (params, file hash) if solves get heavy.

## Roadmap

**This MVP** is the perfect-foresight **ceiling**: dispatch optimally against
historical actuals (real-time LMP + historical MOER). Planned next:

1. **Performance floor** — day-ahead self-schedule / limited lookahead (one plan
   per day, `run_comparison` chained by daily terminal SOC). For markets with a
   day-ahead market (CAISO, NYISO, ERCOT). WEIM/real-time-only nodes get an
   RT-only self-dispatch mode instead.
2. **Realistic middle** — dispatch on a *forecast*, settle on *actuals*, using the
   already-decoupled plan/settle signals (`solve_dispatch` optimizes the forecast;
   `evaluate` scores the actual). Needs paired data: DA-vs-RT LMP, and MOER
   forecast vintages vs actual MOER.
3. **Per-project settlement mode** — DA self-schedule / DA+RT-settled deviations /
   RT-only, selectable per project.

Also deferred: Pareto λ-sweep frontier; WattTime MOER API fetch (per-user auth);
separate MOER/MBER columns; batch/portfolio comparison across projects.

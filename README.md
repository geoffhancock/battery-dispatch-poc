# Battery Dispatch Solver

A lightweight, open-source, perfect-foresight battery dispatch tool — a simple
alternative to EPRI [DER-VET / StorageVET](https://www.der-vet.com/) for the
single-service arbitrage case.

You give it battery parameters (MW, MWh, round-trip efficiency) and a per-MWh
signal timeseries; it returns the optimal dispatch over that horizon. Its
distinguishing feature is **three dispatches in one run**, so you can bracket the
carbon-vs-cost tradeoff of a battery:

| | Optimized against | Answers |
|---|---|---|
| **A — Price Optimized** | price only (usually LMP, $/MWh) | most revenue achievable |
| **B — Price + CO2** | price *and* carbon, co-optimized | the carbon-aware operating point |
| **C — CO2 only** | carbon only | most CO2 avoidable, at any cost |

A and C are the two endpoints; B sits between them. Every dispatch is scored
against *both* real signals, so a carbon-aware run still reports real dollars.

The headline output is the **realized marginal abatement cost** — dollars of
arbitrage revenue foregone per tonne of CO2 abated, B against A. That number, not
the carbon price you dial in, is what tells you whether carbon-aware operation is
worth it.

The carbon signal is usually CO2 MOER (lbs/MWh), but the tool is signal-agnostic:
any $/MWh price column and any mass/MWh carbon column will work, including a
pre-combined MOER+MBER prepared upstream.

## Quick start

```bash
# With pixi (recommended on Windows):
pixi install
pixi run test        # sanity tests
pixi run streamlit run app.py --server.headless true

# Or with a plain venv:
pip install -r requirements.txt
pytest test_solver.py
streamlit run app.py --server.headless true
```

`--server.headless true` skips Streamlit's first-run email prompt, which otherwise
blocks startup when there is no interactive terminal attached. It also binds all
network interfaces, so the app is reachable from other machines on the same
network — not only `localhost`. Drop the flag if you want localhost only and have
a terminal to answer the prompt at, or create a `~/.streamlit/credentials.toml`
containing `[general]` and `email = ""` to silence the prompt permanently.

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

An optional fourth column holding a **metered net-dispatch series** (net MW,
positive = discharge) can be selected in the UI. It is scored with the same
accounting as the modeled scenarios, adding an "Actual Dispatch" column to the
results table and a real-world pattern to the carpet plots.

Timestamps are parsed to a local wall clock: any UTC offset is stripped rather
than converted, so hour-of-day plots read as written. A genuinely-UTC column is
detected and flagged, because its wall clock is not local time. Missing or
non-numeric values are reported with a count and can be linear-interpolated.
Irregular intervals are quantified, and a DST-sized handful is distinguished from
the many-gaps case that usually means missing data.

## Formulation

MILP solved with HiGHS via `scipy.optimize.milp`. Per interval `t` of length `dt`
hours, with `eta = sqrt(rte)`:

```
variables : charge c_t (MW), discharge d_t (MW), soc_t (MWh)
0 <= c_t <= power_mw ; 0 <= d_t <= power_discharge_mw
soc_t = soc_{t-1} + eta*c_t*dt - d_t*dt/eta       (soc_{-1} := soc_init)
soc_min <= soc_t <= energy_mwh
optional  : soc_last = soc_init                    (terminal-SOC lock)
objective : max  sum(signal_t*(d_t - c_t)*dt)  -  cycle_cost*sum(d_t*dt)
```

Charge and discharge power limits can be set independently; leave them equal for a
symmetric battery.

### The three scenarios

All three are the *same solver* fed a different **effective signal**:

```
A: price only   : signal_t = price_t
B: price+carbon : signal_t = price_t + lambda * carbon_t
C: carbon only  : signal_t =            lambda * carbon_t

lambda [$/tonne] = carbon_price / mass_per_tonne
                   (2204.62 lb/tonne for lbs/MWh MOER; 1000 for kg/MWh)
```

Discharging displaces grid emissions (`+carbon_t*d_t`); charging causes them
(`-carbon_t*c_t`), so net avoided emissions `= carbon_t*(d_t - c_t)*dt`, symmetric
with the price term. Round-trip losses are respected automatically — the optimizer
only charges when the price/MOER differential pays for the loss.

Scenario C uses the **dollarized** carbon signal (`lambda * carbon_t`, in $/MWh)
rather than raw tonnes. This matters: a raw-tonnes signal would be
unit-incompatible with the $/MWh `cycle_cost`, and zeroing the cycle cost to
compensate would let C over-cycle and overstate how much CO2 is really avoidable.

Because the input carbon price is somewhat arbitrary (it does not map 1:1 to
revenue foregone), the tool reports the **realized** abatement cost from the two
dispatches:

```
abatement_cost = (baseline_revenue - carbon_aware_revenue)
                 / (baseline_emissions - carbon_aware_emissions)
```

This is an **average** cost over all abatement achieved, so it sits below the
`carbon_price` you dialed in — that input is a *marginal* willingness to pay, and
every tonne cheaper than the marginal one is abated too. The two are not expected
to match.

### Abatement curve

Sweeping `lambda` from 0 (scenario A) upward traces the full tradeoff between
revenue and CO2. `abatement_frontier` runs one solve per sweep point and returns
cumulative tonnes abated, revenue foregone, and the marginal $/tonne between
consecutive points.

Sweep points are packed toward zero, and the range is built from 5th–95th
percentile spans of both signals so that price spikes do not stretch it. The
chart plots **realized** abatement cost on the x-axis — the same quantity as the
headline metric — against percent of maximum revenue and percent of optimal CO2.

### Simultaneous charge+discharge guard

A pure LP will run charge and discharge together to burn energy through round-trip
losses and get paid for the net import — a free dump load. Shrinking both sides
SOC-neutrally (`c -= delta`, `d -= rte*delta`) changes the objective by
`delta*dt*[signal_t*(1 - rte) + cycle_cost*rte]`, so simultaneous dispatch is
strictly suboptimal only above a threshold:

```
T = -cycle_cost * rte / (1 - rte)
```

A nonzero `cycle_cost` therefore does **not** remove the problem on its own — it
only lowers the threshold. At `rte = 0.85` and `cycle_cost = $10/MWh`, T is
-$56.67/MWh, and real negative prices reach past that. Below T the artifact is not
objective-neutral (the LP earns `|signal|*(1 - rte)*P*dt` where the physical
dispatch earns nothing), so no exact LP formulation exists in general.

The solver adds a binary on the affected intervals (`c_t <= P*z_t`,
`d_t <= P*(1-z_t)`), currently keyed on `signal_t < 0`. That is exactly right when
`cycle_cost == 0` and conservative otherwise.

Because the binaries are added per affected interval, a price series with many
negative hours is the expensive case. Scenarios B and C add a non-negative carbon
term, so they usually carry fewer binaries than A — C is typically a pure LP.

### Solver bounds

Large inputs are bounded rather than solved to proven optimality: `mip_gap`
(1% in the UI) and `time_limit` (30 s per solve). Proving optimality is what
explodes — on a week of CAISO data at ~765 binaries it takes minutes, against
well under a second at a 1% gap. A feasible incumbent is accepted when a bound
stops the solve early.

One consequence worth knowing when comparing two runs: with a nonzero gap neither
result is a proven optimum, so a *smaller* problem can report slightly more value
than a larger one that contains it. That is the gap, not an inconsistency.

Measured effect of the 1% gap on the headline metric, across the four test weeks:
average abatement cost moves by at most 2.7% against a tight-gap solve, and under
1% in three of four. Revenue foregone is a small difference between two large
revenues, so it amplifies the gap roughly 30-400x in principle — but the errors on
the two scenarios correlate almost exactly and cancel.

### Long horizons: chunked solving

Solver memory scales with interval count — roughly 10 KB each, dominated by HiGHS
working memory rather than by the binaries. A full year of 5-minute data peaks near
**1.2 GB**, which will not fit a small container. Above `CHUNK_THRESHOLD_INTERVALS`
the app switches to `solve_chunked`, a receding horizon: each chunk is solved over
`chunk + overlap` intervals, only the chunk is kept, and the state of charge carries
forward. The overlap is what makes it nearly free — without it the optimizer drains
the battery at every boundary, having no reason to hold energy it will never sell.

This trades perfect foresight over the horizon for perfect foresight within each
chunk. Measured on CAISO 2025 (105,120 intervals, 10 MW / 40 MWh, 30-day chunks
with 5 days of overlap): peak **319 MB instead of 1,188 MB**, revenue within
**0.004%**, and average abatement cost identical to the cent. It costs about 25%
more wall-clock, which is the trade for fitting in memory.

The terminal-SOC lock belongs to the whole horizon, not to a chunk: the last chunk
starts at whatever SOC it inherited and must still land on the SOC the run began
at. That is what `terminal_soc_value` on `solve_dispatch` is for.

`chunk_intervals` is also the limited-lookahead knob from the roadmap below. Large
chunks approximate perfect foresight; one day with no overlap is a day-ahead
self-schedule, the performance *floor* rather than the ceiling.

## Module layout

| File | Purpose |
|------|---------|
| `dispatch_core.py` | Solver + accounting. No I/O or Streamlit — importable and testable on its own. |
| `app.py` | Streamlit UI. |
| `test_solver.py` | Sanity tests (`pixi run test`, or `python test_solver.py` for a standalone run + performance smoke test). |
| `sample_signals_week.csv` | Synthetic hourly week for the demo/upload. |
| `test_data/` | Real market weeks used by the tests (see below). |

### Test data

`test_data/` holds one week per region (2,016 five-minute intervals, ~77 KB each)
of real RTM LMP, DAM LMP and CO2 MOER, as `timestamp,lmp_rtm,lmp_dam,moer`.

**Either price column is a valid dispatch signal.** Transacting day-ahead only,
real-time only, or a mix of both are all real ways to operate a battery, so pick
whichever column matches the strategy being modeled. DAM clears hourly, so at
5-minute resolution it is a step function — twelve identical intervals per hour,
which means many tied objective coefficients and, on a negative hour, twelve
consecutive intervals entering the guard at once.

The choice is not cosmetic. Over these same weeks, day-ahead-only revenue ranges
from 24% of real-time-only (NYISO, whose RTM week contains a $2,382 scarcity spike
that DAM never sees) to 118% (ERCOT, where the day-ahead market is simply the
better one for a perfect-foresight battery). CAISO's average abatement cost is
$12.76/tonne against RTM and $3.85/tonne against DAM — a 3.3x difference in the
headline metric from the price column alone.

The weeks were picked to contain what synthetic signals do not:

| Fixture | Negative RTM | Zero MOER | Longest negative run |
|---|---|---|---|
| `CAISO_PALMSPRINGS_week.csv` | 38.0% | 43.4% | 11.0 h |
| `ERCOT_EASTTX_week.csv` | 17.1% | 17.3% | 21.6 h |
| `SRP_week.csv` | 36.6% | 35.2% | 11.0 h |
| `NYISO_WEST_week.csv` | 0% | 0% | — (RTM peaks at $2,382) |

Zero MOER is renewable curtailment on the margin — charging then induces no
emissions — and is concentrated in spring midday hours in the solar-heavy regions.
It is signal, not missing data.

These weeks are what makes the simultaneous-charge guard testable. Unguarded, the
LP finds the dump load on all three negative-price weeks and overstates revenue by
1.5–2.3%; NYISO has no negative intervals, so the guard adds no binaries there.

Tests also run against full-year (105,120-interval) versions of the same signals
when the source drive happens to be mapped, and skip cleanly when it is not.

Key functions in `dispatch_core.py`:

| Function | Returns |
|---|---|
| `solve_dispatch` | `DispatchResult` — one MILP solve against one effective signal |
| `solve_chunked` | `DispatchResult` — a long horizon as overlapping chunks; drop-in for `solve_dispatch` |
| `run_comparison` | `Comparison` — scenarios A, B and C plus the realized abatement cost |
| `evaluate` | `Metrics` — scores any dispatch against the real price and carbon signals |
| `evaluate_actual` | `Metrics` for a metered net-MW series, using identical accounting |
| `abatement_frontier` | `Frontier` — the carbon-price sweep |
| `default_carbon_prices` | a sweep range inferred from the signals |
| `signal_alignment` | Spearman / Pearson price-vs-carbon correlation, overall and over active intervals only |
| `parse_timestamps` | `(naive_series, is_utc)` — offsets stripped to local wall clock |
| `infer_dt_hours` | `(dt_hours, n_irregular, max_gap_hours)` |

`run_comparison` and `abatement_frontier` both accept a `solve_fn`, which is the
seam the UI uses to inject a cached solver, and a `_progress` callback.

## Known simplifications (vs StorageVET)

Perfect foresight; price-taker (dispatch does not move the price); single service
(no ancillary/capacity stacking); symmetric efficiency split (`eta = sqrt(rte)` on
both charge and discharge); throughput penalty only (no SOC-dependent degradation,
calendar aging, or auxiliary load); no separate import/export tariffs. Rows are
treated as uniform, in-order steps, so irregular intervals are not re-timed.

### Validating against StorageVET

Run one matched scenario through both tools — same MW/MWh/RTE, price signal only
(`carbon_price = 0`), no extra services, terminal SOC locked. Dispatch and total
value should agree closely; any divergence points to a StorageVET default this
model omits.

## Deploying

Free options: [Streamlit Community Cloud](https://share.streamlit.io) or Hugging
Face Spaces (push the repo, point at `app.py`). Community Cloud deploys from a
private repo and private apps take a viewer email allowlist, which is the simplest
way to share with a named set of people.

The defaults are already sized for a ~1 GB container, from measurement rather than
guesswork:

| Setting | Value | Why |
|---|---|---|
| `MAX_INTERVALS` | 200,000 | ~2 years of 5-minute data. Safe because long horizons are chunked |
| `CHUNK_THRESHOLD_INTERVALS` | 15,000 | Above this, chunk. Below, one solve is cheap |
| `CHUNK_DAYS` / `CHUNK_OVERLAP_DAYS` | 30 / 5 | 319 MB peak for a full 5-minute year |
| `SOLVE_TIME_LIMIT` | 30 s | Chunks solve in well under a second |
| Abatement-curve points | 4–12 | Each point is a full solve across every chunk |
| `cached_solve` cache | `max_entries=24, ttl=1800` | Bounds retained dispatch arrays |
| `server.maxUploadSize` | 20 MB | See below |
| `_SOLVE_LOCK` | one solve at a time | See below |

Two of those are less obvious:

- **The upload limit matters because the row cap is checked too late.** `app.py`
  tests `len(df) > MAX_INTERVALS` only *after* pandas has parsed the upload, so an
  oversized file exhausts memory before the cap can apply. The real defence is
  `server.maxUploadSize` in `.streamlit/config.toml`.
- **Solves are serialized.** Streamlit serves every viewer from one process, so
  concurrent solves would stack their peaks in the same address space. Queueing is
  a far better failure mode than an out-of-memory restart that drops every session.

Also pin exact versions (`==`) in `requirements.txt` so rebuilds cannot break —
already done.

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

Also deferred: targeting a realized abatement cost directly instead of a carbon
price; WattTime MOER API fetch (per-user auth); separate MOER/MBER columns;
batch/portfolio comparison across projects.

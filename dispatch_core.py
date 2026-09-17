"""Perfect-foresight battery dispatch solver.

A lightweight, open-source alternative to EPRI DER-VET / StorageVET for the
simplest useful case: a price-taking battery optimized against a per-MWh signal
under perfect foresight.

The module is deliberately free of any I/O or Streamlit dependency so it can be
imported and unit-tested on its own.

Formulation (MILP, solved with HiGHS via ``scipy.optimize.milp``)
-----------------------------------------------------------------
Per interval ``t`` of length ``dt`` hours, with ``eta = sqrt(rte)``:

    variables : charge c_t (MW), discharge d_t (MW), soc_t (MWh)
    0 <= c_t, d_t <= power_mw
    soc_t = soc_{t-1} + eta*c_t*dt - d_t*dt/eta      (soc_{-1} := soc_init)
    soc_min <= soc_t <= energy_mwh
    optional: soc_{last} = soc_init                  (terminal_soc)
    objective: max  sum(signal_t * (d_t - c_t) * dt)  -  cycle_cost * sum(d_t*dt)

The co-optimization of price and carbon is handled *outside* this function by
building an effective ``signal``:

    baseline      : signal = price
    carbon-aware  : signal = price + carbon_price * carbon      (see run_comparison)

so the same solver serves both scenarios. This is also the natural seam for a
future Pareto sweep (loop run_comparison over carbon_price).

Simultaneous charge+discharge guard
-----------------------------------
A pure LP will run charge and discharge together on any interval where the
*effective* signal is negative, to burn energy through round-trip losses and get
paid/credited for the net import. We add a binary z_t only on those intervals
(c_t <= P*z_t ; d_t <= P*(1-z_t)). A nonzero cycle_cost independently suppresses
this, but the guard is needed when cycle_cost == 0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.stats import spearmanr

# Charge/discharge power below this (MW) is treated as numerical zero.
_TOL = 1e-6


@dataclass
class DispatchResult:
    """Raw solver output. All power arrays are MW, soc is MWh, length n."""

    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    soc_mwh: np.ndarray
    objective: float          # maximized objective value, in signal*MWh units
    n_binaries: int
    solve_s: float
    status: str
    success: bool


@dataclass
class Metrics:
    """Dispatch scored against the real price and carbon signals."""

    revenue: float                # $ (positive = net earnings)
    net_emissions_tonnes: float   # tonnes CO2 (positive = added, negative = avoided)
    mwh_discharged: float
    mwh_charged: float
    equiv_cycles: float
    simultaneous_intervals: int


@dataclass
class Comparison:
    """Baseline vs carbon-aware, plus the realized abatement cost."""

    baseline: DispatchResult
    carbon_aware: DispatchResult
    carbon_max: DispatchResult                 # pure-CO2 (MOER-only) dispatch
    baseline_metrics: Metrics
    carbon_aware_metrics: Metrics
    carbon_max_metrics: Metrics
    tonnes_abated: float                       # baseline_emis - carbon_aware_emis
    revenue_foregone: float                    # baseline_rev - carbon_aware_rev
    abatement_cost_per_tonne: float            # revenue_foregone / tonnes_abated
    max_avoided_tonnes: float                  # CO2 avoided by the carbon-max dispatch
    dt: float
    price: np.ndarray = field(repr=False)
    carbon_tonnes_per_mwh: np.ndarray = field(repr=False)
    baseline_signal: np.ndarray = field(repr=False)
    carbon_aware_signal: np.ndarray = field(repr=False)


def solve_dispatch(
    signal,
    dt,
    power_mw,
    energy_mwh,
    rte,
    soc_init=0.0,
    soc_min=0.0,
    cycle_cost=0.0,
    terminal_soc=True,
    terminal_soc_value=None,
    guard_simultaneous=True,
    power_discharge_mw=None,
    mip_gap=None,
    time_limit=None,
):
    """Optimize battery dispatch against ``signal`` ($/MWh) under perfect foresight.

    Parameters
    ----------
    signal : array-like
        Effective per-interval objective coefficient in $/MWh. Discharging earns
        ``signal_t``; charging pays it. Must be all-finite (no NaN/inf).
    dt : float
        Interval length in hours (e.g. 1.0 hourly, 0.0833... for 5-minute).
    power_mw, energy_mwh, rte : float
        Charge power limit (MW), usable energy (MWh), and round-trip efficiency (0-1).
    power_discharge_mw : float or None
        Discharge power limit (MW). If None, the battery is symmetric and the
        discharge limit equals ``power_mw``.
    soc_init, soc_min : float
        Initial and minimum state of charge in MWh.
    cycle_cost : float
        Penalty in $/MWh of energy *discharged* (throughput / degradation proxy).
    terminal_soc : bool
        If True, force final SOC back to ``soc_init`` (prevents value inflation
        from simply draining the battery over the horizon).
    terminal_soc_value : float or None
        Where the terminal lock points, when it is not ``soc_init``. Only used when
        ``terminal_soc`` is True. Chunked solving needs this: a later chunk starts
        at whatever SOC it inherited, but the lock has to target the SOC the whole
        horizon began at, not the chunk's own.
    guard_simultaneous : bool
        If True, add binaries on negative-signal intervals to forbid simultaneous
        charge and discharge.

    Returns
    -------
    DispatchResult
    """
    signal = np.asarray(signal, dtype=float)
    n = signal.size
    if n == 0:
        raise ValueError("signal is empty")
    if not np.all(np.isfinite(signal)):
        raise ValueError(
            f"signal contains {int(np.sum(~np.isfinite(signal)))} non-finite "
            "value(s) (NaN/inf); clean the input before solving"
        )
    if not (0.0 < rte <= 1.0):
        raise ValueError(f"rte must be in (0, 1], got {rte}")
    P_c = float(power_mw)
    P_d = float(power_mw if power_discharge_mw is None else power_discharge_mw)
    if energy_mwh <= 0 or P_c <= 0 or P_d <= 0:
        raise ValueError("charge/discharge power and energy_mwh must be positive")
    if not (soc_min <= soc_init <= energy_mwh):
        raise ValueError("require soc_min <= soc_init <= energy_mwh")

    eta = np.sqrt(rte)

    # Variable layout: [c_0..c_{n-1}, d_0..d_{n-1}, soc_0..soc_{n-1}, z_0..z_{k-1}]
    ci = np.arange(n)                 # charge columns
    di = n + np.arange(n)             # discharge columns
    si = 2 * n + np.arange(n)         # soc columns

    guarded = np.where(signal < 0)[0] if guard_simultaneous else np.array([], dtype=int)
    k = guarded.size
    zi = 3 * n + np.arange(k)         # binary columns (one per guarded interval)
    n_vars = 3 * n + k

    # ---- Objective (milp minimizes c @ x); maximize signal*(d-c)*dt - cycle*d*dt
    c_obj = np.zeros(n_vars)
    c_obj[ci] = signal * dt                       # minimizing => +cost to charge value
    c_obj[di] = -signal * dt + cycle_cost * dt    # reward discharge, penalize throughput

    # ---- SOC balance equalities: soc_t - soc_{t-1} - eta*dt*c_t + (dt/eta)*d_t = rhs
    rows, cols, data = [], [], []
    # +soc_t
    rows += list(range(n)); cols += list(si); data += [1.0] * n
    # -soc_{t-1} for t >= 1
    rows += list(range(1, n)); cols += list(si[:-1]); data += [-1.0] * (n - 1)
    # -eta*dt*c_t
    rows += list(range(n)); cols += list(ci); data += [-eta * dt] * n
    # +dt/eta*d_t
    rows += list(range(n)); cols += list(di); data += [dt / eta] * n
    A_soc = coo_matrix((data, (rows, cols)), shape=(n, n_vars))
    soc_rhs = np.zeros(n)
    soc_rhs[0] = soc_init             # soc_{-1} := soc_init moved to RHS
    con_soc = LinearConstraint(A_soc, soc_rhs, soc_rhs)

    constraints = [con_soc]

    # ---- Simultaneous-charge/discharge guard (2 rows per guarded interval)
    if k:
        grows, gcols, gdata = [], [], []
        gub = np.empty(2 * k)
        for j, t in enumerate(guarded):
            # c_t - P_c*z_j <= 0
            grows += [j, j]; gcols += [int(ci[t]), int(zi[j])]; gdata += [1.0, -P_c]
            gub[j] = 0.0
            # d_t + P_d*z_j <= P_d
            grows += [k + j, k + j]; gcols += [int(di[t]), int(zi[j])]; gdata += [1.0, P_d]
            gub[k + j] = P_d
        A_g = coo_matrix((gdata, (grows, gcols)), shape=(2 * k, n_vars))
        con_g = LinearConstraint(A_g, -np.inf, gub)
        constraints.append(con_g)

    # ---- Bounds
    lb = np.zeros(n_vars)
    ub = np.empty(n_vars)
    ub[ci] = P_c
    ub[di] = P_d
    lb[si] = soc_min
    ub[si] = energy_mwh
    if terminal_soc:
        target = soc_init if terminal_soc_value is None else float(terminal_soc_value)
        if not (soc_min <= target <= energy_mwh):
            raise ValueError(
                f"terminal SOC target {target} outside [{soc_min}, {energy_mwh}]")
        lb[si[-1]] = target
        ub[si[-1]] = target
    if k:
        lb[zi] = 0.0
        ub[zi] = 1.0
    bounds = Bounds(lb, ub)

    integrality = np.zeros(n_vars)
    if k:
        integrality[zi] = 1

    options = {}
    if mip_gap is not None:
        options["mip_rel_gap"] = mip_gap
    if time_limit is not None:
        options["time_limit"] = time_limit

    t0 = time.perf_counter()
    res = milp(c=c_obj, constraints=constraints, bounds=bounds, integrality=integrality,
               options=options)
    solve_s = time.perf_counter() - t0

    # res.x may hold a feasible (near-optimal) incumbent even when a gap/time limit
    # stopped the solve before proving optimality; accept it if present.
    if res.x is None:
        return DispatchResult(
            charge_mw=np.zeros(n), discharge_mw=np.zeros(n), soc_mwh=np.full(n, np.nan),
            objective=float("nan"), n_binaries=k, solve_s=solve_s,
            status=res.message, success=False,
        )

    x = res.x
    return DispatchResult(
        charge_mw=x[ci],
        discharge_mw=x[di],
        soc_mwh=x[si],
        objective=-res.fun,
        n_binaries=k,
        solve_s=solve_s,
        status=res.message,
        success=True,
    )


def solve_chunked(
    signal,
    dt,
    *,
    chunk_intervals,
    overlap_intervals=0,
    power_mw,
    energy_mwh,
    rte,
    soc_init=0.0,
    soc_min=0.0,
    cycle_cost=0.0,
    terminal_soc=True,
    guard_simultaneous=True,
    power_discharge_mw=None,
    mip_gap=None,
    time_limit=None,
    _progress=None,
):
    """Solve a long horizon as a series of overlapping chunks. Drop-in for
    :func:`solve_dispatch` -- same arguments, same :class:`DispatchResult`.

    Solver memory grows with the number of intervals (roughly 10 KB each, dominated
    by HiGHS working memory rather than by the binaries), so a full year of
    5-minute data peaks near a gigabyte and will not fit a small container. Chunking
    caps the peak at one chunk's worth regardless of horizon length.

    Memory is the only reason to use this -- it is not a speed optimization.
    Repeated timings of identical runs varied by 28-128% on one machine, swamping
    any difference between chunked and unchunked.

    Receding horizon: each chunk is solved over ``chunk_intervals +
    overlap_intervals`` and only the first ``chunk_intervals`` are kept, with the
    state of charge carried into the next chunk. The overlap is what makes this
    nearly free -- without it the optimizer drains the battery at every boundary,
    having no reason to hold energy it will never get to sell.

    This trades perfect foresight over the whole horizon for perfect foresight
    within each chunk. Measured on CAISO 2025 (105,120 five-minute intervals, a
    10 MW / 40 MWh battery) at 30-day chunks with 5 days of overlap: 319 MB peak
    instead of 1,188 MB, revenue within 0.004% of the unchunked answer, and average
    abatement cost identical to the cent. Error grows with storage duration, since
    longer-duration assets couple across boundaries more.

    ``chunk_intervals`` is also the limited-lookahead knob: large chunks approximate
    perfect foresight, while one day with no overlap is a day-ahead self-schedule.
    """
    signal = np.asarray(signal, dtype=float)
    n = signal.size
    if n == 0:
        raise ValueError("signal is empty")
    if chunk_intervals < 1:
        raise ValueError(f"chunk_intervals must be >= 1, got {chunk_intervals}")
    if overlap_intervals < 0:
        raise ValueError(f"overlap_intervals must be >= 0, got {overlap_intervals}")

    # Chunk spans. A short tail is absorbed into the previous chunk rather than
    # solved as a stub, which would force a near-empty horizon to hit terminal SOC.
    spans, s = [], 0
    while s < n:
        e = min(s + chunk_intervals, n)
        if n - e < chunk_intervals // 2:
            e = n
        spans.append((s, e))
        s = e

    charge = np.empty(n)
    discharge = np.empty(n)
    soc_out = np.empty(n)
    soc = float(soc_init)
    n_bin = 0
    solve_s = 0.0
    failures = []

    for i, (s, e) in enumerate(spans):
        end = min(e + overlap_intervals, n)
        last = e == n
        res = solve_dispatch(
            signal[s:end], dt=dt, power_mw=power_mw,
            power_discharge_mw=power_discharge_mw, energy_mwh=energy_mwh, rte=rte,
            soc_init=soc, soc_min=soc_min, cycle_cost=cycle_cost,
            terminal_soc=(terminal_soc and last),
            # The lock belongs to the whole horizon, not to this chunk: the final
            # chunk starts at whatever SOC it inherited but must still land on the
            # SOC the run began at.
            terminal_soc_value=soc_init,
            guard_simultaneous=guard_simultaneous,
            mip_gap=mip_gap, time_limit=time_limit,
        )
        n_bin += res.n_binaries
        solve_s += res.solve_s
        if not res.success:
            failures.append(f"chunk {i + 1}/{len(spans)}: {res.status}")
            break
        keep = e - s
        charge[s:e] = res.charge_mw[:keep]
        discharge[s:e] = res.discharge_mw[:keep]
        soc_out[s:e] = res.soc_mwh[:keep]
        soc = float(res.soc_mwh[keep - 1])
        if _progress is not None:
            _progress((i + 1) / len(spans),
                      f"Chunk {i + 1} of {len(spans)}…")

    if failures:
        return DispatchResult(
            charge_mw=np.zeros(n), discharge_mw=np.zeros(n), soc_mwh=np.full(n, np.nan),
            objective=float("nan"), n_binaries=n_bin, solve_s=solve_s,
            status="; ".join(failures), success=False,
        )

    # Recomputed from the stitched dispatch: each chunk's own objective covers its
    # overlap tail, which is discarded, so the per-chunk values do not sum correctly.
    objective = float(np.sum(signal * (discharge - charge) * dt)
                      - cycle_cost * np.sum(discharge * dt))
    return DispatchResult(
        charge_mw=charge, discharge_mw=discharge, soc_mwh=soc_out,
        objective=objective, n_binaries=n_bin, solve_s=solve_s,
        status=f"chunked: {len(spans)} chunks of {chunk_intervals} "
               f"(+{overlap_intervals} overlap)",
        success=True,
    )


def evaluate(dispatch, price, carbon_tonnes_per_mwh, dt, energy_mwh):
    """Score a dispatch against the real price ($/MWh) and carbon (tonnes/MWh) signals.

    ``carbon_tonnes_per_mwh`` must already be in tonnes CO2 per MWh (unit
    conversion happens in :func:`run_comparison`). Any dispatch can be scored
    against both signals, so a carbon-aware dispatch still reports real dollars.
    """
    price = np.asarray(price, dtype=float)
    carbon = np.asarray(carbon_tonnes_per_mwh, dtype=float)
    c = dispatch.charge_mw
    d = dispatch.discharge_mw

    net_inject = (d - c) * dt                     # MWh delivered to grid per interval
    revenue = float(np.sum(price * net_inject))
    # Charging draws grid energy (emits carbon*c); discharging displaces it
    # (avoids carbon*d). Net emissions caused = carbon*(c - d).
    net_emissions = float(np.sum(carbon * (c - d) * dt))
    mwh_discharged = float(np.sum(d * dt))
    mwh_charged = float(np.sum(c * dt))
    equiv_cycles = mwh_discharged / energy_mwh if energy_mwh else float("nan")
    simultaneous = int(np.sum((c > _TOL) & (d > _TOL)))

    return Metrics(
        revenue=revenue,
        net_emissions_tonnes=net_emissions,
        mwh_discharged=mwh_discharged,
        mwh_charged=mwh_charged,
        equiv_cycles=equiv_cycles,
        simultaneous_intervals=simultaneous,
    )


# lbs / kg per metric tonne, for converting a native MOER to tonnes/MWh.
MASS_PER_TONNE = {"lbs/MWh": 2204.6226, "kg/MWh": 1000.0, "tonnes/MWh": 1.0}


def run_comparison(
    price,
    carbon,
    dt,
    *,
    power_mw,
    energy_mwh,
    rte,
    carbon_price_per_tonne,
    carbon_units="lbs/MWh",
    power_discharge_mw=None,
    soc_init=0.0,
    soc_min=0.0,
    cycle_cost=0.0,
    terminal_soc=True,
    guard_simultaneous=True,
    mip_gap=None,
    time_limit=None,
    solve_fn=None,
    _progress=None,
):
    """Run baseline (price-only) and carbon-aware dispatch, return the comparison.

    ``carbon_price_per_tonne`` ($/tonne CO2) sets how aggressively the carbon-aware
    run trades dollars for emissions. The headline output is the *realized*
    abatement cost = revenue foregone / tonnes abated -- the number that actually
    informs a decision, since the input carbon price is somewhat arbitrary.

    ``mip_gap`` / ``time_limit`` bound the solver (see solve_dispatch) -- important
    at large scale, where the negative-price guard adds many binaries.
    """
    price = np.asarray(price, dtype=float)
    carbon = np.asarray(carbon, dtype=float)
    if price.shape != carbon.shape:
        raise ValueError("price and carbon must have the same length")
    if carbon_units not in MASS_PER_TONNE:
        raise ValueError(f"carbon_units must be one of {list(MASS_PER_TONNE)}")

    carbon_tonnes = carbon / MASS_PER_TONNE[carbon_units]

    common = dict(
        dt=dt, power_mw=power_mw, power_discharge_mw=power_discharge_mw,
        energy_mwh=energy_mwh, rte=rte,
        soc_init=soc_init, soc_min=soc_min, cycle_cost=cycle_cost,
        terminal_soc=terminal_soc, guard_simultaneous=guard_simultaneous,
        mip_gap=mip_gap, time_limit=time_limit,
    )

    baseline_signal = price
    carbon_aware_signal = price + carbon_price_per_tonne * carbon_tonnes

    solve = solve_fn or solve_dispatch   # app injects a cached solver; drives progress

    def _p(frac, msg):
        if _progress is not None:
            _progress(frac, msg)

    _p(0.0, "Solving A: price-only…")
    base = solve(baseline_signal, **common)
    _p(1 / 3, "Solving B: price + CO2…")
    ca = solve(carbon_aware_signal, **common)
    # C = "Model B with LMP set to 0": optimize on the dollarized carbon value only
    # (carbon_price * carbon_tonnes, $/MWh), so the $/MWh cycle_cost is respected and
    # weighed against the cost of carbon exactly as in B. (A raw tonnes signal would
    # be unit-incompatible with cycle_cost; zeroing cycle_cost would over-cycle.)
    _p(2 / 3, "Solving C: CO2-only…")
    cmax = solve(carbon_price_per_tonne * carbon_tonnes, **common)
    _p(1.0, "Scoring scenarios…")

    base_m = evaluate(base, price, carbon_tonnes, dt, energy_mwh)
    ca_m = evaluate(ca, price, carbon_tonnes, dt, energy_mwh)
    cmax_m = evaluate(cmax, price, carbon_tonnes, dt, energy_mwh)

    tonnes_abated = base_m.net_emissions_tonnes - ca_m.net_emissions_tonnes
    revenue_foregone = base_m.revenue - ca_m.revenue
    abatement = (
        revenue_foregone / tonnes_abated if abs(tonnes_abated) > _TOL else float("nan")
    )

    return Comparison(
        baseline=base,
        carbon_aware=ca,
        carbon_max=cmax,
        baseline_metrics=base_m,
        carbon_aware_metrics=ca_m,
        carbon_max_metrics=cmax_m,
        tonnes_abated=tonnes_abated,
        revenue_foregone=revenue_foregone,
        abatement_cost_per_tonne=abatement,
        max_avoided_tonnes=-cmax_m.net_emissions_tonnes,
        dt=dt,
        price=price,
        carbon_tonnes_per_mwh=carbon_tonnes,
        baseline_signal=baseline_signal,
        carbon_aware_signal=carbon_aware_signal,
    )


def evaluate_actual(net_mw, price, carbon, dt, energy_mwh, carbon_units="lbs/MWh"):
    """Score a metered net-dispatch series (+ = discharge, - = charge).

    Reconstructs charge/discharge from the net (assuming no simultaneous
    charge+discharge, which real batteries don't do) and scores it with the same
    accounting as the model, so metered revenue/CO2 are directly comparable to the
    modeled scenarios.
    """
    net = np.asarray(net_mw, dtype=float)
    if carbon_units not in MASS_PER_TONNE:
        raise ValueError(f"carbon_units must be one of {list(MASS_PER_TONNE)}")
    carbon_tonnes = np.asarray(carbon, dtype=float) / MASS_PER_TONNE[carbon_units]
    disp = DispatchResult(
        charge_mw=np.maximum(-net, 0.0),
        discharge_mw=np.maximum(net, 0.0),
        soc_mwh=np.full(net.size, np.nan),
        objective=float("nan"), n_binaries=0, solve_s=0.0,
        status="metered", success=True,
    )
    return evaluate(disp, price, carbon_tonnes, dt, energy_mwh)


@dataclass
class Frontier:
    """Marginal abatement cost curve: price-only -> carbon-max, swept over carbon price."""

    carbon_prices: np.ndarray          # $/tonne swept (== marginal abatement cost)
    tonnes_abated: np.ndarray          # cumulative, vs the price-only baseline
    revenue_foregone: np.ndarray       # cumulative $, vs the price-only baseline
    marginal_cost: np.ndarray          # $/tonne between consecutive points (len n-1)
    revenue: np.ndarray                # absolute $ at each sweep point
    avoided_tonnes: np.ndarray         # absolute CO2 avoided (vs idle) at each point
    baseline_revenue: float            # revenue at carbon_price=0 (the max revenue)
    baseline_avoided_tonnes: float     # CO2 avoided by price-only dispatch (vs idle)
    max_avoided_tonnes: float          # CO2 avoided by the carbon-max dispatch
    capture_fraction: float            # baseline_avoided / max_avoided


def default_carbon_prices(price, carbon_tonnes_per_mwh, n=10, headroom=8.0, skew=2.0):
    """A carbon-price sweep wide enough to push dispatch from price-optimal to
    carbon-optimal. Uses robust (5th-95th pct) spans so spikes don't blow it up.

    Points are packed toward zero (``skew`` > 1 => quadratic-ish spacing) because
    the low-carbon-price / low-revenue-loss region is where the tradeoff curves
    vary most and where operators actually want to explore.
    """
    price = np.asarray(price, dtype=float)
    carbon = np.asarray(carbon_tonnes_per_mwh, dtype=float)
    p_span = np.percentile(price, 95) - np.percentile(price, 5)
    c_span = np.percentile(carbon, 95) - np.percentile(carbon, 5)
    if c_span <= 0:
        c_span = abs(np.mean(carbon)) or 1.0
    lam_max = headroom * (p_span if p_span > 0 else abs(np.mean(price)) or 1.0) / c_span
    t = np.linspace(0.0, 1.0, n)
    return lam_max * t ** skew


def abatement_frontier(
    price,
    carbon,
    dt,
    *,
    power_mw,
    energy_mwh,
    rte,
    carbon_units="lbs/MWh",
    carbon_prices=None,
    n_points=10,
    power_discharge_mw=None,
    soc_init=0.0,
    soc_min=0.0,
    cycle_cost=0.0,
    terminal_soc=True,
    guard_simultaneous=True,
    mip_gap=None,
    time_limit=None,
    solve_fn=None,
    _progress=None,
):
    """Sweep the carbon price and trace the marginal abatement cost curve.

    Solves the price-only baseline once, then one carbon-aware dispatch per carbon
    price. Everything is measured against the price-only baseline, so ``tonnes_abated``
    and ``revenue_foregone`` are the *incremental* carbon and cost of being
    carbon-aware. ``capture_fraction`` is how much of the maximum avoidable CO2 the
    price-only dispatch already gets for free.
    """
    price = np.asarray(price, dtype=float)
    carbon = np.asarray(carbon, dtype=float)
    carbon_tonnes = carbon / MASS_PER_TONNE[carbon_units]
    if carbon_prices is None:
        carbon_prices = default_carbon_prices(price, carbon_tonnes, n=n_points)
    carbon_prices = np.asarray(carbon_prices, dtype=float)

    common = dict(
        dt=dt, power_mw=power_mw, power_discharge_mw=power_discharge_mw,
        energy_mwh=energy_mwh, rte=rte, soc_init=soc_init, soc_min=soc_min,
        cycle_cost=cycle_cost, terminal_soc=terminal_soc,
        guard_simultaneous=guard_simultaneous, mip_gap=mip_gap, time_limit=time_limit,
    )

    solve = solve_fn or solve_dispatch
    base = solve(price, **common)
    base_m = evaluate(base, price, carbon_tonnes, dt, energy_mwh)

    n_pts = len(carbon_prices)
    if _progress is not None:
        _progress(1 / (n_pts + 1), "curve: baseline solved…")
    abated, foregone, net_emissions, revenue = [], [], [], []
    for i, cp in enumerate(carbon_prices):
        disp = solve(price + cp * carbon_tonnes, **common)
        m = evaluate(disp, price, carbon_tonnes, dt, energy_mwh)
        abated.append(base_m.net_emissions_tonnes - m.net_emissions_tonnes)
        foregone.append(base_m.revenue - m.revenue)
        net_emissions.append(m.net_emissions_tonnes)
        revenue.append(m.revenue)
        if _progress is not None:
            _progress((i + 2) / (n_pts + 1), f"curve point {i + 1}/{n_pts}…")

    abated = np.asarray(abated)
    foregone = np.asarray(foregone)
    revenue = np.asarray(revenue)
    avoided_tonnes = -np.asarray(net_emissions)
    # Marginal $/tonne between consecutive sweep points (sorted by abatement).
    order = np.argsort(abated)
    a_s, f_s = abated[order], foregone[order]
    d_a = np.diff(a_s)
    marginal = np.where(np.abs(d_a) > _TOL, np.diff(f_s) / np.where(d_a == 0, np.nan, d_a), np.nan)

    baseline_avoided = -base_m.net_emissions_tonnes
    max_avoided = -min(net_emissions)          # most CO2 avoided over the sweep (top of curve)
    capture = baseline_avoided / max_avoided if abs(max_avoided) > _TOL else float("nan")

    return Frontier(
        carbon_prices=carbon_prices,
        tonnes_abated=abated,
        revenue_foregone=foregone,
        marginal_cost=marginal,
        revenue=revenue,
        avoided_tonnes=avoided_tonnes,
        baseline_revenue=base_m.revenue,
        baseline_avoided_tonnes=baseline_avoided,
        max_avoided_tonnes=max_avoided,
        capture_fraction=capture,
    )


# How the carbon aggressiveness of a run can be specified. All three resolve to the
# same lambda ($/tonne) that the solver actually consumes.
#   marginal_cost        -- lambda itself. The cost of the LAST (most expensive) tonne
#                           abated; what the optimizer is willing to pay at the margin.
#                           Needs no inversion: lambda IS this number by construction.
#   average_cost         -- revenue foregone / tonnes abated, over ALL abatement. Sits
#                           BELOW marginal cost, because every inframarginal tonne cost
#                           less than the last one. Requires inversion.
#   pct_revenue_foregone -- revenue given up as a % of the price-only maximum. Requires
#                           inversion. Best-behaved of the three: monotone, bounded
#                           below by 0, and free of the 0/0 instability that average
#                           cost has as abatement approaches zero.
TARGET_KINDS = ("marginal_cost", "average_cost", "pct_revenue_foregone")


@dataclass
class TargetSolution:
    """The carbon price that hits a requested average cost or revenue-foregone target."""

    carbon_price: float        # lambda ($/tonne) to feed the solver
    achieved: float            # the target metric actually delivered by that lambda
    target: float              # what was asked for
    target_kind: str
    reached: bool              # False => target lies outside the achievable range
    reachable_min: float       # smallest achievable value of the metric (NaN if none)
    reachable_max: float       # largest achievable value of the metric (NaN if none)
    n_solves: int
    note: str                  # why the target was missed; "" when reached


def target_value(kind, baseline_metrics, metrics):
    """Score one carbon-aware dispatch on a target metric, against the price-only run.

    Returns NaN where the metric is undefined (no abatement, or zero baseline
    revenue) rather than a sentinel, so callers must filter explicitly.
    """
    abated = baseline_metrics.net_emissions_tonnes - metrics.net_emissions_tonnes
    foregone = baseline_metrics.revenue - metrics.revenue
    if kind == "average_cost":
        return foregone / abated if abated > _TOL else float("nan")
    if kind == "pct_revenue_foregone":
        base_rev = baseline_metrics.revenue
        return 100.0 * foregone / base_rev if abs(base_rev) > _TOL else float("nan")
    if kind == "marginal_cost":
        raise ValueError("marginal_cost needs no inversion -- lambda is the value")
    raise ValueError(f"target_kind must be one of {TARGET_KINDS}, got {kind!r}")


def solve_for_target(
    price,
    carbon,
    dt,
    *,
    target,
    target_kind,
    power_mw,
    energy_mwh,
    rte,
    carbon_units="lbs/MWh",
    frontier=None,
    n_bracket=6,
    max_iter=8,
    rel_tol=0.02,
    power_discharge_mw=None,
    soc_init=0.0,
    soc_min=0.0,
    cycle_cost=0.0,
    terminal_soc=True,
    guard_simultaneous=True,
    mip_gap=None,
    time_limit=None,
    solve_fn=None,
    _progress=None,
):
    """Invert a dispatch outcome back to the carbon price (lambda) that produces it.

    ``target_kind`` is ``"average_cost"`` ($/tonne) or ``"pct_revenue_foregone"`` (%).
    Both are non-decreasing in lambda, so the method is a coarse bracketing sweep
    followed by bisection. Non-decreasing is not *strictly* increasing: MILP
    discreteness, the terminal-SOC lock and the guard binaries all produce plateaus,
    and a plateau spanning the target is why this bisects to a tolerance instead of
    root-finding to machine precision.

    A target outside the achievable range does NOT raise and is NOT silently clamped.
    The returned :class:`TargetSolution` carries ``reached=False``, the nearest
    achievable lambda, and the achievable range, so the caller can report honestly.

    ``frontier`` -- pass an already-computed :class:`Frontier` to bracket the target
    on its sweep points instead of running a private one. The two sweeps would
    otherwise land on different lambdas (both are ``lam_max * linspace(0,1,n)**skew``,
    which share only their endpoints for different ``n``), so nearly every solve would
    be repeated. With a frontier supplied the cost drops to at most ``1 + max_iter``
    solves, all of which the cache can serve; without one it is ``1 + n_bracket +
    max_iter``.
    """
    if target_kind not in ("average_cost", "pct_revenue_foregone"):
        raise ValueError(
            f"target_kind must be 'average_cost' or 'pct_revenue_foregone', got "
            f"{target_kind!r} (marginal_cost needs no inversion)"
        )
    price = np.asarray(price, dtype=float)
    carbon = np.asarray(carbon, dtype=float)
    if price.shape != carbon.shape:
        raise ValueError("price and carbon must have the same length")
    if carbon_units not in MASS_PER_TONNE:
        raise ValueError(f"carbon_units must be one of {list(MASS_PER_TONNE)}")
    carbon_tonnes = carbon / MASS_PER_TONNE[carbon_units]

    common = dict(
        dt=dt, power_mw=power_mw, power_discharge_mw=power_discharge_mw,
        energy_mwh=energy_mwh, rte=rte, soc_init=soc_init, soc_min=soc_min,
        cycle_cost=cycle_cost, terminal_soc=terminal_soc,
        guard_simultaneous=guard_simultaneous, mip_gap=mip_gap, time_limit=time_limit,
    )
    solve = solve_fn or solve_dispatch
    total = 1 + (0 if frontier is not None else n_bracket) + max_iter
    n_done = 0

    def _tick(msg):
        if _progress is not None:
            _progress(min(n_done / total, 1.0), msg)

    _tick("Solving the price-only baseline…")
    base_m = evaluate(solve(price, **common), price, carbon_tonnes, dt, energy_mwh)
    n_done += 1

    def value_at(lam):
        nonlocal n_done
        m = evaluate(solve(price + lam * carbon_tonnes, **common),
                     price, carbon_tonnes, dt, energy_mwh)
        n_done += 1
        return target_value(target_kind, base_m, m)

    # ---- Bracket. Reuse an existing sweep when one was handed over; both target
    # metrics are recoverable from the Frontier's own arrays without re-solving.
    if frontier is not None:
        lams = [float(l) for l in frontier.carbon_prices]
        if target_kind == "average_cost":
            with np.errstate(divide="ignore", invalid="ignore"):
                vals = [float(f / a) if a > _TOL else float("nan")
                        for f, a in zip(frontier.revenue_foregone, frontier.tonnes_abated)]
        else:
            br = frontier.baseline_revenue
            vals = [float(100.0 * f / br) if abs(br) > _TOL else float("nan")
                    for f in frontier.revenue_foregone]
        grid = np.asarray(lams, dtype=float)
    else:
        # default_carbon_prices starts at 0, where abatement is zero and average cost
        # is 0/0, so the private sweep is taken from the first nonzero point up.
        grid = default_carbon_prices(price, carbon_tonnes, n=n_bracket + 1)[1:]
        lams = [0.0]
        vals = [0.0 if target_kind == "pct_revenue_foregone" else float("nan")]
        for lam in grid:
            _tick(f"Bracketing: ${lam:,.0f}/tonne…")
            lams.append(float(lam))
            vals.append(value_at(float(lam)))

    finite = [(l, v) for l, v in zip(lams, vals) if np.isfinite(v)]
    if not finite:
        return TargetSolution(
            carbon_price=float(grid[-1]), achieved=float("nan"), target=target,
            target_kind=target_kind, reached=False, reachable_min=float("nan"),
            reachable_max=float("nan"), n_solves=n_done,
            note=("No carbon price produced any abatement, so the target metric is "
                  "undefined everywhere. The carbon signal may be flat, or too weak "
                  "to overcome round-trip losses."),
        )

    v_min = min(v for _, v in finite)
    v_max = max(v for _, v in finite)
    if target > v_max:
        l_at_max = max((l for l, v in finite if v == v_max))
        return TargetSolution(
            carbon_price=l_at_max, achieved=v_max, target=target,
            target_kind=target_kind, reached=False, reachable_min=v_min,
            reachable_max=v_max, n_solves=n_done,
            note=("Target is above everything this battery and signal can reach. The "
                  "most any carbon price achieves is the maximum shown; beyond it the "
                  "dispatch is already fully carbon-optimal."),
        )
    if target < v_min:
        l_at_min = min((l for l, v in finite if v == v_min))
        return TargetSolution(
            carbon_price=l_at_min, achieved=v_min, target=target,
            target_kind=target_kind, reached=False, reachable_min=v_min,
            reachable_max=v_max, n_solves=n_done,
            note=("Target is below the cheapest abatement available. The first tonne "
                  "the battery gives up revenue for already costs more than this."),
        )

    # ---- Bisect the bracketing pair.
    lo, hi = 0.0, float(grid[-1])
    lo_v = float("-inf")
    for (l, v) in finite:
        if v <= target and l >= lo:
            lo, lo_v = l, v
    for (l, v) in sorted(finite):
        if v >= target:
            hi = l
            break

    best_l, best_v = (lo, lo_v) if np.isfinite(lo_v) else (hi, v_max)
    for _ in range(max_iter):
        if np.isfinite(best_v) and abs(best_v - target) <= rel_tol * max(abs(target), _TOL):
            break
        mid = 0.5 * (lo + hi)
        _tick(f"Refining: ${mid:,.0f}/tonne…")
        v = value_at(mid)
        if not np.isfinite(v):
            lo = mid                      # still in the no-abatement region
            continue
        if abs(v - target) < abs(best_v - target) or not np.isfinite(best_v):
            best_l, best_v = mid, v
        if v < target:
            lo = mid
        else:
            hi = mid

    if _progress is not None:
        _progress(1.0, "Target resolved.")
    hit = np.isfinite(best_v) and abs(best_v - target) <= rel_tol * max(abs(target), _TOL)
    return TargetSolution(
        carbon_price=best_l, achieved=best_v, target=target, target_kind=target_kind,
        reached=bool(hit), reachable_min=v_min, reachable_max=v_max, n_solves=n_done,
        note=("" if hit else
              "The achievable values step rather than vary continuously here, so the "
              "closest reachable operating point is shown instead of an exact match."),
    )


def signal_alignment(price, carbon, dispatch=None):
    """Rank/level correlation between the price and carbon signals.

    Spearman (rank) is the headline: dispatch is about *ordering* intervals, and
    rank correlation is robust to price spikes. If ``dispatch`` is given, also report
    the correlation over just the intervals the battery actually uses -- the
    alignment it experiences (tails matter more than the whole distribution).
    """
    price = np.asarray(price, dtype=float)
    carbon = np.asarray(carbon, dtype=float)
    out = {
        "spearman": float(spearmanr(price, carbon).statistic),
        "pearson": float(np.corrcoef(price, carbon)[0, 1]),
        "spearman_active": float("nan"),
    }
    if dispatch is not None:
        active = (dispatch.charge_mw > _TOL) | (dispatch.discharge_mw > _TOL)
        if int(active.sum()) > 2:
            out["spearman_active"] = float(spearmanr(price[active], carbon[active]).statistic)
    return out


def parse_timestamps(series):
    """Parse a timestamp column to a timezone-NAIVE local-wall-clock series.

    Any timezone offset is dropped so plots use the wall clock as written:
      - naive input stays as-is (local),
      - a fixed offset (e.g. -06:00) is stripped to that offset's wall clock (local),
      - mixed offsets (civil local time across a DST change) are stripped per row,
      - a UTC column (zero offset / 'Z') yields the UTC wall clock -> is_utc=True.

    Returns ``(naive_series, is_utc)``. ``is_utc`` flags a genuinely-UTC column,
    whose "wall clock" is UTC time and therefore NOT local for hour-of-day plots.
    Needs no IANA zone / tzdata -- it only strips offsets, never converts zones.
    """
    try:
        ts = pd.to_datetime(series)
    except ValueError:
        # Mixed offsets: pandas won't hold them in one column. Strip the offset text
        # and read the local wall clock as written.
        stripped = series.astype(str).str.replace(r"(Z|[+-]\d{2}:?\d{2})\s*$", "", regex=True)
        return pd.to_datetime(stripped), False
    if ts.dt.tz is None:
        return ts, False
    is_utc = ts.iloc[0].utcoffset() == pd.Timedelta(0)
    return ts.dt.tz_localize(None), bool(is_utc)


def infer_dt_hours(timestamps):
    """Infer interval length (hours) from the median timestamp diff.

    Returns ``(dt_hours, n_irregular, max_gap_hours)`` where ``n_irregular`` counts
    gaps deviating from the median by more than 1% (e.g. DST transitions in
    naive local time produce ~2/year) and ``max_gap_hours`` is the largest gap.
    The solver treats rows as uniform, in-order steps, so a couple of DST-sized
    irregularities are harmless; many usually mean missing data.
    """
    ts = np.asarray(timestamps, dtype="datetime64[ns]")
    if ts.size < 2:
        raise ValueError("need at least two timestamps to infer interval")
    diffs = np.diff(ts).astype("timedelta64[s]").astype(float)  # seconds
    median = float(np.median(diffs))
    if median <= 0:
        raise ValueError("non-increasing timestamps")
    n_irregular = int(np.sum(np.abs(diffs - median) > 0.01 * median))
    max_gap_hours = float(np.max(diffs) / 3600.0)
    return median / 3600.0, n_irregular, max_gap_hours

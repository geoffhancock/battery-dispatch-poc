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
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

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
    baseline_metrics: Metrics
    carbon_aware_metrics: Metrics
    tonnes_abated: float                       # baseline_emis - carbon_aware_emis
    revenue_foregone: float                    # baseline_rev - carbon_aware_rev
    abatement_cost_per_tonne: float            # revenue_foregone / tonnes_abated
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
    guard_simultaneous=True,
):
    """Optimize battery dispatch against ``signal`` ($/MWh) under perfect foresight.

    Parameters
    ----------
    signal : array-like
        Effective per-interval objective coefficient in $/MWh. Discharging earns
        ``signal_t``; charging pays it.
    dt : float
        Interval length in hours (e.g. 1.0 hourly, 0.0833... for 5-minute).
    power_mw, energy_mwh, rte : float
        Battery power (MW), usable energy (MWh), and round-trip efficiency (0-1).
    soc_init, soc_min : float
        Initial and minimum state of charge in MWh.
    cycle_cost : float
        Penalty in $/MWh of energy *discharged* (throughput / degradation proxy).
    terminal_soc : bool
        If True, force final SOC back to ``soc_init`` (prevents value inflation
        from simply draining the battery over the horizon).
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
    if not (0.0 < rte <= 1.0):
        raise ValueError(f"rte must be in (0, 1], got {rte}")
    if energy_mwh <= 0 or power_mw <= 0:
        raise ValueError("power_mw and energy_mwh must be positive")
    if not (soc_min <= soc_init <= energy_mwh):
        raise ValueError("require soc_min <= soc_init <= energy_mwh")

    eta = np.sqrt(rte)
    P = float(power_mw)

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
            # c_t - P*z_j <= 0
            grows += [j, j]; gcols += [int(ci[t]), int(zi[j])]; gdata += [1.0, -P]
            gub[j] = 0.0
            # d_t + P*z_j <= P
            grows += [k + j, k + j]; gcols += [int(di[t]), int(zi[j])]; gdata += [1.0, P]
            gub[k + j] = P
        A_g = coo_matrix((gdata, (grows, gcols)), shape=(2 * k, n_vars))
        con_g = LinearConstraint(A_g, -np.inf, gub)
        constraints.append(con_g)

    # ---- Bounds
    lb = np.zeros(n_vars)
    ub = np.empty(n_vars)
    ub[ci] = P
    ub[di] = P
    lb[si] = soc_min
    ub[si] = energy_mwh
    if terminal_soc:
        lb[si[-1]] = soc_init
        ub[si[-1]] = soc_init
    if k:
        lb[zi] = 0.0
        ub[zi] = 1.0
    bounds = Bounds(lb, ub)

    integrality = np.zeros(n_vars)
    if k:
        integrality[zi] = 1

    t0 = time.perf_counter()
    res = milp(c=c_obj, constraints=constraints, bounds=bounds, integrality=integrality)
    solve_s = time.perf_counter() - t0

    if not res.success or res.x is None:
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
    soc_init=0.0,
    soc_min=0.0,
    cycle_cost=0.0,
    terminal_soc=True,
    guard_simultaneous=True,
):
    """Run baseline (price-only) and carbon-aware dispatch, return the comparison.

    ``carbon_price_per_tonne`` ($/tonne CO2) sets how aggressively the carbon-aware
    run trades dollars for emissions. The headline output is the *realized*
    abatement cost = revenue foregone / tonnes abated -- the number that actually
    informs a decision, since the input carbon price is somewhat arbitrary.
    """
    price = np.asarray(price, dtype=float)
    carbon = np.asarray(carbon, dtype=float)
    if price.shape != carbon.shape:
        raise ValueError("price and carbon must have the same length")
    if carbon_units not in MASS_PER_TONNE:
        raise ValueError(f"carbon_units must be one of {list(MASS_PER_TONNE)}")

    carbon_tonnes = carbon / MASS_PER_TONNE[carbon_units]

    common = dict(
        dt=dt, power_mw=power_mw, energy_mwh=energy_mwh, rte=rte,
        soc_init=soc_init, soc_min=soc_min, cycle_cost=cycle_cost,
        terminal_soc=terminal_soc, guard_simultaneous=guard_simultaneous,
    )

    baseline_signal = price
    carbon_aware_signal = price + carbon_price_per_tonne * carbon_tonnes

    base = solve_dispatch(baseline_signal, **common)
    ca = solve_dispatch(carbon_aware_signal, **common)

    base_m = evaluate(base, price, carbon_tonnes, dt, energy_mwh)
    ca_m = evaluate(ca, price, carbon_tonnes, dt, energy_mwh)

    tonnes_abated = base_m.net_emissions_tonnes - ca_m.net_emissions_tonnes
    revenue_foregone = base_m.revenue - ca_m.revenue
    abatement = (
        revenue_foregone / tonnes_abated if abs(tonnes_abated) > _TOL else float("nan")
    )

    return Comparison(
        baseline=base,
        carbon_aware=ca,
        baseline_metrics=base_m,
        carbon_aware_metrics=ca_m,
        tonnes_abated=tonnes_abated,
        revenue_foregone=revenue_foregone,
        abatement_cost_per_tonne=abatement,
        dt=dt,
        price=price,
        carbon_tonnes_per_mwh=carbon_tonnes,
        baseline_signal=baseline_signal,
        carbon_aware_signal=carbon_aware_signal,
    )


def infer_dt_hours(timestamps):
    """Infer interval length (hours) from the median timestamp diff.

    Returns ``(dt_hours, irregular)`` where ``irregular`` is True if any gap
    deviates from the median by more than 1%.
    """
    ts = np.asarray(timestamps, dtype="datetime64[ns]")
    if ts.size < 2:
        raise ValueError("need at least two timestamps to infer interval")
    diffs = np.diff(ts).astype("timedelta64[s]").astype(float)  # seconds
    median = float(np.median(diffs))
    if median <= 0:
        raise ValueError("non-increasing timestamps")
    irregular = bool(np.any(np.abs(diffs - median) > 0.01 * median))
    return median / 3600.0, irregular

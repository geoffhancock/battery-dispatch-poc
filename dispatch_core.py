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
        lb[si[-1]] = soc_init
        ub[si[-1]] = soc_init
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

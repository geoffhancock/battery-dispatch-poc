"""Sanity tests for dispatch_core.

Run with:  pixi run pytest test_solver.py
Or standalone:  pixi run python test_solver.py
All console output uses ASCII only (Windows cp1252-safe).
"""

import numpy as np
import pytest

from dispatch_core import (
    MASS_PER_TONNE,
    abatement_frontier,
    default_carbon_prices,
    evaluate,
    evaluate_actual,
    infer_dt_hours,
    parse_timestamps,
    run_comparison,
    signal_alignment,
    solve_chunked,
    solve_dispatch,
    solve_for_target,
    target_value,
)
import pandas as pd

# A synthetic day with a clear price shape: cheap at night, expensive in evening.
HOURS = 24
_t = np.arange(HOURS)
PRICE = 30 + 40 * np.sin((_t - 8) / 24 * 2 * np.pi)          # ~ -10 to 70 $/MWh
# Carbon roughly anti-correlated with price is unrealistic; make it correlated
# with load (dirtier in the evening peak) so carbon-aware differs from baseline.
CARBON = 900 + 300 * np.sin((_t - 10) / 24 * 2 * np.pi)       # lbs/MWh

PARAMS = dict(power_mw=10.0, energy_mwh=40.0, rte=0.85, soc_init=0.0, soc_min=0.0)

# A 24-hour horizon admits only a handful of distinct dispatches, so outcome metrics
# move in large steps (on the signal above: 0%, 1.2%, 6.1%, 9.6% revenue foregone and
# nothing between). Targeting tests need a longer, noisier signal where the achievable
# set is fine enough for a target to land inside it.
_WEEK_N = 168
_wt = np.arange(_WEEK_N)
_wrng = np.random.default_rng(0)
WEEK_PRICE = 30 + 40 * np.sin((_wt - 8) / 24 * 2 * np.pi) + _wrng.normal(0, 8, _WEEK_N)
WEEK_CARBON = 900 + 300 * np.sin((_wt - 10) / 24 * 2 * np.pi) + _wrng.normal(0, 60, _WEEK_N)


def _bounds_hold(res, dt, params, tol=1e-4):
    assert np.all(res.charge_mw <= params["power_mw"] + tol)
    assert np.all(res.discharge_mw <= params["power_mw"] + tol)
    assert np.all(res.charge_mw >= -tol)
    assert np.all(res.discharge_mw >= -tol)
    assert np.all(res.soc_mwh <= params["energy_mwh"] + tol)
    assert np.all(res.soc_mwh >= params["soc_min"] - tol)


def test_bounds_and_terminal_soc():
    res = solve_dispatch(PRICE, dt=1.0, terminal_soc=True, **PARAMS)
    assert res.success
    _bounds_hold(res, 1.0, PARAMS)
    assert abs(res.soc_mwh[-1] - PARAMS["soc_init"]) < 1e-4


def test_realized_rte_matches_input():
    """With terminal SOC fixed, energy in * eta^2 should equal energy out."""
    res = solve_dispatch(PRICE, dt=1.0, terminal_soc=True, **PARAMS)
    e_in = res.charge_mw.sum() * 1.0
    e_out = res.discharge_mw.sum() * 1.0
    assert e_in > 0, "battery should cycle at all"
    realized = e_out / e_in
    assert abs(realized - PARAMS["rte"]) < 1e-3, f"realized RTE {realized:.4f}"


def test_negative_price_exploit_and_guard():
    """A sustained deeply-negative price tempts simultaneous charge+discharge."""
    # Small battery, long negative stretch so it fills and would idle-cycle.
    price = np.full(12, -200.0)
    small = dict(power_mw=10.0, energy_mwh=5.0, rte=0.85, soc_init=0.0, soc_min=0.0)

    off = solve_dispatch(price, dt=1.0, guard_simultaneous=False,
                         terminal_soc=False, **small)
    on = solve_dispatch(price, dt=1.0, guard_simultaneous=True,
                        terminal_soc=False, **small)
    off_sim = int(np.sum((off.charge_mw > 1e-6) & (off.discharge_mw > 1e-6)))
    on_sim = int(np.sum((on.charge_mw > 1e-6) & (on.discharge_mw > 1e-6)))
    assert off_sim > 0, "expected the LP to exploit simultaneous charge+discharge"
    assert on_sim == 0, "guard should eliminate simultaneous charge+discharge"


def test_carbon_aware_reduces_emissions_and_costs_money():
    comp = run_comparison(
        PRICE, CARBON, dt=1.0, carbon_price_per_tonne=100.0, carbon_units="lbs/MWh",
        **PARAMS,
    )
    assert comp.baseline.success and comp.carbon_aware.success
    # Carbon-aware run should not emit more than baseline (>= 0 tonnes abated).
    assert comp.tonnes_abated >= -1e-6
    # Baseline maximizes revenue, so carbon-aware cannot earn more.
    assert comp.revenue_foregone >= -1e-6
    # Abatement cost is non-negative (or NaN if nothing was abated).
    if not np.isnan(comp.abatement_cost_per_tonne):
        assert comp.abatement_cost_per_tonne >= -1e-6


def test_carbon_max_is_the_maximum():
    comp = run_comparison(PRICE, CARBON, dt=1.0, carbon_price_per_tonne=100.0, **PARAMS)
    a_av = -comp.baseline_metrics.net_emissions_tonnes
    b_av = -comp.carbon_aware_metrics.net_emissions_tonnes
    c_av = -comp.carbon_max_metrics.net_emissions_tonnes
    # The pure-CO2 dispatch avoids the most; A and B can't exceed it (capture <= 100%).
    assert c_av >= a_av - 1e-6 and c_av >= b_av - 1e-6
    assert abs(c_av - comp.max_avoided_tonnes) < 1e-6
    assert comp.carbon_max.success


def test_zero_carbon_price_matches_baseline():
    comp = run_comparison(
        PRICE, CARBON, dt=1.0, carbon_price_per_tonne=0.0, **PARAMS,
    )
    # With no carbon price the two dispatches are identical.
    np.testing.assert_allclose(
        comp.baseline.discharge_mw, comp.carbon_aware.discharge_mw, atol=1e-6
    )
    assert abs(comp.tonnes_abated) < 1e-6


def test_nonfinite_signal_raises():
    sig = PRICE.copy()
    sig[3] = np.nan
    try:
        solve_dispatch(sig, dt=1.0, **PARAMS)
    except ValueError as e:
        assert "non-finite" in str(e)
    else:
        raise AssertionError("expected ValueError on a NaN signal")


def test_asymmetric_power_limits():
    """A tighter discharge cap must bind while the charge cap stays higher."""
    sym = solve_dispatch(PRICE, dt=1.0, power_mw=10.0, energy_mwh=40.0, rte=0.85,
                         soc_init=0.0, soc_min=0.0, terminal_soc=True)
    asy = solve_dispatch(PRICE, dt=1.0, power_mw=10.0, power_discharge_mw=3.0,
                         energy_mwh=40.0, rte=0.85, soc_init=0.0, soc_min=0.0,
                         terminal_soc=True)
    assert asy.success
    assert sym.discharge_mw.max() > 3.0 + 1e-3      # symmetric would discharge faster
    assert asy.discharge_mw.max() <= 3.0 + 1e-4     # asymmetric discharge cap binds
    assert asy.charge_mw.max() <= 10.0 + 1e-4       # charge still allowed up to 10


def test_abatement_frontier_monotone():
    fr = abatement_frontier(PRICE, CARBON, dt=1.0, carbon_units="lbs/MWh",
                            n_points=8, **PARAMS)
    # More carbon price -> at least as much abatement and at least as much cost.
    assert np.all(np.diff(fr.tonnes_abated) >= -1e-6)
    assert np.all(np.diff(fr.revenue_foregone) >= -1e-6)
    # First sweep point is carbon_price=0 -> no abatement, no cost.
    assert abs(fr.tonnes_abated[0]) < 1e-6 and abs(fr.revenue_foregone[0]) < 1e-6
    # Price-only captures between 0 and 100% of the max avoidable CO2.
    if not np.isnan(fr.capture_fraction):
        assert -1e-6 <= fr.capture_fraction <= 1 + 1e-6
    # Marginal cost is one shorter than the number of points.
    assert fr.marginal_cost.size == fr.carbon_prices.size - 1


def test_evaluate_actual():
    net = np.array([-5.0, 0.0, 5.0, 5.0])       # charge, idle, discharge, discharge
    price = np.array([10.0, 20.0, 50.0, 40.0])
    carbon = np.array([1000.0, 900.0, 1200.0, 1100.0])  # lbs/MWh
    m = evaluate_actual(net, price, carbon, dt=1.0, energy_mwh=20.0, carbon_units="lbs/MWh")
    assert abs(m.revenue - float(np.sum(price * net))) < 1e-9
    exp_emis = float(np.sum((carbon / MASS_PER_TONNE["lbs/MWh"]) * (-net)))
    assert abs(m.net_emissions_tonnes - exp_emis) < 1e-9
    assert abs(m.mwh_discharged - 10.0) < 1e-9   # 5 + 5
    assert m.simultaneous_intervals == 0


def test_signal_alignment():
    a = signal_alignment(PRICE, CARBON)
    assert -1.0 <= a["spearman"] <= 1.0
    assert -1.0 <= a["pearson"] <= 1.0
    # Perfectly monotonic signals -> Spearman == 1.
    perfect = signal_alignment(np.arange(10.0), 2 * np.arange(10.0) + 3)
    assert abs(perfect["spearman"] - 1.0) < 1e-9


def test_default_carbon_prices():
    cps = default_carbon_prices(PRICE, CARBON / MASS_PER_TONNE["lbs/MWh"], n=6)
    assert cps.size == 6 and cps[0] == 0.0 and cps[-1] > 0


def test_parse_timestamps():
    # naive local -> unchanged, not UTC
    ts, u = parse_timestamps(pd.Series(["2025-01-01 00:00", "2025-01-01 00:05"]))
    assert ts.dt.tz is None and u is False and ts.dt.hour.tolist() == [0, 0]

    # fixed offset -06:00 -> local wall clock kept, not UTC
    ts, u = parse_timestamps(pd.Series(["2025-01-01 00:00:00-06:00",
                                         "2025-07-01 12:00:00-06:00"]))
    assert ts.dt.tz is None and u is False and ts.dt.hour.tolist() == [0, 12]

    # mixed offsets (civil time across DST) -> local wall clock as written, not UTC
    ts, u = parse_timestamps(pd.Series(["2025-01-01 00:00:00-06:00",
                                         "2025-07-01 00:00:00-05:00"]))
    assert ts.dt.tz is None and u is False and ts.dt.hour.tolist() == [0, 0]

    # genuine UTC column -> flagged, wall clock is UTC
    ts, u = parse_timestamps(pd.Series(["2025-01-01 06:00:00+00:00", "2025-07-01 05:00:00Z"]))
    assert ts.dt.tz is None and u is True and ts.dt.hour.tolist() == [6, 5]


def test_infer_dt():
    ts = np.arange("2026-01-01T00:00", "2026-01-02T00:00",
                   np.timedelta64(1, "h"), dtype="datetime64[m]")
    dt, n_irr, max_gap = infer_dt_hours(ts)
    assert abs(dt - 1.0) < 1e-9 and n_irr == 0 and abs(max_gap - 1.0) < 1e-9

    ts5 = np.arange("2026-01-01T00:00", "2026-01-01T02:00",
                    np.timedelta64(5, "m"), dtype="datetime64[m]")
    dt5, n5, _ = infer_dt_hours(ts5)
    assert abs(dt5 - 5 / 60) < 1e-9 and n5 == 0

    # A single ~1-hour gap (DST-style) in an otherwise 5-min series = one irregular.
    before = np.arange("2026-03-09T00:00", "2026-03-09T01:00",
                       np.timedelta64(5, "m"), dtype="datetime64[m]")   # ...00:55
    after = np.arange("2026-03-09T02:00", "2026-03-09T03:00",
                      np.timedelta64(5, "m"), dtype="datetime64[m]")    # 02:00 (65-min gap)
    ts_gap = np.concatenate([before, after])
    _, n_gap, mg = infer_dt_hours(ts_gap)
    assert n_gap == 1 and abs(mg - 65 / 60) < 1e-6


def test_unit_conversion():
    assert abs(MASS_PER_TONNE["lbs/MWh"] - 2204.6226) < 1e-3
    assert MASS_PER_TONNE["kg/MWh"] == 1000.0


def test_average_cost_is_below_marginal():
    """The core framing claim: average abatement cost < the marginal cost that set it.

    lambda is the marginal willingness to pay, so every inframarginal tonne costs
    less and drags the average down. If this ever inverts, the two labels in the UI
    are lying.
    """
    for lam in (150.0, 300.0, 1000.0):
        comp = run_comparison(PRICE, CARBON, dt=1.0, carbon_price_per_tonne=lam, **PARAMS)
        assert comp.tonnes_abated > 1e-6, f"no abatement at lambda={lam}"
        assert comp.abatement_cost_per_tonne < lam, (
            f"average {comp.abatement_cost_per_tonne:.1f} should be below marginal {lam}")


def test_solve_for_target_hits_pct_revenue():
    """Revenue-foregone targeting is the well-behaved case: bounded and monotone."""
    sol = solve_for_target(WEEK_PRICE, WEEK_CARBON, dt=1.0, target=5.0,
                           target_kind="pct_revenue_foregone", **PARAMS)
    assert sol.reached, f"should hit 5% revenue foregone; note={sol.note}"
    assert abs(sol.achieved - 5.0) <= 0.02 * 5.0
    # Re-running at the resolved lambda must reproduce the achieved value.
    comp = run_comparison(WEEK_PRICE, WEEK_CARBON, dt=1.0,
                          carbon_price_per_tonne=sol.carbon_price, **PARAMS)
    pct = 100 * comp.revenue_foregone / comp.baseline_metrics.revenue
    assert abs(pct - sol.achieved) < 1e-6


def test_solve_for_target_hits_average_cost():
    sol = solve_for_target(WEEK_PRICE, WEEK_CARBON, dt=1.0, target=40.0,
                           target_kind="average_cost", **PARAMS)
    assert sol.reached, f"should hit $40/tonne average; note={sol.note}"
    comp = run_comparison(WEEK_PRICE, WEEK_CARBON, dt=1.0,
                          carbon_price_per_tonne=sol.carbon_price, **PARAMS)
    assert abs(comp.abatement_cost_per_tonne - sol.achieved) < 1e-6
    # The whole point of the two labels: what you asked for is the average, and the
    # marginal cost the solver needed to get there is higher.
    assert sol.carbon_price > sol.achieved


def test_solve_for_target_unreachable_is_flagged_not_clamped():
    """An impossible target must come back reached=False -- never silently nearest."""
    sol = solve_for_target(WEEK_PRICE, WEEK_CARBON, dt=1.0, target=100.0,
                           target_kind="pct_revenue_foregone", **PARAMS)
    assert not sol.reached
    assert sol.note
    assert sol.achieved < 100.0


def test_frontier_reuse_matches_private_sweep_with_fewer_solves():
    """Handing solve_for_target an existing frontier must not change the answer.

    The two sweeps land on different lambdas (lam_max*linspace(0,1,n)**skew shares
    only its endpoints across different n), so without reuse the curve and the target
    search repeat nearly every solve.
    """
    fr = abatement_frontier(WEEK_PRICE, WEEK_CARBON, dt=1.0, n_points=10, **PARAMS)
    counter = {"n": 0}

    def counting_solve(signal, **kw):
        counter["n"] += 1
        return solve_dispatch(signal, **kw)

    reused = solve_for_target(WEEK_PRICE, WEEK_CARBON, dt=1.0, target=5.0,
                              target_kind="pct_revenue_foregone", frontier=fr,
                              solve_fn=counting_solve, **PARAMS)
    n_reused = counter["n"]
    counter["n"] = 0
    private = solve_for_target(WEEK_PRICE, WEEK_CARBON, dt=1.0, target=5.0,
                               target_kind="pct_revenue_foregone",
                               solve_fn=counting_solve, **PARAMS)
    n_private = counter["n"]

    assert n_reused < n_private, (
        f"reuse should cost fewer solves, got {n_reused} vs {n_private}")
    assert reused.reached and private.reached
    # Both land on the same plateau of the achievable set.
    assert abs(reused.achieved - private.achieved) <= 0.02 * 5.0


def test_coarse_horizon_target_is_reported_not_faked():
    """A short horizon has few distinct dispatches, so most targets are unreachable.

    On the 24-hour signal the only achievable revenue-foregone values are roughly
    0%, 1.2%, 6.1% and 9.6%. Asking for 5% cannot succeed; the contract is that it
    comes back reached=False with the bracketing range, never a fabricated match.
    """
    sol = solve_for_target(PRICE, CARBON, dt=1.0, target=5.0,
                           target_kind="pct_revenue_foregone", **PARAMS)
    assert not sol.reached
    assert np.isfinite(sol.reachable_min) and np.isfinite(sol.reachable_max)
    assert sol.reachable_min <= sol.achieved <= sol.reachable_max
    assert sol.note


def test_solve_for_target_rejects_marginal_kind():
    """marginal_cost needs no inversion; asking for it is a caller error."""
    for bad in ("marginal_cost", "nonsense"):
        try:
            solve_for_target(PRICE, CARBON, dt=1.0, target=10.0, target_kind=bad, **PARAMS)
        except ValueError:
            continue
        raise AssertionError(f"target_kind={bad!r} should have raised")


def test_target_value_undefined_is_nan_not_zero():
    """No abatement => the average cost is 0/0. It must be NaN, not a fake 0."""
    comp = run_comparison(PRICE, CARBON, dt=1.0, carbon_price_per_tonne=0.0, **PARAMS)
    v = target_value("average_cost", comp.baseline_metrics, comp.carbon_aware_metrics)
    assert np.isnan(v)


# --------------------------------------------------------------------------- #
# Real market data
#
# Committed fixtures are one week (2,016 five-minute intervals) per region, chosen
# to contain the cases synthetic sine signals never produce: long negative-price
# blocks and long zero-MOER curtailment blocks. The full-year sources live on a
# shared drive and are exercised only when it happens to be mapped.
# --------------------------------------------------------------------------- #
import pathlib

FIXTURES = pathlib.Path(__file__).parent / "test_data"
SHARED = pathlib.Path(
    r"I:\Shared drives\WattTime-Team\Partnerships\Partners"
    r"\REDACTED\analysis\run_files")
REAL_PARAMS = dict(power_mw=10.0, energy_mwh=40.0, rte=0.85, soc_init=0.0, soc_min=0.0)
DT_5MIN = 5.0 / 60.0
# Match the app, so test timings reflect what a user actually waits for.
BOUNDS = dict(mip_gap=0.01, time_limit=120)


WEEKS = ("CAISO_PALMSPRINGS", "ERCOT_EASTTX", "NYISO_WEST", "SRP")


def _load_week(name, price_col="lmp_rtm"):
    """Load a fixture week. Either price column is a valid dispatch signal: a
    battery can transact day-ahead only, real-time only, or a mix of both."""
    df = pd.read_csv(FIXTURES / f"{name}_week.csv")
    return (df[price_col].to_numpy(float), df["moer"].to_numpy(float),
            df["timestamp"])


def test_real_weeks_solve_and_respect_physics():
    """Every committed week must solve and obey the battery's physical limits.

    Run against BOTH price columns: day-ahead-only and real-time-only are each a
    valid way to operate a battery, so both must be first-class dispatch signals.
    """
    for name in WEEKS:
        for price_col in ("lmp_rtm", "lmp_dam"):
            price, moer, _ = _load_week(name, price_col)
            assert len(price) == 2016, f"{name}: expected a 7-day 5-minute week"
            comp = run_comparison(price, moer, dt=DT_5MIN, carbon_price_per_tonne=50.0,
                                  **REAL_PARAMS, **BOUNDS)
            tag = f"{name}/{price_col}"
            for label, res in (("A", comp.baseline), ("B", comp.carbon_aware),
                               ("C", comp.carbon_max)):
                assert res.success, f"{tag} scenario {label} failed: {res.status}"
                _bounds_hold(res, DT_5MIN, REAL_PARAMS)
            # A is revenue-optimal and C is carbon-optimal, by construction.
            assert comp.baseline_metrics.revenue >= comp.carbon_aware_metrics.revenue - 1e-6
            assert (-comp.carbon_max_metrics.net_emissions_tonnes
                    >= -comp.carbon_aware_metrics.net_emissions_tonnes - 1e-6)


def test_day_ahead_prices_are_hourly_steps():
    """DAM clears hourly, so at 5-minute resolution it is a step function.

    Twelve identical intervals per hour means many tied objective coefficients, and
    a negative DAM hour puts twelve consecutive intervals into the guard at once
    rather than one. Worth pinning: a fixture regenerated from the wrong column
    would silently lose this structure.
    """
    for name in WEEKS:
        dam, _, _ = _load_week(name, "lmp_dam")
        # A 7-day week is 168 hours; allow for consecutive hours clearing equal.
        assert len(np.unique(dam)) <= 200, f"{name}: DAM looks sub-hourly"
        run_lengths = [len(list(g)) for g in
                       np.split(dam, np.flatnonzero(np.diff(dam)) + 1)]
        assert min(run_lengths) >= 12, (
            f"{name}: DAM changes within an hour (shortest hold "
            f"{min(run_lengths)} intervals)")
        assert max(run_lengths) % 12 == 0, f"{name}: DAM holds are not hour-aligned"


def test_real_weeks_guard_actually_suppresses_the_dump_load():
    """The guard is load-bearing on real negative prices, not a theoretical nicety.

    CAISO's fixture week is 38% negative with an 11-hour longest run, well past the
    energy/power saturation point where an unguarded LP starts running charge and
    discharge together to burn energy for the net import. With the guard on there
    must be none; with it off and no cycle cost there must be some.
    """
    price, moer, _ = _load_week("CAISO_PALMSPRINGS")
    assert (price < 0).mean() > 0.3, "fixture should be negative-price heavy"

    guarded = solve_dispatch(price, dt=DT_5MIN, cycle_cost=0.0,
                             guard_simultaneous=True, **REAL_PARAMS, **BOUNDS)
    m_guard = evaluate(guarded, price, moer / MASS_PER_TONNE["lbs/MWh"],
                       DT_5MIN, REAL_PARAMS["energy_mwh"])
    assert m_guard.simultaneous_intervals == 0
    assert guarded.n_binaries > 0

    unguarded = solve_dispatch(price, dt=DT_5MIN, cycle_cost=0.0,
                               guard_simultaneous=False, **REAL_PARAMS, **BOUNDS)
    m_un = evaluate(unguarded, price, moer / MASS_PER_TONNE["lbs/MWh"],
                    DT_5MIN, REAL_PARAMS["energy_mwh"])
    assert unguarded.n_binaries == 0
    assert m_un.simultaneous_intervals > 0, (
        "an unguarded LP should exploit this week; if this fails the artifact is "
        "not reachable here and the guard's scope should be revisited")


def test_real_week_average_below_marginal():
    """The marginal-vs-average framing, on real LMP and MOER rather than sine waves."""
    for name in ("CAISO_PALMSPRINGS", "NYISO_WEST"):
        price, moer, _ = _load_week(name)
        lam = 100.0
        comp = run_comparison(price, moer, dt=DT_5MIN, carbon_price_per_tonne=lam,
                              **REAL_PARAMS, **BOUNDS)
        if comp.tonnes_abated > 1e-6:
            assert comp.abatement_cost_per_tonne < lam, (
                f"{name}: average {comp.abatement_cost_per_tonne:.1f} should be "
                f"below marginal {lam}")


def test_real_week_timestamps_parse_as_uniform_five_minute():
    """Fixtures are local wall clock, so the interval must infer cleanly at 5 min."""
    for name in ("CAISO_PALMSPRINGS", "ERCOT_EASTTX", "NYISO_WEST", "SRP"):
        _, _, ts = _load_week(name)
        parsed, is_utc = parse_timestamps(ts)
        assert not is_utc, f"{name}: fixtures are local, not UTC"
        dt_h, n_irr, max_gap = infer_dt_hours(parsed.values)
        assert abs(dt_h - DT_5MIN) < 1e-9, f"{name}: inferred {dt_h * 60:.2f} min"
        assert n_irr == 0, f"{name}: {n_irr} irregular gaps, max {max_gap:.2f} h"


def test_chunked_matches_whole_closely_on_real_data():
    """Chunked dispatch must track the unchunked optimum and never beat it.

    Chunking trades whole-horizon foresight for per-chunk foresight, so it can only
    lose value. Beating the unchunked answer would mean the chunked path is solving
    a different, easier problem -- a bug, not a win.
    """
    price, moer, _ = _load_week("CAISO_PALMSPRINGS")
    whole = solve_dispatch(price, dt=DT_5MIN, **REAL_PARAMS, **BOUNDS)
    # 2 days per chunk with 8 hours of overlap, over a 7-day week.
    chunked = solve_chunked(price, dt=DT_5MIN, chunk_intervals=576,
                            overlap_intervals=96, **REAL_PARAMS, **BOUNDS)
    assert chunked.success, chunked.status
    _bounds_hold(chunked, DT_5MIN, REAL_PARAMS)
    assert len(chunked.charge_mw) == len(price)

    ct = moer / MASS_PER_TONNE["lbs/MWh"]
    m_w = evaluate(whole, price, ct, DT_5MIN, REAL_PARAMS["energy_mwh"])
    m_c = evaluate(chunked, price, ct, DT_5MIN, REAL_PARAMS["energy_mwh"])
    assert m_c.simultaneous_intervals == 0, "the guard must still bind per chunk"
    # A two-sided band, deliberately: under the shipped mip_gap=0.01 neither result
    # is a proven optimum, so "chunked cannot beat whole" is NOT assertable here --
    # the smaller chunk problems land nearer true optimality and their stitched
    # total can legitimately come out slightly ahead. The one-sided property is
    # tested against a real optimum in the test below.
    assert abs(m_c.revenue - m_w.revenue) <= 0.05 * abs(m_w.revenue), (
        f"chunked ${m_c.revenue:,.0f} differs from unchunked ${m_w.revenue:,.0f} "
        f"by more than 5%; the handoff is losing value")


def test_chunked_cannot_beat_a_proven_optimum():
    """Chunking trades horizon foresight for per-chunk foresight, so it can only
    lose value -- provided both sides are solved to proven optimality.

    Kept to a single day so the tight-gap solve stays quick. Branch-and-bound time
    scales badly with binary count: a full CAISO week (~765 binaries) takes minutes
    to prove, three days (~330) takes over a minute, one day (~110) takes seconds.
    """
    price, _, _ = _load_week("CAISO_PALMSPRINGS")
    price = price[:288]                              # 1 day at 5-minute resolution
    tight = dict(mip_gap=1e-9, time_limit=120)
    whole = solve_dispatch(price, dt=DT_5MIN, **REAL_PARAMS, **tight)
    chunked = solve_chunked(price, dt=DT_5MIN, chunk_intervals=96,
                            overlap_intervals=24, **REAL_PARAMS, **tight)
    assert whole.success and chunked.success
    assert chunked.objective <= whole.objective + 1e-6, (
        f"chunked {chunked.objective:,.2f} beat the proven optimum "
        f"{whole.objective:,.2f}")


def test_chunked_respects_soc_continuity_and_terminal_lock():
    """SOC must carry across boundaries and still return home at the very end."""
    price, _, _ = _load_week("SRP")
    r = solve_chunked(price, dt=DT_5MIN, chunk_intervals=576, overlap_intervals=96,
                      soc_init=10.0, terminal_soc=True,
                      power_mw=10.0, energy_mwh=40.0, rte=0.85, soc_min=0.0,
                      **BOUNDS)
    assert r.success
    assert abs(r.soc_mwh[-1] - 10.0) < 1e-4, "terminal SOC lock not applied at the end"
    # The stitched SOC path must obey the battery's own energy balance everywhere,
    # which is what catches an off-by-one at a chunk seam.
    eta = np.sqrt(0.85)
    expected = 10.0 + np.cumsum(eta * r.charge_mw * DT_5MIN
                                - r.discharge_mw * DT_5MIN / eta)
    assert np.allclose(r.soc_mwh, expected, atol=1e-4), "SOC path breaks at a seam"


def test_chunked_absorbs_a_short_tail():
    """A remainder shorter than half a chunk is absorbed, not solved as a stub."""
    price, _, _ = _load_week("NYISO_WEST")          # 2,016 intervals
    r = solve_chunked(price, dt=DT_5MIN, chunk_intervals=900, overlap_intervals=0,
                      **REAL_PARAMS, **BOUNDS)
    assert r.success
    assert len(r.charge_mw) == len(price)
    # 2,016 = 900 + 900 + 216; the 216 tail is under half a chunk, so 2 chunks.
    assert "2 chunks" in r.status, r.status


def test_chunked_rejects_bad_parameters():
    price, _, _ = _load_week("NYISO_WEST")
    for kw in (dict(chunk_intervals=0), dict(chunk_intervals=100, overlap_intervals=-1)):
        try:
            solve_chunked(price, dt=DT_5MIN, **kw, **REAL_PARAMS)
        except ValueError:
            continue
        raise AssertionError(f"solve_chunked({kw}) should have raised")


def test_full_year_shared_drive_if_available():
    """Full-scale check against a year of 5-minute data, when the drive is mapped.

    Skipped rather than failed off-network: the fixtures above are the portable
    coverage, and this only adds the 105k-interval scale that MAX_INTERVALS and the
    solver's time bound exist for.
    """
    src = SHARED / "NYISO_WEST_BESS_control_signal.csv"
    if not src.exists():
        pytest.skip(f"shared drive not available: {src}")
    df = pd.read_csv(src)
    price = pd.to_numeric(df["lmp_rtm"], errors="coerce").to_numpy(float)
    moer = pd.to_numeric(df["co2_moer_lb_per_mwh"], errors="coerce").to_numpy(float)
    assert len(price) == 105_120, f"expected a full 5-minute year, got {len(price):,}"
    # NYISO is the cheap one to solve (0.2% negative intervals => few binaries).
    res = solve_dispatch(price, dt=DT_5MIN, **REAL_PARAMS, **BOUNDS)
    assert res.success, res.status
    _bounds_hold(res, DT_5MIN, REAL_PARAMS)
    assert res.solve_s < 120 * 1.5, f"solve took {res.solve_s:.1f}s past its bound"


def _perf_smoke():
    """Not a pytest assertion -- prints solve times for large horizons."""
    rng = np.random.default_rng(0)
    for label, n, dt in [("8760 hourly", 8760, 1.0), ("30d 5-min", 8640, 5 / 60)]:
        price = 30 + 40 * np.sin(np.arange(n) / 24 * 2 * np.pi) + rng.normal(0, 5, n)
        carbon = 900 + 200 * np.sin(np.arange(n) / 24 * 2 * np.pi)
        comp = run_comparison(price, carbon, dt=dt, carbon_price_per_tonne=50.0,
                              power_mw=10, energy_mwh=40, rte=0.85)
        print(f"  {label}: baseline {comp.baseline.solve_s:.2f}s "
              f"({comp.baseline.n_binaries} bin), "
              f"carbon-aware {comp.carbon_aware.solve_s:.2f}s "
              f"({comp.carbon_aware.n_binaries} bin), "
              f"abatement ${comp.abatement_cost_per_tonne:.1f}/tonne")


if __name__ == "__main__":
    # Standalone runner (no pytest needed).
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except pytest.skip.Exception as exc:
                print(f"SKIP {name}: {exc}")
                continue
            print(f"PASS {name}")
    print("Performance smoke test:")
    _perf_smoke()
    print("All tests passed.")

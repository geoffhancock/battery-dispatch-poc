"""Sanity tests for dispatch_core.

Run with:  pixi run pytest test_solver.py
Or standalone:  pixi run python test_solver.py
All console output uses ASCII only (Windows cp1252-safe).
"""

import numpy as np

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
    solve_dispatch,
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
            fn()
            print(f"PASS {name}")
    print("Performance smoke test:")
    _perf_smoke()
    print("All tests passed.")

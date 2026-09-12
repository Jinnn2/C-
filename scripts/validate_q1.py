"""Independent physical and accounting audit of exported Q1 dispatch rows."""
import math


def validate(rows, expected_objective=None, require_exclusive=True, tol=1e-5):
    assert len(rows) == 144, 'Expected 144 intervals'
    assert [r['slot'] for r in rows] == list(range(1, 145)), 'Slot order'
    balance = max(abs(r['purchase_kwh'] + r['pv_forecast_kwh'] + r['discharge_kwh']
                      - r['load_kwh'] - r['charge_kwh'] - r['curtailment_kwh']) for r in rows)
    dynamics = max(abs(r['soc_end_kwh'] - r['soc_start_kwh'] - .9*r['charge_kwh']
                       + r['discharge_kwh']/.9) for r in rows)
    continuity = max(abs(a['soc_end_kwh']-b['soc_start_kwh']) for a,b in zip(rows,rows[1:]))
    soc = [rows[0]['soc_start_kwh']] + [r['soc_end_kwh'] for r in rows]
    charge = math.fsum(r['charge_kwh'] for r in rows)
    discharge = math.fsum(r['discharge_kwh'] for r in rows)
    purchase = math.fsum(r['purchase_kwh'] for r in rows)
    curtailed = math.fsum(r['curtailment_kwh'] for r in rows)
    load = math.fsum(r['load_kwh'] for r in rows)
    pv = math.fsum(r['pv_forecast_kwh'] for r in rows)
    cost = math.fsum(r['purchase_kwh']*r['price_yuan_per_kwh'] for r in rows)
    nonnegative = min(r[k] for r in rows for k in ('purchase_kwh','charge_kwh','discharge_kwh','curtailment_kwh'))
    overlap = max(min(r['charge_kwh'],r['discharge_kwh']) for r in rows)
    checks = {
        'finite': all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v,(int,float))),
        'nonnegative': nonnegative >= -tol,
        'power_limit': max(max(r['charge_kwh'],r['discharge_kwh']) for r in rows) <= 5000/6+tol,
        'soc_bounds': min(soc)>=1200-tol and max(soc)<=10800+tol,
        'initial_final_soc': abs(soc[0]-6000)<=tol and abs(soc[-1]-6000)<=tol,
        'supply_balance': balance<=tol, 'storage_dynamics': dynamics<=tol, 'soc_continuity':continuity<=tol,
        'round_trip_energy': abs(discharge-.81*charge)<=tol*144,
        'daily_energy_identity': abs(purchase-(load-pv+.19*charge+curtailed))<=tol*144,
        'cost_columns': all(abs(r['cost_yuan']-r['price_yuan_per_kwh']*r['purchase_kwh'])<=tol for r in rows),
    }
    if require_exclusive:
        checks['charge_discharge_exclusive'] = overlap<=tol
    if expected_objective is not None:
        checks['solver_objective'] = abs(cost-expected_objective)<=tol*144
    failed = [k for k,v in checks.items() if not v]
    assert not failed, f'Failed audit: {failed}'
    return dict(passed=True, checks=checks, max_balance_error_kwh=balance,
                max_soc_dynamics_error_kwh=dynamics, max_simultaneous_kwh=overlap,
                soc_min_kwh=min(soc), soc_max_kwh=max(soc), purchase_kwh=purchase,
                charge_kwh=charge, discharge_kwh=discharge, curtailed_kwh=curtailed,
                cost_yuan=cost, storage_loss_kwh=charge-discharge)

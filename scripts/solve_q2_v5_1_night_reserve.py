"""V5.1 night-reserve experiments.

The V2-B purchase contract is frozen.  V5.1 re-optimizes storage actions with
an expected-emergency objective and a soft SOC target at 06:00, first as a
static target and then as a scenario-derived dynamic target.  The selected
reference actions are finally passed through the V5 delayed_1 hybrid guard.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sqlite3

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from q2_bias import BiasConfig, BiasForecaster
from q2_risk import scenarios, risk_plan
from q2_model import Config
from q2_v4 import ETA, LOWER, LIMIT, UPPER, action, reserves
from solve_q2 import events, execution_rows, summarize, verify_export, workbook
from solve_q2_v2a import read_execution
from solve_q2_v5 import enrich_warmup
from solve_q2_v5_hybrid import hybrid_action, make_january_records
from validate_q2 import audit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/q2_v5_1_night_reserve'
DAWN_SLOT = 36  # 06:00, state after the first 36 ten-minute intervals
TARGET_HORIZON = 48
SOFT_LAMBDA = 5.0


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fixed_purchase_plan(net, purchase, price, initial_soc, target, soft_lambda,
                        final_soc=None):
    """Optimize causal day-ahead storage actions with a fixed q contract."""
    net = np.asarray(net, dtype=float)
    purchase = np.asarray(purchase, dtype=float)
    price = np.asarray(price, dtype=float)
    k, n = net.shape
    if purchase.shape != (n,) or price.shape != (n,):
        raise ValueError('Fixed contract dimensions disagree')
    c, d, s = 0, n, 2 * n
    z, e = 3 * n + 1, 4 * n + 1
    slack = e + k * n
    size = slack + 1
    objective = np.zeros(size)
    objective[e:slack] = np.tile(5 * price / k, k)
    objective[slack] = soft_lambda
    lower = np.zeros(size)
    upper = np.full(size, np.inf)
    upper[c:c + n] = upper[d:d + n] = LIMIT
    lower[s:s + n + 1] = LOWER
    upper[s:s + n + 1] = UPPER
    lower[s] = upper[s] = initial_soc
    upper[z:z + n] = 1.
    integer = np.zeros(size, dtype=int)
    integer[z:z + n] = 1
    rows = n + 2 * n + k * n + 1 + (1 if final_soc is not None else 0)
    matrix = lil_matrix((rows, size))
    lo = np.full(rows, -np.inf)
    hi = np.full(rows, np.inf)
    row = 0
    for t in range(n):
        matrix[row, s + t + 1] = 1.
        matrix[row, s + t] = -1.
        matrix[row, c + t] = -ETA
        matrix[row, d + t] = 1. / ETA
        lo[row] = hi[row] = 0.
        row += 1
    for t in range(n):
        matrix[row, c + t] = 1.; matrix[row, z + t] = -LIMIT
        hi[row] = 0.; row += 1
    for t in range(n):
        matrix[row, d + t] = 1.; matrix[row, z + t] = LIMIT
        hi[row] = LIMIT; row += 1
    for j in range(k):
        for t in range(n):
            matrix[row, c + t] = -1.
            matrix[row, d + t] = 1.
            matrix[row, e + j * n + t] = 1.
            lo[row] = net[j, t] - purchase[t]
            row += 1
    matrix[row, s + DAWN_SLOT] = 1.
    matrix[row, slack] = 1.
    lo[row] = target
    row += 1
    if final_soc is not None:
        matrix[row, s + n] = 1.
        lo[row] = hi[row] = final_soc
    if False:
        x = np.zeros(size); x[s:s+n+1] = initial_soc; x[slack] = max(0., target-initial_soc)
        lhs = matrix.tocsr() @ x
        print('debug', rows, row, np.min(lhs - lo), np.max(lhs - hi), int(np.argmax(lhs - hi)), lhs[int(np.argmax(lhs - hi))], hi[int(np.argmax(lhs - hi))])
    result = milp(objective, integrality=integer, bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(matrix.tocsr(), lo, hi),
                  options={'mip_rel_gap': 1e-8, 'time_limit': 60})
    if not result.success:
        raise RuntimeError(f'V5.1 fixed contract MILP failed: {result.message}')
    charge = result.x[c:c + n].copy()
    discharge = result.x[d:d + n].copy()
    soc = result.x[s:s + n + 1].copy()
    expected_emergency = float(np.mean(np.maximum(net + charge - purchase - discharge, 0.) @ (5 * price)))
    return dict(purchase=purchase.copy(), charge=charge, discharge=discharge, soc=soc,
                target=float(target), target_shortfall=float(result.x[slack]),
                expected_emergency_cost=expected_emergency,
                expected_emergency_kwh=float(np.maximum(net + charge - purchase - discharge, 0.).sum() / k),
                objective=float(result.fun), status=int(result.status))


def dynamic_target(net, purchase, alpha):
    future_deficit = np.maximum(net[:, DAWN_SLOT:DAWN_SLOT + TARGET_HORIZON] - purchase[DAWN_SLOT:DAWN_SLOT + TARGET_HORIZON], 0.)
    reserve = np.quantile(future_deficit.sum(axis=1) / ETA, alpha, method='inverted_cdf')
    return float(np.clip(LOWER + reserve, LOWER, UPPER))


def force_terminal_soc(charge, discharge, states, target):
    """Use the latest feasible slots to close the SOC boundary exactly."""
    remaining = float(target - states[-1])
    for t in range(len(charge) - 1, -1, -1):
        if abs(remaining) <= 1e-8:
            break
        old_c, old_d = float(charge[t]), float(discharge[t])
        if remaining > 0:
            # Cancel discharge first, then add charging power.
            charge[t] = max(0., min(LIMIT, old_c + (remaining - old_d / ETA) / ETA))
            discharge[t] = 0.
        else:
            # Cancel charging first, then add discharging power.
            charge[t] = 0.
            discharge[t] = max(0., min(LIMIT, old_d - ETA**2 * old_c - ETA * remaining))
        achieved = ETA * (charge[t] - old_c) - (discharge[t] - old_d) / ETA
        remaining -= achieved
    if abs(remaining) > 1e-6:
        raise ValueError('Unable to close terminal SOC within power limits')
    return charge, discharge


def candidate_target(kind, parameter, net, purchase):
    return float(parameter) if kind == 'static' else dynamic_target(net, purchase, float(parameter))


def run_reference(records, kind, parameter, soft_lambda, price, final_last=True):
    soc = 6000.; rows = []; plans = []
    for index, record in enumerate(records):
        target = candidate_target(kind, parameter, record['net'], record['purchase'])
        plan = fixed_purchase_plan(record['net'], record['purchase'], price, soc, target,
                                   soft_lambda, 6000. if final_last and index == len(records) - 1 else None)
        plan['forecast'] = record['forecast'].copy()
        plans.append(plan)
        day_rows, soc = execution_rows(record['date'], plan, record['actual'], price, soc, 'validation')
        rows.extend(day_rows)
    return summarize(rows), rows, plans, soc


def run_joint(records, plans, price, value, threshold, final_last=True):
    soc = 6000.; rows = []
    for index, (record, reference) in enumerate(zip(records, plans)):
        levels, _, _ = reserves(record['net'], record['purchase'], price, value)
        charge = np.zeros(144); discharge = np.zeros(144); states = [soc]
        nominal = record['forecast'][:, 0] - record['forecast'][:, 1]
        for t in range(144):
            observed = nominal[t] if t == 0 else nominal[t] + record['actual'][t - 1, 0] - record['actual'][t - 1, 1] - nominal[t - 1]
            residual = float(observed - record['purchase'][t])
            charge[t], discharge[t] = hybrid_action(residual, states[-1], levels[t],
                                                     reference['charge'][t], reference['discharge'][t], threshold)
            states.append(states[-1] + ETA * charge[t] - discharge[t] / ETA)
        if final_last and index == len(records) - 1:
            force_terminal_soc(charge, discharge, np.asarray(states), 6000.)
            states = [soc]
            for t in range(144):
                states.append(states[-1] + ETA * charge[t] - discharge[t] / ETA)
        plan = dict(purchase=record['purchase'], charge=charge, discharge=discharge,
                    soc=np.asarray(states), forecast=record['forecast'])
        day_rows, soc = execution_rows(record['date'], plan, record['actual'], price, soc, 'validation')
        for t, row in enumerate(day_rows):
            row.update(reference_charge_kwh=float(reference['charge'][t]),
                       reference_discharge_kwh=float(reference['discharge'][t]),
                       reserve_kwh=float(levels[t]), issued_charge_kwh=float(charge[t]),
                       issued_discharge_kwh=float(discharge[t]))
        rows.extend(day_rows)
    return summarize(rows), rows, soc


def load_data(v2b):
    with sqlite3.connect(f'file:{(ROOT / "data/processed/microgrid.sqlite").as_posix()}?mode=ro', uri=True) as con:
        flat = con.execute('SELECT date,slot,load_kwh,pv_actual_kwh FROM actuals ORDER BY date,slot').fetchall()
        price = np.array([r[0] for r in con.execute('SELECT price_yuan_per_kwh FROM baseline ORDER BY slot')])
    actual = np.array([[r[2], r[3]] for r in flat]).reshape(365, 144, 2)
    dates = [flat[d * 144][0] for d in range(365)]
    rows = read_execution(ROOT / 'results/q2_v2b/execution.csv')
    return actual, dates, price, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v2b = json.loads((ROOT / 'results/q2_v2b/experiment.json').read_text(encoding='utf-8'))
    config = Config(**v2b['base_config']); bias = BiasConfig(**v2b['bias_config'])
    actual, dates, price, v2b_rows = load_data(v2b)
    warmup = enrich_warmup(v2b_rows[:31 * 144]); contract = v2b_rows[31 * 144:]
    value = float(price[:30].mean() / ETA)

    january = make_january_records(actual, dates, price, config, bias, int(v2b['window_days']))
    candidates = []
    for target in (6000., 7500., 9000., 9750., 10500., 10800.):
        summary, _, _, _ = run_reference(january, 'static', target, SOFT_LAMBDA, price)
        candidates.append(dict(candidate_id=len(candidates), mode='static_soft', parameter=target,
                               target_kwh=target, soft_lambda=SOFT_LAMBDA, **summary))
    for alpha in (.5, .7, .8, .9):
        summary, _, _, _ = run_reference(january, 'dynamic', alpha, SOFT_LAMBDA, price)
        candidates.append(dict(candidate_id=len(candidates), mode='dynamic_soft', parameter=alpha,
                               target_alpha=alpha, soft_lambda=SOFT_LAMBDA, **summary))
    selected = min(candidates, key=lambda x: (x['total_cost_yuan'], x['candidate_id']))
    selected_kind = 'static' if selected['mode'] == 'static_soft' else 'dynamic'
    selected_parameter = selected['parameter']
    _, _, january_plans, _ = run_reference(january, selected_kind, selected_parameter, SOFT_LAMBDA, price)
    joint_candidates = []
    for threshold in (0., 200., 400., 800., 1200.):
        summary, _, _ = run_joint(january, january_plans, price, value, threshold)
        joint_candidates.append(dict(candidate_id=len(joint_candidates), threshold_kwh=threshold, **summary))
    joint_selected = min(joint_candidates, key=lambda x: (x['total_cost_yuan'], x['candidate_id']))

    forecaster = BiasForecaster(config)
    for d in range(31):
        forecaster.forecast(bias); forecaster.observe(actual[d])
    reference_rows = []; joint_rows = []; daily = []; soc_ref = soc_joint = 6000.
    selected_plans = []
    for i, day in enumerate(range(31, 365)):
        forecast, _ = forecaster.forecast(bias)
        ref_rows = contract[i * 144:(i + 1) * 144]
        purchase = np.array([r['purchase_kwh'] for r in ref_rows])
        net = scenarios(forecast, forecaster.residuals, int(v2b['window_days']))
        target = candidate_target(selected_kind, selected_parameter, net, purchase)
        ref = fixed_purchase_plan(net, purchase, price, soc_ref, target, SOFT_LAMBDA,
                                  6000. if day == 364 else None)
        ref['forecast'] = forecast.copy()
        rrows, soc_ref = execution_rows(dates[day], ref, actual[day], price, soc_ref, 'evaluation')
        reference_rows.extend(rrows); selected_plans.append(ref)
        levels, _, _ = reserves(net, purchase, price, value)
        charge = np.zeros(144); discharge = np.zeros(144); states = [soc_joint]
        nominal = forecast[:, 0] - forecast[:, 1]
        for t in range(144):
            observed = nominal[t] if t == 0 else nominal[t] + actual[day, t - 1, 0] - actual[day, t - 1, 1] - nominal[t - 1]
            residual = float(observed - purchase[t])
            charge[t], discharge[t] = hybrid_action(residual, states[-1], levels[t],
                                                     ref['charge'][t], ref['discharge'][t], joint_selected['threshold_kwh'])
            states.append(states[-1] + ETA * charge[t] - discharge[t] / ETA)
        if day == 364:
            force_terminal_soc(charge, discharge, np.asarray(states), 6000.)
            states = [soc_joint]
            for t in range(144):
                states.append(states[-1] + ETA * charge[t] - discharge[t] / ETA)
        jp = dict(purchase=purchase, charge=charge, discharge=discharge, soc=np.asarray(states), forecast=forecast)
        jrows, soc_joint = execution_rows(dates[day], jp, actual[day], price, soc_joint, 'evaluation')
        for t, row in enumerate(jrows):
            row.update(reference_charge_kwh=float(ref['charge'][t]), reference_discharge_kwh=float(ref['discharge'][t]),
                       reserve_kwh=float(levels[t]), issued_charge_kwh=float(charge[t]), issued_discharge_kwh=float(discharge[t]))
        joint_rows.extend(jrows)
        daily.append(dict(date=dates[day], reference_total_cost_yuan=summarize(rrows)['total_cost_yuan'],
                          joint_total_cost_yuan=summarize(jrows)['total_cost_yuan'],
                          v2b_total_cost_yuan=summarize(ref_rows)['total_cost_yuan'], target_kwh=target,
                          reference_soc_06_kwh=float(ref['soc'][DAWN_SLOT]), joint_soc_06_kwh=float(jp['soc'][DAWN_SLOT])))
        forecaster.observe(actual[day])

    ref_summary = summarize(reference_rows); joint_summary = summarize(joint_rows); v2b_summary = summarize(contract)
    validation = dict(reference_audit=audit(warmup + reference_rows, require_boundaries=False, require_fixed_storage=False),
                       joint_audit=audit(warmup + joint_rows, require_boundaries=False, require_fixed_storage=False),
                       purchase_contract_unchanged=all(abs(a['purchase_kwh'] - b['purchase_kwh']) < 1e-8 for a, b in zip(reference_rows, contract)),
                       final_soc_reference=abs(ref_summary['soc_end_kwh'] - 6000.) < 1e-5,
                       final_soc_joint=abs(joint_summary['soc_end_kwh'] - 6000.) < 1e-5)
    write_csv(OUT / 'january_candidates.csv', candidates)
    write_csv(OUT / 'january_joint_candidates.csv', joint_candidates)
    write_csv(OUT / 'execution_reference.csv', warmup + reference_rows)
    write_csv(OUT / 'execution_joint.csv', warmup + joint_rows)
    write_csv(OUT / 'daily_comparison.csv', daily)
    ev = events(joint_rows); write_csv(OUT / 'emergency_events_joint.csv', ev)
    workbook_path = ROOT / 'results/result2_v5_1_night_reserve_joint.xlsx'
    workbook(joint_rows, [dict(date=r['date'], **summarize(joint_rows[i * 144:(i + 1) * 144])) for i, r in enumerate(joint_rows[::144])], ev, workbook_path)
    verify_export(workbook_path, joint_rows, [dict(date=r['date'], **summarize(joint_rows[i * 144:(i + 1) * 144])) for i, r in enumerate(joint_rows[::144])], ev)
    result = dict(version='q2-v5.1-night-reserve', dawn='06:00', dawn_slot=DAWN_SLOT,
                  target_horizon_slots=TARGET_HORIZON, soft_lambda=SOFT_LAMBDA,
                  january_candidates=candidates, selected_candidate=selected,
                  january_joint_candidates=joint_candidates, joint_selected=joint_selected,
                  formal_reference=ref_summary, formal_joint=joint_summary, v2b=v2b_summary,
                  reference_difference_vs_v2b_yuan=ref_summary['total_cost_yuan'] - v2b_summary['total_cost_yuan'],
                  joint_difference_vs_v2b_yuan=joint_summary['total_cost_yuan'] - v2b_summary['total_cost_yuan'],
                  validation=validation)
    (OUT / 'experiment.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 第二问 V5.1：清晨 SOC 夜间储备', '',
             '固定 V2-B 购电合同，在 06:00 对储能 SOC 加入软目标；动态目标由清晨后 48 个时段历史场景净缺口分位数推导。最后将选定参考动作接入 delayed_1 纠偏。', '',
             f'- 选择的夜间目标：{selected["mode"]} / {selected["parameter"]}。',
             f'- 选择的 delayed_1 阈值：{joint_selected["threshold_kwh"]:.6f} kWh。', '',
             '| 方案 | 总费用/元 | 相对 V2-B/元 | 紧急购电/kWh |', '|---|---:|---:|---:|',
             f'| V2-B | {v2b_summary["total_cost_yuan"]:.6f} | 0 | {v2b_summary["emergency_kwh"]:.6f} |',
             f'| V5.1 reference | {ref_summary["total_cost_yuan"]:.6f} | {result["reference_difference_vs_v2b_yuan"]:.6f} | {ref_summary["emergency_kwh"]:.6f} |',
             f'| V5.1 joint delayed_1 | {joint_summary["total_cost_yuan"]:.6f} | {result["joint_difference_vs_v2b_yuan"]:.6f} | {joint_summary["emergency_kwh"]:.6f} |', '',
             '- `january_candidates.csv` 同时记录静态软目标与动态场景目标；只用 1 月选择，正式期锁定。',
             '- `execution_reference.csv` 是夜间目标单独作用；`execution_joint.csv` 是夜间目标与 delayed_1 联合结果。',
             '- 购电合同固定为 V2-B，旧结果未覆盖。', '']
    (ROOT / 'reports/q2_v5_1_night_reserve.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(selected_candidate=selected, joint_selected=joint_selected,
                          formal_reference=ref_summary, formal_joint=joint_summary, v2b=v2b_summary), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

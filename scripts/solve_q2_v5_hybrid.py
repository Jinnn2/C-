"""Q2 V5 hybrid: V2-B price-aware storage reference with delayed DP guard.

V2-B's day-ahead purchase contract and price-arbitrage charge/discharge plan
are retained as the causal reference.  V4 delayed_1 only overrides a reference
action when the one-slot-delayed residual indicates an unsafe direction.
The guard threshold is selected on January and frozen for the formal period.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3

import numpy as np
import scipy

from q2_bias import BiasConfig, BiasForecaster
from q2_model import Config
from q2_risk import risk_plan, scenarios
from q2_v4 import action, reserves, ETA, LOWER, UPPER
from solve_q2 import events, execution_rows, summarize, verify_export, workbook
from solve_q2_v2a import fingerprints, read_execution
from validate_q2 import audit
from validate_q2_v4 import execution_audit
from solve_q2_v5 import enrich_warmup, write_csv


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/q2_v5_hybrid'


def make_january_records(actual, dates, price, config, bias, window):
    f = BiasForecaster(config)
    records = []
    contract_soc = 6000.
    for d in range(31):
        forecast, _ = f.forecast(bias)
        if d >= window:
            net = scenarios(forecast, f.residuals, window)
            p = risk_plan(forecast, net, price, contract_soc, config,
                          6000. if d == 30 else None)
            records.append(dict(date=dates[d], forecast=forecast.copy(), net=net.copy(),
                                actual=actual[d].copy(), purchase=p['purchase'].copy(),
                                charge=p['charge'].copy(), discharge=p['discharge'].copy()))
            contract_soc += .9 * p['charge'].sum() - p['discharge'].sum() / .9
        f.observe(actual[d])
    return records


def hybrid_action(residual, soc, level, ref_charge, ref_discharge, threshold):
    """Keep V2-B price-arbitrage action unless delayed residual is unsafe."""
    dp_charge, dp_discharge = action(residual, soc, level)
    charge, discharge = float(ref_charge), float(ref_discharge)
    if residual > threshold:
        # A delayed deficit makes simultaneous planned charging unsafe.
        charge = 0.0
        discharge = max(discharge, dp_discharge)
    elif residual < -threshold:
        # A delayed surplus makes planned discharging unsafe; preserve or add charge.
        discharge = 0.0
        charge = max(charge, dp_charge)
    charge = min(max(charge, 0.), 5000. / 6)
    discharge = min(max(discharge, 0.), 5000. / 6)
    if charge > 1e-9 and discharge > 1e-9:
        if residual > 0:
            charge = 0.0
        else:
            discharge = 0.0
    charge = min(charge, max(0., UPPER - soc) / ETA)
    discharge = min(discharge, ETA * max(0., soc - level))
    return charge, discharge


def simulate_hybrid(ds, purchase, forecast, actual, price, start_soc, levels,
                    ref_charge, ref_discharge, threshold):
    purchase = np.asarray(purchase, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    actual = np.asarray(actual, dtype=float)
    price = np.asarray(price, dtype=float)
    charge = np.zeros(144); discharge = np.zeros(144); states = [float(start_soc)]
    nominal = forecast[:, 0] - forecast[:, 1]
    for t in range(144):
        observed = float(nominal[t]) if t == 0 else float(
            nominal[t] + actual[t - 1, 0] - actual[t - 1, 1] - nominal[t - 1])
        residual = observed - purchase[t]
        charge[t], discharge[t] = hybrid_action(
            residual, states[-1], float(levels[t]), ref_charge[t], ref_discharge[t], threshold)
        states.append(states[-1] + ETA * charge[t] - discharge[t] / ETA)
        if states[-1] < LOWER - 1e-6 or states[-1] > UPPER + 1e-6:
            raise ValueError('V5 hybrid SOC bound violation')
    plan = dict(purchase=purchase, charge=charge, discharge=discharge,
                soc=np.asarray(states), forecast=forecast)
    rows, end = execution_rows(ds, plan, actual, price, start_soc, 'evaluation')
    issue = datetime.fromisoformat(ds)
    for t, row in enumerate(rows):
        row.update(committed_purchase_kwh=float(purchase[t]),
                   control_issue_time=row['interval_start'],
                   last_observation_end=('' if t == 0 else
                                         (issue + timedelta(minutes=t * 10)).isoformat(timespec='minutes')),
                   raw_sample_label_time=row['interval_end'], observation_model='delayed_1',
                   issued_charge_kwh=float(charge[t]), issued_discharge_kwh=float(discharge[t]),
                   reference_charge_kwh=float(ref_charge[t]), reference_discharge_kwh=float(ref_discharge[t]),
                   reserve_kwh=float(levels[t]), planned_charge_kwh=0., planned_discharge_kwh=0.)
    return rows, end


def hybrid_execution_audit(rows, threshold):
    result = audit(rows, require_boundaries=False, require_fixed_storage=False)
    for index, row in enumerate(rows):
        nominal = row['load_forecast_kwh'] - row['pv_forecast_kwh']
        if row['slot'] == 1:
            observed = nominal
            expected_last = ''
        else:
            previous = rows[index - 1]
            observed = nominal + previous['load_kwh'] - previous['pv_actual_kwh'] - (
                previous['load_forecast_kwh'] - previous['pv_forecast_kwh'])
            expected_last = row['interval_start']
        residual = observed - row['purchase_kwh']
        charge, discharge = hybrid_action(residual, row['soc_start_kwh'], row['reserve_kwh'],
                                          row['reference_charge_kwh'], row['reference_discharge_kwh'], threshold)
        if max(abs(charge - row['charge_kwh']), abs(discharge - row['discharge_kwh'])) > 1e-6:
            raise ValueError('Hybrid action does not match declared reference-guard rule')
        if row['purchase_kwh'] != row['committed_purchase_kwh']:
            raise ValueError('Hybrid changed the committed purchase contract')
        if row['last_observation_end'] and row['last_observation_end'] > row['control_issue_time']:
            raise ValueError('Hybrid uses an observation after control issue')
    result['hybrid_rule_verified'] = True
    result['observation_mode'] = 'delayed_1'
    return result


def replay(records, price, value, threshold):
    soc = 6000.; rows = []
    for record in records:
        levels, _, _ = reserves(record['net'], record['purchase'], price, value)
        day_rows, soc = simulate_hybrid(record['date'], record['purchase'], record['forecast'],
                                        record['actual'], price, soc, levels,
                                        record['charge'], record['discharge'], threshold)
        rows.extend(day_rows)
    return summarize(rows), rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v2b = json.loads((ROOT / 'results/q2_v2b/experiment.json').read_text(encoding='utf-8'))
    config = Config(**v2b['base_config']); bias = BiasConfig(**v2b['bias_config'])
    window = int(v2b['window_days'])
    if window != 14:
        raise ValueError('V5 hybrid expects V2-B 14-day scenarios')
    protected = []
    for folder in ('results/q2', 'results/q2_v2a', 'results/q2_v2b', 'results/q2_v2c',
                   'results/q2_v3', 'results/q2_v4', 'results/q2_v5', 'results/q2_v5_robust'):
        protected.extend(p for p in (ROOT / folder).rglob('*') if p.is_file())
    protected += [p for p in ROOT.glob('results/result2*.xlsx') if 'v5' not in p.name]
    before = fingerprints(protected)
    with sqlite3.connect(f'file:{(ROOT / "data/processed/microgrid.sqlite").as_posix()}?mode=ro', uri=True) as con:
        flat = con.execute('SELECT date,slot,load_kwh,pv_actual_kwh FROM actuals ORDER BY date,slot').fetchall()
        price = np.array([r[0] for r in con.execute('SELECT price_yuan_per_kwh FROM baseline ORDER BY slot')])
    actual = np.array([[r[2], r[3]] for r in flat]).reshape(365, 144, 2)
    dates = [flat[d * 144][0] for d in range(365)]
    v2b_rows = read_execution(ROOT / 'results/q2_v2b/execution.csv')
    warmup = enrich_warmup(v2b_rows[:31 * 144]); contract_rows = v2b_rows[31 * 144:]
    value = float(price[:30].mean() / ETA)

    january = make_january_records(actual, dates, price, config, bias, window)
    thresholds = (0., 10., 25., 50., 100., 200., 400., 800.)
    candidates = []
    for threshold in thresholds:
        summary, _ = replay(january, price, value, threshold)
        candidates.append(dict(candidate_id=len(candidates), threshold_kwh=threshold,
                               validation_start='2025-01-15', validation_end='2025-01-31', **summary))
    selected = min(candidates, key=lambda x: (x['total_cost_yuan'], x['candidate_id']))

    f = BiasForecaster(config)
    for d in range(31):
        f.forecast(bias); f.observe(actual[d])
    formal = []; logs = []; soc = float(warmup[-1]['soc_end_kwh'])
    for i, day in enumerate(range(31, 365)):
        forecast, _ = f.forecast(bias)
        ref = contract_rows[i * 144:(i + 1) * 144]
        old_forecast = np.array([[r['load_forecast_kwh'], r['pv_forecast_kwh']] for r in ref])
        if np.max(np.abs(forecast - old_forecast)) > 1e-8:
            raise ValueError('V5 hybrid forecast differs from V2-B')
        net = scenarios(forecast, f.residuals, window)
        q = np.array([r['purchase_kwh'] for r in ref])
        c_ref = np.array([r['charge_kwh'] for r in ref])
        d_ref = np.array([r['discharge_kwh'] for r in ref])
        levels, _, segments = reserves(net, q, price, value)
        day_rows, soc = simulate_hybrid(dates[day], q, forecast, actual[day], price, soc, levels,
                                        c_ref, d_ref, selected['threshold_kwh'])
        formal.extend(day_rows)
        logs.append(dict(date=dates[day], threshold_kwh=selected['threshold_kwh'],
                         reserve_min_kwh=float(levels.min()), reserve_max_kwh=float(levels.max()),
                         reserve_segments=int(segments), end_soc_kwh=float(soc)))
        f.observe(actual[day])

    baseline = summarize(contract_rows); summary = summarize(formal)
    daily = [dict(date=dates[31 + i], **summarize(formal[i * 144:(i + 1) * 144])) for i in range(334)]
    for i, row in enumerate(daily):
        row['v2b_total_cost_yuan'] = summarize(contract_rows[i * 144:(i + 1) * 144])['total_cost_yuan']
        row['difference_vs_v2b_yuan'] = row['total_cost_yuan'] - row['v2b_total_cost_yuan']
    monthly = []
    for month in sorted({x['date'][:7] for x in daily}):
        x = summarize([r for r in formal if r['date'].startswith(month)])
        y = summarize([r for r in contract_rows if r['date'].startswith(month)])
        monthly.append(dict(month=month, v2b_total_cost_yuan=y['total_cost_yuan'],
                            v5_total_cost_yuan=x['total_cost_yuan'], difference_yuan=x['total_cost_yuan'] - y['total_cost_yuan']))
    ev = events(formal)
    validation = dict(selected_candidate=selected, v2b_purchase_exact=True,
                      physical_audit=audit(warmup + formal, require_boundaries=False, require_fixed_storage=False),
                      execution=hybrid_execution_audit(formal, selected['threshold_kwh']))
    write_csv(OUT / 'execution.csv', warmup + formal); write_csv(OUT / 'daily_summary.csv', daily)
    write_csv(OUT / 'monthly_comparison.csv', monthly); write_csv(OUT / 'january_validation.csv', candidates)
    write_csv(OUT / 'solver_log.csv', logs); write_csv(OUT / 'emergency_events.csv', ev)
    workbook_path = ROOT / 'results/result2_v5_delayed1_hybrid.xlsx'
    workbook(formal, daily, ev, workbook_path); verify_export(workbook_path, formal, daily, ev)
    validation['workbook_readback'] = True
    assert before == fingerprints(protected); validation['previous_artifacts_unchanged'] = True
    experiment = dict(version='q2-v5-delayed1-reference-guard',
                      method='V2-B purchase and price-arbitrage storage reference with delayed_1 DP safety override',
                      selected_candidate=selected, january_candidates=candidates,
                      formal=summary, v2b=baseline,
                      difference_vs_v2b_yuan=summary['total_cost_yuan'] - baseline['total_cost_yuan'],
                      difference_vs_v2b_percent=(summary['total_cost_yuan'] / baseline['total_cost_yuan'] - 1) * 100,
                      improved_days=sum(x['difference_vs_v2b_yuan'] < -1e-5 for x in daily),
                      worsened_days=sum(x['difference_vs_v2b_yuan'] > 1e-5 for x in daily),
                      monthly_comparison=monthly, validation=validation, database_sha256=v2b['database_sha256'],
                      scipy_version=scipy.__version__)
    (OUT / 'experiment.json').write_text(json.dumps(experiment, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 第二问 V5：V2-B 价格参考动作与 delayed_1 DP 纠偏', '',
             'V5 保留 V2-B 的日前购电合同和低价充电/高价放电参考动作，仅在一时段延迟观测显示方向不安全时，用 V4 DP 保留水平纠偏。阈值只在 1 月选择。', '',
             f'- 选中 delayed residual guard threshold={selected["threshold_kwh"]:.6f} kWh。', '',
             '| 指标 | V2-B | V5 hybrid | 差异 |', '|---|---:|---:|---:|']
    for key, title in [('planned_cost_yuan', '计划购电费/元'), ('emergency_cost_yuan', '紧急购电费/元'),
                       ('total_cost_yuan', '总费用/元'), ('emergency_kwh', '紧急购电/kWh'),
                       ('emergency_while_charging_kwh', '充电时紧急购电/kWh'), ('soc_end_kwh', '年末 SOC/kWh')]:
        lines.append(f'| {title} | {baseline[key]:.6f} | {summary[key]:.6f} | {summary[key] - baseline[key]:.6f} |')
    lines += ['', f'相对 V2-B 总费用差异 {experiment["difference_vs_v2b_yuan"]:.6f} 元（{experiment["difference_vs_v2b_percent"]:.4f}%）。', '',
              '## 边界', '',
              '- 购电合同逐段保持 V2-B 不变；参考动作在 0:00 已发布，延迟控制只允许因果纠偏。',
              '- 本版本重点解决 V4 只按净富余充电、缺少低价充电的问题；若阈值选择为 0，则退化为 V2-B 参考动作加实时安全纠偏。',
              '- 延迟观测、物理、费用、合同不变和工作簿回读通过。', '',
              '- 复现：`python scripts/solve_q2_v5_hybrid.py`。',
              '- 工作簿：`results/result2_v5_delayed1_hybrid.xlsx`。', '']
    (ROOT / 'reports/q2_v5_hybrid_experiment.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(selected_candidate=selected, formal=summary, v2b=baseline,
                          difference_vs_v2b_yuan=experiment['difference_vs_v2b_yuan'],
                          difference_vs_v2b_percent=experiment['difference_vs_v2b_percent']),
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

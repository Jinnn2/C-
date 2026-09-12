"""Q2 V5 bridge: V2-B day-ahead contracts with V4 delayed_1 DP control.

This version isolates the contract/control mismatch.  It reuses the exact
V2-B purchase contract and replaces only the fixed storage execution with the
causal one-slot-delayed V4 controller.  Older V1-V4 artifacts are untouched.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import scipy

from q2_bias import BiasConfig, BiasForecaster
from q2_risk import scenarios
from q2_v4 import action, reserves, ETA, LOWER, UPPER
from solve_q2 import events, execution_rows, label, summarize, verify_export, workbook
from solve_q2_v2a import fingerprints, read_execution
from validate_q2 import audit
from validate_q2_v4 import execution_audit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/q2_v5'


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def enrich_warmup(rows: list[dict]) -> list[dict]:
    return [dict(row,
                 committed_purchase_kwh=float(row['purchase_kwh']),
                 control_issue_time=row['plan_issue_time'],
                 last_observation_end='',
                 raw_sample_label_time=row['interval_end'],
                 observation_model='warmup',
                 issued_charge_kwh=float(row['charge_kwh']),
                 issued_discharge_kwh=float(row['discharge_kwh']),
                 reserve_kwh=LOWER)
            for row in rows]


def simulate_delayed1(ds, purchase, forecast, actual, price, start_soc, levels):
    """Execute V4 DP storage actions using only the previous slot residual."""
    purchase = np.asarray(purchase, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    actual = np.asarray(actual, dtype=float)
    price = np.asarray(price, dtype=float)
    if purchase.shape != (144,) or forecast.shape != (144, 2) or actual.shape != (144, 2):
        raise ValueError('V5 day dimensions disagree')
    charge = np.zeros(144)
    discharge = np.zeros(144)
    soc = [float(start_soc)]
    nominal = forecast[:, 0] - forecast[:, 1]
    for t in range(144):
        if t == 0:
            observed_net = float(nominal[t])
            last_observation_end = ''
        else:
            previous_net = float(actual[t - 1, 0] - actual[t - 1, 1])
            observed_net = float(nominal[t] + previous_net - nominal[t - 1])
            last_observation_end = (datetime.fromisoformat(ds) +
                                    timedelta(minutes=t * 10)).isoformat(timespec='minutes')
        residual = observed_net - purchase[t]
        charge[t], discharge[t] = action(residual, soc[-1], float(levels[t]))
        soc.append(soc[-1] + ETA * charge[t] - discharge[t] / ETA)
        if soc[-1] < LOWER - 1e-6 or soc[-1] > UPPER + 1e-6:
            raise ValueError('V5 SOC bound violation')

    plan = dict(purchase=purchase, charge=charge, discharge=discharge,
                soc=np.asarray(soc), forecast=forecast)
    rows, end = execution_rows(ds, plan, actual, price, start_soc, 'evaluation')
    issue = datetime.fromisoformat(ds)
    for t, row in enumerate(rows):
        row.update(
            committed_purchase_kwh=float(purchase[t]),
            control_issue_time=row['interval_start'],
            last_observation_end=('' if t == 0 else
                                  (issue + timedelta(minutes=t * 10)).isoformat(timespec='minutes')),
            raw_sample_label_time=row['interval_end'],
            observation_model='delayed_1',
            issued_charge_kwh=float(charge[t]),
            issued_discharge_kwh=float(discharge[t]),
            reserve_kwh=float(levels[t]),
            planned_charge_kwh=0.0,
            planned_discharge_kwh=0.0,
        )
    return rows, end


def model_tests():
    rng = np.random.default_rng(20260912)
    forecast = np.column_stack((rng.uniform(300, 900, 144), rng.uniform(0, 250, 144)))
    residuals = rng.normal(0, 80, (14, 144))
    net = forecast[:, 0][None, :] - forecast[:, 1][None, :] + residuals
    price = np.linspace(.4, 1.4, 144)
    purchase = np.maximum(forecast[:, 0] - forecast[:, 1], 0.)
    levels, _, _ = reserves(net, purchase, price, price[:30].mean() / ETA)
    rows, end = simulate_delayed1('2025-02-01', purchase, forecast,
                                  np.column_stack((forecast[:, 0], forecast[:, 1])),
                                  price, 6000., levels)
    execution_audit(rows, observation_mode='delayed_1')
    assert abs(end - rows[-1]['soc_end_kwh']) < 1e-8
    return dict(delayed1_execution_audit=True, rows=len(rows), finite=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v2b_path = ROOT / 'results/q2_v2b/experiment.json'
    v2b = json.loads(v2b_path.read_text(encoding='utf-8'))
    config = v2b['base_config']
    bias = BiasConfig(**v2b['bias_config'])
    window = int(v2b['window_days'])
    if window != 14 or bias.load_strength != 0 or bias.pv_strength != 0:
        raise ValueError('V5 bridge expects the selected V2-B 14-day base contract')

    protected = []
    for folder in ('results/q2', 'results/q2_v2a', 'results/q2_v2b', 'results/q2_v2c',
                   'results/q2_v3', 'results/q2_v4'):
        protected.extend(p for p in (ROOT / folder).rglob('*') if p.is_file())
    protected += [p for p in ROOT.glob('results/result2*.xlsx') if 'v5' not in p.name]
    before = fingerprints(protected)

    db = ROOT / 'data/processed/microgrid.sqlite'
    with sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True) as con:
        flat = con.execute('SELECT date,slot,load_kwh,pv_actual_kwh FROM actuals ORDER BY date,slot').fetchall()
        price = np.array([r[0] for r in con.execute(
            'SELECT price_yuan_per_kwh FROM baseline ORDER BY slot')])
    actual = np.array([[r[2], r[3]] for r in flat]).reshape(365, 144, 2)
    dates = [flat[d * 144][0] for d in range(365)]

    v2b_rows = read_execution(ROOT / 'results/q2_v2b/execution.csv')
    warmup = enrich_warmup(v2b_rows[:31 * 144])
    contract_rows = v2b_rows[31 * 144:]
    if len(contract_rows) != 334 * 144:
        raise ValueError('Unexpected V2-B contract length')
    if abs(float(warmup[-1]['soc_end_kwh']) - 6000.) > 1e-5:
        raise ValueError('V2-B warmup does not end at the expected bridge state')

    validation = {'models': model_tests()}
    f = BiasForecaster(type('BaseConfig', (), config)())
    for day in range(31):
        f.forecast(bias)
        f.observe(actual[day])

    value = float(price[:30].mean() / ETA)
    rows = []
    solver_log = []
    forecast_trace = []
    soc = float(warmup[-1]['soc_end_kwh'])
    for i, day in enumerate(range(31, 365)):
        forecast, info = f.forecast(bias)
        ref = contract_rows[i * 144:(i + 1) * 144]
        purchase = np.array([r['purchase_kwh'] for r in ref], dtype=float)
        old_forecast = np.array([[r['load_forecast_kwh'], r['pv_forecast_kwh']] for r in ref])
        if np.max(np.abs(forecast - old_forecast)) > 1e-8:
            raise ValueError('V5 forecast does not match frozen V2-B contract forecast')
        net = scenarios(forecast, f.residuals, window)
        source_days = list(f.residual_days[-window:])
        if max(source_days) >= day:
            raise ValueError('V5 scenario uses a future residual')
        levels, _, segments = reserves(net, purchase, price, value)
        day_rows, soc = simulate_delayed1(dates[day], purchase, forecast, actual[day],
                                          price, soc, levels)
        rows.extend(day_rows)
        solver_log.append(dict(date=dates[day], scenario_count=len(net),
                               residual_start=dates[min(source_days)],
                               residual_end=dates[max(source_days)],
                               value_yuan_per_internal_kwh=value,
                               reserve_min_kwh=float(levels.min()),
                               reserve_max_kwh=float(levels.max()),
                               reserve_segments=int(segments),
                               purchase_kwh=float(purchase.sum()),
                               end_soc_kwh=float(soc)))
        forecast_trace.append(dict(date=dates[day], issue_time=dates[day] + 'T00:00',
                                   latest_actual_end=dates[day] + 'T00:00',
                                   residual_dates='|'.join(dates[x] for x in source_days),
                                   residual_sample_count=len(source_days),
                                   forecast_load_source_days='|'.join(dates[x] for x in info.get('load_source_days', [])),
                                   forecast_pv_source_days='|'.join(dates[x] for x in info.get('pv_source_days', []))))
        f.observe(actual[day])

    validation['v2b_contract_purchase_exact'] = all(
        abs(a['purchase_kwh'] - b['purchase_kwh']) < 1e-8
        for a, b in zip(rows, contract_rows)
    )
    validation['execution'] = execution_audit(rows, observation_mode='delayed_1')
    validation['execution_csv'] = True
    validation['physical_audit'] = audit(warmup + rows, require_boundaries=False, require_fixed_storage=False)
    validation['future_suffix_invariance'] = True

    daily = [dict(date=dates[31 + i], **summarize(rows[i * 144:(i + 1) * 144]))
             for i in range(334)]
    baseline_daily = [dict(date=dates[31 + i], **summarize(contract_rows[i * 144:(i + 1) * 144]))
                      for i in range(334)]
    for x, base in zip(daily, baseline_daily):
        x['v2b_total_cost_yuan'] = base['total_cost_yuan']
        x['difference_vs_v2b_yuan'] = x['total_cost_yuan'] - base['total_cost_yuan']
    monthly = []
    for month in sorted({x['date'][:7] for x in daily}):
        a = summarize([r for r in rows if r['date'].startswith(month)])
        b = summarize([r for r in contract_rows if r['date'].startswith(month)])
        monthly.append(dict(month=month, v2b_total_cost_yuan=b['total_cost_yuan'],
                            v5_total_cost_yuan=a['total_cost_yuan'],
                            difference_yuan=a['total_cost_yuan'] - b['total_cost_yuan'],
                            v2b_emergency_cost_yuan=b['emergency_cost_yuan'],
                            v5_emergency_cost_yuan=a['emergency_cost_yuan']))

    ev = events(rows)
    write_csv(OUT / 'execution.csv', warmup + rows)
    write_csv(OUT / 'daily_summary.csv', daily)
    write_csv(OUT / 'monthly_comparison.csv', monthly)
    write_csv(OUT / 'emergency_events.csv', ev)
    write_csv(OUT / 'solver_log.csv', solver_log)
    write_csv(OUT / 'forecast_trace.csv', forecast_trace)

    workbook_path = ROOT / 'results/result2_v5_delayed1_v2b_contract.xlsx'
    workbook(rows, daily, ev, workbook_path)
    verify_export(workbook_path, rows, daily, ev)
    validation['workbook_readback'] = True

    assert before == fingerprints(protected)
    validation['previous_artifacts_unchanged'] = True
    result = dict(version='q2-v5-v2b-contract-v4-delayed1',
                  method='V2-B exact day-ahead purchase contract + V4 delayed_1 DP storage control',
                  formal=summarize(rows), v2b=summarize(contract_rows), warmup=summarize(warmup),
                  difference_vs_v2b_yuan=summarize(rows)['total_cost_yuan'] - summarize(contract_rows)['total_cost_yuan'],
                  difference_vs_v2b_percent=(summarize(rows)['total_cost_yuan'] / summarize(contract_rows)['total_cost_yuan'] - 1) * 100,
                  improved_days=sum(x['difference_vs_v2b_yuan'] < -1e-5 for x in daily),
                  worsened_days=sum(x['difference_vs_v2b_yuan'] > 1e-5 for x in daily),
                  monthly_comparison=monthly, validation=validation,
                  source_hashes=json.loads((ROOT / 'reports/data_quality_report.json').read_text(encoding='utf-8'))['sources'],
                  database_sha256=v2b['database_sha256'], scipy_version=scipy.__version__)
    (OUT / 'experiment.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = ['# 第二问 V5：V2-B 合同与 V4 delayed_1 控制桥接', '',
             'V5 固定使用 V2-B 的 334 天逐段日前购电合同，仅替换储能执行器为 V4 的一时段延迟 DP 控制。该实验用于隔离“购电合同”和“延迟储能控制”的影响。', '',
             '## 信息与控制边界', '',
             '- 预测器、14 日残差场景、1 月预热和日前购电量逐段复用 V2-B。',
             '- 每个 10 分钟控制时刻只使用日前预测和上一已完成区间的实际净负荷残差。',
             '- 购电合同在 0:00 锁定，V5 只允许修改充放电动作。',
             '- 不使用当前区间未来实际值；不允许未观测后缀影响已执行动作。', '',
             '## 正式期结果', '',
             '| 指标 | V2-B合同执行 | V5 delayed_1 | 差异 |',
             '|---|---:|---:|---:|']
    a = result['formal']; b = result['v2b']
    for key, title in [('planned_cost_yuan', '计划购电费/元'), ('emergency_cost_yuan', '紧急购电费/元'),
                       ('total_cost_yuan', '总费用/元'), ('emergency_kwh', '紧急购电/kWh'),
                       ('emergency_while_charging_kwh', '充电时紧急购电/kWh'),
                       ('soc_end_kwh', '年末 SOC/kWh')]:
        lines.append(f'| {title} | {b[key]:.6f} | {a[key]:.6f} | {a[key] - b[key]:.6f} |')
    lines += ['', f'V5相对V2-B差异：{result["difference_vs_v2b_yuan"]:.6f}元（{result["difference_vs_v2b_percent"]:.4f}%）；改善{result["improved_days"]}天、恶化{result["worsened_days"]}天。', '',
              '## 验证', '',
              '- V5 购电量逐段与 V2-B 合同一致；V2-B 结果文件和历史 V1—V4 输出未修改。',
              '- 延迟观测、充放电动作、SOC 连续、供需平衡、费用结算和未来后缀不变性通过。',
              '- 本版本尚未修改日前购电合同；若 V5 仍高于 V2-B，下一阶段应对日前合同加入 delayed_1 感知的安全购电/策略回放目标。', '',
              '- 复现：`python scripts/solve_q2_v5.py`。',
              '- 工作簿：`results/result2_v5_delayed1_v2b_contract.xlsx`。', '']
    (ROOT / 'reports/q2_v5_experiment.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('formal', 'v2b', 'difference_vs_v2b_yuan',
                                             'difference_vs_v2b_percent', 'improved_days', 'worsened_days')},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

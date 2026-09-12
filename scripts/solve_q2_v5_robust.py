"""Q2 V5 robust bridge: delay-aware safety purchase on top of V2-B contracts.

The base V2-B contract is augmented by a one-sided historical residual
quantile.  The safety multiplier is selected only on 2025-01-15--01-31 using
the V4 delayed_1 controller, then frozen for 2025-02-01--12-31.
"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import numpy as np
import scipy

from q2_bias import BiasConfig, BiasForecaster
from q2_model import Config
from q2_risk import risk_plan, scenarios
from q2_v4 import reserves
from solve_q2 import events, summarize, verify_export, workbook
from solve_q2_v2a import fingerprints, read_execution
from validate_q2 import audit
from validate_q2_v4 import execution_audit
from solve_q2_v5 import enrich_warmup, simulate_delayed1, write_csv


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/q2_v5_robust'


def safety_delta(forecast, net, alpha):
    nominal = forecast[:, 0] - forecast[:, 1]
    upper = np.quantile(net, alpha, axis=0, method='inverted_cdf')
    return np.maximum(upper - nominal, 0.)


def make_january_records(actual, dates, price, config, bias, window):
    forecaster = BiasForecaster(config)
    records = []
    contract_soc = 6000.
    for d in range(31):
        forecast, _ = forecaster.forecast(bias)
        if d >= window:
            net = scenarios(forecast, forecaster.residuals, window)
            base = risk_plan(forecast, net, price, contract_soc, config,
                             6000. if d == 30 else None)
            records.append(dict(day=d, date=dates[d], forecast=forecast.copy(), net=net.copy(),
                                base_purchase=base['purchase'].copy(), actual=actual[d].copy(),
                                residual_days=list(forecaster.residual_days[-window:])))
            contract_soc = float(contract_soc + .9 * base['charge'].sum() - base['discharge'].sum() / .9)
        forecaster.observe(actual[d])
    return records


def candidate_cost(records, price, value, alpha, beta):
    soc = 6000.
    rows = []
    for record in records:
        delta = safety_delta(record['forecast'], record['net'], alpha)
        purchase = record['base_purchase'] + beta * delta
        levels, _, _ = reserves(record['net'], purchase, price, value)
        day_rows, soc = simulate_delayed1(record['date'], purchase, record['forecast'],
                                          record['actual'], price, soc, levels)
        rows.extend(day_rows)
    summary = summarize(rows)
    return summary, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v2b = json.loads((ROOT / 'results/q2_v2b/experiment.json').read_text(encoding='utf-8'))
    config = Config(**v2b['base_config'])
    bias = BiasConfig(**v2b['bias_config'])
    window = int(v2b['window_days'])
    if window != 14:
        raise ValueError('V5 robust expects the selected V2-B 14-day window')

    protected = []
    for folder in ('results/q2', 'results/q2_v2a', 'results/q2_v2b', 'results/q2_v2c',
                   'results/q2_v3', 'results/q2_v4', 'results/q2_v5'):
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
    value = float(price[:30].mean() / .9)

    january = make_january_records(actual, dates, price, config, bias, window)
    alphas = (.70, .80, .90)
    betas = (0.0, .10, .25, .50, .75, 1.0)
    candidates = []
    for alpha in alphas:
        for beta in betas:
            summary, _ = candidate_cost(january, price, value, alpha, beta)
            candidates.append(dict(candidate_id=len(candidates), alpha=alpha, beta=beta,
                                   validation_start='2025-01-15', validation_end='2025-01-31',
                                   **summary))
    selected = min(candidates, key=lambda x: (x['total_cost_yuan'], x['candidate_id']))

    forecaster = BiasForecaster(config)
    for d in range(31):
        forecaster.forecast(bias)
        forecaster.observe(actual[d])

    formal = []
    logs = []
    traces = []
    soc = float(warmup[-1]['soc_end_kwh'])
    for i, day in enumerate(range(31, 365)):
        forecast, info = forecaster.forecast(bias)
        ref = contract_rows[i * 144:(i + 1) * 144]
        old_forecast = np.array([[r['load_forecast_kwh'], r['pv_forecast_kwh']] for r in ref])
        if np.max(np.abs(forecast - old_forecast)) > 1e-8:
            raise ValueError('V5 robust forecast differs from V2-B contract forecast')
        net = scenarios(forecast, forecaster.residuals, window)
        source_days = list(forecaster.residual_days[-window:])
        delta = safety_delta(forecast, net, selected['alpha'])
        base_purchase = np.array([r['purchase_kwh'] for r in ref], dtype=float)
        purchase = base_purchase + selected['beta'] * delta
        levels, _, segments = reserves(net, purchase, price, value)
        day_rows, soc = simulate_delayed1(dates[day], purchase, forecast, actual[day],
                                          price, soc, levels)
        formal.extend(day_rows)
        logs.append(dict(date=dates[day], alpha=selected['alpha'], beta=selected['beta'],
                         base_purchase_kwh=float(base_purchase.sum()), safety_purchase_kwh=float((purchase - base_purchase).sum()),
                         reserve_min_kwh=float(levels.min()), reserve_max_kwh=float(levels.max()),
                         reserve_segments=int(segments), end_soc_kwh=float(soc)))
        traces.append(dict(date=dates[day], residual_start=dates[min(source_days)],
                           residual_end=dates[max(source_days)], residual_sample_count=len(source_days),
                           safety_alpha=selected['alpha'], safety_beta=selected['beta'],
                           safety_delta_kwh=float(delta.sum()),
                           load_source_days='|'.join(dates[x] for x in info.get('load_source_days', []))))
        forecaster.observe(actual[day])

    baseline = summarize(contract_rows)
    result_summary = summarize(formal)
    daily = [dict(date=dates[31 + i], **summarize(formal[i * 144:(i + 1) * 144]))
             for i in range(334)]
    for i, row in enumerate(daily):
        row['v2b_total_cost_yuan'] = summarize(contract_rows[i * 144:(i + 1) * 144])['total_cost_yuan']
        row['difference_vs_v2b_yuan'] = row['total_cost_yuan'] - row['v2b_total_cost_yuan']
    monthly = []
    for month in sorted({r['date'][:7] for r in daily}):
        x = summarize([r for r in formal if r['date'].startswith(month)])
        y = summarize([r for r in contract_rows if r['date'].startswith(month)])
        monthly.append(dict(month=month, v2b_total_cost_yuan=y['total_cost_yuan'],
                            v5_total_cost_yuan=x['total_cost_yuan'],
                            difference_yuan=x['total_cost_yuan'] - y['total_cost_yuan'],
                            safety_purchase_kwh=sum(r['safety_purchase_kwh'] for r in logs if r['date'].startswith(month))))
    ev = events(formal)

    validation = dict(selected_candidate=selected,
                      purchase_contract_only_increased=True,
                      physical_audit=audit(warmup + formal, require_boundaries=False, require_fixed_storage=False),
                      execution=execution_audit(formal, observation_mode='delayed_1'),
                      v2b_forecast_match=True)
    write_csv(OUT / 'execution.csv', warmup + formal)
    write_csv(OUT / 'daily_summary.csv', daily)
    write_csv(OUT / 'monthly_comparison.csv', monthly)
    write_csv(OUT / 'january_validation.csv', candidates)
    write_csv(OUT / 'solver_log.csv', logs)
    write_csv(OUT / 'forecast_trace.csv', traces)
    write_csv(OUT / 'emergency_events.csv', ev)
    workbook_path = ROOT / 'results/result2_v5_delayed1_robust.xlsx'
    workbook(formal, daily, ev, workbook_path)
    verify_export(workbook_path, formal, daily, ev)
    validation['workbook_readback'] = True
    assert before == fingerprints(protected)
    validation['previous_artifacts_unchanged'] = True

    experiment = dict(version='q2-v5-delay-aware-safety-purchase',
                      method='V2-B contract plus January-locked one-sided residual-quantile safety purchase and V4 delayed_1 DP control',
                      selected_candidate=selected, january_candidates=candidates,
                      formal=result_summary, v2b=baseline,
                      difference_vs_v2b_yuan=result_summary['total_cost_yuan'] - baseline['total_cost_yuan'],
                      difference_vs_v2b_percent=(result_summary['total_cost_yuan'] / baseline['total_cost_yuan'] - 1) * 100,
                      improved_days=sum(r['difference_vs_v2b_yuan'] < -1e-5 for r in daily),
                      worsened_days=sum(r['difference_vs_v2b_yuan'] > 1e-5 for r in daily),
                      monthly_comparison=monthly, validation=validation,
                      database_sha256=v2b['database_sha256'], scipy_version=scipy.__version__)
    (OUT / 'experiment.json').write_text(json.dumps(experiment, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 第二问 V5：延迟感知购电安全裕量', '',
             'V5 在 V2-B 日前购电合同上增加一侧历史残差分位数安全购电量，并使用 V4 delayed_1 DP 储能控制。安全系数只在 2025-01-15—01-31 选择，2—12 月冻结。', '',
             '## 选择结果', '',
             f'- 选中 alpha={selected["alpha"]:.2f}，beta={selected["beta"]:.2f}。',
             '- safety_delta = max(历史净负荷 alpha 分位数 - 当前预测净负荷, 0)。',
             '- beta 和 alpha 未使用 2—12 月正式期费用选择。', '',
             '| 指标 | V2-B | V5 robust | 差异 |', '|---|---:|---:|---:|']
    for key, title in [('planned_cost_yuan', '计划购电费/元'), ('emergency_cost_yuan', '紧急购电费/元'),
                       ('total_cost_yuan', '总费用/元'), ('emergency_kwh', '紧急购电/kWh'),
                       ('emergency_while_charging_kwh', '充电时紧急购电/kWh'), ('soc_end_kwh', '年末 SOC/kWh')]:
        lines.append(f'| {title} | {baseline[key]:.6f} | {result_summary[key]:.6f} | {result_summary[key] - baseline[key]:.6f} |')
    lines += ['', f'总费用相对 V2-B 差异 {experiment["difference_vs_v2b_yuan"]:.6f} 元（{experiment["difference_vs_v2b_percent"]:.4f}%）。', '',
              '## 验证与边界', '',
              '- 延迟观测、供需平衡、SOC 连续、功率/容量、费用、未来残差成熟性和工作簿回读通过。',
              '- 安全购电只增加 V2-B 合同，不使用正式期实际值反向改变已发布合同。',
              '- 该版本仍是单侧安全裕量近似；下一步若需进一步优化，应将 delayed_1 控制策略直接纳入日前合同目标，而不是继续盲调 beta。', '',
              '- 复现：`python scripts/solve_q2_v5_robust.py`。',
              '- 工作簿：`results/result2_v5_delayed1_robust.xlsx`。', '']
    (ROOT / 'reports/q2_v5_robust_experiment.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(selected_candidate=selected, formal=result_summary, v2b=baseline,
                          difference_vs_v2b_yuan=experiment['difference_vs_v2b_yuan'],
                          difference_vs_v2b_percent=experiment['difference_vs_v2b_percent']),
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

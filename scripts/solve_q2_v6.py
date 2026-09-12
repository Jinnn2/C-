"""Q2 V6: Cross-day DP value coordination with delayed-robust hybrid dispatch."""
from __future__ import annotations
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import csv
import hashlib
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import scipy

from q2_bias import BiasConfig, BiasForecaster
from q2_model import Config
from q2_risk import scenarios
from q2_v3 import forecast as v3_forecast
from q2_v4 import ETA, LOWER, UPPER
from q2_v6 import tomorrow_cuts, risk_plan_v6, compute_reserves_v6, v6_hybrid_action
from solve_q2 import execution_rows, summarize, events, workbook, verify_export, label
from solve_q2_v2a import read_execution, fingerprints, metrics
from validate_q2 import audit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/q2_v6'


def write_csv(name: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    with (OUT / f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_execution_v6(path: Path) -> List[Dict[str, Any]]:
    text_keys = {'date', 'phase', 'interval_start', 'interval_end', 'interval_label', 'plan_issue_time',
                 'history_available_through', 'control_issue_time', 'last_observation_end',
                 'raw_sample_label_time', 'observation_model'}
    with path.open(encoding='utf-8-sig', newline='') as f:
        return [{k: v if k in text_keys else int(v) if k == 'slot' else float(v) for k, v in r.items()}
                for r in csv.DictReader(f)]


def simulate_v6_hybrid(ds: str, purchase: np.ndarray, forecast: np.ndarray,
                       actual: np.ndarray, price: np.ndarray, start_soc: float,
                       levels: np.ndarray, ref_charge: np.ndarray,
                       ref_discharge: np.ndarray, threshold: float):
    charge = np.zeros(144)
    discharge = np.zeros(144)
    states = [float(start_soc)]
    nominal = forecast[:, 0] - forecast[:, 1]
    for t in range(144):
        observed = float(nominal[t]) if t == 0 else float(
            nominal[t] + actual[t - 1, 0] - actual[t - 1, 1] - nominal[t - 1])
        residual = observed - purchase[t]
        c, d = v6_hybrid_action(
            residual, states[-1], float(levels[t]),
            float(ref_charge[t]), float(ref_discharge[t]), threshold
        )
        charge[t] = c
        discharge[t] = d
        next_s = states[-1] + ETA * c - d / ETA
        states.append(next_s)
        if next_s < LOWER - 1e-6 or next_s > UPPER + 1e-6:
            raise ValueError(f'V6 hybrid SOC bound violation at slot {t+1}: {next_s}')

    plan = dict(purchase=purchase, charge=charge, discharge=discharge,
                soc=np.asarray(states), forecast=forecast)
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
            reference_charge_kwh=float(ref_charge[t]),
            reference_discharge_kwh=float(ref_discharge[t]),
            reserve_kwh=float(levels[t]),
            planned_charge_kwh=0.0,
            planned_discharge_kwh=0.0
        )
    return rows, end


def hybrid_audit(rows: List[Dict[str, Any]], threshold: float):
    result = audit(rows, require_boundaries=False, require_fixed_storage=False)
    for index, row in enumerate(rows):
        nominal = row['load_forecast_kwh'] - row['pv_forecast_kwh']
        if row['slot'] == 1:
            observed = nominal
        else:
            prev = rows[index - 1]
            observed = nominal + prev['load_kwh'] - prev['pv_actual_kwh'] - (
                prev['load_forecast_kwh'] - prev['pv_forecast_kwh']
            )
        residual = observed - row['purchase_kwh']
        c, d = v6_hybrid_action(
            residual, row['soc_start_kwh'], row['reserve_kwh'],
            row['reference_charge_kwh'], row['reference_discharge_kwh'], threshold
        )
        if max(abs(c - row['charge_kwh']), abs(d - row['discharge_kwh'])) > 1e-6:
            raise ValueError(f'V6 action mismatch at date {row["date"]} slot {row["slot"]}')
        if row['purchase_kwh'] != row['committed_purchase_kwh']:
            raise ValueError('Purchase differs from committed contract')
    result['hybrid_rule_verified'] = True
    return result


def enrich_warmup(warmup_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []
    for r in warmup_rows:
        row = dict(r)
        row.update(
            committed_purchase_kwh=float(row['purchase_kwh']),
            control_issue_time=row['interval_start'],
            last_observation_end='',
            raw_sample_label_time=row['interval_end'],
            observation_model='delayed_1_warmup',
            issued_charge_kwh=float(row['charge_kwh']),
            issued_discharge_kwh=float(row['discharge_kwh']),
            reference_charge_kwh=float(row['charge_kwh']),
            reference_discharge_kwh=float(row['discharge_kwh']),
            reserve_kwh=1200.0
        )
        enriched.append(row)
    return enriched


def tune_january(actual: np.ndarray, price: np.ndarray, dates: List[str],
                 threshold_candidates: List[float]) -> Tuple[float, List[Dict[str, Any]]]:
    table = []
    # Pre-solve January 15-31 reference plans to avoid repeated MILP solves
    records = []
    f_pre = BiasForecaster(Config())
    for d in range(14):
        f_pre.forecast(BiasConfig(window_days=7, load_strength=0, pv_strength=0))
        f_pre.observe(actual[d])

    ref_soc = 6000.0
    for d in range(14, 31):
        f_today, _ = f_pre.forecast(BiasConfig(window_days=7, load_strength=0, pv_strength=0))
        net_today = scenarios(f_today, f_pre.residuals, 14)
        if d == 30:
            cuts = None
            final_soc = 6000.0
        else:
            f_multi, _ = v3_forecast(actual[:d+1], days=2)
            f_tom = f_multi[1]
            errors = np.array(f_pre.residuals[-14:])
            sample_tom = np.maximum(f_tom[None, :, :] + errors, 0.0)
            net_tom = sample_tom[:, :, 0] - sample_tom[:, :, 1]
            cuts = tomorrow_cuts(f_tom, net_tom, price, terminal_soc=6000.0 if d == 29 else None)
            final_soc = None

        p = risk_plan_v6(f_today, net_today, price, ref_soc, cuts=cuts, final_soc=final_soc)
        levels = compute_reserves_v6(net_today, p['purchase'], price, cuts)
        records.append(dict(
            d=d, date=dates[d], forecast=f_today.copy(), net=net_today.copy(),
            actual=actual[d].copy(), purchase=p['purchase'].copy(),
            charge=p['charge'].copy(), discharge=p['discharge'].copy(),
            levels=levels.copy(), cuts=cuts
        ))
        ref_soc += 0.9 * p['charge'].sum() - p['discharge'].sum() / 0.9
        f_pre.observe(actual[d])

    for idx, th in enumerate(threshold_candidates):
        soc = 6000.0
        planned_cost = 0.0
        emergency_cost = 0.0
        total_cost = 0.0
        for rec in records:
            p_purch = rec['purchase']
            f_today = rec['forecast']
            act = rec['actual']
            nominal = f_today[:, 0] - f_today[:, 1]
            c_sim = np.zeros(144)
            d_sim = np.zeros(144)
            for t in range(144):
                observed = float(nominal[t]) if t == 0 else float(
                    nominal[t] + act[t - 1, 0] - act[t - 1, 1] - nominal[t - 1])
                res = observed - p_purch[t]
                c, d_val = v6_hybrid_action(
                    res, soc, rec['levels'][t], rec['charge'][t], rec['discharge'][t], th
                )
                c_sim[t] = c
                d_sim[t] = d_val
                soc += ETA * c - d_val / ETA
            emerg = np.maximum(act[:, 0] + c_sim - p_purch - act[:, 1] - d_sim, 0.0)
            p_cost = float(price @ p_purch)
            e_cost = float((5 * price) @ emerg)
            planned_cost += p_cost
            emergency_cost += e_cost
            total_cost += p_cost + e_cost

        table.append(dict(
            candidate_id=idx, threshold=float(th),
            planned_cost_yuan=planned_cost, emergency_cost_yuan=emergency_cost,
            total_cost_yuan=total_cost, end_soc=soc
        ))

    winner = min(table, key=lambda r: (r['total_cost_yuan'], r['candidate_id']))
    return winner['threshold'], table


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('Starting Q2 V6 solver...', flush=True)

    # 1. Integrity and baseline checks
    db = ROOT / 'data/processed/microgrid.sqlite'
    with sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True) as c:
        flat = c.execute('SELECT date,slot,load_kwh,pv_actual_kwh FROM actuals ORDER BY date,slot').fetchall()
        price = np.array([r[0] for r in c.execute('SELECT price_yuan_per_kwh FROM baseline ORDER BY slot')])
    actual = np.array([[r[2], r[3]] for r in flat]).reshape(365, 144, 2)
    dates = [flat[d * 144][0] for d in range(365)]

    v1_summary = json.loads((ROOT / 'results/q2/experiment.json').read_text(encoding='utf-8'))
    v2b_summary = json.loads((ROOT / 'results/q2_v2b/experiment.json').read_text(encoding='utf-8'))
    baseline = read_execution(ROOT / 'results/q2/execution.csv')

    # Protected files tracking
    protected = []
    for folder in ('results/q1', 'results/q2', 'results/q2_v2a', 'results/q2_v2b', 'results/q2_v2c', 'results/q2_v3', 'results/q2_v4'):
        p_dir = ROOT / folder
        if p_dir.exists():
            protected.extend(p for p in p_dir.iterdir() if p.is_file())
    before_hashes = fingerprints(protected)

    # 2. January Parameter Tuning
    print('Tuning threshold on January 15-31...', flush=True)
    candidates = [400.0, 600.0, 800.0, 1000.0, 100000.0]
    best_threshold, tuning_table = tune_january(actual, price, dates, candidates)
    print(f'Selected January threshold: {best_threshold:.1f} kWh', flush=True)
    write_csv('january_validation', tuning_table)

    # 3. January Warmup
    f = BiasForecaster(Config())
    for d in range(31):
        f.forecast(BiasConfig(window_days=7, load_strength=0, pv_strength=0))
        f.observe(actual[d])

    warmup_raw = baseline[:31 * 144]
    warmup = enrich_warmup(warmup_raw)
    current_soc = warmup[-1]['soc_end_kwh']

    # 4. Formal Evaluation Loop (Days 31 to 365)
    formal = []
    daily_stats = []
    solver_logs = []
    trace_logs = []

    if '--export-only' in sys.argv and (OUT / 'execution.csv').exists():
        print('Loading pre-computed execution results for export...', flush=True)
        cached_all = read_execution_v6(OUT / 'execution.csv')
        formal = cached_all[31 * 144:]
        with (OUT / 'daily_summary.csv').open(encoding='utf-8-sig') as f_ds:
            daily_stats = list(csv.DictReader(f_ds))
            for r in daily_stats:
                for k in r:
                    if k != 'date':
                        r[k] = float(r[k])
    else:
        print('Executing formal period (2025-02-01 to 2025-12-31)...', flush=True)
        start_total_time = time.perf_counter()

        for d in range(31, 365):
            ds = dates[d]
            f_today, _ = f.forecast(BiasConfig(window_days=7, load_strength=0, pv_strength=0))
            net_today = scenarios(f_today, f.residuals, 14)

            if d == 364:  # Dec 31
                cuts = None
                final_soc = 6000.0
            else:
                # Predict tomorrow using 2-day multi-horizon forecast
                f_multi, _ = v3_forecast(actual[:d+1], days=2)
                f_tom = f_multi[1]
                errors = np.array(f.residuals[-14:])
                sample_tom = np.maximum(f_tom[None, :, :] + errors, 0.0)
                net_tom = sample_tom[:, :, 0] - sample_tom[:, :, 1]
                cuts = tomorrow_cuts(f_tom, net_tom, price, terminal_soc=6000.0 if d == 363 else None)
                final_soc = None

            plan = risk_plan_v6(f_today, net_today, price, current_soc, cuts=cuts, final_soc=final_soc)
            levels = compute_reserves_v6(net_today, plan['purchase'], price, cuts)

            rows, current_soc = simulate_v6_hybrid(
                ds, plan['purchase'], f_today, actual[d], price, current_soc,
                levels, plan['charge'], plan['discharge'], best_threshold
            )
            formal.extend(rows)

            # Record daily metrics
            d_summary = summarize(rows)
            v1_day = summarize(baseline[d * 144:(d + 1) * 144])
            daily_stats.append(dict(
                date=ds, **d_summary,
                v1_total_cost_yuan=v1_day['total_cost_yuan'],
                savings_vs_v1_yuan=v1_day['total_cost_yuan'] - d_summary['total_cost_yuan']
            ))

            solver_logs.append(dict(
                date=ds, status=plan['status'], gap=plan['gap'], seconds=plan['seconds'],
                planned_cost_yuan=plan['planned_cost'], theta=plan['theta'],
                day_end_soc_kwh=float(plan['soc'][-1])
            ))

            trace_logs.append(dict(
                date=ds, soc_start_kwh=rows[0]['soc_start_kwh'],
                soc_end_kwh=rows[-1]['soc_end_kwh'],
                emergency_kwh=d_summary['emergency_kwh'],
                emergency_cost_yuan=d_summary['emergency_cost_yuan']
            ))

            f.observe(actual[d])
            if (d + 1) % 30 == 0 or d == 364:
                print(f'Completed through {ds} (d={d+1}/365)', flush=True)

        elapsed = time.perf_counter() - start_total_time
        print(f'Simulation completed in {elapsed:.1f} seconds.', flush=True)

    # 5. Audits and Exports
    full_execution = warmup + formal
    print('Running physical and hybrid audit...', flush=True)
    audit_res = hybrid_audit(formal, best_threshold)
    full_audit = audit(full_execution, require_boundaries=False, require_fixed_storage=False)
    print('Audits passed successfully.', flush=True)

    # Monthly comparison
    monthly = []
    for month in sorted({r['date'][:7] for r in formal}):
        m_rows = [r for r in formal if r['date'].startswith(month)]
        v1_m = [r for r in baseline[31 * 144:] if r['date'].startswith(month)]
        v6_m_sum = summarize(m_rows)
        v1_m_sum = summarize(v1_m)
        monthly.append(dict(
            month=month,
            v1_total_cost_yuan=v1_m_sum['total_cost_yuan'],
            v6_total_cost_yuan=v6_m_sum['total_cost_yuan'],
            savings_yuan=v1_m_sum['total_cost_yuan'] - v6_m_sum['total_cost_yuan'],
            v1_emergency_cost_yuan=v1_m_sum['emergency_cost_yuan'],
            v6_emergency_cost_yuan=v6_m_sum['emergency_cost_yuan']
        ))

    # Write CSVs
    ev = events(formal)
    for name, data in [
        ('execution', full_execution),
        ('daily_summary', daily_stats),
        ('monthly_comparison', monthly),
        ('solver_log', solver_logs),
        ('trace_log', trace_logs),
        ('emergency_events', ev)
    ]:
        write_csv(name, data)

    # Export Excel Workbook
    wb_path = ROOT / 'results/result2_v6.xlsx'
    print('Exporting Excel workbook results/result2_v6.xlsx...', flush=True)
    workbook(formal, daily_stats, ev, wb_path)
    verify_export(wb_path, formal, daily_stats, ev)
    print('Workbook verified.', flush=True)

    # Check protected files
    assert before_hashes == fingerprints(protected), 'Protected files altered!'

    # Generate JSON summary
    v6_formal_sum = summarize(formal)
    v1_formal_sum = summarize(baseline[31 * 144:])
    saving = v1_formal_sum['total_cost_yuan'] - v6_formal_sum['total_cost_yuan']

    result = dict(
        version='q2-v6-crossday-dp-hybrid',
        selected_threshold=best_threshold,
        january_candidates=tuning_table,
        formal=v6_formal_sum,
        v1=v1_formal_sum,
        v2b=v2b_summary['formal'],
        savings_vs_v1_yuan=saving,
        savings_vs_v1_percent=saving / v1_formal_sum['total_cost_yuan'] * 100,
        savings_vs_v2b_yuan=v2b_summary['formal']['total_cost_yuan'] - v6_formal_sum['total_cost_yuan'],
        monthly_comparison=monthly,
        improved_days=sum(r['savings_vs_v1_yuan'] > 1e-5 for r in daily_stats),
        worsened_days=sum(r['savings_vs_v1_yuan'] < -1e-5 for r in daily_stats),
        audit=audit_res
    )
    (OUT / 'experiment.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    # Generate Report
    report_lines = [
        '# 第二问 V6 实验报告：跨日 DP 终值函数协调与延迟鲁棒混合调度',
        '',
        '基于次日场景多阶段 DP 成本函数作为日前终端边界，彻底取消 6000 kWh 软惩罚；日内执行基于跨日保留水平与时延门禁。',
        '',
        '## 1. 1月参数优选 (2025-01-15 — 2025-01-31)',
        '',
        '| 候选阈值 / kWh | 计划购电费 / 元 | 紧急购电费 / 元 | 总费用 / 元 | 日末 SOC / kWh |',
        '|---|---:|---:|---:|---:|'
    ]
    for r in tuning_table:
        report_lines.append(
            f"| {r['threshold']} | {r['planned_cost_yuan']:.2f} | {r['emergency_cost_yuan']:.2f} | {r['total_cost_yuan']:.2f} | {r['end_soc']:.2f} |"
        )
    report_lines.extend([
        '',
        f'优选门禁阈值：`{best_threshold:.1f} kWh`。',
        '',
        '## 2. 334 天正式期主对照 (2025-02-01 — 2025-12-31)',
        '',
        '| 指标 | v1 基线 | V2-B (14日场景) | V6 (跨日DP协调) |',
        '|---|---:|---:|---:|',
        f"| 计划购电量 / kWh | {v1_formal_sum['purchase_kwh']:.2f} | {v2b_summary['formal']['purchase_kwh']:.2f} | {v6_formal_sum['purchase_kwh']:.2f} |",
        f"| 紧急购电量 / kWh | {v1_formal_sum['emergency_kwh']:.2f} | {v2b_summary['formal']['emergency_kwh']:.2f} | {v6_formal_sum['emergency_kwh']:.2f} |",
        f"| 计划购电费 / 元 | {v1_formal_sum['planned_cost_yuan']:.2f} | {v2b_summary['formal']['planned_cost_yuan']:.2f} | {v6_formal_sum['planned_cost_yuan']:.2f} |",
        f"| 紧急购电费 / 元 | {v1_formal_sum['emergency_cost_yuan']:.2f} | {v2b_summary['formal']['emergency_cost_yuan']:.2f} | {v6_formal_sum['emergency_cost_yuan']:.2f} |",
        f"| **全年总费用 / 元** | **{v1_formal_sum['total_cost_yuan']:.2f}** | **{v2b_summary['formal']['total_cost_yuan']:.2f}** | **{v6_formal_sum['total_cost_yuan']:.2f}** |",
        f"| 弃用电量 / kWh | {v1_formal_sum['curtailment_kwh']:.2f} | {v2b_summary['formal']['curtailment_kwh']:.2f} | {v6_formal_sum['curtailment_kwh']:.2f} |",
        f"| 充电时紧急电量 / kWh | {v1_formal_sum['emergency_while_charging_kwh']:.2f} | {v2b_summary['formal']['emergency_while_charging_kwh']:.2f} | {v6_formal_sum['emergency_while_charging_kwh']:.2f} |",
        f"| 年末 SOC / kWh | {v1_formal_sum['soc_end_kwh']:.2f} | {v2b_summary['formal']['soc_end_kwh']:.2f} | {v6_formal_sum['soc_end_kwh']:.2f} |",
        '',
        f"相对 v1 节费 **{saving:.2f} 元 ({result['savings_vs_v1_percent']:.4f}%)**；",
        f"相对 V2-B 节费 **{result['savings_vs_v2b_yuan']:.2f} 元**；",
        f"改善天数：{result['improved_days']} 天，恶化天数：{result['worsened_days']} 天。",
        '',
        '## 3. 分月对比',
        '',
        '| 月份 | v1 费用 / 元 | V6 费用 / 元 | 节费 / 元 |',
        '|---|---:|---:|---:|'
    ])
    for m in monthly:
        report_lines.append(f"| {m['month']} | {m['v1_total_cost_yuan']:.2f} | {m['v6_total_cost_yuan']:.2f} | {m['savings_yuan']:.2f} |")

    (ROOT / 'reports/q2_v6_experiment.md').write_text('\n'.join(report_lines), encoding='utf-8')
    print('Report reports/q2_v6_experiment.md generated successfully.', flush=True)


if __name__ == '__main__':
    main()


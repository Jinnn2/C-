"""Reproduce manuscript main Q2 experiment, preserving all older outputs.

Run normally for greedy/DP and their fixed-plan control. --include-mpc also
runs the two deterministic MPC comparisons (substantially more CPU time).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import scipy

from q2_v4 import Archive, procure, reserves, action, mpc_level, ETA, LOWER, UPPER
from solve_q2 import execution_rows, summarize, events, workbook, verify_export
from validate_q2_v4 import model_tests, execution_audit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/q2_v4'


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def read_rows(path):
    text_keys = {'date', 'phase', 'interval_start', 'interval_end', 'interval_label', 'plan_issue_time',
                 'history_available_through', 'control_issue_time', 'last_observation_end',
                 'raw_sample_label_time', 'observation_model'}
    with path.open(encoding='utf-8-sig', newline='') as f:
        return [{k: v if k in text_keys else int(v) if k == 'slot' else float(v) for k, v in r.items()}
                for r in csv.DictReader(f)]


def simulate(ds, purchase, forecast, actual, price, start_soc, levels, mode, value, rho,
             warmup=False, observation_mode='instant', risk_upper=None, risk_lower=None):
    """Only actual[t] is passed to each action; later actuals cannot influence it."""
    charge, discharge = np.zeros(144), np.zeros(144)
    used = np.full(144, LOWER); states = [start_soc]
    if observation_mode in ('delayed_2', 'delayed_3') and risk_upper is None:
        risk_upper = np.zeros(144); risk_lower = np.zeros(144)
    nominal = forecast[:, 0]-forecast[:, 1]
    for t in range(144):
        actual_net = float(actual[t, 0]-actual[t, 1])
        if observation_mode == 'instant':
            current_net = actual_net
        elif observation_mode == 'delayed_1':
            if t == 0:
                current_net = float(nominal[t])
            else:
                previous_net = float(actual[t-1, 0]-actual[t-1, 1])
                current_net = float(nominal[t] + previous_net - nominal[t-1])
        elif observation_mode in ('delayed_2', 'delayed_3'):
            if t == 0:
                current_net = float(nominal[t])
            else:
                previous_net = float(actual[t-1, 0]-actual[t-1, 1])
                current_net = float(nominal[t] + previous_net - nominal[t-1])
        else:
            raise ValueError(f'Unknown observation mode: {observation_mode}')
        if not warmup:
            level = (mpc_level(t, current_net, nominal, purchase, price, value, rho)
                     if mode == 'mpc' else levels[t])
            used[t] = level
            residual = current_net-purchase[t]
            if observation_mode == 'delayed_2':
                upper = current_net + float(risk_upper[t])
                lower = current_net + float(risk_lower[t])
                if upper < purchase[t]:
                    residual = upper-purchase[t]
                elif lower > purchase[t]:
                    residual = lower-purchase[t]
                else:
                    residual = 0.
            elif observation_mode == 'delayed_3':
                # Gate only charging with the conservative upper bound;
                # retain delayed_1's original discharge decision.
                upper = current_net + float(risk_upper[t])
                if current_net <= purchase[t] and upper < purchase[t]:
                    residual = upper-purchase[t]
                elif current_net > purchase[t]:
                    residual = current_net-purchase[t]
                else:
                    residual = 0.
            charge[t], discharge[t] = action(residual, states[-1], level)
        states.append(states[-1]+ETA*charge[t]-discharge[t]/ETA)
    p = dict(purchase=purchase, forecast=forecast, charge=charge, discharge=discharge, soc=np.array(states))
    rows, end = execution_rows(ds, p, actual, price, start_soc, 'warmup' if warmup else 'evaluation')
    for t, r in enumerate(rows):
        last_observation_end = (r['interval_start'] if observation_mode == 'instant' else
                                ('' if t == 0 else rows[t-1]['interval_end']))
        r.update(committed_purchase_kwh=float(purchase[t]),
                 control_issue_time=r['interval_start'], last_observation_end=last_observation_end,
                 raw_sample_label_time=r['interval_end'], observation_model=observation_mode,
                 risk_upper_kwh=(0. if risk_upper is None else float(risk_upper[t])),
                 risk_lower_kwh=(0. if risk_lower is None else float(risk_lower[t])),
                 issued_charge_kwh=float(charge[t]), issued_discharge_kwh=float(discharge[t]),
                 reserve_kwh=float(used[t]))
        # There is no committed day-ahead storage trajectory; these legacy fields
        # are zero placeholders. Actual/issued fields carry the feedback actions.
        r['planned_charge_kwh'] = 0.; r['planned_discharge_kwh'] = 0.
    return rows, end


def run_family(tag, ds, net, forecast, actual, price, states, value, rho, include_mpc, observation_mode, risk_alpha):
    """Independent same-day family; only procure's forecast inputs affect q."""
    plan = procure(net, price, float(np.clip(states[tag], LOWER, UPPER)), value)
    family = [tag]
    if tag == 'greedy_closed':
        family += ['dp_fixed']+(['mpc_fixed'] if include_mpc else [])
    output = {}; saved = []
    nominal = forecast[:, 0]-forecast[:, 1]
    scenario_error = net-nominal
    risk_upper = np.quantile(scenario_error, risk_alpha, axis=0) if observation_mode in ('delayed_2', 'delayed_3') else np.zeros(144)
    risk_lower = np.quantile(scenario_error, 1.-risk_alpha, axis=0) if observation_mode == 'delayed_2' else np.zeros(144)
    for variant in family:
        mode = variant.split('_')[0]
        levels = np.full(144, LOWER)
        if mode == 'dp':
            levels, _, segments = reserves(net, plan['purchase'], price, value)
            saved.extend(dict(date=ds, variant=variant, slot=t+1, reserve_kwh=float(x), max_path_segments=segments)
                         for t, x in enumerate(levels))
        day_rows, end = simulate(ds, plan['purchase'], forecast, actual, price, states[variant], levels, mode, value, rho,
                                 observation_mode=observation_mode, risk_upper=risk_upper, risk_lower=risk_lower)
        output[variant] = (day_rows, end)
    log = dict(date=ds, variant=tag, initial_soc_kwh=states[tag], rho=rho,
               **{k: v for k, v in plan.items() if k != 'purchase'})
    return output, log, saved, plan['purchase']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-mpc', action='store_true')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--observation-mode', choices=('instant', 'delayed_1', 'delayed_2', 'delayed_3'), default='instant')
    parser.add_argument('--risk-alpha', type=float, default=0.90)
    parser.add_argument('--output-root', default='results/q2_v4')
    args = parser.parse_args()
    if not 0.5 < args.risk_alpha < 1.0:
        parser.error('--risk-alpha must be in (0.5, 1)')
    global OUT
    OUT = ROOT / args.output_root
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [p for p in (ROOT/'results').rglob('*') if p.is_file() and OUT not in p.parents
                 and not p.name.startswith('result2_v4')]
    protected += [ROOT/'C题.pdf', ROOT/'第一二问_论文完整稿3.pdf']
    before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    quality = json.loads((ROOT/'reports/data_quality_report.json').read_text(encoding='utf-8'))
    for source in quality['sources']:
        assert hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest() == source['sha256']
    validation = {'models': model_tests()}
    print('Model oracles passed', validation['models'], flush=True)
    db = ROOT/'data/processed/microgrid.sqlite'
    with sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True) as con:
        flat = con.execute('SELECT date,slot,load_kwh,pv_actual_kwh FROM actuals ORDER BY date,slot').fetchall()
        price = np.array([x[0] for x in con.execute('SELECT price_yuan_per_kwh FROM baseline ORDER BY slot')])
    actual = np.array([[x[2], x[3]] for x in flat]).reshape(365, 144, 2)
    dates = [flat[i*144][0] for i in range(365)]
    value = float(price[:30].mean()/ETA)
    tags = ['greedy_closed', 'dp_fixed', 'dp_closed']
    if args.include_mpc:
        tags += ['mpc_fixed', 'mpc_closed']
    rows = {tag: [] for tag in tags}; states = {tag: 6000. for tag in tags}
    logs = []; traces = []; forecasts = []; archived = []; levels_saved = []; contracts = []
    archive = Archive(); start = time.perf_counter()
    pool = ProcessPoolExecutor(max_workers=max(1, args.workers))
    for day, ds in enumerate(dates):
        forecast, net, ids, load_ids, pv_ids = archive.issue()
        if forecast is not None:
            forecasts.append(forecast.copy())
            traces.append(dict(date=ds, issue_time=ds+'T00:00',
                               load_sources=json.dumps([dates[i] for i in load_ids]),
                               pv_sources=json.dumps([dates[i] for i in pv_ids]),
                               residual_sources=json.dumps([dates[i] for i in ids]),
                               latest_residual_day=dates[ids[-1]] if ids else '', scenario_count=len(net)))
            assert all(i < day for i in ids+load_ids+pv_ids)
        if day < 31:
            f = np.zeros((144, 2)) if forecast is None else forecast
            warm, end = simulate(ds, np.zeros(144), f, actual[day], price, 6000., np.full(144, LOWER),
                                 'greedy', value, 0., warmup=True, observation_mode=args.observation_mode)
            for tag in tags:
                rows[tag].extend(warm); states[tag] = end
        else:
            assert len(net) == 30
            rho = archive.rho()
            # Families are independent; procurement inside each worker accepts
            # only net scenarios, price and initial SOC, never actual observations.
            families = ['greedy_closed', 'dp_closed']+(['mpc_closed'] if args.include_mpc else [])
            futures = [pool.submit(run_family, tag, ds, net, forecast, actual[day], price,
                                   states.copy(), value, rho, args.include_mpc, args.observation_mode, args.risk_alpha) for tag in families]
            purchases = {}
            for future in futures:
                output, log, saved, q = future.result()
                logs.append(log); levels_saved.extend(saved)
                for tag, (day_rows, end) in output.items():
                    rows[tag].extend(day_rows); states[tag] = end; purchases[tag] = q
            contracts.append(np.array([purchases[tag] for tag in tags]))
        archive.observe(actual[day])
        if (day+1) % 15 == 0 or day == 364:
            print(f'{ds} completed; elapsed {time.perf_counter()-start:.1f}s', flush=True)
    pool.shutdown()
    results = {}; comparisons = []; all_daily = {}
    for tag in tags:
        folder = OUT/tag; folder.mkdir(exist_ok=True)
        validation[tag] = execution_audit(rows[tag], observation_mode=args.observation_mode)
        formal = rows[tag][31*144:]
        daily = [dict(date=dates[31+i], **summarize(formal[i*144:(i+1)*144])) for i in range(334)]
        ev = events(formal); all_daily[tag] = daily
        for name, data in [('execution', rows[tag]), ('daily_summary', daily), ('emergency_events', ev)]:
            write_csv(folder/f'{name}.csv', data)
        validation[tag+'_persisted'] = execution_audit(read_rows(folder/'execution.csv'), observation_mode=args.observation_mode)
        path = ROOT/f'results/result2_v4_{tag}.xlsx'
        workbook(formal, daily, ev, path); verify_export(path, formal, daily, ev)
        validation[tag+'_workbook'] = True
        results[tag] = dict(formal=summarize(formal), warmup=summarize(rows[tag][:31*144]),
                            emergency_events=len(ev),
                            end_soc_min=min(x['soc_end_kwh'] for x in daily), end_soc_max=max(x['soc_end_kwh'] for x in daily))
    # Fixed-plan replay must preserve every baseline purchase exactly.
    for tag in [t for t in tags if t.endswith('_fixed')]:
        assert all(a['purchase_kwh'] == b['purchase_kwh'] for a, b in zip(rows[tag], rows['greedy_closed']))
    validation['fixed_contracts_identical'] = True
    broken = [dict(x) for x in rows['dp_closed'][-144:]]; broken[0]['purchase_kwh'] += 1
    try:
        execution_audit(broken)
    except (ValueError, AssertionError):
        validation['injected_balance_error_rejected'] = True
    else:
        raise ValueError('Auditor accepted injected defect')
    # Changing unseen future actuals cannot change the already-executed prefix.
    f = np.array([[x['load_forecast_kwh'], x['pv_forecast_kwh']] for x in rows['dp_closed'][31*144:32*144]])
    q = np.array([x['purchase_kwh'] for x in rows['dp_closed'][31*144:32*144]])
    level = np.array([x['reserve_kwh'] for x in rows['dp_closed'][31*144:32*144]])
    changed = actual[31].copy(); changed[60:] = 1e6
    baseline_actual = actual[31].copy()
    if args.observation_mode in ('delayed_2', 'delayed_3'):
        baseline_rows, _ = simulate(dates[31], q, f, baseline_actual, price, 6000., level, 'dp', value, 0.,
                                    observation_mode=args.observation_mode)
    else:
        baseline_rows = rows['dp_closed'][31*144:31*144+144]
    changed_rows, _ = simulate(dates[31], q, f, changed, price, 6000., level, 'dp', value, 0.,
                               observation_mode=args.observation_mode)
    assert all(a['charge_kwh'] == b['charge_kwh'] and a['discharge_kwh'] == b['discharge_kwh']
               for a, b in zip(changed_rows[:60], baseline_rows[:60]))
    validation['future_actual_suffix_invariance'] = True
    for tag, paper in [('greedy_closed', 14174236.10), ('dp_fixed', 14022321.93), ('dp_closed', 14022279.33)]:
        measured = results[tag]['formal']['total_cost_yuan']
        comparisons.append(dict(variant=tag, paper_total_yuan=paper, reproduced_total_yuan=measured, difference_yuan=measured-paper))
    metrics = {}
    formal_rows = rows['dp_closed'][31*144:]
    for name, terms in [('load', [('load_forecast_kwh', 1), ('load_kwh', -1)]),
                        ('pv', [('pv_forecast_kwh', 1), ('pv_actual_kwh', -1)]),
                        ('net', [('load_forecast_kwh', 1), ('pv_forecast_kwh', -1), ('load_kwh', -1), ('pv_actual_kwh', 1)])]:
        metrics[name+'_mae_kw'] = float(np.mean([abs(sum(r[k]*s for k, s in terms))*6 for r in formal_rows]))
    write_csv(OUT/'paper_comparison.csv', comparisons); write_csv(OUT/'solver_log.csv', logs)
    write_csv(OUT/'forecast_trace.csv', traces); write_csv(OUT/'reserve_levels.csv', levels_saved)
    write_csv(OUT/'specified_dates.csv', [dict(variant=tag, **x) for tag, daily in all_daily.items() for x in daily
                                        if x['date'] in ('2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21')])
    np.savez_compressed(OUT/'forecast_archive.npz', forecast=np.array(forecasts),
                        residual=np.array([x[1] for x in archive.errors]), residual_day=np.array([x[0] for x in archive.errors]))
    np.savez_compressed(OUT/'committed_contracts.npz', variants=np.array(tags), dates=np.array(dates[31:]), purchase=np.array(contracts))
    for rel, expected in before.items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == expected, rel
    for source in quality['sources']:
        assert hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest() == source['sha256']
    validation['older_results_and_sources_unchanged'] = True
    experiment = dict(version='v4-paper-reproduction', value_yuan_per_internal_kwh=value,
                      risk_alpha=args.risk_alpha,
                      results=results, forecast_metrics=metrics, validation=validation,
                      paper_comparison=comparisons, seconds=time.perf_counter()-start, scipy_version=scipy.__version__,
                      manuscript_sha256=before['第一二问_论文完整稿3.pdf'], source_hashes=quality['sources'],
                      observation_mode=args.observation_mode,
                      current_slot_assumption=('Instantaneously observable constant power proxy.' if args.observation_mode == 'instant'
                                               else 'One-slot delayed observation; current net is forecast plus previous-slot residual.'),
                      scope='Main paper policies; candidate mixing and online kernel weighting excluded: kernel/bandwidth/scaling unspecified.')
    (OUT/'experiment.json').write_text(json.dumps(experiment, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 第二问 V4：论文主方案复现', '',
             '来源：第一二问_论文完整稿3.pdf，第8—18页。独立实现，未获得作者代码；同价LP多解可能影响轨迹。', '',
             '## 实现与假设', '',
             '- 35天内同星期等权负载预测，少于两天回退最近7天；光伏最近7天均值。30条完整历史配对残差，分别截断负负载/光伏。',
             '- 共同日前购电、逐情景充放电追索；日末线性残值，不设每日目标。实际SOC跨日传递。',
             f'- 残值系数 {value:.12f} 元/内部kWh；1月储能待机、零计划、紧急补缺，2月从6000开始。',
             ('- 解析控制和DP控制使用当期实际功率即时观测，禁止紧急购电充电。' if args.observation_mode == 'instant'
              else ('- 解析控制和DP控制使用一时段延迟观测：当前预测加上一段已结束观测残差；延迟下允许并统计充电时紧急购电。' if args.observation_mode == 'delayed_1'
                    else (f'- delayed_2 使用一时段延迟观测与历史场景 {args.risk_alpha:.2f} 分位数边界；仅在保守富余/缺口条件下动作。' if args.observation_mode == 'delayed_2'
                          else f'- delayed_3 使用一时段延迟观测；仅对充电使用历史场景 {args.risk_alpha:.2f} 分位数上界，放电沿用 delayed_1。'))),
             '- DP采用连续凸折线斜率合并及最大最优端点；路径平均是乐观未来价值近似，不是精确随机DP。',
             '- fixed沿用解析方案全年合同；closed每天按各自实际SOC重订。MPC用无截距相邻残差回归（不跨午夜），论文对此未明确。', '',
             '## 正式期费用', '', '| 方案 | 总费/元 | 紧急费/元 | 年末电量/kWh |', '|---|---:|---:|---:|']
    for tag, result in results.items():
        s = result['formal']; lines.append(f"| {tag} | {s['total_cost_yuan']:.6f} | {s['emergency_cost_yuan']:.6f} | {s['soc_end_kwh']:.6f} |")
    lines += ['', '## 与论文对照', '', '| 方案 | 论文费用/元 | 复现减论文/元 |', '|---|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['variant']} | {r['paper_total_yuan']:.2f} | {r['difference_yuan']:.6f} |")
    lines += ['', f'预测MAE/kW：{metrics}', '',
              '## 验证与边界', '',
              '- 随机多时域路径DP对独立LP、平均价值阈值和可达区间投影、情景LP对严格MILP均通过。',
              '- 全年物理、互斥、SOC连续、观测规则、费用、CSV回读、Excel334天逐项回读、固定合同一致通过。',
              '- 预测残差日期早于发布日；未来实际后缀修改不影响已执行前缀；注入能量错误被拒绝。',
              '- 所有旧结果和原始输入哈希不变；既有审计器中planned_charge/discharge在V4为零占位，issued/actual字段才是执行动作。',
              '- 主方案已复现；论文第14页候选混合和在线核权重属于后续增强，核函数、带宽、标准化及收益基线未充分给出，本轮不声称复现。',
              '- 论文价格/功率时间解释和0.9单程效率继续作为建模假设；未声称策略全局最优。',
              '- 运行：python scripts/solve_q2_v4.py --include-mpc；主要工作簿为results/result2_v4_dp_closed.xlsx。', '']
    report_name = {'instant': 'q2_v4_experiment.md',
                   'delayed_1': 'q2_v4_delayed1_experiment.md',
                   'delayed_2': 'q2_v4_delayed2_experiment.md',
                   'delayed_3': 'q2_v4_delayed3_experiment.md'}[args.observation_mode]
    report_path = ROOT / 'reports' / report_name
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(results={k: v['formal']['total_cost_yuan'] for k, v in results.items()},
                          paper_comparison=comparisons, seconds=experiment['seconds']), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

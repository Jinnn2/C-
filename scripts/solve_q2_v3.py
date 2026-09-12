"""V3 R0/R1/R2: target-free 24/48/72h backtests with one shared warm-up."""
import csv
from dataclasses import asdict
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import scipy

from q2_model import idle_plan
from q2_v3 import V3Config, ForecastArchive, forecast, optimize, first_day
from solve_q2 import execution_rows, summarize, events, workbook, verify_export
from solve_q2_v2a import fingerprints, read_execution
from validate_q2 import audit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/q2_v3'


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def check_trace(rows):
    for r in rows:
        issue = datetime.fromisoformat(r['issue_time'])
        target = datetime.fromisoformat(r['target_start'])
        assert target == issue+timedelta(days=int(r['lead_day'])-1)
        assert datetime.fromisoformat(r['latest_actual_end']) <= issue
        for value in r['residual_issue_dates'].split('|'):
            if value:
                assert datetime.fromisoformat(value)+timedelta(days=3) <= issue
        for value in (r['load_source_dates']+'|'+r['pv_source_dates']).split('|'):
            if value:
                assert datetime.fromisoformat(value)+timedelta(days=1) <= issue
        assert int(r['residual_sample_count']) == len([v for v in r['residual_issue_dates'].split('|') if v])
    return True


def model_tests():
    rng = np.random.default_rng(31)
    errors = []
    for n in (6, 24, 144, 288):
        f = np.column_stack((rng.uniform(100, 900, n), rng.uniform(0, 450, n)))
        net = f[:, 0]-f[:, 1]+rng.normal(0, 70, (3, n))
        price = rng.uniform(.3, 1.5, n)
        lp = optimize(f, net, price, 4500)
        exact = optimize(f, net, price, 4500, strict=True)
        errors.append(abs(lp['objective']-exact['objective']))
    assert max(errors) < 1e-5
    # Synthetic late cheap slot followed by expensive morning: do not empty at midnight.
    f = np.zeros((288, 2)); f[144:156, 0] = 700
    price = np.ones(288); price[:144] = .2
    full = optimize(f, f[:, 0][None], price, 1200)
    short = optimize(f[:144], f[:144, 0][None], price[:144], 1200)
    next_day = optimize(f[144:], f[144:, 0][None], price[144:], short['soc'][-1])
    saving = short['objective']+next_day['objective']-full['objective']
    assert saving > 1000 and full['soc'][144] > 1200+1000
    assert abs(full['soc'][144]-6000) > 1
    # Day-1 predictor equivalence, plus distinct lead-day weekday indexing.
    from q2_model import Config, predict
    history = rng.uniform(0, 1000, (35, 144, 2))
    f, sources = forecast(history)
    assert np.allclose(f[0], predict(history, Config()))
    assert sources[1]['load_source_days'] == [29, 22, 15, 8]
    archive = ForecastArchive()
    for d in range(12):
        issued, net, ids, sources = archive.issue()
        assert all(i+3 <= d for i in ids)
        if d == 9: assert not ids
        if d == 10: assert ids == [7]
        archive.observe(history[d])
    return dict(lp_vs_strict_milp_max_error_yuan=max(errors),
                cross_midnight_saving_yuan=saving, synthetic_midnight_soc_kwh=float(full['soc'][144]),
                first_lead_matches_frozen_predictor=True, residual_maturity_test=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = V3Config()
    protected = [p for folder in ('results/q1', 'results/q2', 'results/q2_v2a', 'results/q2_v2b', 'results/q2_v2c')
                 for p in (ROOT/folder).rglob('*') if p.is_file()]
    protected += [p for p in (ROOT/'results').glob('*.xlsx') if '_v3' not in p.name]
    protected += list((ROOT/'reports').glob('q*_experiment.md'))
    protected = [p for p in protected if 'v3' not in p.name]
    before = fingerprints(protected)
    quality = json.loads((ROOT/'reports/data_quality_report.json').read_text(encoding='utf-8'))
    for r in quality['sources']:
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest() == r['sha256']
    db = ROOT/'data/processed/microgrid.sqlite'
    database_hash = hashlib.sha256(db.read_bytes()).hexdigest()
    old = json.loads((ROOT/'results/q2_v2b/experiment.json').read_text(encoding='utf-8'))
    assert database_hash == old['database_sha256']
    with sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True) as c:
        flat = c.execute('SELECT date,slot,load_kwh,pv_actual_kwh FROM actuals ORDER BY date,slot').fetchall()
        price = np.array([r[0] for r in c.execute('SELECT price_yuan_per_kwh FROM baseline ORDER BY slot')])
    actual = np.array([[r[2], r[3]] for r in flat]).reshape(365, 144, 2)
    dates = [flat[d*144][0] for d in range(365)]
    epoch = datetime.fromisoformat(dates[0])
    def date_at(index): return (epoch+timedelta(days=int(index))).date().isoformat()

    validation = model_tests()
    print('V3 model checks passed', flush=True)
    archive = ForecastArchive(config)
    warmup = []; formal = {h: [] for h in (1, 2, 3)}
    soc = {h: 6000. for h in (1, 2, 3)}
    logs = {h: [] for h in (1, 2, 3)}
    traces = []; forecasts = []; common = []
    for day, ds in enumerate(dates):
        f, net, ids, sources = archive.issue()
        if f is not None:
            forecasts.append(f.copy())
            for lead, source in enumerate(sources, 1):
                traces.append(dict(issue_time=ds+'T00:00', latest_actual_end=ds+'T00:00',
                                   lead_day=lead, target_start=date_at(day+lead-1)+'T00:00',
                                   load_source_dates='|'.join(date_at(i) for i in source['load_source_days']),
                                   pv_source_dates='|'.join(date_at(i) for i in source['pv_source_days']),
                                   residual_issue_dates='|'.join(date_at(i) for i in ids),
                                   residual_sample_count=len(ids), optimizer_scenario_count=len(net),
                                   point_fallback=int(not ids)))
        if day == 0:
            p = idle_plan(soc[2])
            rows, end = execution_rows(ds, p, actual[day], price, soc[2], 'warmup')
            warmup.extend(rows)
            for h in soc: soc[h] = end
        elif day < 31:
            h = config.warmup_horizon_days
            p = optimize(f[:h].reshape(-1, 2), net[:, :h*144], np.tile(price, h), soc[2])
            rows, end = execution_rows(ds, first_day(p), actual[day], price, soc[2], 'warmup')
            warmup.extend(rows)
            for key in soc: soc[key] = end
            logs[2].append(dict(date=ds, phase='warmup', horizon_hours=h*24,
                               horizon_objective_yuan=p['objective'], horizon_planned_cost_yuan=p['planned_cost'],
                               first_day_planned_cost_yuan=float(price@p['purchase'][:144]),
                               horizon_final_soc_kwh=float(p['soc'][-1]), first_day_end_soc_kwh=end,
                               seconds=p['seconds'], objective_error=p['objective_error']))
        else:
            start_r1 = soc[2]
            plans = {}
            # Solve all decisions before revealing today's observations to the executor.
            for h in (1, 2, 3):
                plans[h] = optimize(f[:h].reshape(-1, 2), net[:, :h*144], np.tile(price, h), soc[h])
            # Isolate horizon effect at identical *current* SOC, in addition to annual policy paths.
            p72 = plans[3] if abs(soc[3]-start_r1) < 1e-9 else optimize(f.reshape(-1, 2), net, np.tile(price, 3), start_r1)
            p48 = plans[2]
            common.append(dict(date=ds, common_initial_soc_kwh=start_r1,
                               purchase_l1_difference_kwh=float(np.abs(p72['purchase'][:144]-p48['purchase'][:144]).sum()),
                               charge_l1_difference_kwh=float(np.abs(p72['charge'][:144]-p48['charge'][:144]).sum()),
                               discharge_l1_difference_kwh=float(np.abs(p72['discharge'][:144]-p48['discharge'][:144]).sum()),
                               r1_end_soc_kwh=float(p48['soc'][144]), r2_end_soc_kwh=float(p72['soc'][144]),
                               end_soc_difference_kwh=float(p72['soc'][144]-p48['soc'][144])))
            for h, p in plans.items():
                rows, soc[h] = execution_rows(ds, first_day(p), actual[day], price, soc[h], 'evaluation')
                formal[h].extend(rows)
                logs[h].append(dict(date=ds, phase='evaluation', horizon_hours=h*24,
                                    horizon_objective_yuan=p['objective'], horizon_planned_cost_yuan=p['planned_cost'],
                                    first_day_planned_cost_yuan=float(price@p['purchase'][:144]),
                                    horizon_final_soc_kwh=float(p['soc'][-1]), first_day_end_soc_kwh=soc[h],
                                    seconds=p['seconds'], objective_error=p['objective_error']))
        archive.observe(actual[day])
        if (day+1) % 30 == 0:
            print('V3 executed through', ds, flush=True)

    validation['forecast_trace'] = check_trace(traces)
    bad = [dict(traces[-1])]; bad[0]['residual_issue_dates'] = dates[-1]
    try: check_trace(bad)
    except AssertionError: validation['immature_residual_rejected'] = True
    else: raise ValueError('Failed trace defect detection')
    assert abs(warmup[0]['soc_start_kwh']-6000) < 1e-8
    validation['true_initial_condition'] = True
    write_csv(OUT/'forecast_trace.csv', traces)
    write_csv(OUT/'common_state_horizon_comparison.csv', common)
    np.savez_compressed(OUT/'forecast_archive.npz', issue_dates=np.array(dates[1:]),
                        forecast_kwh=np.asarray(forecasts),
                        residual_issue_days=np.array([i for i, _ in archive.residuals]),
                        residual_kwh=np.array([e for _, e in archive.residuals]))
    with np.load(OUT/'forecast_archive.npz') as saved:
        assert list(saved['issue_dates']) == dates[1:]
        for i, issued in enumerate(saved['forecast_kwh'], 1):
            independently_rebuilt, _ = forecast(actual[:i].copy())
            assert np.array_equal(issued, independently_rebuilt)
        for i, error in zip(saved['residual_issue_days'], saved['residual_kwh']):
            assert i+3 <= len(actual)
            expected = actual[i:i+3]-saved['forecast_kwh'][i-1]
            assert np.array_equal(error, expected)
    validation['persisted_forecasts_and_residuals_reconstructed'] = True
    results = {}; daily_all = {}; monthly = []
    for h, rows in formal.items():
        tag = f'r{h-1}_{h*24}h'; folder = OUT/tag; folder.mkdir(exist_ok=True)
        full = warmup+rows
        validation[tag] = audit(full, require_boundaries=False)
        for r in rows:
            assert r['plan_issue_time'] == r['date']+'T00:00'
        assert abs(rows[0]['soc_start_kwh']-warmup[-1]['soc_end_kwh']) < 1e-8
        daily = [dict(date=dates[31+i], **summarize(rows[i*144:(i+1)*144])) for i in range(334)]
        daily_all[h] = daily
        ev = events(rows)
        for filename, data in [('execution', full), ('daily_summary', daily), ('emergency_events', ev), ('solver_log', logs[h])]:
            write_csv(folder/f'{filename}.csv', data)
        validation[tag+'_csv'] = audit(read_execution(folder/'execution.csv'), require_boundaries=False)
        path = ROOT/f'results/result2_v3_r{h-1}.xlsx'
        workbook(rows, daily, ev, path); verify_export(path, rows, daily, ev)
        validation[tag+'_workbook'] = True
        # Horizon objectives include provisional future purchases; settled cost is first-day only.
        charged = sum(r['first_day_planned_cost_yuan'] for r in logs[h] if r['phase'] == 'evaluation')
        assert abs(charged-summarize(rows)['planned_cost_yuan']) < 1e-5
        results[tag] = dict(formal=summarize(rows), annual=summarize(full),
                            daily_end_soc_min_kwh=min(r['soc_end_kwh'] for r in daily),
                            daily_end_soc_max_kwh=max(r['soc_end_kwh'] for r in daily),
                            days_ending_at_6000=sum(abs(r['soc_end_kwh']-6000) < 1e-5 for r in daily),
                            days_ending_at_lower_bound=sum(abs(r['soc_end_kwh']-1200) < 1e-5 for r in daily),
                            savings_vs_restricted_v2b_yuan=old['formal']['total_cost_yuan']-summarize(rows)['total_cost_yuan'],
                            solver_seconds=sum(r['seconds'] for r in logs[h]))
    for month in sorted({ds[:7] for ds in dates[31:]}):
        row = dict(month=month)
        for h in formal:
            row[f'r{h-1}_cost_yuan'] = summarize([r for r in formal[h] if r['date'].startswith(month)])['total_cost_yuan']
        monthly.append(row)
    write_csv(OUT/'monthly_comparison.csv', monthly)
    # Auxiliary energy valuation: same usable-energy convention and price for every strategy.
    # Zero plus low/high tariff replacement costs; not actual revenue or a year-end transaction.
    valuations = [0., .9*float(price.min()), .9*float(price.max())]
    residual_value = []
    for value in valuations:
        for tag, result in results.items():
            r = result['formal']
            residual_value.append(dict(variant=tag, value_yuan_per_stored_kwh=value,
                                      terminal_soc_kwh=r['soc_end_kwh'],
                                      auxiliary_cost_yuan=r['total_cost_yuan']-value*(r['soc_end_kwh']-1200)))
    write_csv(OUT/'terminal_value_sensitivity.csv', residual_value)
    assert before == fingerprints(protected)
    for r in quality['sources']:
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest() == r['sha256']
    validation['original_and_previous_artifacts_unchanged'] = True
    validation['only_committed_first_day_billed'] = True
    validation['common_warmup_and_continuous_soc'] = True
    comparison = dict(common_state_days=334,
                      max_abs_day_end_soc_difference_kwh=max(abs(r['end_soc_difference_kwh']) for r in common),
                      mean_purchase_l1_difference_kwh=float(np.mean([r['purchase_l1_difference_kwh'] for r in common])),
                      max_purchase_l1_difference_kwh=max(r['purchase_l1_difference_kwh'] for r in common),
                      days_different_end_soc=sum(abs(r['end_soc_difference_kwh']) > 1e-5 for r in common),
                      r2_minus_r1_actual_cost_yuan=results['r2_72h']['formal']['total_cost_yuan']-results['r1_48h']['formal']['total_cost_yuan'])
    experiment = dict(version='q2-v3-cross-day-r0-r1-r2', config=asdict(config),
                      warmup=summarize(warmup), results=results, horizon_comparison=comparison,
                      terminal_value_sensitivity=residual_value, validation=validation,
                      database_sha256=database_hash, source_hashes=quality['sources'],
                      previous_artifact_hashes=before, scipy_version=scipy.__version__,
                      scope='Fixed daily storage controls; R3 intraday feedback remains a separate next experiment')
    (OUT/'experiment.json').write_text(json.dumps(experiment, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 第二问 V3 跨日优化实验', '',
             '已完成 R0/R1/R2 全年顺序回测。48小时是预先指定的主候选，72小时用于视野敏感性检查；本轮不重新选参。', '',
             '## 模型与信息边界', '',
             '- 删除每日6000目标、SOC偏差惩罚、1月31日/12月31日硬终点。只有1月1日初始6000与1200—10800物理界限。',
             '- 三种策略共用48小时无目标1月预热；首日没有历史时零计划购电、储能闲置，需求由紧急供电平衡。预热费另列。',
             '- 负载按目标星期选取截止前最多4个历史周、权重0.7递减，光伏取过去7日均值；不使用附件3、4。',
             '- 3日完整误差路径在结束后才入库，且其发布时至少有7日历史；取最近14条成熟路径，三种视野用相同样本前缀。',
             '- 日前优化未来24/48/72小时，只有首日购电成为合同并结算，未来日为暂定计划；次日以连续SOC重算。',
             '- 12月31日仍预测2026年初；没有2026实际观测、虚构结算或年末强制清仓。',
             '- 每日充放电动作本轮固定，共享场景控制是近似；每小时反馈R3尚未纳入本轮。', '',
             '## 费用与状态', '',
             f'共同预热费用 {summarize(warmup)["total_cost_yuan"]:.6f} 元；自然形成的2月1日SOC {warmup[-1]["soc_end_kwh"]:.6f} kWh。', '',
             '| 方案 | 2—12月总费/元 | 紧急费/元 | 弃用/kWh | 年末SOC/kWh | 日末SOC范围/kWh |',
             '|---|---:|---:|---:|---:|---:|']
    for tag, result in results.items():
        r = result['formal']
        lines.append(f'| {tag} | {r["total_cost_yuan"]:.6f} | {r["emergency_cost_yuan"]:.6f} | {r["curtailment_kwh"]:.3f} | {r["soc_end_kwh"]:.6f} | {result["daily_end_soc_min_kwh"]:.3f}—{result["daily_end_soc_max_kwh"]:.3f} |')
    lines += ['', f'受限V2-B原费用 {old["formal"]["total_cost_yuan"]:.6f} 元。旧版与V3同时存在SOC约束、误差样本成熟口径及预热状态差异，因此节费不能全部归因于某一算法变化。', '',
              f'48小时相对24小时实际节费 {results["r0_24h"]["formal"]["total_cost_yuan"]-results["r1_48h"]["formal"]["total_cost_yuan"]:.6f} 元，但比旧V2-B多支出 {-results["r1_48h"]["savings_vs_restricted_v2b_yuan"]:.6f} 元。本轮实现了题意边界纠正，不能宣称成本已优于旧版。',
              f'24小时方案有 {results["r0_24h"]["days_ending_at_lower_bound"]} 天在日末耗到下限；48小时有 {results["r1_48h"]["days_ending_at_lower_bound"]} 天。这是有限视野效果的诊断，而非人为指定日末状态。', '',
              '## 48/72小时敏感性', '',
              f'- 独立全年策略费用差（72h−48h）：{comparison["r2_minus_r1_actual_cost_yuan"]:.6f} 元。',
              f'- 在逐日相同R1初始SOC、相同预测和样本下，首日购电逐段绝对差之和均值 {comparison["mean_purchase_l1_difference_kwh"]:.6f} kWh，最大 {comparison["max_purchase_l1_difference_kwh"]:.6f} kWh。',
              f'- 共同状态下日末SOC差最大 {comparison["max_abs_day_end_soc_difference_kwh"]:.6f} kWh，有 {comparison["days_different_end_soc"]} 天不同。',
              '- 计划动作存在多个同价最优解，动作差本身不证明经济上显著；需结合SOC和实际费用判断。48/72小时相近也不证明无限期最优。', '',
              '## 终端能量评价', '',
              '- 实际费用不扣残值。辅助表以高于1200的储能量，分别按0、0.9×最低电价、0.9×最高电价估值，统一反映后续放电替代计划购电的价格范围；不把它视为实际卖电收入。',
              '- 该线性残值仅是敏感性包络，并非已求得真实续运行成本；若方案年末SOC相同，则这一修正不改变它们的费用差。详见terminal_value_sensitivity.csv。', '',
              '## 验证及复现', '',
              '- 6/24/144/288时段随机实例的LP恢复与严格MILP目标一致；跨午夜合成样例验证跨日储电获益且无需6000日末。',
              '- 全年物理平衡、充放互斥、SOC连续、计划不变、费用重算、CSV与334天Excel逐项回读通过。只检查真实初始状态，不要求期末相等。',
              '- 预测来源和残差成熟日期检查通过，并注入未成熟残差验证拒绝；旧版本和原始文件哈希不变。',
              '- LP互斥恢复依赖免费弃用、无循环奖励、外部充放电各90%；更换这些假设后需重新证明。',
              '- 本项目已多轮使用历史数据开发；这是历史回测，不称为未触碰测试集。保留原有右端点和效率计量假设。',
              '- 后续先诊断统一72小时成熟样本对近期误差响应的影响，再推进R3日内反馈；未经受控实验，不将本轮旧版差额归咎于单个因素。',
              '- 执行 `python scripts/solve_q2_v3.py`；输出results/q2_v3/及result2_v3_r0/r1/r2.xlsx。r1是本轮主候选工作簿。',
              '- forecast_archive.npz保存逐发布时刻三日预测和已成熟残差；forecast_trace.csv记录来源日期，solver_log.csv区分视野目标与首日计划费。', '']
    (ROOT/'reports/q2_v3_experiment.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(results=results, comparison=comparison), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

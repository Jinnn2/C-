"""V2-A experiment: frozen Q2 v1 policy with January-selected forecast bias correction."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import scipy

from q2_model import Config, execute, idle_plan, plan
from q2_bias import BiasConfig, BiasForecaster, candidates
from solve_q2 import execution_rows, summarize, events, workbook, verify_export, label
from validate_q2 import audit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/q2_v2a'
STRINGS={'date','phase','interval_start','interval_end','interval_label','plan_issue_time','history_available_through'}


def read_execution(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return [{k:v if k in STRINGS else int(v) if k=='slot' else float(v) for k,v in r.items()} for r in csv.DictReader(f)]


def write_csv(name,rows):
    with (OUT/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def fingerprints(paths):
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_bias(base_config):
    f=BiasForecaster(base_config)
    rejected=False
    try:f.observe(np.ones((144,2)))
    except ValueError:rejected=True
    assert rejected
    # A synthetic day-8 residual: constant load shift, daylight-only PV shift.
    actual=np.zeros((144,2));actual[:,0]=100;actual[48:96,1]=50
    for _ in range(7):f.forecast(BiasConfig());f.observe(actual)
    _,info=f.forecast(BiasConfig())
    shifted=actual.copy();shifted[:,0]+=12;shifted[48:96,1]+=6
    f.observe(shifted);shifted[:]=-999  # Mutating caller's array cannot change stored history.
    corrected,info=f.forecast(BiasConfig(3,.5,1.))
    assert np.allclose(corrected[:,0]-info['base'][:,0],6)
    assert np.allclose(corrected[48:96,1]-info['base'][48:96,1],6)
    assert np.all(corrected[info['base'][:,1]==0,1]==0)
    assert info['last_residual_day']==7 and info['residual_days_used']==1
    rejected=False
    try:f.forecast(BiasConfig())
    except ValueError:rejected=True
    assert rejected
    return True


def select(january,price,base_config):
    table=[]
    for index,bias_config in enumerate(candidates()):
        forecaster=BiasForecaster(base_config);soc=6000.;total=0.;emergency=0.
        errors=[];planned_cost=0.;emergency_cost=0.
        for d,actual in enumerate(january):
            forecast,info=forecaster.forecast(bias_config)
            if d>=14:
                p=plan(forecast,price,soc,base_config,final_soc=6000 if d==30 else None)
                e,_,cost=execute(p,actual,price)
                total+=float(cost.sum());emergency+=float(e.sum())
                planned_cost+=float(np.dot(price,p['purchase']))
                emergency_cost+=float(np.dot(5*price,e))
                errors.append((actual-forecast)*6)
                soc+=float(.9*p['charge'].sum()-p['discharge'].sum()/.9)
            forecaster.observe(actual)
        assert abs(soc-6000)<1e-5
        error=np.concatenate(errors)
        table.append(dict(candidate_id=index,**bias_config.record(),validation_start='2025-01-15',
                          validation_end='2025-01-31',total_cost_yuan=total,emergency_kwh=emergency,
                          planned_cost_yuan=planned_cost,emergency_cost_yuan=emergency_cost,
                          load_mae_kw=float(np.abs(error[:,0]).mean()),pv_mae_kw=float(np.abs(error[:,1]).mean()),
                          net_mae_kw=float(np.abs(error[:,0]-error[:,1]).mean()),
                          net_bias_kw=float((error[:,0]-error[:,1]).mean())))
    best=min(table,key=lambda r:(r['total_cost_yuan'],r['candidate_id']))
    return BiasConfig(**{k:best[k] for k in BiasConfig().record()}),table,best['candidate_id']


def metrics(rows):
    n=len(rows)
    load=np.array([r['load_kwh']-r['load_forecast_kwh'] for r in rows])*6
    pv=np.array([r['pv_actual_kwh']-r['pv_forecast_kwh'] for r in rows])*6
    return dict(load_mae_kw=float(np.abs(load).mean()),pv_mae_kw=float(np.abs(pv).mean()),
                net_mae_kw=float(np.abs(load-pv).mean()),load_bias_kw=float(load.mean()),
                pv_bias_kw=float(pv.mean()),net_bias_kw=float((load-pv).mean()))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    v1=json.loads((ROOT/'results/q2/experiment.json').read_text(encoding='utf-8'))
    config=Config(**v1['selected_config'])
    quality=json.loads((ROOT/'reports/data_quality_report.json').read_text(encoding='utf-8'))
    assert quality['status']=='PASS_WITH_WARNINGS'
    for r in quality['sources']:assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
    protected=[p for p in (ROOT/'results/q2').iterdir() if p.is_file()]+[ROOT/'results/result2.xlsx',ROOT/'reports/q2_experiment.md']
    before=fingerprints(protected)
    db=ROOT/'data/processed/microgrid.sqlite'
    assert hashlib.sha256(db.read_bytes()).hexdigest()==v1['database_sha256'],'v1 input database changed'
    with sqlite3.connect(f'file:{db.as_posix()}?mode=ro',uri=True) as c:
        flat=c.execute('SELECT date,slot,load_kwh,pv_actual_kwh FROM actuals ORDER BY date,slot').fetchall()
        price=np.array([r[0] for r in c.execute('SELECT price_yuan_per_kwh FROM baseline ORDER BY slot')])
    actual=np.array([[r[2],r[3]] for r in flat]).reshape(365,144,2)
    dates=[flat[d*144][0] for d in range(365)]
    baseline=read_execution(ROOT/'results/q2/execution.csv')
    assert len(baseline)==365*144
    validation={'bias_unit_checks':test_bias(config)}
    bias_config,selection,selected_id=select(actual[:31].copy(),price,config)
    with (ROOT/'results/q2/january_validation.csv').open(encoding='utf-8-sig',newline='') as f:
        old=next(r for r in csv.DictReader(f) if int(r['candidate_id'])==v1['selected_candidate_id'])
    assert abs(selection[0]['total_cost_yuan']-float(old['actual_cost_yuan']))<1e-6
    validation['zero_correction_matches_v1_january']=True
    write_csv('january_validation',selection)
    print('January selected correction:',bias_config.record(),flush=True)
    # Initialize residuals by historical replay; actual warm-up execution remains byte-for-byte v1.
    forecaster=BiasForecaster(config)
    for d in range(31):forecaster.forecast(bias_config);forecaster.observe(actual[d])
    warmup=baseline[:31*144];soc=warmup[-1]['soc_end_kwh']
    formal=[];control=[];traces=[];solver_rows=[]
    for d in range(31,365):
        forecast,info=forecaster.forecast(bias_config)
        v1_rows=baseline[d*144:(d+1)*144]
        old_forecast=np.array([[r['load_forecast_kwh'],r['pv_forecast_kwh']] for r in v1_rows])
        if np.max(np.abs(old_forecast-info['base']))>1e-8:raise ValueError('Frozen base forecast differs from v1')
        assert info['last_residual_day']<d
        p=plan(forecast,price,soc,config,final_soc=6000 if d==364 else None)
        rs,soc=execution_rows(dates[d],p,actual[d],price,soc,'evaluation');formal.extend(rs)
        cr,_=execution_rows(dates[d],idle_plan(6000.,forecast),actual[d],price,6000.,'evaluation');control.extend(cr)
        for group in range(6):
            a,b=group*24,(group+1)*24
            traces.append(dict(date=dates[d],interval_label=label(group*240,(group+1)*240),
                               residual_days_used=info['residual_days_used'],last_residual_date=dates[info['last_residual_day']],
                               load_correction_mean_kw=float(info['correction'][a:b,0].mean()*6),
                               pv_correction_mean_kw=float(info['correction'][a:b,1].mean()*6)))
        solver_rows.append(dict(date=dates[d],status=p['status'],gap=p['gap'],seconds=p['seconds'],
                                terminal_penalty_yuan=p['terminal_penalty'],predicted_objective_yuan=p['objective']))
        forecaster.observe(actual[d])
        if (d+1)%60==0:print('Executed through',dates[d],flush=True)
    all_rows=warmup+formal
    validation.update(full=audit(all_rows),formal=audit(formal),same_forecast_no_storage=audit(control),
                      frozen_base_forecast_matches_v1=True,warmup_execution_identical=all_rows[:31*144]==baseline[:31*144])
    # Sequential invariance: later observations cannot change a saved earlier forecast.
    fa=BiasForecaster(config);fb=BiasForecaster(config)
    for d in range(31):
        fa.forecast(bias_config);fb.forecast(bias_config);fa.observe(actual[d]);fb.observe(actual[d].copy())
    pa,_=fa.forecast(bias_config);pb,_=fb.forecast(bias_config)
    fb.observe(np.full((144,2),1e6))
    assert np.array_equal(pa,pb)
    validation['future_observation_does_not_mutate_issued_forecast']=True
    broken=[dict(r) for r in formal[:144]];broken[0]['purchase_kwh']+=1
    try:audit(broken,require_boundaries=False)
    except ValueError:validation['injected_imbalance_rejected']=True
    else:raise ValueError('Auditor did not reject imbalance')
    daily=[]
    for i in range(334):
        rs=formal[i*144:(i+1)*144];old_rows=baseline[(i+31)*144:(i+32)*144]
        old_summary=summarize(old_rows)
        daily.append(dict(date=dates[i+31],**summarize(rs),**metrics(rs),
                          v1_total_cost_yuan=old_summary['total_cost_yuan'],v1_emergency_cost_yuan=old_summary['emergency_cost_yuan'],
                          savings_vs_v1_yuan=old_summary['total_cost_yuan']-summarize(rs)['total_cost_yuan']))
    monthly=[]
    for month in sorted({r['date'][:7] for r in formal}):
        new=[r for r in formal if r['date'].startswith(month)]
        old_rows=[r for r in baseline[31*144:] if r['date'].startswith(month)]
        ns=summarize(new);os=summarize(old_rows)
        monthly.append(dict(month=month,v1_total_cost_yuan=os['total_cost_yuan'],v2a_total_cost_yuan=ns['total_cost_yuan'],
                            savings_yuan=os['total_cost_yuan']-ns['total_cost_yuan'],
                            v1_emergency_cost_yuan=os['emergency_cost_yuan'],v2a_emergency_cost_yuan=ns['emergency_cost_yuan'],
                            v1_net_bias_kw=metrics(old_rows)['net_bias_kw'],v2a_net_bias_kw=metrics(new)['net_bias_kw'],
                            v1_net_mae_kw=metrics(old_rows)['net_mae_kw'],v2a_net_mae_kw=metrics(new)['net_mae_kw']))
    ev=events(formal)
    write_csv('execution',all_rows);write_csv('daily_summary',daily);write_csv('monthly_comparison',monthly)
    write_csv('bias_trace',traces);write_csv('solver_log',solver_rows)
    write_csv('emergency_events',ev)
    write_csv('same_forecast_no_storage_daily',[dict(date=dates[i+31],**summarize(control[i*144:(i+1)*144])) for i in range(334)])
    validation['persisted_execution']=audit(read_execution(OUT/'execution.csv'))
    path=workbook(formal,daily,ev,ROOT/'results/result2_v2a.xlsx');verify_export(path,formal,daily,ev)
    validation['workbook_readback']=True
    assert before==fingerprints(protected)
    validation['v1_artifacts_unchanged']=True
    for r in quality['sources']:assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
    validation['originals_unchanged']=True
    ns=summarize(formal);os=summarize(baseline[31*144:]);saving=os['total_cost_yuan']-ns['total_cost_yuan']
    best_nonzero=min(selection[1:],key=lambda r:(r['total_cost_yuan'],r['candidate_id']))
    best_accuracy=min(selection,key=lambda r:(r['net_mae_kw'],r['candidate_id']))
    result=dict(version='q2-v2a-bias-only-fixed-plan',base_config=config.record(),bias_config=bias_config.record(),
                selected_candidate_id=selected_id,selection_period='2025-01-15/2025-01-31',
                january_control=selection[0],january_best_nonzero=best_nonzero,january_best_accuracy=best_accuracy,
                evaluation_period='2025-02-01/2025-12-31',formal=ns,v1=os,
                same_forecast_no_storage=summarize(control),forecast_metrics=metrics(formal),
                v1_forecast_metrics=metrics(baseline[31*144:]),savings_vs_v1_yuan=saving,
                savings_vs_v1_percent=saving/os['total_cost_yuan']*100,
                monthly_comparison=monthly,validation=validation,v1_artifact_sha256=before,
                database_sha256=v1['database_sha256'],source_hashes=quality['sources'],scipy_version=scipy.__version__,
                solver_max_relative_gap=max(r['gap'] for r in solver_rows),solver_total_seconds=sum(r['seconds'] for r in solver_rows),
                improved_days=sum(r['savings_vs_v1_yuan']>1e-5 for r in daily),
                worsened_days=sum(r['savings_vs_v1_yuan']< -1e-5 for r in daily))
    (OUT/'experiment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 第二问V2-A实验：近期偏差修正','',
           '仅修改预测。复用v1购电优化、固定全天储能执行及费用结算，保留同一1月预热和6000 kWh评价边界。',
           '不是安全裕量、场景优化或日内储能重调度；全年结果属于已用于诊断的历史数据回测。','',
           '## 参数与因果性','',f'- 基础参数冻结：`{json.dumps(config.record())}`。',
           f'- 1月17个候选选中的修正参数：`{json.dumps(bias_config.record())}`。',
           '- 6个4小时组，分别估计实际减基础预测的历史残差；只用已完成日，零光伏时段保持零。',
           '- 基础参数与偏差参数均在1月开发选择，这不是独立测试；2—12月只更新残差，不改参数。',
           '- 零修正候选与v1同基础参数的1月费用一致；正式期基础预测逐条与v1一致。','',
           '## 1月候选诊断','',
           '| 候选 | 净负载MAE/kW | 计划费/元 | 紧急费/元 | 总费用/元 |',
           '|---|---:|---:|---:|---:|']
    for title,r in [('不修正',selection[0]),('费用最低的非零修正',best_nonzero),('净负载MAE最低',best_accuracy)]:
        lines.append(f'| {title}（ID {r["candidate_id"]}） | {r["net_mae_kw"]:.6f} | {r["planned_cost_yuan"]:.6f} | {r["emergency_cost_yuan"]:.6f} | {r["total_cost_yuan"]:.6f} |')
    if selected_id==0:
        lines += ['', '按1月实际费用选择，不修正优于全部非零候选，因此正式期仍使用v1预测，收益为零。不能据此断言所有偏差修正无效，只能说明当前候选在此选择期未获支持。']
    lines += ['',
           '## 正式期对照','', '| 指标 | v1 | V2-A |','|---|---:|---:|']
    for key,title in [('planned_cost_yuan','计划费/元'),('emergency_cost_yuan','紧急费/元'),('total_cost_yuan','总费用/元'),
                      ('emergency_kwh','紧急电量/kWh'),('curtailment_kwh','富余弃用/kWh'),
                      ('emergency_while_charging_kwh','充电区间紧急电量/kWh')]:
        lines.append(f'| {title} | {os[key]:.6f} | {ns[key]:.6f} |')
    lines += ['',f'相对v1节费{saving:.6f}元（{result["savings_vs_v1_percent"]:.4f}%）；改善{result["improved_days"]}天、恶化{result["worsened_days"]}天。',
              f'新预测下无储能费用另为{result["same_forecast_no_storage"]["total_cost_yuan"]:.6f}元，不替代上述v1主对照。','',
              '| 预测指标 | v1 | V2-A |','|---|---:|---:|']
    for key,title in [('load_mae_kw','负载MAE/kW'),('pv_mae_kw','光伏MAE/kW'),('net_mae_kw','净负载MAE/kW'),('net_bias_kw','净负载偏差/kW（实际减预测）')]:
        lines.append(f'| {title} | {result["v1_forecast_metrics"][key]:.6f} | {result["forecast_metrics"][key]:.6f} |')
    lines += ['', '## 分月变化','', '| 月份 | v1费用/元 | V2-A费用/元 | 节费/元 | v1净负载偏差/kW | V2-A偏差/kW |',
              '|---|---:|---:|---:|---:|---:|']
    for m in monthly:lines.append(f'| {m["month"]} | {m["v1_total_cost_yuan"]:.2f} | {m["v2a_total_cost_yuan"]:.2f} | {m["savings_yuan"]:.2f} | {m["v1_net_bias_kw"]:.2f} | {m["v2a_net_bias_kw"]:.2f} |')
    lines += ['', '## 题目指定日期','']
    for ds in ('2025-03-20','2025-06-21','2025-09-23','2025-12-21'):
        dr=next(r for r in daily if r['date']==ds);rs=[r for r in formal if r['date']==ds]
        lines += [f'### {ds}','',f'计划量{dr["purchase_kwh"]:.6f} kWh；计划费{dr["planned_cost_yuan"]:.6f}元；紧急量{dr["emergency_kwh"]:.6f} kWh；紧急费{dr["emergency_cost_yuan"]:.6f}元；总费用{dr["total_cost_yuan"]:.6f}元。','',
                  '| 时间段 | 计划购电/kWh |','|---|---:|']
        for h in (10,12,14,16,18,20):lines.append(f'| {rs[h*6]["interval_label"]} | {rs[h*6]["purchase_kwh"]:.6f} |')
        lines += ['', '| 时间段 | 充电/kWh | 放电/kWh |','|---|---:|---:|']
        for j in range(6):
            block=rs[j*24:(j+1)*24]
            lines.append(f'| {label(j*240,(j+1)*240)} | {sum(r["charge_kwh"] for r in block):.6f} | {sum(r["discharge_kwh"] for r in block):.6f} |')
        lines += ['',f'日初SOC {dr["soc_start_kwh"]:.6f} kWh，日末SOC {dr["soc_end_kwh"]:.6f} kWh。','', '| 紧急时间段 | 电量/kWh |','|---|---:|']
        lines += [f'| {e["interval_label"]} | {e["emergency_kwh"]:.6f} |' for e in ev if e['date']==ds] or ['| 无 | 0 |']
        lines.append('')
    lines += ['## 验证与局限','',
              '- 物理与费用独立审计、CSV回读、Excel所有计划/充放电/SOC/事件回读通过。',
              '- 时序调用与已知残差修正检查通过；未来观测不改变已发布预测；故障注入被拒绝。',
              '- v1结果文件和原始附件哈希保持不变。偏差参数、每天实际修正量和分月结果均保存。',
              '- 固定储能计划仍可能造成充电时紧急购电、放电时富余；本版没有为5倍紧急费显式建模风险。',
              '- 沿用右端点代表区间功率、充放电各90%、电网侧计量；这些仍是建模口径。',
              '- `python scripts/solve_q2_v2a.py` 复现；结果 `results/result2_v2a.xlsx`，明细 `results/q2_v2a/`。',
              '- 工作簿计划表全天量/费用仅为计划量/计划费；紧急费与实际总费用单独在报告和daily_summary.csv列出。','']
    (ROOT/'reports/q2_v2a_experiment.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('bias_config','formal','savings_vs_v1_yuan','savings_vs_v1_percent','forecast_metrics','improved_days','worsened_days')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

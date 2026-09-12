"""Q2 V2-B: January-selected empirical emergency-cost optimization."""
from __future__ import annotations
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np
import scipy

from q2_model import Config, execute, idle_plan, plan
from q2_bias import BiasConfig, BiasForecaster
from q2_risk import risk_plan, scenarios
from solve_q2 import execution_rows, summarize, events, workbook, verify_export, label
from solve_q2_v2a import read_execution, fingerprints, metrics
from validate_q2 import audit
import csv

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/q2_v2b'


def write_csv(name,rows):
    with (OUT/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def model_tests(config):
    forecast=np.zeros((144,2));forecast[:,0]=400
    price=np.linspace(.4,1.4,144)
    deterministic=plan(forecast,price,6000,config,6000)
    single=risk_plan(forecast,np.array([forecast[:,0]]),price,6000,config,6000)
    assert abs(deterministic['objective']-single['objective'])<1e-5
    samples=np.repeat(np.arange(100,701,100)[:,None],144,axis=1)
    no_storage=risk_plan(forecast,samples,np.ones(144),6000,config,6000,False)
    assert np.max(np.abs(no_storage['purchase']-600))<1e-5
    assert abs(no_storage['expected_emergency_kwh']-144*100/7)<1e-5
    return dict(single_scenario_matches_deterministic=True,no_storage_quantile_identity=True)


def choose(january,price,config,bias):
    table=[]
    for index,window in enumerate((0,7,14)):
        f=BiasForecaster(config);soc=6000.;total=0.;emergency=0.;planned=0.
        for d,actual in enumerate(january):
            forecast,_=f.forecast(bias)
            if d>=14:
                if window:
                    net=scenarios(forecast,f.residuals,window)
                    p=risk_plan(forecast,net,price,soc,config,6000 if d==30 else None)
                else:p=plan(forecast,price,soc,config,6000 if d==30 else None)
                e,_,cost=execute(p,actual,price)
                total+=float(cost.sum());emergency+=float((5*price)@e);planned+=float(price@p['purchase'])
                soc+=float(.9*p['charge'].sum()-p['discharge'].sum()/.9)
            f.observe(actual)
        assert abs(soc-6000)<1e-5
        table.append(dict(candidate_id=index,window_days=window,mode='risk' if window else 'point',
                          validation_start='2025-01-15',validation_end='2025-01-31',
                          planned_cost_yuan=planned,emergency_cost_yuan=emergency,total_cost_yuan=total))
    winner=min(table,key=lambda r:(r['total_cost_yuan'],r['candidate_id']))
    return winner['window_days'],table,winner['candidate_id']


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    v1=json.loads((ROOT/'results/q2/experiment.json').read_text(encoding='utf-8'))
    v2a=json.loads((ROOT/'results/q2_v2a/experiment.json').read_text(encoding='utf-8'))
    config=Config(**v2a['base_config']);bias=BiasConfig(**v2a['bias_config'])
    # Residuals currently are relative to the unchanged base forecast. Do not silently double-correct.
    assert bias.load_strength==0 and bias.pv_strength==0,'Nonzero bias requires a revised residual contract'
    quality=json.loads((ROOT/'reports/data_quality_report.json').read_text(encoding='utf-8'))
    assert quality['status']=='PASS_WITH_WARNINGS'
    for r in quality['sources']:assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
    protected=[]
    for folder in ('results/q2','results/q2_v2a'):
        protected.extend(p for p in (ROOT/folder).iterdir() if p.is_file())
    protected.extend(ROOT/p for p in ('results/result2.xlsx','results/result2_v2a.xlsx','reports/q2_experiment.md','reports/q2_v2a_experiment.md'))
    before=fingerprints(protected)
    db=ROOT/'data/processed/microgrid.sqlite'
    assert hashlib.sha256(db.read_bytes()).hexdigest()==v1['database_sha256']
    with sqlite3.connect(f'file:{db.as_posix()}?mode=ro',uri=True) as c:
        flat=c.execute('SELECT date,slot,load_kwh,pv_actual_kwh FROM actuals ORDER BY date,slot').fetchall()
        price=np.array([r[0] for r in c.execute('SELECT price_yuan_per_kwh FROM baseline ORDER BY slot')])
    actual=np.array([[r[2],r[3]] for r in flat]).reshape(365,144,2)
    dates=[flat[d*144][0] for d in range(365)]
    baseline=read_execution(ROOT/'results/q2/execution.csv')
    previous=read_execution(ROOT/'results/q2_v2a/execution.csv')
    validation=model_tests(config)
    window,table,selected_id=choose(actual[:31].copy(),price,config,bias)
    assert abs(table[0]['total_cost_yuan']-v2a['january_control']['total_cost_yuan'])<1e-5
    validation['point_control_matches_previous_january']=True
    write_csv('january_validation',table)
    print('January-selected window (0=point):',window,flush=True)
    f=BiasForecaster(config)
    for d in range(31):f.forecast(bias);f.observe(actual[d])
    warmup=baseline[:31*144];soc=warmup[-1]['soc_end_kwh']
    formal=[];control=[];traces=[];solver=[]
    for d in range(31,365):
        forecast,info=f.forecast(bias)
        old=np.array([[r['load_forecast_kwh'],r['pv_forecast_kwh']] for r in previous[d*144:(d+1)*144]])
        assert np.max(np.abs(forecast-old))<1e-8,'Predictor changed during risk-only upgrade'
        if window:
            net=scenarios(forecast,f.residuals,window)
            source_days=f.residual_days[-window:]
            assert max(source_days)<d
            p=risk_plan(forecast,net,price,soc,config,6000 if d==364 else None)
        else:
            net=(forecast[:,0]-forecast[:,1])[None,:];source_days=[]
            p=plan(forecast,price,soc,config,6000 if d==364 else None)
        # All decisions made before revealing this day's observations.
        rs,soc=execution_rows(dates[d],p,actual[d],price,soc,'evaluation');formal.extend(rs)
        no_storage=idle_plan(6000.,forecast)
        if window:no_storage['purchase']=np.maximum(np.quantile(net,.8,axis=0,method='inverted_cdf'),0)
        cr,_=execution_rows(dates[d],no_storage,actual[d],price,6000.,'evaluation');control.extend(cr)
        traces.append(dict(date=dates[d],scenario_count=len(net),
                           earliest_residual_date=dates[min(source_days)] if source_days else '',
                           latest_residual_date=dates[max(source_days)] if source_days else '',
                           residual_dates='|'.join(dates[i] for i in source_days),
                           empirical_net_p80_daily_sum_kwh=float(np.quantile(net,.8,axis=0,method='inverted_cdf').sum()),
                           expected_emergency_kwh=p.get('expected_emergency_kwh',0.),
                           expected_emergency_cost_yuan=p.get('expected_emergency_cost',0.)))
        solver.append(dict(date=dates[d],status=p['status'],gap=p['gap'],seconds=p['seconds'],
                           objective_yuan=p['objective'],planned_cost_yuan=p['planned_cost'],
                           terminal_penalty_yuan=p['terminal_penalty'],
                           expected_emergency_cost_yuan=p.get('expected_emergency_cost',0.),
                           objective_reconstruction_error=p.get('objective_reconstruction_error',0.)))
        f.observe(actual[d])
        if (d+1)%30==0:print('Executed through',dates[d],flush=True)
    full=warmup+formal
    validation.update(full=audit(full),formal=audit(formal),same_risk_no_storage=audit(control),
                      frozen_forecasts_match_v2a=True,warmup_identical=full[:31*144]==baseline[:31*144])
    broken=[dict(r) for r in formal[:144]];broken[0]['purchase_kwh']+=1
    try:audit(broken,require_boundaries=False)
    except ValueError:validation['injected_imbalance_rejected']=True
    else:raise ValueError('Failed defect detection')
    # Scenario arrays copy past residuals; later mutation cannot alter an issued scenario set.
    sample_forecast=forecast.copy();residual=[np.ones((144,2))]
    snapshot=scenarios(sample_forecast,residual,7);residual[0][:]=1e6
    assert np.array_equal(snapshot,scenarios(sample_forecast,[np.ones((144,2))],7))
    validation['issued_scenario_copy_isolation']=True
    daily=[]
    for i in range(334):
        rs=formal[i*144:(i+1)*144];old=baseline[(i+31)*144:(i+32)*144]
        ns=summarize(rs);os=summarize(old)
        daily.append(dict(date=dates[i+31],**ns,v1_total_cost_yuan=os['total_cost_yuan'],
                          savings_vs_v1_yuan=os['total_cost_yuan']-ns['total_cost_yuan'],
                          same_risk_no_storage_cost_yuan=summarize(control[i*144:(i+1)*144])['total_cost_yuan']))
    monthly=[]
    for month in sorted({r['date'][:7] for r in formal}):
        ns=summarize([r for r in formal if r['date'].startswith(month)])
        os=summarize([r for r in baseline[31*144:] if r['date'].startswith(month)])
        monthly.append(dict(month=month,v1_total_cost_yuan=os['total_cost_yuan'],v2b_total_cost_yuan=ns['total_cost_yuan'],
                            savings_yuan=os['total_cost_yuan']-ns['total_cost_yuan'],
                            v1_emergency_cost_yuan=os['emergency_cost_yuan'],v2b_emergency_cost_yuan=ns['emergency_cost_yuan'],
                            v1_curtailment_kwh=os['curtailment_kwh'],v2b_curtailment_kwh=ns['curtailment_kwh']))
    ev=events(formal)
    for name,rows in [('execution',full),('daily_summary',daily),('monthly_comparison',monthly),
                      ('scenario_trace',traces),('solver_log',solver),('emergency_events',ev)]:write_csv(name,rows)
    write_csv('same_risk_no_storage_daily',[dict(date=dates[i+31],**summarize(control[i*144:(i+1)*144])) for i in range(334)])
    validation['persisted_execution']=audit(read_execution(OUT/'execution.csv'))
    path=workbook(formal,daily,ev,ROOT/'results/result2_v2b.xlsx');verify_export(path,formal,daily,ev)
    validation['workbook_readback']=True
    assert before==fingerprints(protected)
    validation['previous_artifacts_unchanged']=True
    for r in quality['sources']:assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
    validation['originals_unchanged']=True
    ns=summarize(formal);os=summarize(baseline[31*144:]);saving=os['total_cost_yuan']-ns['total_cost_yuan']
    result=dict(version='q2-v2b-expected-emergency-fixed-plan',base_config=config.record(),bias_config=bias.record(),
                window_days=window,selected_candidate_id=selected_id,january_candidates=table,
                formal=ns,v1=os,v2a=summarize(previous[31*144:]),same_risk_no_storage=summarize(control),
                savings_vs_v1_yuan=saving,savings_vs_v1_percent=saving/os['total_cost_yuan']*100,
                forecast_metrics=metrics(formal),monthly_comparison=monthly,validation=validation,
                improved_days=sum(r['savings_vs_v1_yuan']>1e-5 for r in daily),
                worsened_days=sum(r['savings_vs_v1_yuan']< -1e-5 for r in daily),
                solver_max_gap=max(r['gap'] for r in solver),solver_total_seconds=sum(r['seconds'] for r in solver),
                max_objective_reconstruction_error=max(r['objective_reconstruction_error'] for r in solver),
                previous_artifact_hashes=before,source_hashes=quality['sources'],database_sha256=v1['database_sha256'],
                scipy_version=scipy.__version__)
    (OUT/'experiment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 第二问V2-B实验：历史误差场景购电','',
           '仅在日前优化加入期望紧急购电费，预测与固定全天储能执行保持V2-A/v1口径。',
           '共享场景控制；不提前知道真实场景，不是日内滚动控制。','',
           '## 选择与信息边界','',f'- 1月15—31日选中窗口：{window}日（0表示退回点预测）。',
           '- 基础参数和预测冻结；仅使用已完成日残差。每个历史日负载/光伏残差配对，叠加预测后裁为非负。',
           '- 候选及正式期均保持6000 kWh评价边界；预热直接保留v1，真实费用不计SOC虚拟惩罚。',
           '- 1月为反复使用的开发期；2—12月不重新选参，属于既有历史数据回测，不声称全新测试集。','',
           '| 1月候选窗口 | 计划费/元 | 紧急费/元 | 总费用/元 |','|---|---:|---:|---:|']
    for r in table:lines.append(f'| {r["window_days"]} | {r["planned_cost_yuan"]:.6f} | {r["emergency_cost_yuan"]:.6f} | {r["total_cost_yuan"]:.6f} |')
    lines += ['', '## 334天主对照','', '| 指标 | v1/V2-A | V2-B |','|---|---:|---:|']
    for key,title in [('purchase_kwh','计划购电/kWh'),('emergency_kwh','紧急购电/kWh'),('planned_cost_yuan','计划费/元'),
                      ('emergency_cost_yuan','紧急费/元'),('total_cost_yuan','总费用/元'),('curtailment_kwh','弃用/kWh'),
                      ('emergency_while_charging_kwh','充电区间紧急电量/kWh')]:
        lines.append(f'| {title} | {os[key]:.6f} | {ns[key]:.6f} |')
    lines += ['',f'相对v1节费{saving:.6f}元（{result["savings_vs_v1_percent"]:.4f}%），改善{result["improved_days"]}天、恶化{result["worsened_days"]}天。',
              f'同风险购电规则下无储能费用{result["same_risk_no_storage"]["total_cost_yuan"]:.6f}元，单独作为诊断，不替换冻结v1主对照。','',
              '## 分月对照','', '| 月份 | v1费用/元 | V2-B费用/元 | 节费/元 |','|---|---:|---:|---:|']
    for m in monthly:lines.append(f'| {m["month"]} | {m["v1_total_cost_yuan"]:.2f} | {m["v2b_total_cost_yuan"]:.2f} | {m["savings_yuan"]:.2f} |')
    lines += ['', '## 指定日期','']
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
    lines += ['## 验证与边界','',
              '- 单场景退化与无储能80%分位数子问题验证通过；场景期望费用由输出重新计算。',
              '- 物理约束、固定控制、费用、SOC与时间连续、CSV和Excel回读、故障注入通过；原件及旧版结果不变。',
              f'- 最大求解相对间隙{result["solver_max_gap"]:.3g}，期望目标重算误差{result["max_objective_reconstruction_error"]:.3g}元。',
              '- 概率目标使用样本边际误差；不等于已实现跨时段自适应风险控制。固定计划的充电时紧急购电等局限仍存在。',
              '- 仍沿用时间右端点、充放电各90%、设备外部功率/电量计量的假设。',
              '- `python scripts/solve_q2_v2b.py`复现；结果为results/result2_v2b.xlsx，明细和场景日期见results/q2_v2b/。',
              '- 工作簿计划表的全天量/费为计划量/计划费；实际总费用和紧急费单独列在报告与daily_summary.csv。','']
    (ROOT/'reports/q2_v2b_experiment.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('window_days','formal','savings_vs_v1_yuan','savings_vs_v1_percent','improved_days','worsened_days','solver_total_seconds')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

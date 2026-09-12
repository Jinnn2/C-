"""C0/C1 hourly storage MPC, holding the entire V2-B purchase table fixed."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import sqlite3
from pathlib import Path
import numpy as np
import scipy

from q2_model import Config
from q2_bias import BiasConfig,BiasForecaster
from q2_risk import risk_plan,scenarios
from q2_rolling import Feedback,feedback_candidates,rolling_plan,update_scenarios
from solve_q2 import execution_rows,summarize,events,workbook,verify_export,label
from solve_q2_v2a import read_execution,fingerprints
from validate_q2 import audit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/q2_v2c'


def write_csv(name,rows):
    with (OUT/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def read_rolling(path):
    strings={'date','phase','interval_start','interval_end','interval_label','plan_issue_time','history_available_through',
             'variant','control_issue_time','last_observation_end'}
    with path.open(encoding='utf-8-sig',newline='') as f:
        return [{k:v if k in strings else int(v) if k=='slot' else float(v) for k,v in r.items()} for r in csv.DictReader(f)]


def day_control(reference_rows,original_scenarios,price,initial_soc,config,params,variant,final_soc=None):
    q=np.array([r['purchase_kwh'] for r in reference_rows])
    remaining_c=np.array([r['charge_kwh'] for r in reference_rows])
    remaining_d=np.array([r['discharge_kwh'] for r in reference_rows])
    observed=[];soc=initial_soc;rows=[];logs=[]
    for t in range(0,144,6):
        # Only completed observations enter the updater/optimizer.
        updated,innovation=update_scenarios(original_scenarios,observed,params)
        p=rolling_plan(q[t:],updated,price[t:],soc,config,final_soc,
                       reference=(remaining_c[t:],remaining_d[t:]))
        remaining_c[t:]=p['charge'];remaining_d[t:]=p['discharge']
        issued=reference_rows[t]['interval_start']
        last_observed=reference_rows[t-1]['interval_end'] if t else ''
        logs.append(dict(date=reference_rows[0]['date'],variant=variant,control_issue_time=issued,
                         last_observation_end=last_observed,initial_soc_kwh=soc,innovation_kwh=innovation,
                         objective_yuan=p['objective'],lower_bound_yuan=p['lower_bound'],recovery_error=p['recovery_error'],
                         used_reference=int(p['used_reference']),reference_gap='' if p['reference_gap'] is None else p['reference_gap'],
                         removed_simultaneous_kwh=p['removed_simultaneous_kwh'],seconds=p['seconds']))
        for j in range(t,t+6):
            # Target interval actuals are first used here, after its command was issued.
            base=reference_rows[j];r=dict(base)
            charge=float(remaining_c[j]);discharge=float(remaining_d[j])
            net=base['load_kwh']-base['pv_actual_kwh']
            emergency=max(net+charge-q[j]-discharge,0.)
            waste=max(q[j]+discharge-net-charge,0.)
            end_soc=soc+.9*charge-discharge/.9
            r.update(charge_kwh=charge,discharge_kwh=discharge,emergency_kwh=float(emergency),curtailment_kwh=float(waste),
                     soc_start_kwh=soc,soc_end_kwh=end_soc,emergency_cost_yuan=float(5*price[j]*emergency),
                     total_cost_yuan=float(price[j]*q[j]+5*price[j]*emergency),variant=variant,
                     control_issue_time=issued,last_observation_end=last_observed,
                     committed_purchase_kwh=float(q[j]),issued_charge_kwh=charge,issued_discharge_kwh=discharge)
            rows.append(r);soc=end_soc;observed.append(net)
    return rows,soc,logs


def enrich_warmup(rows):
    return [dict(r,variant='warmup',control_issue_time=r['plan_issue_time'],last_observation_end='',
                 committed_purchase_kwh=r['purchase_kwh'],issued_charge_kwh=r['charge_kwh'],issued_discharge_kwh=r['discharge_kwh']) for r in rows]


def build_inputs(actual,dates,price,config,bias,window,baseline):
    f=BiasForecaster(config);jan=[];formal=[];jan_soc=6000.
    for d in range(365):
        forecast,_=f.forecast(bias)
        if d>=14:
            net=scenarios(forecast,f.residuals,window)
            assert max(f.residual_days[-window:])<d
            if d<31:
                p=risk_plan(forecast,net,price,jan_soc,config,6000 if d==30 else None)
                rs,jan_soc=execution_rows(dates[d],p,actual[d],price,jan_soc,'validation')
                jan.append((rs,net))
            else:
                rs=baseline[d*144:(d+1)*144]
                assert np.max(np.abs(forecast-np.array([[r['load_forecast_kwh'],r['pv_forecast_kwh']] for r in rs])))<1e-8
                formal.append((rs,net))
        f.observe(actual[d])
    return jan,formal


def model_tests(config):
    rng=np.random.default_rng(5)
    checks=[]
    for n in (6,24,144):
        q=np.full(n,400.);net=rng.normal(420,70,(14,n));price=np.linspace(.4,1.4,n)
        p=rolling_plan(q,net,price,6000,config,6000,strict_check=True)
        assert np.max(np.minimum(p['charge'],p['discharge']))<1e-6
        checks.append(dict(horizon=n,strict_lp_gap=p['strict_gap'],recovery_error=p['recovery_error']))
    original=np.zeros((14,144));observed=[100.]*6
    updated,_=update_scenarios(original,observed,Feedback(1.,.8))
    assert abs(updated[0,0]-80)<1e-8
    observed[-1]=1e6
    assert abs(updated[0,0]-80)<1e-8
    return dict(strict_milp_crosschecks=checks,causal_prefix_update_check=True)


def choose(january,price,config):
    table=[]
    for index,params in enumerate(feedback_candidates()):
        soc=6000.;rows=[];max_recovery=0.;solver_seconds=0.
        for d,(reference,net) in enumerate(january):
            rs,soc,logs=day_control(reference,net,price,soc,config,params,'validation',6000 if d==len(january)-1 else None)
            rows.extend(rs);max_recovery=max(max_recovery,max(r['recovery_error'] for r in logs))
            solver_seconds+=sum(r['seconds'] for r in logs)
        audit(rows,require_fixed_storage=False)
        totals=summarize(rows)
        table.append(dict(candidate_id=index,**params.record(),validation_start='2025-01-15',validation_end='2025-01-31',
                          planned_cost_yuan=totals['planned_cost_yuan'],emergency_cost_yuan=totals['emergency_cost_yuan'],
                          total_cost_yuan=totals['total_cost_yuan'],curtailment_kwh=totals['curtailment_kwh'],
                          max_recovery_error=max_recovery,solver_seconds=solver_seconds))
        print('January candidate',index,params.record(),'cost',round(totals['total_cost_yuan'],2),flush=True)
    winner=min(table,key=lambda r:(r['total_cost_yuan'],r['candidate_id']))
    return Feedback(winner['gamma'],winner['rho']),table,winner['candidate_id']


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    v2b=json.loads((ROOT/'results/q2_v2b/experiment.json').read_text(encoding='utf-8'))
    config=Config(**v2b['base_config']);bias=BiasConfig(**v2b['bias_config']);window=v2b['window_days']
    assert window>0 and bias.load_strength==bias.pv_strength==0
    quality=json.loads((ROOT/'reports/data_quality_report.json').read_text(encoding='utf-8'))
    for r in quality['sources']:assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
    protected=[]
    for folder in ('results/q2','results/q2_v2a','results/q2_v2b'):
        protected.extend(p for p in (ROOT/folder).iterdir() if p.is_file())
    protected.extend(ROOT/p for p in ('results/result2.xlsx','results/result2_v2a.xlsx','results/result2_v2b.xlsx',
                                      'reports/q2_experiment.md','reports/q2_v2a_experiment.md','reports/q2_v2b_experiment.md'))
    before=fingerprints(protected);db=ROOT/'data/processed/microgrid.sqlite'
    assert hashlib.sha256(db.read_bytes()).hexdigest()==v2b['database_sha256']
    with sqlite3.connect(f'file:{db.as_posix()}?mode=ro',uri=True) as c:
        flat=c.execute('SELECT date,slot,load_kwh,pv_actual_kwh FROM actuals ORDER BY date,slot').fetchall()
        price=np.array([r[0] for r in c.execute('SELECT price_yuan_per_kwh FROM baseline ORDER BY slot')])
    actual=np.array([[r[2],r[3]] for r in flat]).reshape(365,144,2);dates=[flat[d*144][0] for d in range(365)]
    baseline=read_execution(ROOT/'results/q2_v2b/execution.csv')
    validation=model_tests(config)
    jan,inputs=build_inputs(actual,dates,price,config,bias,window,baseline)
    jan_reference=summarize([r for rows,_ in jan for r in rows])
    old_jan=next(r for r in v2b['january_candidates'] if r['window_days']==window)
    assert abs(jan_reference['total_cost_yuan']-old_jan['total_cost_yuan'])<1e-5
    validation['january_fixed_purchase_reference_matches_v2b']=True
    params,selection,selected_id=choose(jan,price,config)
    assert abs(selection[0]['total_cost_yuan']-jan_reference['total_cost_yuan'])<1e-4,'January C0 unexpectedly changed cost'
    write_csv('january_validation',selection)
    print('Selected feedback:',params.record(),flush=True)
    warmup=enrich_warmup(baseline[:31*144]);c0=[];c1=[];logs0=[];logs1=[]
    soc0=soc1=warmup[-1]['soc_end_kwh']
    for i,(reference,net) in enumerate(inputs):
        final=6000 if i==333 else None
        rs,soc0,logs=day_control(reference,net,price,soc0,config,Feedback(),'C0',final)
        c0.extend(rs);logs0.extend(logs)
        rs,soc1,logs=day_control(reference,net,price,soc1,config,params,'C1',final)
        c1.extend(rs);logs1.extend(logs)
        if (i+1)%7==0:print('C0/C1 executed through',reference[0]['date'],flush=True)
    old=baseline[31*144:]
    for a,b,c in zip(old,c0,c1):assert a['purchase_kwh']==b['purchase_kwh']==c['purchase_kwh']
    max_c0_action_error=max(max(abs(a['charge_kwh']-b['charge_kwh']),abs(a['discharge_kwh']-b['discharge_kwh'])) for a,b in zip(old,c0))
    assert max_c0_action_error<1e-5,'C0 action drift beyond numerical tolerance'
    assert abs(summarize(c0)['total_cost_yuan']-summarize(old)['total_cost_yuan'])<1e-3
    validation.update(c0=audit(warmup+c0,require_fixed_storage=False),c1=audit(warmup+c1,require_fixed_storage=False),
                      frozen_purchase_exact=True,c0_max_action_difference_kwh=max_c0_action_error,
                      c0_reference_retained_all=all(r['used_reference'] for r in logs0))
    broken=[dict(r) for r in c1[:144]];broken[0]['last_observation_end']=broken[0]['interval_end']
    try:audit(broken,require_boundaries=False,require_fixed_storage=False)
    except ValueError:validation['future_observation_stamp_rejected']=True
    else:raise ValueError('Audit failed to detect future observation')
    broken=[dict(r) for r in c1[:144]];broken[0]['purchase_kwh']+=1
    try:audit(broken,require_boundaries=False,require_fixed_storage=False)
    except ValueError:validation['modified_contract_rejected']=True
    else:raise ValueError('Audit failed to detect changed purchase')
    daily=[]
    for i in range(334):
        rs=c1[i*144:(i+1)*144];os=summarize(old[i*144:(i+1)*144]);ns=summarize(rs)
        daily.append(dict(date=rs[0]['date'],**ns,v2b_total_cost_yuan=os['total_cost_yuan'],
                          savings_vs_v2b_yuan=os['total_cost_yuan']-ns['total_cost_yuan']))
    monthly=[]
    for month in sorted({r['date'][:7] for r in c1}):
        ns=summarize([r for r in c1 if r['date'].startswith(month)]);os=summarize([r for r in old if r['date'].startswith(month)])
        monthly.append(dict(month=month,v2b_cost_yuan=os['total_cost_yuan'],c1_cost_yuan=ns['total_cost_yuan'],
                            savings_yuan=os['total_cost_yuan']-ns['total_cost_yuan'],v2b_waste_kwh=os['curtailment_kwh'],
                            c1_waste_kwh=ns['curtailment_kwh']))
    ev=events(c1)
    for name,rows in [('c0_execution',warmup+c0),('execution',warmup+c1),('c0_update_log',logs0),('update_log',logs1),
                      ('daily_summary',daily),('monthly_comparison',monthly),('emergency_events',ev)]:write_csv(name,rows)
    validation['persisted_c1_execution']=audit(read_rolling(OUT/'execution.csv'),require_fixed_storage=False)
    validation['persisted_c0_execution']=audit(read_rolling(OUT/'c0_execution.csv'),require_fixed_storage=False)
    path=workbook(c1,daily,ev,ROOT/'results/result2_v2c.xlsx');verify_export(path,c1,daily,ev)
    validation['workbook_readback']=True
    assert before==fingerprints(protected);validation['previous_artifacts_unchanged']=True
    for r in quality['sources']:assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
    validation['originals_unchanged']=True
    ns=summarize(c1);os=summarize(old);saving=os['total_cost_yuan']-ns['total_cost_yuan']
    result=dict(version='q2-v2c-hourly-fixed-contract',feedback=params.record(),selected_candidate_id=selected_id,
                base_config=config.record(),scenario_window=window,january_candidates=selection,
                c0=summarize(c0),c1=ns,v2b=os,savings_vs_v2b_yuan=saving,savings_vs_v2b_percent=saving/os['total_cost_yuan']*100,
                improved_days=sum(r['savings_vs_v2b_yuan']>1e-5 for r in daily),worsened_days=sum(r['savings_vs_v2b_yuan']< -1e-5 for r in daily),
                monthly_comparison=monthly,validation=validation,solver_c0_seconds=sum(r['seconds'] for r in logs0),
                solver_c1_seconds=sum(r['seconds'] for r in logs1),max_recovery_error=max(r['recovery_error'] for r in logs0+logs1),
                update_count_per_variant=len(logs1),previous_artifact_hashes=before,source_hashes=quality['sources'],
                database_sha256=v2b['database_sha256'],scipy_version=scipy.__version__)
    (OUT/'experiment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 第二问V2-C：固定购电表的逐小时储能反馈','',
           '全年逐段购电量完全冻结V2-B，仅修改储能执行。C0不更新场景；C1使用已结束区间的观测更新场景。',
           '这是储能控制的隔离实验，尚未接入次日重新制定购电的完整闭环。','',
           '## 1月选择','',f'选中反馈参数：`{json.dumps(params.record())}`。',
           '仅使用1月15—31日选择；参考购电表由该期V2-B策略重建，候选从6000到6000。1月是重复使用的开发期，全年为历史回测。','',
           '| 候选 | γ | ρ | 总费用/元 |','|---|---:|---:|---:|']
    for r in selection:lines.append(f'| {r["candidate_id"]} | {r["gamma"]} | {r["rho"]} | {r["total_cost_yuan"]:.6f} |')
    lines += ['', '## 334天对照','', '| 指标 | V2-B | C0 | C1 |','|---|---:|---:|---:|']
    for key,title in [('planned_cost_yuan','计划费/元'),('emergency_cost_yuan','紧急费/元'),('total_cost_yuan','总费用/元'),
                      ('emergency_kwh','紧急购电/kWh'),('curtailment_kwh','富余弃用/kWh'),
                      ('emergency_while_charging_kwh','充电区间紧急电量/kWh'),('waste_while_discharging_kwh','放电区间弃用/kWh')]:
        lines.append(f'| {title} | {os[key]:.6f} | {result["c0"][key]:.6f} | {ns[key]:.6f} |')
    lines += ['',f'C1相对V2-B节费{saving:.6f}元（{result["savings_vs_v2b_percent"]:.4f}%）；改善{result["improved_days"]}天、恶化{result["worsened_days"]}天。',
              f'C0最大动作差{max_c0_action_error:.3g} kWh；每次旧方案仍最优并保留：{validation["c0_reference_retained_all"]}。','',
              '## 分月对照','', '| 月份 | V2-B费用/元 | C1费用/元 | 节费/元 |','|---|---:|---:|---:|']
    for m in monthly:lines.append(f'| {m["month"]} | {m["v2b_cost_yuan"]:.2f} | {m["c1_cost_yuan"]:.2f} | {m["savings_yuan"]:.2f} |')
    lines += ['', '## 指定日期','']
    for ds in ('2025-03-20','2025-06-21','2025-09-23','2025-12-21'):
        dr=next(r for r in daily if r['date']==ds);rs=[r for r in c1 if r['date']==ds]
        lines += [f'### {ds}','',f'计划量{dr["purchase_kwh"]:.6f} kWh；计划费{dr["planned_cost_yuan"]:.6f}元；紧急量{dr["emergency_kwh"]:.6f} kWh；紧急费{dr["emergency_cost_yuan"]:.6f}元；总费用{dr["total_cost_yuan"]:.6f}元。','',
                  '| 时间段 | 计划购电/kWh |','|---|---:|']
        for h in (10,12,14,16,18,20):lines.append(f'| {rs[h*6]["interval_label"]} | {rs[h*6]["purchase_kwh"]:.6f} |')
        lines += ['', '| 时间段 | 实际充电/kWh | 实际放电/kWh |','|---|---:|---:|']
        for j in range(6):
            block=rs[j*24:(j+1)*24]
            lines.append(f'| {label(j*240,(j+1)*240)} | {sum(r["charge_kwh"] for r in block):.6f} | {sum(r["discharge_kwh"] for r in block):.6f} |')
        lines += ['',f'日初SOC {dr["soc_start_kwh"]:.6f} kWh，日末SOC {dr["soc_end_kwh"]:.6f} kWh。','', '| 紧急时间段 | 电量/kWh |','|---|---:|']
        lines += [f'| {e["interval_label"]} | {e["emergency_kwh"]:.6f} |' for e in ev if e['date']==ds] or ['| 无 | 0 |']
        lines.append('')
    lines += ['## 验证与边界','',
              '- 6/24/144段LP与严格MILP交叉验证通过；每次消除同时充放电后重新验证目标等于LP下界。',
              '- C0与V2-B动作及费用一致；新观测、而非重复求解本身，是C1改变控制的来源。',
              '- 购电合同逐条完全一致；实际动作等于最新发布指令，观测截止不晚于控制发布；原物理、SOC和费用审计通过。',
              '- 未来观测时间戳与修改购电合同的故障注入被拒绝；CSV、工作簿回读通过；原件及旧版输出哈希不变。',
              f'- 每种策略重算{len(logs1)}次；C0/C1求解器累计耗时{result["solver_c0_seconds"]:.3f}/{result["solver_c1_seconds"]:.3f}秒；最大恢复目标误差{result["max_recovery_error"]:.3g}元。',
              '- 不使用当前区间末真实值实施即时停充；没有同时改变日前购电策略。时间与效率仍沿用原假设。',
              '- `python scripts/solve_q2_v2c.py`复现；结果results/result2_v2c.xlsx；C0/C1轨迹和每小时日志在results/q2_v2c/。','']
    (ROOT/'reports/q2_v2c_experiment.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('feedback','c0','c1','savings_vs_v2b_yuan','savings_vs_v2b_percent','improved_days','worsened_days')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

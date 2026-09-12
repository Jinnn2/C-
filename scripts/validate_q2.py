"""Physical and settlement validation independent of Q2 optimization matrices."""
import math
from datetime import datetime, timedelta


def audit(rows, require_boundaries=True, tolerance=1e-5, require_fixed_storage=True):
    if not rows:
        raise ValueError('No execution rows')
    checks=dict(finite=True,nonnegative=True,balance=True,dynamics=True,bounds=True,power=True,
                exclusive=True,settlement=True,plan_unchanged=True,continuity=True,
                chronology=True,forecast_cutoff=True,no_emergency_and_waste=True)
    maxima=dict(balance=0.,dynamics=0.,cost=0.)
    previous=None
    for r in rows:
        checks['finite'] &= all(math.isfinite(v) for v in r.values() if isinstance(v,(int,float)))
        balance=r['purchase_kwh']+r['emergency_kwh']+r['pv_actual_kwh']+r['discharge_kwh']-r['load_kwh']-r['charge_kwh']-r['curtailment_kwh']
        dynamics=r['soc_end_kwh']-r['soc_start_kwh']-.9*r['charge_kwh']+r['discharge_kwh']/.9
        expected=r['price_yuan_per_kwh']*(r['purchase_kwh']+5*r['emergency_kwh'])
        maxima['balance']=max(maxima['balance'],abs(balance));maxima['dynamics']=max(maxima['dynamics'],abs(dynamics))
        maxima['cost']=max(maxima['cost'],abs(expected-r['total_cost_yuan']))
        checks['nonnegative'] &= min(r[k] for k in ('purchase_kwh','emergency_kwh','charge_kwh','discharge_kwh','curtailment_kwh'))>=-tolerance
        checks['balance'] &= abs(balance)<=tolerance
        checks['dynamics'] &= abs(dynamics)<=tolerance
        checks['bounds'] &= min(r['soc_start_kwh'],r['soc_end_kwh'])>=1200-tolerance and max(r['soc_start_kwh'],r['soc_end_kwh'])<=10800+tolerance
        checks['power'] &= max(r['charge_kwh'],r['discharge_kwh'])<=5000/6+tolerance
        checks['exclusive'] &= min(r['charge_kwh'],r['discharge_kwh'])<=tolerance
        checks['no_emergency_and_waste'] &= min(r['emergency_kwh'],r['curtailment_kwh'])<=tolerance
        checks['settlement'] &= abs(expected-r['total_cost_yuan'])<=tolerance and abs(r['planned_cost_yuan']-r['price_yuan_per_kwh']*r['purchase_kwh'])<=tolerance and abs(r['emergency_cost_yuan']-5*r['price_yuan_per_kwh']*r['emergency_kwh'])<=tolerance
        if require_fixed_storage or 'control_issue_time' not in r:
            checks['plan_unchanged'] &= abs(r['charge_kwh']-r['planned_charge_kwh'])<=tolerance and abs(r['discharge_kwh']-r['planned_discharge_kwh'])<=tolerance
        else:
            checks['plan_unchanged'] &= abs(r['purchase_kwh']-r['committed_purchase_kwh'])<=tolerance
            checks['issued_control_executed']=checks.get('issued_control_executed',True) and abs(r['charge_kwh']-r['issued_charge_kwh'])<=tolerance and abs(r['discharge_kwh']-r['issued_discharge_kwh'])<=tolerance
            control_time=datetime.fromisoformat(r['control_issue_time'])
            checks['causal_control']=checks.get('causal_control',True) and control_time<=datetime.fromisoformat(r['interval_start']) and (not r['last_observation_end'] or datetime.fromisoformat(r['last_observation_end'])<=control_time)
        begin=datetime.fromisoformat(r['interval_start']);end=datetime.fromisoformat(r['interval_end'])
        issue=datetime.fromisoformat(r['plan_issue_time'])
        checks['chronology'] &= end-begin==timedelta(minutes=10) and issue<=begin and issue.date().isoformat()==r['date']
        checks['forecast_cutoff'] &= not r['history_available_through'] or datetime.fromisoformat(r['history_available_through'])<=issue
        if previous:
            checks['continuity'] &= abs(previous['soc_end_kwh']-r['soc_start_kwh'])<=tolerance and previous['interval_end']==r['interval_start']
        previous=r
    if require_boundaries:
        checks['equal_comparison_boundaries']=abs(rows[0]['soc_start_kwh']-6000)<=tolerance and abs(rows[-1]['soc_end_kwh']-6000)<=tolerance
    energy_error=math.fsum(r['purchase_kwh']+r['emergency_kwh']+r['pv_actual_kwh']-r['load_kwh']-r['curtailment_kwh']-.1*r['charge_kwh']-(1/.9-1)*r['discharge_kwh'] for r in rows)-(rows[-1]['soc_end_kwh']-rows[0]['soc_start_kwh'])
    checks['global_energy_identity']=abs(energy_error)<=tolerance*len(rows)
    failed=[k for k,v in checks.items() if not v]
    if failed: raise ValueError(f'Q2 audit failed: {failed}')
    return dict(passed=True,rows=len(rows),checks=checks,max_residuals=maxima,global_energy_error_kwh=energy_error)

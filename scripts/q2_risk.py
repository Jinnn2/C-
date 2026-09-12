"""Shared day-ahead controls with empirical expected emergency purchase cost."""
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


def scenarios(forecast, residuals, window):
    """Paired full-day load/PV residuals; no target-day actual input."""
    if not residuals:
        raise ValueError('Historical residuals required')
    errors=np.asarray(residuals[-window:],dtype=float)
    samples=np.maximum(forecast[None,:,:]+errors,0.)
    return samples[:,:,0]-samples[:,:,1]


def expected_cost(day_plan, net_scenarios, price, config):
    shortfall=np.maximum(net_scenarios+day_plan['charge']-day_plan['purchase']-day_plan['discharge'],0)
    emergency_cost=float(np.mean(shortfall@(5*price)))
    soc=float(day_plan['soc'][-1])
    penalty=config.shortage_penalty*max(config.target_soc-soc,0)+config.surplus_penalty*max(soc-config.target_soc,0)
    return float(price@day_plan['purchase'])+emergency_cost+penalty,emergency_cost


def risk_plan(forecast, net_scenarios, price, initial_soc, config, final_soc=None, battery_enabled=True):
    n=144;k=len(net_scenarios)
    if net_scenarios.shape!=(k,n) or k<1 or not np.all(np.isfinite(net_scenarios)):
        raise ValueError('Invalid scenarios')
    q,c,d,s,z,minus,plus,e=0,144,288,432,577,721,722,723
    size=e+k*n
    cost=np.zeros(size);cost[q:q+n]=price
    cost[minus]=config.shortage_penalty;cost[plus]=config.surplus_penalty
    cost[e:]=np.tile(5*price/k,k)
    low=np.zeros(size);high=np.full(size,np.inf)
    high[c:c+n]=high[d:d+n]=5000/6 if battery_enabled else 0
    low[s:s+n+1]=1200;high[s:s+n+1]=10800
    low[s]=high[s]=initial_soc
    if final_soc is not None:low[s+n]=high[s+n]=final_soc
    high[z:z+n]=1;integer=np.zeros(size,dtype=int);integer[z:z+n]=1
    count=3*n+1+k*n
    a=lil_matrix((count,size));lower=np.full(count,-np.inf);upper=np.zeros(count)
    for t in range(n):
        a[t,s+t+1]=1;a[t,s+t]=-1;a[t,c+t]=-.9;a[t,d+t]=1/.9
        lower[t]=upper[t]=0
        a[n+t,c+t]=1;a[n+t,z+t]=-5000/6
        a[2*n+t,d+t]=1;a[2*n+t,z+t]=5000/6;upper[2*n+t]=5000/6
    a[3*n,s+n]=1;a[3*n,minus]=1;a[3*n,plus]=-1
    lower[3*n]=upper[3*n]=config.target_soc
    for j in range(k):
        for t in range(n):
            row=3*n+1+j*n+t
            a[row,q+t]=1;a[row,d+t]=1;a[row,c+t]=-1;a[row,e+j*n+t]=1
            lower[row]=net_scenarios[j,t];upper[row]=np.inf
    start=time.perf_counter()
    res=milp(cost,integrality=integer,bounds=Bounds(low,high),
             constraints=LinearConstraint(a.tocsr(),lower,upper),
             options={'mip_rel_gap':1e-8,'time_limit':60})
    seconds=time.perf_counter()-start
    if not res.success:raise RuntimeError(f'Risk MILP failed: {res.message}')
    p=dict(purchase=res.x[q:q+n].copy(),charge=res.x[c:c+n].copy(),discharge=res.x[d:d+n].copy(),
           soc=res.x[s:s+n+1].copy(),forecast=forecast.copy(),status=int(res.status),gap=float(res.mip_gap),
           objective=float(res.fun),planned_cost=float(price@res.x[q:q+n]),seconds=seconds,
           terminal_penalty=float(cost[minus]*res.x[minus]+cost[plus]*res.x[plus]),
           predicted_curtailment=np.maximum(res.x[q:q+n]+forecast[:,1]+res.x[d:d+n]-forecast[:,0]-res.x[c:c+n],0),
           scenario_count=k,dual_bound_yuan=float(res.mip_dual_bound))
    reconstructed,expected_emergency=expected_cost(p,net_scenarios,price,config)
    if abs(reconstructed-res.fun)>1e-5:
        raise ValueError('Expected objective accounting mismatch')
    p['expected_emergency_cost']=expected_emergency
    p['expected_emergency_kwh']=float(np.maximum(net_scenarios+p['charge']-p['purchase']-p['discharge'],0).sum()/k)
    p['objective_reconstruction_error']=abs(reconstructed-res.fun)
    return p

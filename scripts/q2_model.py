"""Q2 v1: historical point forecast and a deterministic day-ahead battery plan.

This module receives only historical/forecast data, never the target day's actuals.
"""
from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


@dataclass(frozen=True)
class Config:
    load_method: str = 'weekly_weighted'
    pv_days: int = 7
    shortage_penalty: float = .5
    surplus_penalty: float = .1
    target_soc: float = 6000.

    def record(self):
        return asdict(self)


def predict(history, config):
    """history shape [completed days,144,2], units kWh. No target-day argument."""
    n = len(history)
    if not n:
        raise ValueError('Cold start has no forecast; use explicit zero-purchase idle policy')
    h = np.asarray(history, dtype=float)
    if n < 7:
        load = h[-1, :, 0]
    elif config.load_method == 'weekly_last':
        load = h[-7, :, 0]
    elif config.load_method == 'weekly_weighted':
        ids = list(range(n-7, max(-1, n-29), -7))
        weights = .7 ** np.arange(len(ids))
        load = np.average(h[ids, :, 0], axis=0, weights=weights)
    else:
        raise ValueError(config.load_method)
    pv = h[-config.pv_days:, :, 1].mean(axis=0)
    return np.column_stack((load, pv)).copy()


def plan(forecast, price, initial_soc, config, final_soc=None):
    n = 144
    q,c,d,w,s,z,minus,plus = 0,144,288,432,576,721,865,866
    size = 867
    objective = np.zeros(size)
    objective[q:q+n] = price
    objective[minus] = config.shortage_penalty
    objective[plus] = config.surplus_penalty
    low = np.zeros(size); high = np.full(size, np.inf)
    high[c:c+n] = high[d:d+n] = 5000/6
    low[s:s+n+1] = 1200; high[s:s+n+1] = 10800
    low[s] = high[s] = initial_soc
    if final_soc is not None:
        low[s+n] = high[s+n] = final_soc
    high[z:z+n] = 1
    integer = np.zeros(size, dtype=int); integer[z:z+n] = 1
    a = lil_matrix((4*n+1,size)); lower = np.full(4*n+1,-np.inf); upper = np.zeros(4*n+1)
    for t in range(n):
        a[t,q+t]=1; a[t,d+t]=1; a[t,c+t]=-1; a[t,w+t]=-1
        lower[t]=upper[t]=forecast[t,0]-forecast[t,1]
        a[n+t,s+t+1]=1; a[n+t,s+t]=-1; a[n+t,c+t]=-.9; a[n+t,d+t]=1/.9
        lower[n+t]=upper[n+t]=0
        a[2*n+t,c+t]=1; a[2*n+t,z+t]=-5000/6
        a[3*n+t,d+t]=1; a[3*n+t,z+t]=5000/6; upper[3*n+t]=5000/6
    a[4*n,s+n]=1; a[4*n,minus]=1; a[4*n,plus]=-1
    lower[4*n]=upper[4*n]=config.target_soc
    start = time.perf_counter()
    res = milp(objective, integrality=integer, bounds=Bounds(low,high),
               constraints=LinearConstraint(a.tocsr(),lower,upper),
               options={'mip_rel_gap':1e-8,'time_limit':30})
    if not res.success:
        raise RuntimeError(f'Day-ahead MILP failed: {res.message}')
    # Keep full precision, including harmless solver round-off.
    return dict(purchase=res.x[q:q+n].copy(), charge=res.x[c:c+n].copy(),
                discharge=res.x[d:d+n].copy(), soc=res.x[s:s+n+1].copy(),
                forecast=forecast.copy(), predicted_curtailment=res.x[w:w+n].copy(),
                status=int(res.status), gap=float(res.mip_gap), objective=float(res.fun),
                planned_cost=float(np.dot(price,res.x[q:q+n])),
                terminal_penalty=float(np.dot(objective[[minus,plus]],res.x[[minus,plus]])),
                seconds=time.perf_counter()-start)


def idle_plan(initial_soc=6000., forecast=None):
    n=144
    f=np.zeros((n,2)) if forecast is None else forecast.copy()
    q=np.maximum(f[:,0]-f[:,1],0)
    return dict(purchase=q,charge=np.zeros(n),discharge=np.zeros(n),soc=np.full(n+1,initial_soc),
                forecast=f,predicted_curtailment=np.maximum(f[:,1]-f[:,0],0),
                status=0,gap=0.,objective=0.,planned_cost=0.,terminal_penalty=0.,seconds=0.)


def execute(day_plan, actual, price):
    """Fixed controls precede actuals; emergency supply is an ex-post balance residual.

    No emergency-driven charge cancellation is assumed in this first version.
    """
    residual = actual[:,0]+day_plan['charge']-day_plan['purchase']-actual[:,1]-day_plan['discharge']
    emergency = np.maximum(residual,0)
    waste = np.maximum(-residual,0)
    cost = price*day_plan['purchase']+5*price*emergency
    return emergency,waste,cost

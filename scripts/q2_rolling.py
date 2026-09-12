"""Hourly fixed-purchase MPC with causal scenario updates and an exact LP reduction."""
from dataclasses import asdict, dataclass
from functools import lru_cache
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack


@dataclass(frozen=True)
class Feedback:
    gamma: float = 0.
    rho: float = .8

    def record(self):return asdict(self)


def feedback_candidates():
    yield Feedback(0.,.8)
    for rho in (.5,.8,.95):
        for gamma in (.5,1.):yield Feedback(gamma,rho)


def update_scenarios(original, observed_net, params):
    """Only the completed current-day prefix is accepted, never future actuals."""
    t=len(observed_net)
    if not 0<=t<original.shape[1]:raise ValueError('Invalid observed prefix')
    remaining=original[:,t:].copy()
    innovation=0. if t==0 else float(observed_net[-1]-original[:,t-1].mean())
    shift=params.gamma*params.rho**np.arange(1,remaining.shape[1]+1)*innovation
    return remaining+shift,innovation


@lru_cache(maxsize=64)
def matrices(n,k):
    # [charge n, discharge n, SOC n+1, emergency k*n, terminal short, terminal over]
    c,d,s,e,minus,plus=0,n,2*n,3*n+1,3*n+1+k*n,3*n+2+k*n
    size=plus+1
    t=np.arange(n);scenario_t=np.tile(t,k);j=np.arange(k*n)
    rows=np.concatenate((t,t,n+j,n+j,n+j))
    cols=np.concatenate((c+t,d+t,c+scenario_t,d+scenario_t,e+j))
    values=np.concatenate((np.ones(2*n),np.ones(k*n),-np.ones(2*k*n)))
    ub=coo_matrix((values,(rows,cols)),shape=(n+k*n,size)).tocsr()
    rows=np.concatenate((t,t,t,t,[n,n,n]))
    cols=np.concatenate((s+t+1,s+t,c+t,d+t,[s+n,minus,plus]))
    values=np.concatenate((np.ones(n),-np.ones(n),np.full(n,-.9),np.full(n,1/.9),[1,1,-1]))
    eq=coo_matrix((values,(rows,cols)),shape=(n+1,size)).tocsr()
    return ub,eq,(c,d,s,e,minus,plus,size)


def suffix_objective(charge,discharge,soc_end,purchase,scenarios,price,config):
    emergency=np.maximum(scenarios+charge-purchase-discharge,0)
    return float(np.mean(emergency@(5*price)))+config.shortage_penalty*max(config.target_soc-soc_end,0)+config.surplus_penalty*max(soc_end-config.target_soc,0)


def rolling_plan(purchase,net_scenarios,price,initial_soc,config,final_soc=None,reference=None,strict_check=False):
    n=len(purchase);k=len(net_scenarios)
    if net_scenarios.shape!=(k,n) or not np.all(np.isfinite(net_scenarios)):raise ValueError('Invalid horizon')
    ub,eq,indices=matrices(n,k);c,d,s,e,minus,plus,size=indices
    objective=np.zeros(size);objective[e:e+k*n]=np.tile(5*price/k,k)
    objective[minus]=config.shortage_penalty;objective[plus]=config.surplus_penalty
    b_ub=np.concatenate((np.full(n,5000/6),(purchase[None,:]-net_scenarios).ravel()))
    b_eq=np.concatenate((np.zeros(n),[config.target_soc]))
    lower=np.zeros(size);upper=np.full(size,np.inf)
    upper[c:c+n]=upper[d:d+n]=5000/6
    lower[s:s+n+1]=1200;upper[s:s+n+1]=10800
    lower[s]=upper[s]=initial_soc
    if final_soc is not None:lower[s+n]=upper[s+n]=final_soc
    start=time.perf_counter()
    res=linprog(objective,A_ub=ub,b_ub=b_ub,A_eq=eq,b_eq=b_eq,bounds=np.column_stack((lower,upper)),method='highs')
    seconds=time.perf_counter()-start
    if not res.success:raise RuntimeError(f'MPC LP failed: {res.message}')
    charge=res.x[c:c+n].copy();discharge=res.x[d:d+n].copy()
    # SOC-neutral elimination: reducing c by a and d by .81a preserves SOC,
    # reduces net demand by .19a and cannot increase nonnegative emergency cost.
    remove=np.minimum(charge,discharge/.81)
    charge-=remove;discharge-=.81*remove
    soc=np.concatenate(([initial_soc],initial_soc+np.cumsum(.9*charge-discharge/.9)))
    recovered=suffix_objective(charge,discharge,soc[-1],purchase,net_scenarios,price,config)
    if abs(recovered-res.fun)>1e-5:raise ValueError('LP physical recovery changed optimal cost')
    used_reference=False;reference_gap=None
    if reference is not None:
        rc,rd=reference
        rs=np.concatenate(([initial_soc],initial_soc+np.cumsum(.9*rc-rd/.9)))
        feasible=(min(rs)>=1200-1e-6 and max(rs)<=10800+1e-6 and np.min(rc)>=-1e-7 and np.min(rd)>=-1e-7
                  and max(np.max(rc),np.max(rd))<=5000/6+1e-6 and np.max(np.minimum(rc,rd))<=1e-6
                  and (final_soc is None or abs(rs[-1]-final_soc)<1e-6))
        if feasible:
            reference_value=suffix_objective(rc,rd,rs[-1],purchase,net_scenarios,price,config)
            reference_gap=reference_value-float(res.fun)
            # Keep the existing plan when it remains optimal, avoiding arbitrary changes among ties.
            if abs(reference_gap)<=1e-6:
                charge=rc.copy();discharge=rd.copy();soc=rs;recovered=reference_value;used_reference=True
    strict_gap=None
    if strict_check:
        t=np.arange(n)
        mode_left=coo_matrix((np.ones(2*n),(np.r_[t,n+t],np.r_[c+t,d+t])),shape=(2*n,size)).tocsr()
        mode_right=coo_matrix((np.r_[np.full(n,-5000/6),np.full(n,5000/6)],(np.r_[t,n+t],np.r_[t,t])),shape=(2*n,n)).tocsr()
        mixed_ub=vstack((hstack((ub,csr_matrix((ub.shape[0],n)))),hstack((mode_left,mode_right)))).tocsr()
        mixed_eq=hstack((eq,csr_matrix((eq.shape[0],n)))).tocsr()
        strict=milp(np.r_[objective,np.zeros(n)],integrality=np.r_[np.zeros(size,dtype=int),np.ones(n,dtype=int)],
                    bounds=Bounds(np.r_[lower,np.zeros(n)],np.r_[upper,np.ones(n)]),
                    constraints=[LinearConstraint(mixed_ub,-np.inf,np.r_[b_ub,np.zeros(n),np.full(n,5000/6)]),
                                 LinearConstraint(mixed_eq,b_eq,b_eq)],options={'mip_rel_gap':1e-9,'time_limit':60})
        if not strict.success:raise RuntimeError('Strict MPC check failed')
        strict_gap=float(strict.fun-res.fun)
        if abs(strict_gap)>1e-5:raise ValueError('LP and strict MILP differ')
    return dict(charge=charge,discharge=discharge,soc=soc,objective=recovered,lower_bound=float(res.fun),
                recovery_error=abs(recovered-res.fun),seconds=seconds,used_reference=used_reference,
                reference_gap=reference_gap,strict_gap=strict_gap,removed_simultaneous_kwh=float(remove.sum()))

"""Causal multi-lead forecasts and target-free finite-horizon procurement."""
from dataclasses import dataclass
from functools import lru_cache
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


@dataclass(frozen=True)
class V3Config:
    pv_days: int = 7
    residual_window: int = 14
    archive_days: int = 3
    warmup_horizon_days: int = 2


def forecast(history, days=3, pv_days=7):
    """All indices refer to completed actual days; forecasts never enter history."""
    h = np.asarray(history, dtype=float)
    n = len(h)
    if n == 0:
        raise ValueError('Use explicit cold start without history')
    result = []
    sources = []
    for lead in range(days):
        target = n + lead
        ids = [target - 7*j for j in range(1, 5) if 0 <= target - 7*j < n]
        if not ids:
            ids = [n-1]
        load = np.average(h[ids, :, 0], axis=0, weights=.7**np.arange(len(ids)))
        pv = h[-pv_days:, :, 1].mean(axis=0)
        result.append(np.column_stack((load, pv)))
        sources.append(dict(target_day=target, load_source_days=ids,
                            pv_source_days=list(range(max(0, n-pv_days), n))))
    return np.asarray(result), sources


class ForecastArchive:
    """Issue before observe; only fully matured 72-hour residual paths are sampled."""
    def __init__(self, config=V3Config()):
        self.config = config
        self.history = []
        self.issued = {}
        self.residuals = []
        self.pending = False

    def issue(self):
        if self.pending:
            raise ValueError('Observe the previous issue before issuing again')
        n = len(self.history)
        self.pending = True
        if n == 0:
            return None, None, [], []
        f, sources = forecast(self.history, self.config.archive_days, self.config.pv_days)
        self.issued[n] = f.copy()
        selected = self.residuals[-self.config.residual_window:]
        ids = [i for i, _ in selected]
        assert all(i + self.config.archive_days <= n for i in ids)
        errors = np.asarray([e for _, e in selected]) if selected else np.zeros((1, *f.shape))
        sample = np.maximum(f[None, ...] + errors, 0.)
        net = (sample[..., 0] - sample[..., 1]).reshape(len(sample), -1)
        return f.copy(), net, ids, sources

    def observe(self, actual):
        if not self.pending:
            raise ValueError('Issue a forecast before observing actuals')
        a = np.asarray(actual, dtype=float)
        if a.shape != (144, 2) or not np.all(np.isfinite(a)) or np.min(a) < 0:
            raise ValueError('Invalid actual day')
        self.history.append(a.copy())
        n = len(self.history)
        mature = n - self.config.archive_days
        if mature in self.issued:
            # Exclude forecasts without even one week of history, as in V2.
            if mature >= 7:
                error = np.asarray(self.history[mature:n]) - self.issued[mature]
                self.residuals.append((mature, error.copy()))
        self.pending = False


@lru_cache(maxsize=32)
def matrices(n, k, strict):
    q, c, d, s, e = 0, n, 2*n, 3*n, 4*n+1
    z = e+k*n
    size = z+n if strict else z
    a = lil_matrix((n, size))
    b = lil_matrix(((2 if strict else 1)*n+k*n, size))
    for t in range(n):
        a[t, s+t+1] = 1; a[t, s+t] = -1
        a[t, c+t] = -.9; a[t, d+t] = 1/.9
        b[t, c+t] = 1
        if strict:
            b[t, z+t] = -5000/6
            b[n+t, d+t] = 1; b[n+t, z+t] = 5000/6
        else:
            b[t, d+t] = 1
    offset = (2 if strict else 1)*n
    for j in range(k):
        for t in range(n):
            r = offset+j*n+t
            b[r, c+t] = 1; b[r, d+t] = -1
            b[r, q+t] = -1; b[r, e+j*n+t] = -1
    return a.tocsr(), b.tocsr()


def optimize(f, net, price, initial_soc, strict=False):
    """No terminal equality, target, slack, penalty, or salvage objective.

    LP is exact here: remove a=min(c,d/.81) from charging and .81*a
    from discharge. SOC is unchanged and net demand falls by .19*a.
    Free disposal and no cycling rewards make this feasible and non-worsening.
    Strict MILP is retained for independent objective comparisons.
    """
    price = np.asarray(price, dtype=float)
    n = len(price); k = len(net)
    if f.shape != (n, 2) or net.shape != (k, n) or k < 1:
        raise ValueError('Horizon dimensions disagree')
    if not all(np.all(np.isfinite(x)) for x in (f, net, price)) or np.min(price) <= 0:
        raise ValueError('Invalid model inputs')
    q, c, d, s, e = 0, n, 2*n, 3*n, 4*n+1
    z = e+k*n; size = z+n if strict else z
    cost = np.zeros(size); cost[:n] = price; cost[e:z] = np.tile(5*price/k, k)
    lo = np.zeros(size); hi = np.full(size, np.inf)
    hi[c:d+n] = 5000/6; lo[s:s+n+1] = 1200; hi[s:s+n+1] = 10800
    lo[s] = hi[s] = initial_soc
    integer = np.zeros(size)
    if strict:
        hi[z:] = 1; integer[z:] = 1
    a, b = matrices(n, k, strict)
    upper = np.concatenate((np.r_[np.zeros(n), np.full(n, 5000/6)] if strict
                            else np.full(n, 5000/6), -net.ravel()))
    start = time.perf_counter()
    res = milp(cost, integrality=integer, bounds=Bounds(lo, hi),
               constraints=[LinearConstraint(a, 0, 0), LinearConstraint(b, -np.inf, upper)],
               options={'mip_rel_gap': 1e-8, 'time_limit': 120})
    if not res.success:
        raise RuntimeError(res.message)
    charge = res.x[c:c+n].copy(); discharge = res.x[d:d+n].copy()
    simultaneous = np.minimum(charge, discharge/.81)
    charge -= simultaneous; discharge -= .81*simultaneous
    purchase = res.x[:n].copy()
    soc = np.r_[initial_soc, initial_soc+np.cumsum(.9*charge-discharge/.9)]
    expected = float(np.mean(np.maximum(net+charge-discharge-purchase, 0)@(5*price)))
    objective = float(price@purchase)+expected
    error = abs(objective-res.fun)
    if error > 1e-5 or np.min(soc) < 1200-1e-5 or np.max(soc) > 10800+1e-5:
        raise ValueError('LP recovery/objective/physical validation failed')
    return dict(purchase=purchase, charge=charge, discharge=discharge, soc=soc,
                forecast=f.copy(), objective=objective, expected_emergency_cost=expected,
                planned_cost=float(price@purchase), objective_error=error,
                seconds=time.perf_counter()-start, status=int(res.status),
                recovered_charge_kwh=float(simultaneous.sum()))


def first_day(p):
    return {key: p[key][:145 if key == 'soc' else 144].copy()
            for key in ('purchase', 'charge', 'discharge', 'soc', 'forecast')}

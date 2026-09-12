"""Paper v3 reproduction: scenario recourse LP and exact PWL value control.

Current-slot feedback assumes observable constant power within each ten-minute
interval. Historical end-labelled samples are used as its proxy, not asserted
to have actually been available at interval start. No future actual is accepted
by a planner or controller.
"""
from dataclasses import dataclass
from functools import lru_cache
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

ETA, LOWER, UPPER, LIMIT = .9, 1200., 10800., 5000/6


def predict(history):
    h = np.asarray(history, dtype=float)
    n = len(h)
    if n == 0:
        raise ValueError('No history: January standby is explicit')
    ids = [i for i in range(max(0, n-35), n) if (n-i) % 7 == 0]
    if len(ids) < 2:
        ids = list(range(max(0, n-7), n))
    pv_ids = list(range(max(0, n-7), n))
    return np.column_stack((h[ids, :, 0].mean(axis=0), h[pv_ids, :, 1].mean(axis=0))), ids, pv_ids


class Archive:
    def __init__(self):
        self.history = []
        self.errors = []
        self.pending = False
        self.current = None

    def issue(self):
        if self.pending:
            raise ValueError('Previous day not observed')
        self.pending = True
        if not self.history:
            return None, None, [], [], []
        self.current, load_ids, pv_ids = predict(self.history)
        selected = self.errors[-30:]
        error = np.array([x[1] for x in selected]) if selected else np.zeros((1, 144, 2))
        scenarios = np.maximum(self.current[None]+error, 0)
        return self.current.copy(), scenarios[:, :, 0]-scenarios[:, :, 1], [x[0] for x in selected], load_ids, pv_ids

    def observe(self, actual):
        if not self.pending:
            raise ValueError('Issue before observation')
        a = np.asarray(actual, dtype=float)
        if a.shape != (144, 2) or not np.isfinite(a).all() or a.min() < 0:
            raise ValueError('Invalid actual')
        if self.current is not None:
            self.errors.append((len(self.history), a-self.current))
        self.history.append(a.copy())
        self.pending = False

    def rho(self):
        if not self.errors:
            return 0.
        errors = np.array([x[1][:, 0]-x[1][:, 1] for x in self.errors[-30:]])
        x, y = errors[:, :-1].ravel(), errors[:, 1:].ravel()
        return float(np.clip(x@y/(x@x), 0, .99)) if x@x > 1e-12 else 0.


@lru_cache(maxsize=8)
def recourse_matrices(n, k):
    # common q[n], then each scenario [c[n], d[n], E[n+1], b[n]].
    block = 4*n+1
    size = n+k*block
    er, ec, ev, ur, uc, uv = [], [], [], [], [], []
    for j in range(k):
        base = n+j*block
        c, d, s, b = base, base+n, base+2*n, base+3*n+1
        for t in range(n):
            row = j*n+t
            er.extend([row]*4); ec.extend([s+t+1, s+t, c+t, d+t]); ev.extend([1, -1, -ETA, 1/ETA])
            ur.extend([row]*4); uc.extend([t, c+t, d+t, b+t]); uv.extend([-1, 1, -1, -1])
            row = k*n+j*n+t
            ur.extend([row]*2); uc.extend([c+t, d+t]); uv.extend([1, 1])
    return (coo_matrix((ev, (er, ec)), shape=(k*n, size)).tocsr(),
            coo_matrix((uv, (ur, uc)), shape=(2*k*n, size)).tocsr())


def procure(net, price, initial_soc, value):
    """Perfect-information path recourse is a procurement approximation."""
    net, price = np.asarray(net), np.asarray(price)
    k, n = net.shape
    if price.shape != (n,) or not np.isfinite(net).all() or not (LOWER <= initial_soc <= UPPER):
        raise ValueError('Invalid procurement input')
    block = 4*n+1; size = n+k*block
    cost = np.zeros(size); cost[:n] = price
    bounds = np.column_stack((np.zeros(size), np.full(size, np.inf)))
    for j in range(k):
        base = n+j*block; s = base+2*n; b = base+3*n+1
        bounds[base:base+2*n, 1] = LIMIT
        bounds[s:s+n+1] = [LOWER, UPPER]
        bounds[s] = initial_soc
        cost[b:b+n] = 5*price/k; cost[s+n] = -value/k
    eq, ub = recourse_matrices(n, k)
    start = time.perf_counter()
    result = linprog(cost, A_eq=eq, b_eq=np.zeros(k*n), A_ub=ub,
                     b_ub=np.r_[-net.ravel(), np.full(k*n, LIMIT)], bounds=bounds, method='highs')
    if not result.success:
        raise RuntimeError(result.message)
    q = np.maximum(result.x[:n], 0)
    scenario_charge, scenario_discharge, ends, emergency = [], [], [], []
    for j in range(k):
        base = n+j*block
        c = result.x[base:base+n].copy(); d = result.x[base+n:base+2*n].copy()
        remove = np.minimum(c, d/ETA**2)
        c -= remove; d -= ETA**2*remove
        s = initial_soc+np.cumsum(ETA*c-d/ETA)
        if s.min() < LOWER-1e-5 or s.max() > UPPER+1e-5:
            raise ValueError('Recovered scenario SOC violation')
        scenario_charge.append(c); scenario_discharge.append(d); ends.append(s[-1])
        emergency.append(np.maximum(net[j]+c-d-q, 0))
    emergency = np.array(emergency); scenario_charge = np.array(scenario_charge)
    recovered = float(q@price+np.mean(emergency@(5*price)-value*np.array(ends)))
    if abs(recovered-result.fun) > 1e-5:
        raise ValueError('Recourse LP physical recovery changed objective')
    return dict(purchase=q, objective=recovered, seconds=time.perf_counter()-start,
                recovery_error=abs(recovered-result.fun),
                emergency_charging_slots=int(np.sum((scenario_charge > 1e-7) & (emergency > 1e-7))))


@dataclass
class PWL:
    """Convex function over [LOWER, UPPER], stored without an SOC grid."""
    knots: np.ndarray
    slopes: np.ndarray
    left_value: float

    def evaluate(self, e):
        if e < self.knots[0]-1e-7 or e > self.knots[-1]+1e-7:
            return np.inf
        return float(self.left_value+np.dot(self.slopes, np.clip(e-self.knots[:-1], 0, np.diff(self.knots))))


def terminal(value):
    return PWL(np.array([LOWER, UPPER]), np.array([-value]), -value*LOWER)


def backward(future, residual, price):
    """Infimal convolution with h(y), y = entering SOC - ending SOC."""
    if residual >= 0:
        left, length, slope, stage_value = 0., min(LIMIT, residual)/ETA, -5*price*ETA, 5*price*residual
    else:
        length = ETA*min(LIMIT, -residual)
        left, slope, stage_value = -length, 0., 0.
    slopes = np.r_[future.slopes, slope]
    lengths = np.r_[np.diff(future.knots), length]
    order = np.argsort(slopes, kind='stable')
    slopes, lengths = slopes[order], lengths[order]
    knots = future.knots[0]+left+np.r_[0., np.cumsum(lengths)]
    start = np.maximum(knots[:-1], LOWER); end = np.minimum(knots[1:], UPPER)
    keep = end-start > 1e-9
    below = np.clip(LOWER-knots[:-1], 0, lengths)
    value = future.left_value+stage_value+float(slopes@below)
    slopes, lengths = slopes[keep], (end-start)[keep]
    # Merge identical slopes; repeated prices otherwise accumulate redundant segments.
    begins = np.r_[0, np.flatnonzero(np.diff(slopes) > 1e-12)+1]
    lengths = np.add.reduceat(lengths, begins); slopes = slopes[begins]
    knots = LOWER+np.r_[0., np.cumsum(lengths)]; knots[-1] = UPPER
    return PWL(knots, slopes, value)


def reserve(functions, price):
    """Largest minimizer of 5*price*eta*z + the average path value."""
    derivative = 5*price*ETA+np.mean([f.slopes[0] for f in functions])
    if derivative > 1e-10:
        return LOWER
    positions = np.concatenate([f.knots[1:-1] for f in functions])
    changes = np.concatenate([np.diff(f.slopes) for f in functions])/len(functions)
    order = np.argsort(positions, kind='stable')
    after = derivative+np.cumsum(changes[order])
    positive = np.flatnonzero(after > 1e-10)
    return float(positions[order[positive[0]]]) if len(positive) else UPPER


def reserves(net, purchase, price, value):
    k, n = net.shape
    functions = [terminal(value) for _ in range(k)]
    levels = np.empty(n); max_segments = 1
    for t in range(n-1, -1, -1):
        levels[t] = reserve(functions, price[t])
        functions = [backward(f, net[j, t]-purchase[t], price[t]) for j, f in enumerate(functions)]
        max_segments = max(max_segments, max(len(f.slopes) for f in functions))
    return levels, functions, max_segments


def action(residual, soc, level=LOWER):
    if residual <= 0:
        return min(-residual, LIMIT, max(0., UPPER-soc)/ETA), 0.
    return 0., min(residual, LIMIT, ETA*max(0., soc-level))


def mpc_level(t, current_net, nominal_net, purchase, price, value, rho):
    """Single forecast MPC solved exactly by the same 1-D PWL recursion.

    rho uses within-day adjacent residual regression through the origin;
    the manuscript does not specify intercept handling.
    """
    future = terminal(value)
    innovation = current_net-nominal_net[t]
    for j in range(len(price)-1, t, -1):
        predicted = nominal_net[j]+rho**(j-t)*innovation
        future = backward(future, predicted-purchase[j], price[j])
    return reserve([future], price[t])

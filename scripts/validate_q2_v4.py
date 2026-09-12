"""Independent small-instance oracles and execution checks for paper V4."""
import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import hstack, vstack, eye, csr_matrix, lil_matrix

from q2_v4 import LOWER, UPPER, ETA, LIMIT, backward, terminal, reserve, action, procure, recourse_matrices, predict, Archive
from validate_q2 import audit


def path_lp(residual, price, soc, value):
    """Independent x/E Bellman oracle with direction-restricted actions."""
    n = len(residual)
    cost = np.zeros(2*n+1)
    cost[:n] = np.where(residual > 0, 5*price*ETA, 0.)
    cost[-1] = -value
    a = np.zeros((n, 2*n+1))
    bounds = []
    for t, r in enumerate(residual):
        bounds.append((-min(LIMIT, max(r, 0))/ETA, ETA*min(LIMIT, max(-r, 0))))
        a[t, t] = -1; a[t, n+t] = -1; a[t, n+t+1] = 1
    bounds += [(soc, soc)]+[(LOWER, UPPER)]*n
    result = linprog(cost, A_eq=a, b_eq=np.zeros(n), bounds=bounds, method='highs')
    if not result.success:
        raise ValueError(result.message)
    return float(result.fun+np.maximum(residual, 0)@(5*price))


def model_tests():
    rng = np.random.default_rng(20260911)
    max_error = 0.; count = 0
    for n in (1, 2, 6, 24, 144):
        for _ in range(3):
            residual = rng.uniform(-1500, 1500, n)
            price = rng.choice([0., .43, .88, 1.4], n)
            value = .48154814814814817
            f = terminal(value)
            for t in range(n-1, -1, -1):
                f = backward(f, residual[t], price[t])
            assert np.all(np.diff(f.slopes) >= -1e-10)
            assert max(f.slopes) <= 1e-10
            for e in (LOWER, 6000., UPPER):
                error = abs(f.evaluate(e)-path_lp(residual, price, e, value))
                max_error = max(max_error, error); count += 1
                assert error < 1e-6, (n, error)
    for _ in range(40):
        functions = []
        for j in range(5):
            f = terminal(.48)
            for t in range(8):
                f = backward(f, rng.uniform(-2000, 2000), rng.choice([.4, 1.4]))
            functions.append(f)
        p = rng.choice([0., .4, 1.4]); level = reserve(functions, p)
        knots = np.unique(np.concatenate([f.knots for f in functions]))
        costs = np.array([5*p*ETA*z+np.mean([f.evaluate(z) for f in functions]) for z in knots])
        assert abs(5*p*ETA*level+np.mean([f.evaluate(level) for f in functions])-costs.min()) < 1e-6
        e, r = rng.uniform(LOWER, UPPER), rng.uniform(0, 2000)
        c, d = action(r, e, level)
        lo = max(LOWER, e-min(LIMIT, r)/ETA)
        candidates = np.r_[lo, e, knots[(knots > lo) & (knots < e)]]
        actual = 5*p*(r-d)+np.mean([f.evaluate(e-d/ETA) for f in functions])
        best = min(5*p*(r+ETA*(z-e))+np.mean([f.evaluate(z) for f in functions]) for z in candidates)
        assert abs(actual-best) < 1e-6
    # Small recourse LP versus a strict binary charge/discharge MILP.
    strict_errors = []
    for n in (2, 6):
        k = 3; net = rng.uniform(-500, 1500, (k, n)); price = rng.uniform(.3, 1.4, n); v = .48
        p = procure(net, price, 6000., v)
        eq, ub = recourse_matrices(n, k); size = eq.shape[1]; block = 4*n+1
        cost = np.zeros(size+k*n); cost[:n] = price
        lo = np.zeros(size+k*n); hi = np.full(size+k*n, np.inf); hi[size:] = 1
        mode = lil_matrix((2*k*n, size+k*n))
        for j in range(k):
            base = n+j*block; s = base+2*n; b = base+3*n+1
            hi[base:base+2*n] = LIMIT; lo[s:s+n+1] = LOWER; hi[s:s+n+1] = UPPER
            lo[s] = hi[s] = 6000.; cost[b:b+n] = 5*price/k; cost[s+n] = -v/k
            for t in range(n):
                mode[j*n+t, base+t] = 1; mode[j*n+t, size+j*n+t] = -LIMIT
                mode[k*n+j*n+t, base+n+t] = 1; mode[k*n+j*n+t, size+j*n+t] = LIMIT
        result = milp(cost, integrality=np.r_[np.zeros(size), np.ones(k*n)], bounds=Bounds(lo, hi),
                      constraints=[LinearConstraint(hstack((eq, csr_matrix((k*n, k*n)))), 0, 0),
                                   LinearConstraint(hstack((ub, csr_matrix((2*k*n, k*n)))), -np.inf, np.r_[-net.ravel(), np.full(k*n, LIMIT)]),
                                   LinearConstraint(mode.tocsr(), -np.inf, np.r_[np.zeros(k*n), np.full(k*n, LIMIT)])])
        assert result.success
        strict_errors.append(abs(result.fun-p['objective']))
        assert strict_errors[-1] < 1e-6
    archive = Archive()
    for i in range(32):
        f, net, ids, load_ids, pv_ids = archive.issue()
        assert all(j < i for j in ids+load_ids+pv_ids)
        if i == 31:
            assert ids == list(range(1, 31)) and len(net) == 30
        archive.observe(rng.uniform(0, 1000, (144, 2)))
    return dict(path_lp_comparisons=count, max_path_value_error=max_error,
                reserve_projection_cases=40, recourse_milp_max_error=max(strict_errors),
                archive_chronology=True)


def execution_audit(rows, observation_mode='instant'):
    result = audit(rows, require_boundaries=False, require_fixed_storage=False)
    previous = None
    for r in rows:
        if r['phase'] == 'warmup':
            assert max(abs(r[k]) for k in ('purchase_kwh', 'charge_kwh', 'discharge_kwh')) < 1e-6
            continue
        if r['slot'] == 1:
            previous = None
        actual_net = r['load_kwh']-r['pv_actual_kwh']
        nominal_net = r['load_forecast_kwh']-r['pv_forecast_kwh']
        if observation_mode == 'instant':
            observed_net = actual_net
        elif observation_mode in ('delayed_1', 'delayed_2', 'delayed_3'):
            observed_net = nominal_net if previous is None else nominal_net + previous['actual_net'] - previous['nominal_net']
        else:
            raise ValueError(f'Unknown observation mode: {observation_mode}')
        residual = observed_net-r['purchase_kwh']
        if observation_mode == 'delayed_2':
            upper = observed_net + r['risk_upper_kwh']
            lower = observed_net + r['risk_lower_kwh']
            if upper < r['purchase_kwh']:
                residual = upper-r['purchase_kwh']
            elif lower > r['purchase_kwh']:
                residual = lower-r['purchase_kwh']
            else:
                residual = 0.
        elif observation_mode == 'delayed_3':
            upper = observed_net + r['risk_upper_kwh']
            if observed_net <= r['purchase_kwh'] and upper < r['purchase_kwh']:
                residual = upper-r['purchase_kwh']
            elif observed_net > r['purchase_kwh']:
                residual = observed_net-r['purchase_kwh']
            else:
                residual = 0.
        c, d = action(residual, r['soc_start_kwh'], r['reserve_kwh'])
        if max(abs(c-r['charge_kwh']), abs(d-r['discharge_kwh'])) > 1e-6:
            raise ValueError('Action does not match declared feedback rule')
        if observation_mode == 'instant' and min(r['charge_kwh'], r['emergency_kwh']) > 1e-6:
            raise ValueError('Emergency charging forbidden by V4 execution policy')
        previous = dict(actual_net=actual_net, nominal_net=nominal_net)
    result['feedback_rule_verified'] = True
    result['observation_mode'] = observation_mode
    result['current_observation_is_assumed_instantaneous_proxy'] = observation_mode == 'instant'
    result['emergency_charging_is_allowed_by_delayed_information'] = observation_mode in ('delayed_1', 'delayed_2')
    return result


if __name__ == '__main__':
    print(model_tests())

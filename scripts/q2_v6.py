"""Q2 V6: Cross-day DP value function coordination and delayed-robust hybrid dispatch."""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix

from q2_v4 import PWL, backward, reserve, action, ETA, LOWER, UPPER, LIMIT


def tomorrow_cuts(f_tom: np.ndarray, net_tom: np.ndarray, price: np.ndarray,
                  terminal_soc: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the convex piecewise linear cost-to-go function for tomorrow by sampling exact LP values and convex interpolation.
    
    Returns:
        soc_pts: Support SOC grid points in [1200, 10800]
        costs: Optimal risk procurement costs at each support point
        slopes: Secant slopes (upper interpolation, not supporting subgradients)
    """
    n = 144
    k = len(net_tom)
    q_s, c_s, d_s, s_s, e_s = 0, n, 2*n, 3*n, 4*n + 1
    size = e_s + k*n
    c_obj = np.zeros(size)
    c_obj[q_s:q_s+n] = price
    c_obj[e_s:] = np.tile(5*price/k, k)

    bounds = [(0, np.inf)] * size
    for t in range(n):
        bounds[c_s+t] = (0, 5000/6)
        bounds[d_s+t] = (0, 5000/6)
        bounds[s_s+t] = (1200, 10800)
    bounds[s_s+n] = (1200, 10800)
    if terminal_soc is not None:
        bounds[s_s+n] = (terminal_soc, terminal_soc)

    # SOC dynamic constraints
    eq_rows = n
    A_eq = lil_matrix((eq_rows, size))
    b_eq = np.zeros(eq_rows)
    for t in range(n):
        A_eq[t, s_s+t+1] = 1
        A_eq[t, s_s+t] = -1
        A_eq[t, c_s+t] = -0.9
        A_eq[t, d_s+t] = 1/0.9

    # Power bounds & scenario balance
    ub_rows = n + k*n
    A_ub = lil_matrix((ub_rows, size))
    b_ub = np.zeros(ub_rows)
    for t in range(n):
        A_ub[t, c_s+t] = 1
        A_ub[t, d_s+t] = 1
        b_ub[t] = 5000/6
    for j in range(k):
        for t in range(n):
            row = n + j*n + t
            A_ub[row, q_s+t] = -1
            A_ub[row, d_s+t] = -1
            A_ub[row, c_s+t] = 1
            A_ub[row, e_s+j*n+t] = -1
            b_ub[row] = -net_tom[j, t]

    soc_pts = np.linspace(1200, 10800, 9)
    costs = []
    A_eq_csr = A_eq.tocsr()
    A_ub_csr = A_ub.tocsr()
    for s_val in soc_pts:
        bnd = list(bounds)
        bnd[s_s] = (s_val, s_val)
        res = linprog(c_obj, A_eq=A_eq_csr, b_eq=b_eq, A_ub=A_ub_csr, b_ub=b_ub,
                      bounds=bnd, method='highs')
        if not res.success:
            raise RuntimeError(f'Tomorrow LP failed at SOC {s_val}: {res.message}')
        costs.append(res.fun)

    costs = np.array(costs)
    slopes = np.diff(costs) / np.diff(soc_pts)
    if np.min(np.diff(slopes)) < -1e-7:
        raise ValueError("Nonconvex sampled continuation value")
    return soc_pts, costs, slopes


def risk_plan_v6(forecast: np.ndarray, net_scenarios: np.ndarray, price: np.ndarray,
                 initial_soc: float, cuts: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
                 final_soc: Optional[float] = None) -> Dict[str, Any]:
    """Single-day day-ahead MILP with cross-day DP cutting planes."""
    n = 144
    k = len(net_scenarios)
    q, c, d, s, z, e = 0, n, 2*n, 3*n, 4*n+1, 5*n+1
    has_cuts = (cuts is not None) and (final_soc is None)
    size = e + k*n + (1 if has_cuts else 0)
    theta = size - 1 if has_cuts else None

    cost = np.zeros(size)
    cost[q:q+n] = price
    cost[e:e+k*n] = np.tile(5*price/k, k)
    if has_cuts:
        cost[theta] = 1.0

    low = np.zeros(size)
    high = np.full(size, np.inf)
    high[c:c+n] = high[d:d+n] = 5000/6
    low[s:s+n+1] = 1200
    high[s:s+n+1] = 10800
    low[s] = high[s] = initial_soc
    if final_soc is not None:
        low[s+n] = high[s+n] = final_soc
    high[z:z+n] = 1
    integer = np.zeros(size, dtype=int)
    integer[z:z+n] = 1

    M = len(cuts[2]) if has_cuts else 0
    total_con = 3*n + k*n + M
    A = lil_matrix((total_con, size))
    lower = np.full(total_con, -np.inf)
    upper = np.zeros(total_con)

    for t in range(n):
        A[t, s+t+1] = 1
        A[t, s+t] = -1
        A[t, c+t] = -0.9
        A[t, d+t] = 1/0.9
        lower[t] = upper[t] = 0
        A[n+t, c+t] = 1
        A[n+t, z+t] = -5000/6
        A[2*n+t, d+t] = 1
        A[2*n+t, z+t] = 5000/6
        upper[2*n+t] = 5000/6

    for j in range(k):
        for t in range(n):
            row = 3*n + j*n + t
            A[row, q+t] = 1
            A[row, d+t] = 1
            A[row, c+t] = -1
            A[row, e+j*n+t] = 1
            lower[row] = net_scenarios[j, t]
            upper[row] = np.inf

    if has_cuts:
        soc_pts, costs, slopes = cuts
        for m in range(M):
            row = 3*n + k*n + m
            A[row, theta] = -1
            A[row, s+n] = slopes[m]
            upper[row] = -(costs[m] - slopes[m] * soc_pts[m])

    start = time.perf_counter()
    res = milp(cost, integrality=integer, bounds=Bounds(low, high),
               constraints=LinearConstraint(A.tocsr(), lower, upper),
               options={'mip_rel_gap': 1e-8, 'time_limit': 60})
    seconds = time.perf_counter() - start
    if not res.success:
        raise RuntimeError(f'V6 MILP failed: {res.message}')

    q_opt = res.x[q:q+n].copy()
    c_opt = res.x[c:c+n].copy()
    d_opt = res.x[d:d+n].copy()
    s_opt = res.x[s:s+n+1].copy()
    theta_opt = float(res.x[theta]) if has_cuts else 0.0

    return dict(
        purchase=q_opt, charge=c_opt, discharge=d_opt, soc=s_opt,
        theta=theta_opt, objective=float(res.fun), planned_cost=float(price @ q_opt),
        gap=float(res.mip_gap), seconds=seconds, status=int(res.status)
    )


def compute_reserves_v6(net_today: np.ndarray, purchase: np.ndarray, price: np.ndarray,
                        cuts: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None) -> np.ndarray:
    """Backwards infimal convolution over today's 144 slots using tomorrow's convex value function."""
    k, n = net_today.shape
    if cuts is not None:
        soc_pts, costs, slopes = cuts
        terminal_pwl = PWL(soc_pts, slopes, costs[0])
    else:
        terminal_pwl = PWL(np.array([1200., 10800.]), np.array([-0.48]), -0.48*1200.)

    functions = [terminal_pwl for _ in range(k)]
    levels = np.empty(n)
    for t in range(n-1, -1, -1):
        levels[t] = reserve(functions, price[t])
        functions = [backward(f, net_today[j, t] - purchase[t], price[t]) for j, f in enumerate(functions)]
    return levels


def v6_hybrid_action(residual: float, soc: float, level: float,
                     ref_charge: float, ref_discharge: float, threshold: float) -> Tuple[float, float]:
    """Execute reference arbitrage action unless delayed residual breaches guard threshold."""
    if np.isinf(threshold):
        # Explicit reference-only ablation: no reserve clipping or feedback.
        return (min(max(ref_charge, 0.), LIMIT, max(0., UPPER-soc)/ETA),
                min(max(ref_discharge, 0.), LIMIT, ETA*max(0., soc-LOWER)))
    dp_charge, dp_discharge = action(residual, soc, level)
    charge, discharge = float(ref_charge), float(ref_discharge)
    if residual > threshold:
        # A severe delayed deficit makes scheduled charging unsafe; discharge down to reserve level
        charge = 0.0
        discharge = max(discharge, dp_discharge)
    elif residual < -threshold:
        # A severe delayed surplus makes scheduled discharging unsafe; charge to absorb surplus
        discharge = 0.0
        charge = max(charge, dp_charge)

    charge = min(max(charge, 0.), 5000. / 6)
    discharge = min(max(discharge, 0.), 5000. / 6)
    if charge > 1e-9 and discharge > 1e-9:
        if residual > 0:
            charge = 0.0
        else:
            discharge = 0.0
    charge = min(charge, max(0., UPPER - soc) / ETA)
    discharge = min(discharge, ETA * max(0., soc - level))
    return charge, discharge


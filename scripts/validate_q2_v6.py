"""Targeted causality and continuation approximation checks for corrected V6."""
import json
import sqlite3
from pathlib import Path
import numpy as np
from solve_q2_v6 import causal_inputs
from q2_v6 import tomorrow_cuts, risk_plan_v6, v6_hybrid_action
from q2_v3 import optimize

root=Path(__file__).resolve().parents[1]
with sqlite3.connect(f'file:{root}/data/processed/microgrid.sqlite?mode=ro',uri=True) as con:
    actual=np.array(con.execute('SELECT load_kwh,pv_actual_kwh FROM actuals ORDER BY date,slot').fetchall()).reshape(365,144,2)
    price=np.array([r[0] for r in con.execute('SELECT price_yuan_per_kwh FROM baseline ORDER BY slot')])
checks=[]
for day in (31,171,265):
    inputs=causal_inputs(actual,day)
    changed=actual.copy();changed[day:]+=700
    other=causal_inputs(changed,day)
    assert all(np.array_equal(a,b) for a,b in zip(inputs,other))
    ft,nt,fm,nm=inputs
    cuts=tomorrow_cuts(fm,nm,price)
    p=risk_plan_v6(ft,nt,price,6000.,cuts)
    direct=optimize(np.concatenate((ft,fm)),np.concatenate((nt,nm),axis=1),
                    np.tile(price,2),6000.)
    error=p['objective']-direct['objective']
    assert error>=-1e-4
    # Exact value at selected intermediate SOC versus the chord interpolation.
    state=float(p['soc'][-1])
    exact=optimize(fm,nm,price,state)
    interpolation=float(np.interp(state,cuts[0],cuts[1]))
    assert interpolation>=exact['objective']-1e-4
    checks.append(dict(day_index=day,causality_suffix_invariance=True,
        interpolated_48h_objective=p['objective'],direct_48h_objective=direct['objective'],
        approximation_gap_yuan=error,chosen_soc=state,
        continuation_interpolation_error_yuan=interpolation-exact['objective']))
assert v6_hybrid_action(2000.,6000.,5900.,0.,500.,float('inf'))==(0.,500.)
out=root/'results/q2_v6_corrected';out.mkdir(exist_ok=True)
(out/'targeted_validation.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))

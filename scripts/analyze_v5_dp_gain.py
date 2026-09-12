"""Counterfactual gain audit for V5 hybrid DP corrections.

The audit keeps the V2-B purchase contract and the observed SOC fixed at each
slot.  It compares the immediate expected emergency cost of the reference
action against the hybrid action on the issued historical residual scenarios.
This is deliberately a one-step diagnostic; SOC propagation is reported
separately by the realized V5 execution.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sqlite3

import numpy as np

from q2_bias import BiasConfig, BiasForecaster
from q2_risk import scenarios
from q2_v4 import action
from solve_q2_v2a import read_execution

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/q2_v5_hybrid'


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_v5_execution(path: Path) -> list[dict]:
    string_fields = {'date', 'phase', 'interval_start', 'interval_end', 'interval_label',
                     'plan_issue_time', 'control_issue_time', 'last_observation_end',
                     'raw_sample_label_time', 'observation_model', 'history_available_through'}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = []
        for raw in csv.DictReader(stream):
            rows.append({key: value if key in string_fields or value == '' else int(value) if key == 'slot' else float(value)
                         for key, value in raw.items()})
        return rows


def cost_for(residuals: np.ndarray, charge: float, discharge: float, price: float) -> float:
    emergency = np.maximum(residuals + charge - discharge, 0.)
    return float(np.mean(emergency) * 5. * price)


def main() -> None:
    experiment = json.loads((OUT / 'experiment.json').read_text(encoding='utf-8'))
    v2b = json.loads((ROOT / 'results/q2_v2b/experiment.json').read_text(encoding='utf-8'))
    bias = BiasConfig(**v2b['bias_config'])
    config = type('BaseConfig', (), v2b['base_config'])()
    window = int(v2b['window_days'])
    with sqlite3.connect(f'file:{(ROOT / "data/processed/microgrid.sqlite").as_posix()}?mode=ro', uri=True) as con:
        flat = con.execute('SELECT date,load_kwh,pv_actual_kwh FROM actuals ORDER BY date,slot').fetchall()
        price = np.array([r[0] for r in con.execute('SELECT price_yuan_per_kwh FROM baseline ORDER BY slot')])
    actual = np.array([[r[1], r[2]] for r in flat], dtype=float).reshape(365, 144, 2)
    dates = [flat[d * 144][0] for d in range(365)]
    v2b_rows = read_execution(ROOT / 'results/q2_v2b/execution.csv')
    v5_rows = read_v5_execution(OUT / 'execution.csv')
    formal = v5_rows[31 * 144:]

    forecaster = BiasForecaster(config)
    for d in range(31):
        forecaster.forecast(bias)
        forecaster.observe(actual[d])

    slot_rows: list[dict] = []
    daily_rows: list[dict] = []
    cursor = 0
    for day in range(31, 365):
        forecast, _ = forecaster.forecast(bias)
        net = scenarios(forecast, forecaster.residuals, window)
        day_rows = formal[cursor:cursor + 144]
        cursor += 144
        ref_expected = final_expected = 0.
        ref_actual = final_actual = 0.
        corrected = 0
        expected_gain = actual_gain = 0.
        for t, row in enumerate(day_rows):
            nominal = forecast[t, 0] - forecast[t, 1]
            observed = nominal if t == 0 else nominal + actual[day, t - 1, 0] - actual[day, t - 1, 1] - (forecast[t - 1, 0] - forecast[t - 1, 1])
            residual = float(observed - row['purchase_kwh'])
            dp_charge, dp_discharge = action(residual, row['soc_start_kwh'], row['reserve_kwh'])
            ref_charge = row['reference_charge_kwh']
            ref_discharge = row['reference_discharge_kwh']
            final_charge = row['charge_kwh']
            final_discharge = row['discharge_kwh']
            changed = abs(final_charge - ref_charge) > 1e-8 or abs(final_discharge - ref_discharge) > 1e-8
            if changed:
                corrected += 1
            scen_residual = net[:, t] - row['purchase_kwh']
            ref_c = cost_for(scen_residual, ref_charge, ref_discharge, price[t])
            final_c = cost_for(scen_residual, final_charge, final_discharge, price[t])
            ref_a = cost_for(np.array([residual]), ref_charge, ref_discharge, price[t])
            final_a = cost_for(np.array([residual]), final_charge, final_discharge, price[t])
            dg_c = cost_for(scen_residual, dp_charge, dp_discharge, price[t])
            ref_expected += ref_c
            final_expected += final_c
            ref_actual += ref_a
            final_actual += final_a
            expected_gain += ref_c - final_c
            actual_gain += ref_a - final_a
            slot_rows.append(dict(date=dates[day], slot=t + 1, interval_label=row['interval_label'],
                                  price_yuan_per_kwh=float(price[t]), reserve_kwh=row['reserve_kwh'],
                                  delayed_residual_kwh=residual, reference_charge_kwh=ref_charge,
                                  reference_discharge_kwh=ref_discharge, dp_charge_kwh=dp_charge,
                                  dp_discharge_kwh=dp_discharge, final_charge_kwh=final_charge,
                                  final_discharge_kwh=final_discharge, correction_applied=int(changed),
                                  reference_expected_emergency_cost_yuan=ref_c,
                                  final_expected_emergency_cost_yuan=final_c,
                                  dp_expected_emergency_cost_yuan=dg_c,
                                  expected_gain_yuan=ref_c - final_c,
                                  reference_realized_emergency_cost_yuan=ref_a,
                                  final_realized_emergency_cost_yuan=final_a,
                                  realized_gain_yuan=ref_a - final_a))
        daily_rows.append(dict(date=dates[day], corrected_intervals=corrected,
                               reference_expected_cost_yuan=ref_expected,
                               final_expected_cost_yuan=final_expected,
                               expected_gain_yuan=expected_gain,
                               reference_realized_cost_yuan=ref_actual,
                               final_realized_cost_yuan=final_actual,
                               realized_gain_yuan=actual_gain))
        forecaster.observe(actual[day])

    totals = {key: float(sum(row[key] for row in daily_rows)) for key in (
        'reference_expected_cost_yuan', 'final_expected_cost_yuan', 'expected_gain_yuan',
        'reference_realized_cost_yuan', 'final_realized_cost_yuan', 'realized_gain_yuan')}
    total_corrected = int(sum(row['corrected_intervals'] for row in daily_rows))
    result = dict(version='q2-v5-dp-correction-counterfactual',
                  formal_days=334, corrected_intervals=total_corrected,
                  correction_rate=total_corrected / (334 * 144), totals=totals,
                  expected_gain_per_correction_yuan=totals['expected_gain_yuan'] / max(total_corrected, 1),
                  realized_gain_per_correction_yuan=totals['realized_gain_yuan'] / max(total_corrected, 1),
                  interpretation='one-step emergency-cost counterfactual; SOC propagation remains in V5 execution',
                  selected_threshold=experiment['selected_candidate']['threshold_kwh'])
    write_csv(OUT / 'dp_correction_gain_by_slot.csv', slot_rows)
    write_csv(OUT / 'dp_correction_gain_daily.csv', daily_rows)
    (OUT / 'dp_correction_gain.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# V5 DP 纠偏动作收益诊断', '',
             '固定 V2-B 购电合同和 V5 实际 SOC，在每个时段比较参考动作与最终动作的一步紧急购电期望成本。该指标不把后续 SOC 传播提前折算进来。', '',
             f'- 纠偏时段：{total_corrected}/{334 * 144}（{result["correction_rate"]:.4%}）。',
             f'- 一步期望收益：{totals["expected_gain_yuan"]:.6f} 元。',
             f'- 实际观测收益：{totals["realized_gain_yuan"]:.6f} 元。',
             f'- 每次纠偏期望收益：{result["expected_gain_per_correction_yuan"]:.6f} 元；实际收益：{result["realized_gain_per_correction_yuan"]:.6f} 元。', '',
             '正值表示最终动作相对 V2-B 参考动作减少紧急购电费用；负值表示纠偏动作在该反事实口径下反而增加费用。', '',
             '- 明细：`results/q2_v5_hybrid/dp_correction_gain_by_slot.csv`。',
             '- 日汇总：`results/q2_v5_hybrid/dp_correction_gain_daily.csv`。']
    (ROOT / 'reports/q2_v5_dp_gain.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

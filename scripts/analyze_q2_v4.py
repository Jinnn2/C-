"""Compare persisted V4 output against the manuscript's printed main results."""
import hashlib
import json
from pathlib import Path

import numpy as np

from solve_q2_v4 import OUT, ROOT, read_rows, write_csv


def main():
    experiment = json.loads((OUT/'experiment.json').read_text(encoding='utf-8'))
    dp = [r for r in read_rows(OUT/'dp_closed/execution.csv') if r['phase'] == 'evaluation']
    greedy = [r for r in read_rows(OUT/'greedy_closed/execution.csv') if r['phase'] == 'evaluation']
    # Compare only at the precision actually printed in the paper.
    expected_daily = {
        '2025-03-20': (65969.247, 176.258, 41450.015, 10380.153, 9794.565),
        '2025-06-21': (36166.839, 0., 21588.749, 9278.712, 9851.039),
        '2025-09-23': (64647.304, 86.387, 42581.882, 7486.599, 6462.100),
        '2025-12-21': (96458.551, 0., 62354.119, 10133.788, 10291.557)}
    keys = ['purchase_kwh', 'emergency_kwh', 'total_cost_yuan', 'soc_start_kwh', 'soc_end_kwh']
    comparisons = []
    for date, values in expected_daily.items():
        rows = [r for r in dp if r['date'] == date]
        measured = [sum(r[k] for r in rows) for k in keys[:3]]+[rows[0][keys[3]], rows[-1][keys[4]]]
        for key, actual, paper in zip(keys, measured, values):
            comparisons.append(dict(date=date, metric=key, paper_value=paper, reproduced_value=actual,
                                    difference=actual-paper, within_printed_rounding=abs(actual-paper) <= .000501))
    expected_slots = {
        '2025-03-20': [0., 514.286229, 0., 475.573500, 661.650237, 0.],
        '2025-06-21': [0., 0., 0., 166.574611, 389.515719, 0.],
        '2025-09-23': [0., 307.083611, 0., 590.321211, 693.940577, 79.631807],
        '2025-12-21': [0., 909.778463, 0., 876.473946, 677.748697, 0.]}
    for date, values in expected_slots.items():
        rows = [r for r in dp if r['date'] == date]
        for slot, paper in zip((60, 72, 84, 96, 108, 120), values):
            actual = rows[slot]['purchase_kwh']
            comparisons.append(dict(date=date, metric=rows[slot]['interval_label']+' purchase_kwh', paper_value=paper,
                                    reproduced_value=actual, difference=actual-paper,
                                    within_printed_rounding=abs(actual-paper) <= .00000051))
    monthly = []
    for month in sorted({r['date'][:7] for r in dp}):
        dc = sum(r['total_cost_yuan'] for r in dp if r['date'].startswith(month))
        gc = sum(r['total_cost_yuan'] for r in greedy if r['date'].startswith(month))
        monthly.append(dict(month=month, dp_cost_yuan=dc, greedy_cost_yuan=gc, savings_yuan=gc-dc))
    daily_savings = np.array([sum(r['total_cost_yuan'] for r in greedy[i:i+144])-sum(r['total_cost_yuan'] for r in dp[i:i+144])
                              for i in range(0, len(dp), 144)])
    with np.load(OUT/'forecast_archive.npz') as a:
        forecasts, errors, days = a['forecast'], a['residual'], a['residual_day']
        all_rows = read_rows(OUT/'dp_closed/execution.csv')
        actual = np.array([[r['load_kwh'], r['pv_actual_kwh']] for r in all_rows]).reshape(365, 144, 2)
        from q2_v4 import predict
        for d in range(1, 365):
            assert np.array_equal(forecasts[d-1], predict(actual[:d])[0])
        for d, error in zip(days, errors):
            assert np.array_equal(error, actual[d]-forecasts[d-1])
    result = dict(printed_entries=len(comparisons), matched_entries=sum(r['within_printed_rounding'] for r in comparisons),
                  max_designated_difference=max(abs(r['difference']) for r in comparisons),
                  improved_days=int(sum(daily_savings > .005)), worsened_days=int(sum(daily_savings < -.005)),
                  unchanged_days=int(sum(abs(daily_savings) <= .005)),
                  improved_months=sum(r['savings_yuan'] > .005 for r in monthly),
                  persisted_forecasts_and_residuals_reconstructed=True,
                  code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [ROOT/'scripts/q2_v4.py', ROOT/'scripts/solve_q2_v4.py', ROOT/'scripts/validate_q2_v4.py']})
    write_csv(OUT/'printed_table_comparison.csv', comparisons)
    write_csv(OUT/'monthly_comparison.csv', monthly)
    (OUT/'paper_table_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    report = ROOT/'reports/q2_v4_experiment.md'
    text = report.read_text(encoding='utf-8').split('\n## 印刷表格逐项核对')[0]
    text += '\n## 印刷表格逐项核对\n\n'
    text += f"四个指定日期的总量、费用、首末电量及指定时段购电共{len(comparisons)}项，{result['matched_entries']}项落在论文印刷舍入范围内。最大数值差{result['max_designated_difference']:.9f}。逐项见printed_table_comparison.csv。\n\n"
    text += f"DP闭环相对解析闭环改善{result['improved_days']}天、恶化{result['worsened_days']}天、基本相同{result['unchanged_days']}天（费用差阈值0.005元）；{result['improved_months']}个月改善。\n\n"
    text += '已从持久化实际轨迹独立重建全部历史预测及残差，核对档案一致。论文表格若不匹配，保留差异，不按全年结果反向修改参数。\n'
    report.write_text(text, encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

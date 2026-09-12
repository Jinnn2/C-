"""V4-B: delayed-1 control with a causal emergency-gap purchase margin.

The V4 controller is kept unchanged.  V4-B adds a slot-wise quantile of
historical V4 delayed-1 emergency gaps to the day-ahead contract.  At formal
day d, only gaps from formal days before d are used; the margin is frozen at
the selected quantile and never uses the current day's actual values.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import scipy

from q2_v4 import Archive, LOWER, procure, reserves
from solve_q2 import execution_rows, events, summarize, workbook, verify_export
from solve_q2_v4 import simulate
from validate_q2_v4 import execution_audit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'results/q2_v4b_delayed1'
ALPHAS = (0.0, 0.6, 0.8, 0.9)


def write_csv(path: Path, rows):
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safety_margin(gap_history: np.ndarray, day_index: int, alpha: float) -> np.ndarray:
    """Causal emergency-gap margin, in kWh per ten-minute slot."""
    if alpha <= 0.0 or day_index <= 0:
        return np.zeros(144)
    start = max(0, day_index - 30)
    history = np.asarray(gap_history[start:day_index], dtype=float)
    if history.size == 0:
        return np.zeros(144)
    return np.maximum(np.quantile(history, alpha, axis=0), 0.0)


def procure_with_margin(net, price, initial_soc, value, margin):
    base = procure(net, price, initial_soc, value)
    purchase = base['purchase'] + margin
    return base, purchase


def evaluate_candidate(actual, dates, price, value, alpha):
    """January development replay used only to choose alpha."""
    archive = Archive()
    state = 6000.0
    rows = []
    margin_total = 0.0
    for day, ds in enumerate(dates[:31]):
        forecast, net, *_ = archive.issue()
        if forecast is None:
            warm, state = simulate(ds, np.zeros(144), np.zeros((144, 2)), actual[day], price,
                                    state, np.full(144, LOWER), 'greedy', value, 0.0,
                                    warmup=True, observation_mode='delayed_1')
            rows.extend(warm)
        else:
            # January is only a parameter-development replay.  Forecast-error
            # history is used here because no formal V4 gap archive exists yet.
            if archive.errors and alpha > 0:
                errors = np.asarray([error[:, 0] - error[:, 1]
                                     for _, error in archive.errors[-30:]], dtype=float)
                margin = np.maximum(np.quantile(errors, alpha, axis=0), 0.0)
            else:
                margin = np.zeros(144)
            margin_total += float(margin.sum())
            base, purchase = procure_with_margin(net, price, state, value, margin)
            levels, _, _ = reserves(net, purchase, price, value)
            day_rows, state = simulate(ds, purchase, forecast, actual[day], price, state,
                                       levels, 'dp', value, 0.0,
                                       observation_mode='delayed_1')
            rows.extend(day_rows)
        archive.observe(actual[day])
    selected = [r for r in rows if r['date'] >= '2025-01-15']
    summary = summarize(selected)
    return dict(alpha=alpha, margin_kwh=margin_total,
                total_cost_yuan=summary['total_cost_yuan'],
                planned_cost_yuan=summary['planned_cost_yuan'],
                emergency_cost_yuan=summary['emergency_cost_yuan'],
                emergency_kwh=summary['emergency_kwh'])


def run_formal(actual, dates, price, value, alpha, baseline_gaps, out: Path):
    archive = Archive()
    tags = ('greedy_closed', 'dp_fixed', 'dp_closed')
    states = {tag: 6000.0 for tag in tags}
    rows = {tag: [] for tag in tags}
    logs = []
    margin_trace = []

    for day, ds in enumerate(dates):
        forecast, net, ids, load_ids, pv_ids = archive.issue()
        if day < 31:
            f = np.zeros((144, 2)) if forecast is None else forecast
            warm, end = simulate(ds, np.zeros(144), f, actual[day], price, 6000.0,
                                 np.full(144, LOWER), 'greedy', value, 0.0,
                                 warmup=True, observation_mode='delayed_1')
            for tag in tags:
                rows[tag].extend(warm)
                states[tag] = end
        else:
            if forecast is None or net is None:
                raise ValueError('Formal day has no forecast')
            formal_day = day - 31
            margin = safety_margin(baseline_gaps, formal_day, alpha)
            margin_trace.append(dict(date=ds, alpha=alpha,
                                     margin_kwh=float(margin.sum()),
                                     max_margin_kwh=float(margin.max()),
                                     residual_sample_count=len(archive.errors)))

            # Greedy contract is the fixed-contract baseline for the DP policy.
            base_g, q_g = procure_with_margin(net, price, states['greedy_closed'], value, margin)
            levels_g = np.full(144, LOWER)
            day_rows, end = simulate(ds, q_g, forecast, actual[day], price,
                                     states['greedy_closed'], levels_g, 'greedy', value, 0.0,
                                     observation_mode='delayed_1')
            rows['greedy_closed'].extend(day_rows)
            states['greedy_closed'] = end

            levels, _, segments = reserves(net, q_g, price, value)
            day_rows, end = simulate(ds, q_g, forecast, actual[day], price,
                                     states['dp_fixed'], levels, 'dp', value, 0.0,
                                     observation_mode='delayed_1')
            rows['dp_fixed'].extend(day_rows)
            states['dp_fixed'] = end

            base_d, q_d = procure_with_margin(net, price, states['dp_closed'], value, margin)
            levels_d, _, segments_d = reserves(net, q_d, price, value)
            day_rows, end = simulate(ds, q_d, forecast, actual[day], price,
                                     states['dp_closed'], levels_d, 'dp', value, 0.0,
                                     observation_mode='delayed_1')
            rows['dp_closed'].extend(day_rows)
            states['dp_closed'] = end

            logs.extend([
                dict(date=ds, variant='greedy_closed', alpha=alpha,
                     base_planned_cost_yuan=float(base_g['purchase'] @ price),
                     margin_purchase_cost_yuan=float(margin @ price),
                     purchase_kwh=float(q_g.sum()), margin_kwh=float(margin.sum()),
                     reserve_segments=0),
                dict(date=ds, variant='dp_fixed', alpha=alpha,
                     base_planned_cost_yuan=float(base_g['purchase'] @ price),
                     margin_purchase_cost_yuan=float(margin @ price),
                     purchase_kwh=float(q_g.sum()), margin_kwh=float(margin.sum()),
                     reserve_segments=int(segments)),
                dict(date=ds, variant='dp_closed', alpha=alpha,
                     base_planned_cost_yuan=float(base_d['purchase'] @ price),
                     margin_purchase_cost_yuan=float(margin @ price),
                     purchase_kwh=float(q_d.sum()), margin_kwh=float(margin.sum()),
                     reserve_segments=int(segments_d)),
            ])
        archive.observe(actual[day])

    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for tag in tags:
        validation = execution_audit(rows[tag], observation_mode='delayed_1')
        formal = rows[tag][31 * 144:]
        daily = [dict(date=dates[31 + i], **summarize(formal[i * 144:(i + 1) * 144]))
                 for i in range(334)]
        ev = events(formal)
        folder = out / tag
        folder.mkdir(exist_ok=True)
        write_csv(folder / 'execution.csv', rows[tag])
        write_csv(folder / 'daily_summary.csv', daily)
        write_csv(folder / 'emergency_events.csv', ev)
        validation['persisted'] = execution_audit(
            read_rows(folder / 'execution.csv'), observation_mode='delayed_1')
        path = ROOT / f'results/result2_v4b_{tag}.xlsx'
        workbook(formal, daily, ev, path)
        verify_export(path, formal, daily, ev)
        results[tag] = dict(formal=summarize(formal), warmup=summarize(rows[tag][:31 * 144]),
                            end_soc_min=min(x['soc_end_kwh'] for x in daily),
                            end_soc_max=max(x['soc_end_kwh'] for x in daily),
                            validation=validation, emergency_events=len(ev))
    write_csv(out / 'solver_log.csv', logs)
    write_csv(out / 'margin_trace.csv', margin_trace)
    return results


def read_rows(path: Path):
    text_keys = {'date', 'phase', 'interval_start', 'interval_end', 'interval_label',
                 'plan_issue_time', 'history_available_through', 'control_issue_time',
                 'last_observation_end', 'raw_sample_label_time', 'observation_model'}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return [{k: v if k in text_keys else int(v) if k == 'slot' else float(v)
                 for k, v in row.items()} for row in csv.DictReader(stream)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='Frozen safety quantile (default: 0.8); use --alpha 0 to disable')
    parser.add_argument('--select-alpha', action='store_true',
                        help='Run January candidate diagnostics in addition to the frozen alpha')
    parser.add_argument('--output-root', default='results/q2_v4b_delayed1')
    args = parser.parse_args()
    if args.alpha is not None and not 0.5 < args.alpha < 1.0:
        parser.error('--alpha must be in (0.5, 1)')

    out = ROOT / args.output_root
    out.mkdir(parents=True, exist_ok=True)
    db = ROOT / 'data/processed/microgrid.sqlite'
    with sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True) as con:
        flat = con.execute('SELECT date,slot,load_kwh,pv_actual_kwh FROM actuals ORDER BY date,slot').fetchall()
        price = np.array([x[0] for x in con.execute(
            'SELECT price_yuan_per_kwh FROM baseline ORDER BY slot')])
    actual = np.array([[x[2], x[3]] for x in flat]).reshape(365, 144, 2)
    dates = [flat[i * 144][0] for i in range(365)]
    value = float(price[:30].mean() / 0.9)

    alpha = float(args.alpha)
    # The 80% quantile is the predeclared default: with a 5x emergency price,
    # a scalar purchase-versus-emergency trade-off has an 80% critical fractile.
    # January candidate selection remains available as an explicit diagnostic,
    # but is not allowed to silently select the zero-margin control.
    if args.select_alpha:
        candidates = [evaluate_candidate(actual, dates, price, value, x) for x in ALPHAS]
        selected = next((x for x in candidates if x['alpha'] == alpha),
                        dict(alpha=alpha, selection_mode='predeclared'))
    else:
        candidates = [dict(alpha=alpha, selection_mode='predeclared')]
        selected = candidates[0]

    baseline_path = ROOT / 'results/q2_v4_delayed1/dp_closed/execution.csv'
    baseline_rows = read_rows(baseline_path)
    baseline_formal = [r for r in baseline_rows if r['date'] >= '2025-02-01']
    if len(baseline_formal) != 334 * 144:
        raise ValueError('V4 delayed-1 baseline gap archive has unexpected length')
    baseline_gaps = np.array([r['emergency_kwh'] for r in baseline_formal]).reshape(334, 144)
    results = run_formal(actual, dates, price, value, alpha, baseline_gaps, out)
    v2b = json.loads((ROOT / 'results/q2_v2b/experiment.json').read_text(encoding='utf-8'))
    baseline = json.loads((ROOT / 'results/q2_v4_delayed1/experiment.json').read_text(encoding='utf-8'))
    comparison = {}
    for tag, result in results.items():
        comparison[tag] = dict(v2b_total_yuan=v2b['formal']['total_cost_yuan'],
                               v4_delayed1_total_yuan=baseline['results'][tag]['formal']['total_cost_yuan'],
                               v4b_total_yuan=result['formal']['total_cost_yuan'],
                               vs_v2b_yuan=result['formal']['total_cost_yuan'] - v2b['formal']['total_cost_yuan'],
                               vs_v4_delayed1_yuan=result['formal']['total_cost_yuan'] - baseline['results'][tag]['formal']['total_cost_yuan'])

    experiment = dict(version='q2-v4b-delayed1-safety-margin', observation_mode='delayed_1',
                      alpha=alpha, candidate_selection=candidates,
                      selected_candidate=selected, value_yuan_per_internal_kwh=value,
                      results=results, comparison=comparison,
                      source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in (ROOT / '附件').rglob('*.xlsx')
                                     if not p.name.startswith('~$')},
                      database_sha256=hashlib.sha256(db.read_bytes()).hexdigest(),
                      scipy_version=scipy.__version__,
                      scope='V4 delayed-1 control plus causal quantile of prior V4 emergency gaps')
    (out / 'experiment.json').write_text(json.dumps(experiment, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 第二问 V4-B：延迟观测下的安全购电裕量', '',
             'V4-B 保留 V4 delayed_1 的控制器，只在每天 0:00 的日前购电合同上增加历史紧急缺口分位数裕量。',
             '正式日只使用此前最多 30 个已完成 V4 delayed_1 缺口日；默认 alpha=0.8，候选诊断可单独运行。', '',
             f'选中分位数 alpha={alpha:.2f}。', '',
             '| 方案 | V2-B费用/元 | V4 delayed_1费用/元 | V4-B费用/元 | V4-B−V2-B/元 |',
             '|---|---:|---:|---:|---:|']
    for tag, row in comparison.items():
        lines.append(f'| {tag} | {row["v2b_total_yuan"]:.6f} | {row["v4_delayed1_total_yuan"]:.6f} | '
                     f'{row["v4b_total_yuan"]:.6f} | {row["vs_v2b_yuan"]:.6f} |')
    lines += ['', '## 解释与边界', '',
              '- 安全裕量来自此前已完成日的 V4 delayed_1 紧急缺口，不使用当前日或未来实际值。',
              '- 购电合同在 0:00 固定；日内仍使用一时段延迟观测和 V4 DP 保留水平控制。',
              '- 该版本先验证“延迟控制下多买电”的效果，不改变 V4 预测器、DP 递推或价格模型。',
              '- 年末 SOC 和富余弃用必须与紧急费用一并解释，不能只按总费用排序。', '',
              '复现：`python scripts/solve_q2_v4b.py`。', '']
    (ROOT / 'reports/q2_v4b_experiment.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(alpha=alpha, candidates=candidates, comparison=comparison),
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

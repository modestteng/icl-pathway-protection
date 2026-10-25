"""Fixed windows and paired seed statistics. Reads saved outputs, never models."""
from plan import *
from statistics_core import summarize, holm


def load_window(seed, arm, budget):
    import numpy as np
    arrays, reference_ids, reference_labels = {}, {}, {}
    missing = []
    for stage in window(budget):
        path = OUTPUTS / seed / arm / 'confirm' / f'{stage:011d}.npz'
        if not path.exists():
            missing.append(str(path.relative_to(OUTPUTS)))
            continue
        with np.load(path, allow_pickle=False) as f:
            assert int(f['stage']) == stage and str(f['seed_id']) == seed and str(f['arm']) == arm
            assert str(f['code_manifest_sha256']) == code_identity()
            for task in SETTINGS['reported_tasks']:
                ids, labels = f[task + '/identities'], f[task + '/labels']
                assert ids.shape == (SETTINGS['n'], 3, 2) and labels.shape == (SETTINGS['n'], 3)
                if task in reference_ids:
                    assert np.array_equal(ids, reference_ids[task]) and np.array_equal(labels, reference_labels[task])
                else:
                    reference_ids[task], reference_labels[task] = ids.copy(), labels.copy()
                for metric, field in [('acc', 'acc'), ('ce', 'loss')]:
                    values = np.asarray(f[task + '/' + field], np.float64)
                    assert values.shape == (SETTINGS['n'],) and np.isfinite(values).all()
                    if metric == 'acc':
                        assert np.all((values == 0) | (values == 1))
                    arrays.setdefault(task + '/' + metric, []).append(values)
    if missing:
        return None, missing
    result = {key: np.stack(values).mean(axis=0) for key, values in arrays.items()}
    result.update({task + '/identities': value for task, value in reference_ids.items()})
    result.update({task + '/labels': value for task, value in reference_labels.items()})
    return result, []


def build_analysis():
    import numpy as np
    saved, rows, missing = {}, [], []
    ids = [s['id'] for s in SETTINGS['training_seeds']]
    for seed in ids:
        for arm in SETTINGS['arms']:
            for budget in budgets(seed):
                result, absent = load_window(seed, arm, budget)
                saved[seed, arm, budget] = result
                missing.extend(absent)
                if result is not None:
                    rows.append(dict(seed=seed, arm=arm, budget=budget,
                        metrics={key: float(value.mean()) for key, value in result.items()
                                 if key.endswith(('/acc', '/ce'))}))
    by_budget = {}
    for budget in [10000000, 20000000, 30000000, 40000000]:
        seeds = ids if budget == 10000000 else SETTINGS['long_seeds']
        metrics = {}
        for task in SETTINGS['reported_tasks']:
            for metric in ['acc', 'ce']:
                name = task + '/' + metric
                av, pv, dv = [], [], []
                for seed in seeds:
                    a, p = saved[seed, 'A', budget], saved[seed, 'P25', budget]
                    av.append(float(a[name].mean()) if a is not None else None)
                    pv.append(float(p[name].mean()) if p is not None else None)
                    if a is not None and p is not None:
                        assert np.array_equal(a[task + '/identities'], p[task + '/identities'])
                        assert np.array_equal(a[task + '/labels'], p[task + '/labels'])
                        dv.append(float((p[name] - a[name]).mean()))
                    else:
                        dv.append(None)
                threshold = SETTINGS['analysis']['gain_threshold'] if name == 'icl/acc' else 0.
                metrics[name] = dict(A=summarize(av, len(seeds)), P25=summarize(pv, len(seeds)),
                                     difference=summarize(dv, len(seeds), threshold))
        ni = {}
        for task in ['iwl_copy_avail', 'train_eval']:
            a = metrics[task + '/acc']['difference']
            c = metrics[task + '/ce']['difference']
            ni[task + '/acc'] = bool(a['complete'] and a['lower_one_sided95'] > -SETTINGS['analysis']['ni_accuracy_margin'])
            ni[task + '/ce'] = bool(c['complete'] and c['upper_one_sided95'] < SETTINGS['analysis']['ni_ce_margin'])
        by_budget[str(budget)] = dict(seeds=list(seeds), metrics=metrics, noninferiority=ni,
                                     all_noninferiority_passed=all(ni.values()))
    p = [by_budget[str(b)]['metrics']['icl/acc']['difference']['p_greater'] for b in SETTINGS['primary_budgets']]
    primary = {}
    for budget, adjusted in zip(SETTINGS['primary_budgets'], holm(p)):
        difference = by_budget[str(budget)]['metrics']['icl/acc']['difference']
        primary[str(budget)] = dict(p_holm=adjusted, threshold=.05,
            passed=bool(difference['complete'] and adjusted < SETTINGS['alpha'] and difference['mean'] > .05))
    persistence = {}
    for task in SETTINGS['reported_tasks']:
        for metric in ['acc', 'ce']:
            key = task + '/' + metric
            values = []
            for seed in SETTINGS['long_seeds']:
                early, late = saved[seed, 'P25', 10000000], saved[seed, 'P25', 40000000]
                if early is not None and late is not None:
                    assert np.array_equal(early[task + '/identities'], late[task + '/identities'])
                    values.append(float((late[key] - early[key]).mean()))
                else:
                    values.append(None)
            persistence[key] = summarize(values, 3)
    lasting = persistence['icl/acc']
    lasting['noninferiority_passed'] = bool(lasting['complete'] and lasting['lower_one_sided95'] > -.02)
    return dict(code_manifest_sha256=code_identity(), complete=not missing, missing=missing,
        per_seed=rows, budgets=by_budget, primary=primary, P25_40M_minus_10M=persistence,
        persistent_benefit_supported=bool(primary['40000000']['passed'] and lasting['noninferiority_passed']),
        interval_scope='training-seed variability conditional on the fixed evaluation pool; shared-pool sampling is not integrated',
        scientific_scope='A versus P25 only; no new random-path or task-protection superiority claim')


def render_report(result):
    lines = ['# exp042：新种子与延长训练', '',
             '效果指标为普通ICL、CIWL、原分类任务；窗口11点先按题平均，独立单位为训练种子。', '',
             '|累计训练量|种子数|A普通ICL|P25普通ICL|P25−A / pp|Holm主检验|',
             '|---|---:|---:|---:|---:|---|']
    for budget, row in result['budgets'].items():
        values = row['metrics']['icl/acc']
        def fmt(item, pp=False):
            return '未齐全' if not item['complete'] else f'{item["mean"] * 100:.3f}'
        gate = result['primary'].get(budget)
        lines.append(f'|{int(budget)//1000000}M|{len(row["seeds"])}|{fmt(values["A"])}|{fmt(values["P25"])}|{fmt(values["difference"])}|'
                     + ('通过' if gate and gate['passed'] else ('未通过' if gate else '描述性')) + '|')
    lines.extend(['', '全部种子、三项任务的准确率/交叉熵、区间、CIWL/原任务非劣和P25自身10M→40M变化见analysis.json。',
                  '两种预算的主检验为平均普通ICL收益>5pp、Holm2；P25自身持续保持另按−2pp界限判定。',
                  '机制确认未通过的种子仍纳入预定A/P25效果比较；结构/数值失败和缺失不得当作成功。',
                  '相对A的优势与P25自身是否衰退分别报告；未见类别或其他领域不属于本实验。'])
    if not result['complete']:
        lines.append(f'\n当前缺失{len(result["missing"])}个预定窗口文件，不能报告完整实验完成。')
    return '\n'.join(lines) + '\n'


def main():
    verify_code()
    result = build_analysis()
    write_json(OUTPUTS / 'analysis.json', result)
    (OUTPUTS / 'report.md').write_text(render_report(result))
    print('ANALYSIS_COMPLETE' if result['complete'] else 'ANALYSIS_PARTIAL')


if __name__ == '__main__':
    main()

"""Fixed C/P/D strength grid; paired rows, query-class clustered uncertainty.

This script never executes a model. Historical rho=.25 points are descriptive
anchors; the nine new arms form one frozen intersection-union Holm family.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from reference_analysis import TASKS, METRICS, class_stats, greater, holm, ni, sha256

WORK = Path(__file__).resolve().parent
NEW_ARMS = [kind + suffix for suffix in ('10', '50', '75') for kind in ('P', 'C', 'D')]
ALL_ARMS = ['A'] + [kind + suffix for suffix in ('10', '25', '50', '75') for kind in ('C', 'P', 'D')]
TASK_NAMES = {'icl': '普通 ICL', 'iwl_copy_avail': 'CIWL', 'train_eval': '原分类任务', 'flip_icl': 'Flip'}


def validate_design(s):
    assert s['arms'] == NEW_ARMS and s['training_seeds'] == [{'id': 's5', 'init': 5, 'train': 2}]
    assert s['strength_grid'] == [.1, .25, .5, .75]
    assert s['relaxation'] == {a: int(a[1:])/100 for a in NEW_ARMS}
    assert s['confirm_stages'] == list(range(9000000, 10000001, 100000))
    assert (s['start_stage'], s['end_stage'], s['n']) == (4400000, 10000000, 5000)
    assert s['fresh_tasks'] == list(TASKS)
    assert (s['alpha'], s['min_icl_gain'], s['ni_acc_tolerance'], s['ni_ce_tolerance']) == (.05, .05, .02, .1)
    assert s['primary_joint_family_size'] == s['primary_gain_family_size'] == 9
    assert s['original_Flip_joint_criteria_unchanged'] is True
    assert s['confirmation_data_previously_observed'] is True


def read_window(work, arm, settings, verifier=None, identities=None):
    values = {t: {m: [] for m in METRICS} for t in TASKS}
    for stage in settings['confirm_stages']:
        path = work/'outputs/s5/confirm'/f'{arm}_{stage:011d}.npz'
        if verifier is not None:
            expected = verifier['verified_file_sha256'].get(str(path.resolve()))
            assert expected is not None and sha256(path) == expected, str(path)
        with np.load(path, allow_pickle=False) as raw:
            for task in TASKS:
                if identities is not None:
                    assert np.array_equal(raw[task+'/identities'], identities[task]), (path, task, 'identities')
                for metric in METRICS:
                    row = np.asarray(raw[task+'/'+metric], np.float64)
                    assert row.shape == (settings['n'],) and np.isfinite(row).all(), (path, task, metric)
                    values[task][metric].append(row)
    return {t: {m: np.mean(np.stack(rows), axis=0, dtype=np.float64) for m, rows in ms.items()} for t, ms in values.items()}


def load_references(s, identities):
    manifest = json.loads((WORK/'references_manifest.json').read_text())
    for name, digest in manifest.items():
        assert sha256(WORK/'references'/name) == digest, name
    data = {}
    for experiment, root, arms in [('exp030', Path(s['reference_work']), ['A', 'C25', 'D25']),
                                   ('exp031', Path(s['reference_p25_work']), ['P25'])]:
        v = json.loads((WORK/'references'/experiment/'verification.json').read_text())
        assert v['passed'] is True and not v['failures']
        old_settings = json.loads((WORK/'references'/experiment/'settings.json').read_text())
        for key in ['training_seeds', 'start_stage', 'end_stage', 'train_bs', 'train_microbs',
                    'confirm_stages', 'data_sets', 'preserved_data_sha256', 'protection_bs_per_task', 'candidate_path_head']:
            assert s[key] == old_settings[key], (experiment, key)
        # exp031 reused exp030's immutable confirmation file; its verifier
        # registers that shared source path rather than a copied exp031 file.
        confirm_anchor = Path(s['reference_work'])/'outputs/data/confirm.h5'
        registered = v['verified_file_sha256'].get(str(confirm_anchor.resolve()))
        assert registered == s['preserved_data_sha256']['confirm.h5'], (experiment, 'shared confirmation registration')
        assert sha256(confirm_anchor) == registered
        for arm in arms:
            data[arm] = read_window(root, arm, s, v, identities)
    return data


def compute_statistics(data, query, s):
    descriptive = {a: {t: {m: class_stats(v, query[t], s['alpha']) for m, v in ms.items()}
                          for t, ms in ts.items()} for a, ts in data.items()}
    pairs = [(a, 'A') for a in ALL_ARMS if a != 'A']
    pairs += [(l+suffix, r+suffix) for suffix in ('10', '25', '50', '75')
              for l, r in [('P', 'C'), ('P', 'D'), ('D', 'C')]]
    differences = {a+'-'+b: {t: {m: class_stats(data[a][t][m]-data[b][t][m], query[t], s['alpha'])
                                  for m in METRICS} for t in TASKS} for a, b in pairs}
    gains, joints, noninferiority = [], [], {}
    for arm in NEW_ARMS:
        name = arm+'-A'
        gain = greater(differences[name]['icl']['acc'], s['min_icl_gain'], name+'/icl/acc')
        gains.append(gain)
        tests, components = {}, [gain]
        for task in ('iwl_copy_avail', 'train_eval'):
            for metric in ('acc', 'loss'):
                stat = differences[name][task][metric]
                key = task+'/'+metric
                tests[key] = ni(stat, metric, s)
                if metric == 'acc':
                    one = greater(stat, -s['ni_acc_tolerance'], name+'/'+key)
                else:
                    negative = class_stats(data['A'][task][metric]-data[arm][task][metric], query[task], s['alpha'])
                    one = greater(negative, -s['ni_ce_tolerance'], name+'/'+key+'/negative_delta')
                components.append(one)
        noninferiority[name] = {'tests': tests, 'all_passed': all(t['passed'] for t in tests.values())}
        joints.append({'name': name+'/ordinary_ICL_and_capability_joint',
                       'threshold': 0., 'mean': min(t['mean']-t['threshold'] for t in components),
                       'valid': all(t['valid'] for t in components),
                       'p_raw': max(t['p_raw'] for t in components),
                       'components': components,
                       'alternative': 'ordinary ICL >5pp AND all four capability noninferiority gates',
                       'mean_role': 'minimum component margin used for sign only; mixed units, not a joint effect estimate',
                       'combination': 'intersection_union_maximum_one_sided_p'})
    gain_family, joint_family = holm(gains, s['alpha']), holm(joints, s['alpha'])
    decisions = {arm: {'gain_Holm9_passed': gain_family['tests'][i]['passed'],
                       'all_four_noninferiority_passed': noninferiority[arm+'-A']['all_passed'],
                       'joint_Holm9_passed': joint_family['tests'][i]['passed']}
                 for i, arm in enumerate(NEW_ARMS)}
    objectives = {arm: np.asarray([descriptive[arm][t]['acc']['mean'] for t in ('icl', 'iwl_copy_avail', 'train_eval')])
                  for arm in ALL_ARMS}
    dominators = {arm: [other for other in ALL_ARMS if other != arm
                        and np.all(objectives[other] >= objectives[arm]) and np.any(objectives[other] > objectives[arm])]
                  for arm in ALL_ARMS}
    frontier = {'status': 'descriptive_observed_grid_only', 'objectives': ['icl/acc', 'iwl_copy_avail/acc', 'train_eval/acc'],
                'non_dominated': [a for a in ALL_ARMS if not dominators[a]], 'dominated_by': dominators,
                'warning': 'No best-parameter selection or inferential dominance; CE and Flip remain reported separately.'}
    return {'descriptive': descriptive, 'paired_comparisons': differences, 'ordinary_ICL_gain_Holm9': gain_family,
            'primary_joint_Holm9': joint_family, 'noninferiority': noninferiority, 'arm_decisions': decisions,
            'descriptive_accuracy_frontier': frontier}


def write_tables_and_plots(result, s):
    out, desc = WORK/'outputs', result['descriptive']
    fields = ['arm', 'kind', 'rho', 'source', 'task', 'metric', 'mean', 'se', 'ci95_lower', 'ci95_upper', 'n', 'clusters']
    with (out/'summary.tsv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t'); writer.writeheader()
        for arm in ALL_ARMS:
            for task in TASKS:
                for metric in METRICS:
                    stat = desc[arm][task][metric]
                    writer.writerow(dict(arm=arm, kind=arm[0], rho='' if arm == 'A' else int(arm[1:])/100,
                                         source='new' if arm in NEW_ARMS else 'historical_descriptive', task=task, metric=metric,
                                         **{k: stat[k] for k in fields[6:]}))
    lines = ['# exp038：C/P/D 保护系数敏感性分析', '',
             '结果待独立读取器核验。单轨迹 init5/train2，固定 4.4M→10M、9–10M 十一点，每任务同一 5000 题。',
             '先在每道题内平均 11 点，再按真实查询类别聚类；区间只反映这条训练轨迹的评价不确定性。',
             'rho=.25 的 A/C25/P25/D25 是已观察的历史描述点；九个新臂组成独立冻结的 Holm9 family。', '',
             '| 条件 | rho | 普通 ICL acc / CE | CIWL acc / CE | 原任务 acc / CE | Flip acc / CE |',
             '|---|---:|---:|---:|---:|---:|']
    for arm in ALL_ARMS:
        values = [f"{100*desc[arm][t]['acc']['mean']:.3f}% / {desc[arm][t]['loss']['mean']:.6f}" for t in ('icl', 'iwl_copy_avail', 'train_eval', 'flip_icl')]
        lines.append('| '+arm+' | '+('—' if arm == 'A' else str(int(arm[1:])/100))+' | '+' | '.join(values)+' |')
    lines += ['', '主要联合检验：普通 ICL 相对 A 提升 >5pp，且 CIWL/原任务准确率差 >−2pp、交叉熵差 <+0.1 nat；五项单侧 p 的最大值后跨九新臂作 Holm 校正。',
              '', '| 新臂 | ICL gain Holm9 | 四项 NI | 联合 Holm9 | 联合调整 p |', '|---|---|---|---|---:|']
    for i, arm in enumerate(NEW_ARMS):
        d = result['arm_decisions'][arm]
        lines.append(f"| {arm} | {d['gain_Holm9_passed']} | {d['all_four_noninferiority_passed']} | {d['joint_Holm9_passed']} | {result['primary_joint_Holm9']['tests'][i]['p_holm']:.6g} |")
    lines += ['', '所有同 rho 配对比较；以下为百分点差和普通 95% 区间，未经多重校正，不作“最优点”选择。',
              '', '| 比较 | 普通 ICL | CIWL | 原任务 | Flip |', '|---|---:|---:|---:|---:|']
    for suffix in ('10', '25', '50', '75'):
        for left, right in [('P', 'C'), ('P', 'D'), ('D', 'C')]:
            name = left+suffix+'-'+right+suffix
            def cell(t):
                z = result['paired_comparisons'][name][t]['acc']
                return f"{100*z['mean']:+.3f} [退化]" if z['ci95_lower'] is None else f"{100*z['mean']:+.3f} [{100*z['ci95_lower']:+.3f}, {100*z['ci95_upper']:+.3f}]"
            lines.append('| '+name+' | '+' | '.join(cell(t) for t in ('icl', 'iwl_copy_avail', 'train_eval', 'flip_icl'))+' |')
    lines += ['', '三项准确率的描述性非支配集合：'+', '.join(result['descriptive_accuracy_frontier']['non_dominated'])+'。该集合不检验统计支配，也不涵盖交叉熵与 Flip。',
              '', '## 四任务全部七指标', '', '| 条件 | 任务 | acc | CE | pair acc | pair CE | pair margin | context mass | log context mass |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ALL_ARMS:
        for task in TASKS:
            lines.append('| '+arm+' | '+TASK_NAMES[task]+' | '+' | '.join(f"{desc[arm][task][m]['mean']:.6f}" for m in METRICS)+' |')
    lines += ['', '普通95%区间与全部配对七指标见 analysis.json / summary.tsv。已观察的确认题不再是未见的最终检验；单训练种子不能支持跨种子普遍优势。旧实验包含 Flip 的联合判据及失败记录保持原样。',
              '', '![参数曲线](parameter_curves.png)', '', '![ICL 与 CIWL](icl_ciwl_scatter.png)']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'C': '#b74a41', 'P': '#236fab', 'D': '#329061'}
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for col, task in enumerate(('icl', 'iwl_copy_avail', 'train_eval', 'flip_icl')):
        for row, metric in enumerate(('acc', 'loss')):
            ax = axes[row, col]
            for kind in ('C', 'P', 'D'):
                names = [kind+suffix for suffix in ('10', '25', '50', '75')]
                scale = 100 if metric == 'acc' else 1
                ax.plot(s['strength_grid'], [scale*desc[a][task][metric]['mean'] for a in names], '-o', color=colors[kind], label=kind)
                ax.scatter([.25], [scale*desc[kind+'25'][task][metric]['mean']], marker='s', s=65, facecolors='white', edgecolors=colors[kind], zorder=4)
            ax.axhline(scale*desc['A'][task][metric]['mean'], color='.4', linestyle='--', label='A historical')
            ax.set(title=f'{task}: {metric}', xlabel='rho (smaller = stronger)', ylabel='Accuracy (%)' if metric == 'acc' else 'CE (nat)')
            ax.set_xticks(s['strength_grid']); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8); fig.suptitle('One training trajectory; squares denote historical rho=0.25')
    fig.savefig(out/'parameter_curves.png', dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for kind in ('C', 'P', 'D'):
        names = [kind+suffix for suffix in ('10', '25', '50', '75')]
        xs = [100*desc[a]['iwl_copy_avail']['acc']['mean'] for a in names]
        ys = [100*desc[a]['icl']['acc']['mean'] for a in names]
        ax.plot(xs, ys, '-o', color=colors[kind], label=kind)
        for a, x, y in zip(names, xs, ys): ax.annotate(a, (x, y), xytext=(4, 4), textcoords='offset points', fontsize=8)
    ax.scatter([100*desc['A']['iwl_copy_avail']['acc']['mean']], [100*desc['A']['icl']['acc']['mean']], marker='x', color='black', label='A historical')
    ax.set(xlabel='CIWL accuracy (%)', ylabel='Ordinary ICL accuracy (%)', title='Fixed strength grid; descriptive single-trajectory tradeoff')
    ax.legend(); ax.grid(alpha=.2); fig.savefig(out/'icl_ciwl_scatter.png', dpi=170); plt.close(fig)


def analyze():
    import h5py
    s = json.loads((WORK/'settings.json').read_text()); validate_design(s)
    confirm = WORK/'outputs/data/confirm.h5'
    assert sha256(confirm) == s['preserved_data_sha256']['confirm.h5']
    with h5py.File(confirm, 'r') as f:
        identities = {t: f[t+'/identities'][:] for t in TASKS}
    query = {t: identities[t][:, -1, 0] for t in TASKS}
    data = load_references(s, identities)
    data.update({arm: read_window(WORK, arm, s, identities=identities) for arm in NEW_ARMS})
    result = dict(experiment='exp038', status='analysis_complete_pending_independent_readback',
                  analysis_scope=s['analysis_scope'], training_seeds=s['training_seeds'],
                  code_manifest_sha256=sha256(WORK/'code_manifest.json'), settings_sha256=sha256(WORK/'settings.json'),
                  protocol_sha256=sha256(WORK/'protocol.md'), references_manifest_sha256=sha256(WORK/'references_manifest.json'),
                  confirm_data_sha256=sha256(confirm), historical_rho25_role='descriptive_only_not_in_new_Holm_family',
                  confirmation_data_previously_observed=True, original_Flip_joint_criteria_unchanged=True,
                  claim_boundary='One known seed and previously observed fixed confirmation inputs. No best-setting selection, cross-seed inference, or revised historical Flip criterion.',
                  **compute_statistics(data, query, s))
    (WORK/'outputs/analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    write_tables_and_plots(result, s)
    print(json.dumps(dict(analysis_complete=True, arm_decisions=result['arm_decisions']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    analyze()

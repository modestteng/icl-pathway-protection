"""Exp040 saved-row analysis. No JAX, no model, no training.

Unit of inference: an applicable training seed of one architecture. Rows are
first averaged over the 11 locked window checkpoints; within-seed uncertainty
uses true-query-class clustered standard errors; architecture-level inference
uses paired Student t over seeds. Primary family: H1 retention (P25-A ICL),
H2 specificity (P25-PR25 ICL), H3 selectivity (P25-C25 CIWL) in each of the
three architectures, Holm-adjusted (9 tests). Everything else is descriptive.
All applicable seeds enter, whatever their mechanism-gate outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np
from scipy import stats

WORK = Path(__file__).resolve().parent
SETTINGS = json.loads((WORK / 'settings.json').read_text())
OUTPUTS = Path(os.environ.get('ICL040_OUTPUTS', str(WORK / 'outputs'))).resolve()
TASKS = ('icl', 'iwl_copy_avail', 'train_eval')
METRICS = ('acc', 'loss')
ARMS = ('A',) + tuple(SETTINGS['arms'])
CONTRASTS = (('P25', 'A'), ('P75', 'A'), ('PR25', 'A'), ('C25', 'A'), ('C75', 'A'),
             ('P25', 'PR25'), ('P25', 'C25'), ('P75', 'C75'), ('P25', 'C75'), ('P25', 'P75'), ('C25', 'C75'))
NAMES = {'icl': 'standard ICL', 'iwl_copy_avail': 'CIWL', 'train_eval': 'original task'}


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def clustered(x, groups):
    x = np.asarray(x, np.float64)
    classes, inverse = np.unique(groups, return_inverse=True)
    mean = float(x.mean())
    sums = np.bincount(inverse, weights=x - mean, minlength=len(classes))
    se = math.sqrt(len(classes) / (len(classes) - 1) * float(sums @ sums) / x.size ** 2)
    return dict(mean=mean, se=se, ci95=[mean - 1.959963984540054 * se, mean + 1.959963984540054 * se],
                rows=int(x.size), clusters=int(len(classes)))


def seed_level(values):
    """values: {seed_id: seed mean difference}. Paired t over seeds, H0 mean <= 0."""
    v = np.asarray(list(values.values()), np.float64)
    n = int(v.size)
    out = dict(n=n, seed_values=values, mean=float(v.mean()) if n else None,
               positive=int((v > 0).sum()), negative=int((v < 0).sum()))
    if n >= 2:
        sd = float(v.std(ddof=1))
        se = sd / math.sqrt(n)
        q = float(stats.t.ppf(0.975, n - 1))
        out.update(sd=sd, se=se, df=n - 1, ci95=[out['mean'] - q * se, out['mean'] + q * se])
        if se > 0:
            t = out['mean'] / se
            out.update(t=t, p_one_sided=float(stats.t.sf(t, n - 1)))
        else:
            out.update(t=None, p_one_sided=0.0 if out['mean'] > 0 else 1.0)
        out['sign_test_p_one_sided'] = float(stats.binomtest(out['positive'], n, 0.5, alternative='greater').pvalue) if n else None
    return out


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda k: pvalues[k])
    adjusted, running = [1.0] * len(pvalues), 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[k]))
        adjusted[k] = running
    return adjusted


def read_window(root, arm, stages):
    paths = [root / arm / 'confirm' / f'{arm}_{s:011d}.npz' for s in stages]
    if not all(p.exists() for p in paths):
        return None
    acc = {t: {m: [] for m in METRICS} for t in TASKS}
    for p in paths:
        with np.load(p, allow_pickle=False) as raw:
            for t in TASKS:
                for m in METRICS:
                    a = np.asarray(raw[t + '/' + m], np.float64)
                    if a.shape != (SETTINGS['n'],) or not np.isfinite(a).all():
                        raise ValueError('malformed window file ' + str(p))
                    acc[t][m].append(a)
    return {t: {m: np.stack(v).mean(axis=0) for m, v in g.items()} for t, g in acc.items()}


def arch_analysis(arch, classes):
    root_arch = OUTPUTS / arch
    seeds = {}
    for spec in SETTINGS['training_seeds']:
        sid = spec['id']
        root = root_arch / sid
        lock = load(root / 'timing.json') if (root / 'timing.json').exists() else None
        item = dict(seed=spec, timing=lock and {k: lock.get(k) for k in ('applicable', 'start_stage', 'end_stage', 'window_stages', 'applicability')})
        if not lock or not lock['applicable']:
            item['status'] = 'not_applicable' if lock else 'missing_timing'
            seeds[sid] = item
            continue
        mech = load(root / 'mechanism_complete.json') if (root / 'mechanism_complete.json').exists() else None
        selection = load(root / 'path_selection.json') if (root / 'path_selection.json').exists() else None
        item['mechanism'] = mech and dict(selected=mech['selected'], random_set=mech['random_set'], k=mech['k'],
                                          selection_reason=mech['selection_reason'], gate_passed=mech['gate_passed'],
                                          gate_standard_icl_only_passed=mech['gate_standard_icl_only_passed'],
                                          confirm_coverage=mech['confirm_coverage'],
                                          discovery_prefix_coverage=selection and selection['prefix_coverage'])
        data = {}
        for arm in ARMS:
            if arm != 'A':
                c = root / arm / 'complete.json'
                if not c.exists() or not load(c)['passed']:
                    continue
            window = read_window(root, arm, lock['window_stages'])
            if window is not None:
                data[arm] = window
        item['complete_arms'] = sorted(data)
        item['missing_arms'] = [a for a in ARMS if a not in data]
        item['means'] = {arm: {t: {m: clustered(v, classes[t]) for m, v in g.items()} for t, g in w.items()}
                         for arm, w in data.items()}
        item['contrasts'] = {f'{a}-{b}': {t: {m: clustered(data[a][t][m] - data[b][t][m], classes[t]) for m in METRICS}
                                          for t in TASKS} for a, b in CONTRASTS if a in data and b in data}
        updates = {}
        for arm in SETTINGS['arms']:
            c = root / arm / 'complete.json'
            if c.exists():
                r = load(c)
                updates[arm] = dict(r['update_statistics'], late={k: v['gains'] for k, v in r['late_diagnostics'].items()})
        item['updates'] = updates
        if mech:
            item['label_k_start'] = {k: v['gains'] for k, v in mech['late'].items()}
        item['status'] = 'complete' if not item['missing_arms'] else 'incomplete'
        seeds[sid] = item
    usable = {sid: s for sid, s in seeds.items() if s.get('status') in ('complete', 'incomplete')}
    aggregate = {}
    for a, b in CONTRASTS:
        name = f'{a}-{b}'
        aggregate[name] = {t: {m: seed_level({sid: s['contrasts'][name][t][m]['mean']
                                              for sid, s in usable.items() if name in s['contrasts']})
                               for m in METRICS} for t in TASKS}
    arm_means = {arm: {t: {m: seed_level({sid: s['means'][arm][t][m]['mean'] for sid, s in usable.items() if arm in s['means']})
                           for m in METRICS} for t in TASKS} for arm in ARMS}
    marginal = {}
    for method in ('P', 'C'):
        hi, lo = f'{method}25', f'{method}75'
        icl = arm_means[hi]['icl']['acc']['mean'], arm_means[lo]['icl']['acc']['mean']
        ciwl = arm_means[hi]['iwl_copy_avail']['acc']['mean'], arm_means[lo]['iwl_copy_avail']['acc']['mean']
        if None not in icl + ciwl and icl[0] != icl[1]:
            marginal[method] = dict(icl_gain_75_to_25=icl[0] - icl[1], ciwl_change_75_to_25=ciwl[0] - ciwl[1],
                                    ciwl_cost_per_icl_point=-(ciwl[0] - ciwl[1]) / (icl[0] - icl[1]))
    correction = {arm: seed_level({sid: s['updates'][arm]['mean_correction_norm_ratio']
                                   for sid, s in usable.items() if arm in s.get('updates', {})})
                  for arm in SETTINGS['arms']}
    moderator = []
    for sid, s in usable.items():
        if 'P25-A' in s['contrasts'] and s.get('mechanism'):
            moderator.append(dict(seed=sid, coverage=s['mechanism']['confirm_coverage'],
                                  gate_passed=s['mechanism']['gate_passed'],
                                  icl_gain=s['contrasts']['P25-A']['icl']['acc']['mean'],
                                  specificity=s['contrasts'].get('P25-PR25', {}).get('icl', {}).get('acc', {}).get('mean')))
    return dict(architecture=SETTINGS['architectures'][arch], seeds=seeds,
                applicable_seeds=sorted(usable), not_applicable_seeds=sorted(set(seeds) - set(usable)),
                arm_means=arm_means, contrasts=aggregate, marginal_cost=marginal,
                mean_correction_norm_ratio=correction, moderator_rows=moderator)


def primary(architectures):
    tests = []
    for arch, item in architectures.items():
        for hyp, (a, b, task, metric) in SETTINGS['analysis']['primary_hypotheses'].items():
            stat = item['contrasts'][f'{a}-{b}'][task][metric]
            p = stat.get('p_one_sided', 1.0)
            tests.append(dict(architecture=arch, hypothesis=hyp, contrast=f'{a}-{b}', task=task, metric=metric,
                              n=stat['n'], mean=stat['mean'], ci95=stat.get('ci95'), p_raw=p if p is not None else 1.0,
                              positive_seeds=stat['positive']))
    for test, adj in zip(tests, holm([t['p_raw'] for t in tests])):
        test['p_holm'] = adj
        test['supported'] = bool(adj < SETTINGS['alpha'] and test['mean'] is not None and test['mean'] > 0)
    return tests


def pooled(architectures):
    """All applicable architecture-seed units together (secondary)."""
    out = {}
    for a, b in CONTRASTS:
        name = f'{a}-{b}'
        out[name] = {t: {m: seed_level({f'{arch}/{sid}': v for arch, item in architectures.items()
                                        for sid, v in item['contrasts'][name][t][m]['seed_values'].items()})
                         for m in METRICS} for t in TASKS}
    rows = [r for item in architectures.values() for r in item['moderator_rows'] if r['coverage'] is not None]
    mod = dict(rows=rows)
    if len(rows) >= 3:
        rho, p = stats.spearmanr([r['coverage'] for r in rows], [r['icl_gain'] for r in rows])
        mod.update(spearman_coverage_vs_icl_gain=float(rho), p_two_sided=float(p))
    passed = [r['icl_gain'] for r in rows if r['gate_passed']]
    failed = [r['icl_gain'] for r in rows if not r['gate_passed']]
    mod.update(gate_passed_n=len(passed), gate_failed_n=len(failed),
               gate_passed_mean_icl_gain=float(np.mean(passed)) if passed else None,
               gate_failed_mean_icl_gain=float(np.mean(failed)) if failed else None)
    out['coverage_moderator'] = mod
    return out


def pct(v):
    return '—' if v is None else f'{100 * v:+.2f}'


def report(result):
    L = ['# exp040：规模泛化（宽度与深度）结果', '',
         '单位：训练种子；窗口11点逐题平均；种子内按真实查询类别聚类；架构内配对t。主假设9项Holm。', '',
         '## 适用性与路径集合', '', '| 架构 | 种子 | 适用 | t0 | 终点 | k | 选中路径 | 随机路径 | 确认覆盖率 | 机制门 |', '|---|---|---|---:|---:|---:|---|---|---:|---|']
    for arch, item in result['architectures'].items():
        for sid, s in item['seeds'].items():
            t = s.get('timing') or {}
            m = s.get('mechanism') or {}
            cov = m.get('confirm_coverage')
            L.append(f"| {arch} | {sid} | {t.get('applicable')} | {t.get('start_stage')} | {t.get('end_stage')} | {m.get('k', '—')} | "
                     f"{m.get('selected', '—')} | {m.get('random_set', '—')} | {'—' if cov is None else f'{100 * cov:.1f}%'} | {m.get('gate_passed', '—')} |")
    L += ['', '## 各组末期均值（准确率%，种子均值）', '', '| 架构 | 组 | 普通ICL | CIWL | 原分类任务 | n |', '|---|---|---:|---:|---:|---:|']
    for arch, item in result['architectures'].items():
        for arm in ARMS:
            st = item['arm_means'][arm]
            cells = ['—' if st[t]['acc']['mean'] is None else f"{100 * st[t]['acc']['mean']:.2f}" for t in TASKS]
            L.append(f"| {arch} | {arm} | " + ' | '.join(cells) + f" | {st['icl']['acc']['n']} |")
    L += ['', '## 对比（准确率差，百分点；95% t区间；正向种子数/n）', '', '| 架构 | 对比 | 普通ICL | CIWL | 原分类任务 |', '|---|---|---|---|---|']
    for arch, item in result['architectures'].items():
        for name, g in item['contrasts'].items():
            cells = []
            for t in TASKS:
                s = g[t]['acc']
                ci = s.get('ci95')
                cells.append('—' if s['mean'] is None else f"{pct(s['mean'])} [{pct(ci[0]) if ci else '—'}, {pct(ci[1]) if ci else '—'}] {s['positive']}/{s['n']}")
            L.append(f'| {arch} | {name} | ' + ' | '.join(cells) + ' |')
    L += ['', '## 主假设', '', '| 架构 | 假设 | 对比 | 指标 | 均值 | 原始p | Holm p | 支持 |', '|---|---|---|---|---:|---:|---:|---|']
    for t in result['primary']:
        L.append(f"| {t['architecture']} | {t['hypothesis']} | {t['contrast']} | {NAMES[t['task']]} {t['metric']} | {pct(t['mean'])} | {t['p_raw']:.4g} | {t['p_holm']:.4g} | {t['supported']} |")
    L += ['', '## 选择性：ρ 由 0.75 收紧到 0.25 的边际代价（CIWL下降/普通ICL提高）', '', '| 架构 | P | C |', '|---|---:|---:|']
    for arch, item in result['architectures'].items():
        mc = item['marginal_cost']
        L.append(f"| {arch} | {mc.get('P', {}).get('ciwl_cost_per_icl_point', '—')} | {mc.get('C', {}).get('ciwl_cost_per_icl_point', '—')} |")
    mod = result['pooled']['coverage_moderator']
    L += ['', '## 覆盖率与保护收益', '', f"Spearman(确认覆盖率, P25−A 普通ICL增益) = {mod.get('spearman_coverage_vs_icl_gain')}，p = {mod.get('p_two_sided')}；"
          f"机制门通过 {mod['gate_passed_n']} 个、未通过 {mod['gate_failed_n']} 个，平均增益分别为 {pct(mod['gate_passed_mean_icl_gain'])} / {pct(mod['gate_failed_mean_icl_gain'])}。", '']
    return '\n'.join(L) + '\n'


def analyze():
    with h5py.File(OUTPUTS / 'data' / 'confirm.h5', 'r') as f:
        classes = {t: np.asarray(f[t + '/identities'][:, -1, 0]) for t in TASKS}
    architectures = {arch: arch_analysis(arch, classes) for arch in SETTINGS['architecture_order']}
    result = dict(experiment=SETTINGS['version'], settings_sha256=digest(WORK / 'settings.json'),
                  code_manifest_sha256=digest(WORK / 'code_manifest.json'),
                  protocol_sha256=digest(WORK / 'protocol.md'),
                  confirm_data_sha256=digest(OUTPUTS / 'data' / 'confirm.h5'),
                  architectures=architectures, primary=primary(architectures), pooled=pooled(architectures))
    (OUTPUTS / 'analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    (OUTPUTS / 'report.md').write_text(report(result), encoding='utf-8')
    print(json.dumps(dict(analysis_complete=True, primary=[{k: t[k] for k in ('architecture', 'hypothesis', 'mean', 'p_holm', 'supported')}
                                                       for t in result['primary']]), ensure_ascii=False, indent=1))
    return result


if __name__ == '__main__':
    argparse.ArgumentParser().parse_args()
    analyze()

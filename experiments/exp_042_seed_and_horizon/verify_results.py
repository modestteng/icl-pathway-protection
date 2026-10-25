"""Independent CPU reader. Does not import analyze, statistics_core, or GPU code."""
from pathlib import Path
import hashlib
import json
import math
import os
import statistics

WORK = Path(__file__).resolve().parent
OUT = Path(os.environ.get('ICL042_OUTPUTS', str(WORK / 'outputs'))).resolve()
ASSETS = Path(os.environ.get('ICL042_ASSETS', str(WORK / 'assets'))).resolve()
S = json.loads((WORK / 'settings.json').read_text())
CHECKS, FAILURES = 0, []


def check(value, name):
    global CHECKS
    CHECKS += 1
    if not bool(value):
        FAILURES.append(name)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1048576), b''):
            h.update(data)
    return h.hexdigest()


def cdf(value, df):
    # Exact df=2/4 CDFs, independent of the producer's incomplete-beta code.
    z = value / math.sqrt(df + value * value)
    if df == 2:
        return .5 + z / 2.
    if df == 4:
        return .5 + (3 * z - z ** 3) / 4.
    raise ValueError('unexpected frozen seed count')


def quantile(p, df):
    low, high = -10000., 10000.
    for _ in range(100):
        midpoint = (low + high) * .5
        if cdf(midpoint, df) > p:
            high = midpoint
        else:
            low = midpoint
    return (low + high) * .5


def compare_summary(values, saved, name, threshold=0.):
    n, mean = len(values), statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(n)
    df = n - 1
    check(saved['complete'] and saved['n'] == n, name + '/n')
    expected = dict(mean=mean, sd=statistics.stdev(values), se=se,
        lower95=mean - quantile(.975, df) * se, upper95=mean + quantile(.975, df) * se,
        lower_one_sided95=mean - quantile(.95, df) * se,
        upper_one_sided95=mean + quantile(.95, df) * se,
        p_greater=1 - cdf((mean - threshold) / se, df) if se > 0 else (0. if mean > threshold else 1.))
    for key, value in expected.items():
        check(abs(saved[key] - value) <= 2e-10, name + '/' + key)
    return expected


def verify():
    import numpy as np
    import h5py
    identity = digest(WORK / 'code_manifest.json')
    for name, expected in json.loads((WORK / 'code_manifest.json').read_text()).items():
        check(digest(WORK / name) == expected, 'sealed/' + name)
    data = json.loads((OUT / 'analysis.json').read_text())
    check(data['complete'] and not data['missing'], 'all windows present')
    check(data['code_manifest_sha256'] == identity, 'analysis code identity')
    expected_jobs = ['data', 'validate_A', 'validate_P25', 'short_complete']
    for spec_seed in S['training_seeds']:
        sid = spec_seed['id']
        expected_jobs.extend([sid + '_prefix', sid + '_select', sid + '_confirm', sid + '_preflight'])
        expected_jobs.extend(f'{sid}_discover_{h}' for h in range(8))
        for arm in ['A', 'P25']:
            expected_jobs.extend([f'{sid}_{arm}_10M', f'{sid}_{arm}_10M_diagnostic'])
            if sid in S['long_seeds']:
                expected_jobs.extend([f'{sid}_{arm}_40M', f'{sid}_{arm}_40M_diagnostic'])
    check(len(expected_jobs) == 96 and len(set(expected_jobs)) == 96, 'closed independent job list')
    for job in expected_jobs:
        receipt = json.loads((OUT / 'jobs' / job / 'complete.json').read_text())
        exit_record = json.loads((OUT / 'jobs' / job / 'exit.json').read_text())
        check(receipt['passed'] and receipt['code_manifest_sha256'] == identity, 'job/' + job + '/complete')
        check(exit_record['exit_code'] == 0 and exit_record['real_exit_confirmed'], 'job/' + job + '/exit')
    spec = json.loads((WORK / 'input_spec.json').read_text())
    for name, record in spec['files'].items():
        check(digest(ASSETS / name) == record['sha256'], 'asset/' + name)
    pools = {}
    for name in S['data_sets']:
        expected = spec['files']['frozen_data/' + name + '.h5']['sha256']
        check(digest(OUT / 'data' / (name + '.h5')) == expected, 'purpose/' + name)
    with h5py.File(OUT / 'data/confirm.h5') as f:
        pools = {t: (f[t]['identities'][:], f[t]['labels'][:]) for t in S['reported_tasks']}
    all_values = {}
    for seed in [s['id'] for s in S['training_seeds']]:
        selection = json.loads((OUT / seed / 'path_selection.json').read_text())
        check(selection['selected_before_confirm'] and selection['selected_before_protection'], seed + '/selection_order')
        mechanism = json.loads((OUT / seed / 'mechanism_confirm.json').read_text())
        check(digest(OUT / seed / 'mechanism_confirm.npz') == mechanism['raw_sha256'], seed + '/mechanism_raw')
        check(mechanism['structure_passed'] and mechanism['gate_controls_execution'] is False,
              seed + '/mechanism_inclusion_policy')
        preflight = json.loads((OUT / seed / 'preflight.json').read_text())
        check(preflight['passed'] and preflight['code_manifest_sha256'] == identity, seed + '/preflight')
        scores = {}
        for head in range(8):
            path = OUT / seed / 'discovery' / f'head{head}.npz'
            with np.load(path, allow_pickle=False) as f:
                score = sum(float(np.mean(np.asarray(f[t + '/native/donor_ce'], np.float64)
                    - np.asarray(f[t + '/path/donor_ce'], np.float64))) for t in S['protection_tasks']) / 2.
            scores[head] = score
            check(abs(score - selection['discovery_scores'][str(head)]) < 1e-12, f'{seed}/discovery/{head}')
            check(digest(path) == selection['raw_sha256'][str(head)], f'{seed}/discovery/{head}/sha')
        check(selection['candidate_head'] == min(scores, key=lambda h: (-scores[h], h)), seed + '/selection')
        budgets = [10000000, 20000000, 30000000, 40000000] if seed in S['long_seeds'] else [10000000]
        for arm in ['A', 'P25']:
            for budget in budgets:
                saved = {t + '/' + m: [] for t in S['reported_tasks'] for m in ['acc', 'ce']}
                for stage in range(budget - 1000000, budget + 1, 100000):
                    path = OUT / seed / arm / 'confirm' / f'{stage:011d}.npz'
                    with np.load(path, allow_pickle=False) as f:
                        check(int(f['stage']) == stage and str(f['seed_id']) == seed and str(f['arm']) == arm, str(path) + '/metadata')
                        check(str(f['code_manifest_sha256']) == identity, str(path) + '/code')
                        for task in S['reported_tasks']:
                            check(np.array_equal(f[task + '/identities'], pools[task][0]), str(path) + '/' + task + '/ids')
                            check(np.array_equal(f[task + '/labels'], pools[task][1]), str(path) + '/' + task + '/labels')
                            check(np.all((f[task + '/acc'] == 0) | (f[task + '/acc'] == 1)), str(path) + '/binary')
                            check(np.array_equal(f[task + '/target'], f[task + '/labels'][:, -1]), str(path) + '/targets')
                            check(np.array_equal(f[task + '/acc'], f[task + '/full_pred'] == f[task + '/target']), str(path) + '/predictions')
                            logits = np.asarray(f[task + '/audit_logits'], np.float64)
                            audit_rows = f[task + '/audit_rows']
                            label = f[task + '/target'][audit_rows]
                            maximum = logits.max(axis=1)
                            ce = maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=1)) - logits[np.arange(len(label)), label]
                            check(np.max(np.abs(ce - f[task + '/loss'][audit_rows])) <= S['numeric_eval_mean_ce_max_abs'], str(path) + '/audit_ce')
                            check(np.array_equal(logits.argmax(axis=1), f[task + '/full_pred'][audit_rows]), str(path) + '/audit_pred')
                            saved[task + '/acc'].append(np.asarray(f[task + '/acc'], np.float64))
                            saved[task + '/ce'].append(np.asarray(f[task + '/loss'], np.float64))
                all_values[seed, arm, budget] = {k: np.sum(v, axis=0) / 11. for k, v in saved.items()}
            stages = range(4400000, (40000000 if seed in S['long_seeds'] else 10000000) + 1, 100000)
            for stage in stages:
                rec = json.loads((OUT / seed / arm / 'states' / f'{stage:011d}.json').read_text())
                check(rec['iter'] == stage and rec['adam_count'] == stage // 32, f'{seed}/{arm}/{stage}/count')
                check(rec['evaluated_stage'] == stage and rec['protection_offset'] == (stage - 4400000) // 32,
                      f'{seed}/{arm}/{stage}/continuation')
                if arm == 'P25':
                    other = json.loads((OUT / seed / 'A/states' / f'{stage:011d}.json').read_text())
                    check(rec['prng'] == other['prng'], f'{seed}/{stage}/matched_three_PRNG')
                if stage in [4400000, 10000000, 40000000] or any(stage in range(b - 1000000, b + 1, 100000) for b in budgets):
                    check(digest(OUT / rec['checkpoint_path']) == rec['checkpoint_sha256'], f'{seed}/{arm}/{stage}/checkpoint')
            if seed in S['long_seeds']:
                origin = json.loads((OUT / seed / arm / 'long_origin.json').read_text())
                boundary = json.loads((OUT / seed / arm / 'complete_10000000.json').read_text())
                check(origin['source_10M_sha256'] == boundary['checkpoint_sha256'], f'{seed}/{arm}/10M_origin')
                check(origin['state']['evaluated_stage'] == 10000000 and origin['state']['protection_offset'] == 175000,
                      f'{seed}/{arm}/10M_no_phase_reset')
                acceptance = json.loads((OUT / seed / 'resume_acceptance' / (arm + '.json')).read_text())
                check(acceptance['passed'] and all(acceptance['checks'].values()), f'{seed}/{arm}/resume_acceptance')
        for path in (OUT / seed / 'update_audit').glob('*.npz'):
            with np.load(path, allow_pickle=False) as f:
                u, g, projected = [np.asarray(f[k], np.float64) for k in ['raw', 'gradient', 'projected']]
                dot, norm = float(g @ u), float(g @ g)
                bound = .25 * max(0., dot)
                expected = u - max(0., dot - bound) / norm * g if norm > 0 else u
                check(np.max(np.abs(expected - projected)) <= S['numeric_projection_reference_max_abs'], str(path) + '/projection')
                check(np.linalg.norm(expected - projected) / max(np.linalg.norm(u), 1e-30) <= S['numeric_projection_reference_max_relative'], str(path) + '/relative')
    primary_p, gains = [], {}
    for budget in [10000000, 20000000, 30000000, 40000000]:
        seeds = [s['id'] for s in S['training_seeds']] if budget == 10000000 else S['long_seeds']
        row = data['budgets'][str(budget)]
        for task in S['reported_tasks']:
            for metric in ['acc', 'ce']:
                key = task + '/' + metric
                a = [float(all_values[s, 'A', budget][key].mean()) for s in seeds]
                p = [float(all_values[s, 'P25', budget][key].mean()) for s in seeds]
                d = [float(np.mean(all_values[s, 'P25', budget][key] - all_values[s, 'A', budget][key])) for s in seeds]
                compare_summary(a, row['metrics'][key]['A'], f'{budget}/{key}/A')
                compare_summary(p, row['metrics'][key]['P25'], f'{budget}/{key}/P25')
                expected = compare_summary(d, row['metrics'][key]['difference'], f'{budget}/{key}/difference', .05 if key == 'icl/acc' else 0.)
                if budget in [10000000, 40000000] and key == 'icl/acc':
                    gains[budget] = expected
                if task != 'icl':
                    passed = (expected['lower_one_sided95'] > -.02) if metric == 'acc' else (expected['upper_one_sided95'] < .1)
                    check(row['noninferiority'][key] == passed, f'{budget}/{key}/NI')
        check(row['all_noninferiority_passed'] == all(row['noninferiority'].values()), f'{budget}/NI_intersection')
    primary_p = [gains[b]['p_greater'] for b in [10000000, 40000000]]
    order = sorted(range(2), key=lambda i: primary_p[i])
    adjusted, prior = [1., 1.], 0.
    for rank, i in enumerate(order):
        prior = max(prior, min(1., primary_p[i] * (2 - rank)))
        adjusted[i] = prior
    for b, p in zip([10000000, 40000000], adjusted):
        check(abs(data['primary'][str(b)]['p_holm'] - p) < 2e-10, f'{b}/Holm')
        check(data['primary'][str(b)]['passed'] == (p < .05 and gains[b]['mean'] > .05), f'{b}/gain_gate')
    for task in S['reported_tasks']:
        for metric in ['acc', 'ce']:
            key = task + '/' + metric
            values = [float(np.mean(all_values[s, 'P25', 40000000][key] - all_values[s, 'P25', 10000000][key])) for s in S['long_seeds']]
            compare_summary(values, data['P25_40M_minus_10M'][key], 'persistence/' + key)
    lasting = data['P25_40M_minus_10M']['icl/acc']
    check(lasting['noninferiority_passed'] == (lasting['lower_one_sided95'] > -.02), 'persistence_gate')
    check(data['persistent_benefit_supported'] == (data['primary']['40000000']['passed'] and lasting['noninferiority_passed']), 'persistent_benefit_gate')
    report = (OUT / 'report.md').read_text()
    check('flip_icl' not in report and 'Flip' not in report, 'reported_tasks_only')
    return dict(passed=not FAILURES, checks=CHECKS, failures=FAILURES, independent_reader=True,
                coverage='All saved task rows/predictions, registered data identities, audit-row logits, fixed windows, all seed statistics, retained checkpoint hashes and matched PRNG metadata. No independent training or full-model forward rerun.',
                code_manifest_sha256=identity, analysis_sha256=digest(OUT / 'analysis.json'))


def main():
    try:
        result = verify()
    except Exception as exc:
        result = dict(passed=False, checks=CHECKS, failures=FAILURES + [repr(exc)], independent_reader=True)
    path = OUT / 'verification.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('VERIFIED' if result['passed'] else 'VERIFICATION_FAILED', result['checks'])
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

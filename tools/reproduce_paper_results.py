#!/usr/bin/env python3
"""Recheck saved manuscript results without model execution or experiment imports.

The standard library independently evaluates Student-t probabilities/quantiles
and Holm adjustment. NumPy is needed only to read saved prediction arrays.
This does not train, evaluate a model, recompute logits, or validate checkpoints.
"""
import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import statistics as stats
import sys
import time

TASKS = ('icl', 'iwl_copy_avail', 'train_eval')
SCALE = 'experiments/exp_040_scale_generalization_v2/final_results/cloud4090_20261021'
HORIZON = 'experiments/exp_042_seed_and_horizon/final_results/cloud4090_20261021/outputs/analysis.json'
SUPPLEMENT = 'experiments/exp_042_seed_and_horizon/supplement_s21_s22_40M/final_results/cloud4090_20261021/outputs/analysis.json'
EXTERNAL = 'experiments/exp_043_kuzushiji49_generalization/multiseed_10M_20261022/summary/combined_analysis.json'
COST = 'experiments/exp_045_training_cost/extended_20261022'


def beta_fraction(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 401):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = d if abs(d) > tiny else tiny
        c = 1 + aa / c
        c = c if abs(c) > tiny else tiny
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = d if abs(d) > tiny else tiny
        c = 1 + aa / c
        c = c if abs(c) > tiny else tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 3e-14:
            return h
    raise ArithmeticError('Incomplete beta continued fraction did not converge')


def regularized_beta(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    factor = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                      + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return factor * beta_fraction(a, b, x) / a
    return 1 - factor * beta_fraction(b, a, 1 - x) / b


def t_sf(t, df):
    if t < 0:
        return 1 - t_sf(-t, df)
    return 0.5 * regularized_beta(df / 2, 0.5, df / (df + t * t))


@lru_cache(None)
def t_quantile(p, df):
    assert 0.5 < p < 1 and df > 0
    lo, hi = 0.0, 1.0
    while t_sf(hi, df) > 1 - p:
        hi *= 2
    for _ in range(85):
        mid = (lo + hi) / 2
        if t_sf(mid, df) > 1 - p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def summarize(values, threshold=0):
    values = [float(x) for x in values]
    assert len(values) >= 2 and all(math.isfinite(x) for x in values)
    n = len(values)
    mean, sd = stats.mean(values), stats.stdev(values)
    se = sd / math.sqrt(n)
    q, q1 = t_quantile(.975, n - 1), t_quantile(.95, n - 1)
    p = t_sf((mean - threshold) / se, n - 1) if se > 0 else float(mean <= threshold)
    return dict(n=n, values=values, mean=mean, sd=sd, se=se,
                ci95=[mean - q * se, mean + q * se],
                lower_one_sided95=mean - q1 * se, upper_one_sided95=mean + q1 * se,
                p_one_sided=p, threshold=threshold,
                positive_seeds=sum(x > 0 for x in values))


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result, running = [None] * len(values), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[i]))
        result[i] = running
    return result


class Recheck:
    def __init__(self, root, supplements):
        self.root, self.supplements = root, supplements
        self.metrics, self.groups, self.failures, self.inputs, self.coverage = {}, {}, [], {}, {}

    def path(self, rel, optional=False):
        candidates = ([self.supplements / rel] if self.supplements else []) + [self.root / rel]
        p = next((p for p in candidates if p.is_file()), None)
        if p is None:
            if optional:
                return None
            raise FileNotFoundError(rel)
        self.inputs[rel] = dict(bytes=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            selected_from='supplement' if self.supplements and p.is_relative_to(self.supplements) else 'package')
        return p

    def load(self, rel, optional=False):
        p = self.path(rel, optional)
        return json.loads(p.read_text()) if p else None

    def check(self, group, name, actual, expected, atol=2e-9):
        z = self.groups.setdefault(group, dict(checked=0, failures=0, max_absolute_difference=0.0))
        z['checked'] += 1
        if isinstance(actual, (list, tuple)):
            assert len(actual) == len(expected), name
            for i, (a, b) in enumerate(zip(actual, expected)):
                self.check(group, name + '/' + str(i), a, b, atol)
            return
        delta = abs(float(actual) - float(expected))
        z['max_absolute_difference'] = max(z['max_absolute_difference'], delta)
        if not math.isfinite(delta) or delta > atol:
            z['failures'] += 1
            self.failures.append(dict(group=group, name=name, actual=actual, expected=expected, tolerance=atol))

    def put(self, prefix, summary):
        for k in ('n', 'mean', 'sd', 'se', 'p_one_sided', 'lower_one_sided95', 'upper_one_sided95'):
            if k in summary:
                self.metrics[prefix + '.' + k] = summary[k]
        if 'ci95' in summary:
            self.metrics[prefix + '.ci95.lower'], self.metrics[prefix + '.ci95.upper'] = summary['ci95']

    def scale(self):
        saved = self.load(SCALE + '/outputs/analysis.json')
        with self.path(SCALE + '/review_tables/seed_results.csv').open() as f:
            rows = list(csv.DictReader(f))
        cell = {}
        for r in rows:
            if r['task'] in TASKS and r['metric'] in ('acc', 'loss'):
                key = (r['architecture'], r['arm'], r['task'], r['metric'])
                cell.setdefault(key, {})[r['seed_id']] = float(r['mean'])
        seeds = ['s13', 's14', 's15', 's16', 's17']
        for (arch, arm, task, metric), data in cell.items():
            assert sorted(data) == seeds
            calc = summarize([data[s] for s in seeds])
            prefix = f'scale.arm.{arch}.{arm}.{task}.{metric}'
            self.put(prefix, calc)
            old = saved['architectures'][arch]['arm_means'][arm][task][metric]
            for k in ('mean', 'sd', 'se'):
                self.check('scale_saved_statistics', prefix + '.' + k, calc[k], old[k])
            self.check('scale_saved_statistics', prefix + '.ci95', calc['ci95'], old['ci95'])
            if task == 'icl' and metric == 'acc':
                for seed in seeds:
                    self.metrics[f'scale.seed.{arch}.{seed}.{arm}'] = data[seed]
        for arch in ('base', 'width', 'depth'):
            for arm in ('C25', 'C75', 'P25', 'P75', 'PR25'):
                for task in TASKS:
                    for metric in ('acc', 'loss'):
                        values = [cell[arch, arm, task, metric][s] - cell[arch, 'A', task, metric][s] for s in seeds]
                        self.put(f'scale.comparison.{arch}.{arm}.{task}.{metric}', summarize(values))
        primary = []
        defs = [('H1', 'P25', 'A', 'icl'), ('H2', 'P25', 'PR25', 'icl'),
                ('H3', 'P25', 'C25', 'iwl_copy_avail')]
        for arch in ('base', 'width', 'depth'):
            for h, a, b, task in defs:
                diffs = [cell[arch, a, task, 'acc'][s] - cell[arch, b, task, 'acc'][s] for s in seeds]
                calc = summarize(diffs)
                primary.append((arch, h, calc))
                self.put(f'scale.primary.{arch}.{h}', calc)
                if h == 'H1':
                    for seed, value in zip(seeds, diffs):
                        self.metrics[f'scale.seed.{arch}.{seed}.difference'] = value
        adjusted = holm([c['p_one_sided'] for _, _, c in primary])
        for i, ((arch, h, c), p) in enumerate(zip(primary, adjusted)):
            old = saved['primary'][i]
            assert old['architecture'] == arch and old['hypothesis'].startswith(h)
            prefix = f'scale.primary.{arch}.{h}'
            self.metrics[prefix + '.p_holm'] = p
            self.metrics[prefix + '.supported'] = p < .05 and c['mean'] > 0
            for k, value in [('mean', c['mean']), ('ci95', c['ci95']), ('p_raw', c['p_one_sided']), ('p_holm', p)]:
                self.check('scale_saved_primary', prefix + '.' + k, value, old[k])
        for arch, block in saved['architectures'].items():
            for arm, old in block['mean_correction_norm_ratio'].items():
                calc = summarize(list(old['seed_values'].values()))
                self.metrics[f'scale.correction.{arch}.{arm}.mean'] = calc['mean']
                self.check('scale_saved_correction_seed_values', arch + '.' + arm, calc['mean'], old['mean'])
                if arm == 'P25':
                    for seed, val in old['seed_values'].items():
                        self.metrics[f'scale.seed.{arch}.{seed}.correction'] = val
        self.coverage['scale'] = dict(level='recomputed_from_per_seed_saved_means', seed_count=15,
            primary_tests=9, retained_negative_depth_seeds=['s14', 's16'],
            limit='Within-seed row-level intervals and model forwards are not rerun.')

    def horizon(self):
        sources = [('original', self.load(HORIZON)), ('supplement', self.load(SUPPLEMENT))]
        original = {}
        primary = []
        for cohort, saved in sources:
            index = {(r['seed'], r['arm'], int(r['budget'])): r['metrics'] for r in saved['per_seed']}
            if cohort == 'original':
                original = index
            for budget, item in saved['budgets'].items():
                budget = int(budget)
                seeds = item.get('seeds', ['s18', 's19', 's20', 's21', 's22'])
                if cohort == 'original':
                    assert len(seeds) == (5 if budget == 10000000 else 3)
                for task in TASKS:
                    for metric in ('acc', 'ce'):
                        key = task + '/' + metric
                        av = [index[s, 'A', budget][key] for s in seeds]
                        pv = [index[s, 'P25', budget][key] for s in seeds]
                        threshold = .05 if key == 'icl/acc' else 0
                        for arm, values in [('A', av), ('P25', pv), ('difference', [p-a for p, a in zip(pv, av)])]:
                            calc = summarize(values, threshold if arm == 'difference' else 0)
                            prefix = f'horizon.{cohort}.{budget}.{arm}.{task}.{metric}'
                            self.put(prefix, calc)
                            old = item['metrics'][key][arm]
                            for k in ('mean', 'sd', 'se', 'lower_one_sided95', 'upper_one_sided95'):
                                self.check('horizon_saved_statistics', prefix + '.' + k, calc[k], old[k])
                            self.check('horizon_saved_statistics', prefix + '.ci95', calc['ci95'], [old['lower95'], old['upper95']])
                            if cohort == 'original' and budget in (10000000, 40000000) and key == 'icl/acc' and arm == 'difference':
                                primary.append((budget, calc, saved['primary'][str(budget)]))
                        if task == 'icl' and metric == 'acc':
                            for seed, a, p in zip(seeds, av, pv):
                                for arm, val in [('A', a), ('P25', p), ('difference', p-a)]:
                                    self.metrics[f'horizon.seed.{cohort}.{budget}.{seed}.{arm}'] = val
        adjusted = holm([c['p_one_sided'] for _, c, _ in primary])
        for (budget, c, old), p in zip(primary, adjusted):
            self.metrics[f'horizon.primary.{budget}.p_holm'] = p
            self.check('horizon_saved_primary', str(budget) + '.p_holm', p, old['p_holm'])
        changes = [original[s, 'P25', 40000000]['icl/acc'] - original[s, 'P25', 10000000]['icl/acc'] for s in ('s18', 's19', 's20')]
        persistence = summarize(changes)
        self.put('horizon.within_arm_persistence', persistence)
        self.metrics['horizon.within_arm_persistence.supported'] = persistence['lower_one_sided95'] > -.02
        old = sources[0][1]['P25_40M_minus_10M']['icl/acc']
        self.check('horizon_saved_persistence', 'lower_bound', persistence['lower_one_sided95'], old['lower_one_sided95'])
        self.coverage['horizon'] = dict(level='recomputed_from_saved_seed_window_means',
            original_10M_n=5, original_20M_to_40M_n=3, supplement_n=5,
            primary_gain_threshold=.05, primary_family_size=2,
            supplement_role='descriptive, not substituted for original primary tests',
            limit='No row-level checkpoint-window recomputation or training rerun.')

    def external(self):
        saved = self.load(EXTERNAL)
        for dataset, block in saved['datasets'].items():
            rows = block['per_seed']
            assert [x['seed'] for x in rows] == ['s18', 's19', 's20', 's21', 's22']
            for metric in ('acc', 'ce', 'pair_acc'):
                for arm in ('A', 'P25'):
                    calc = summarize([x['models'][arm][metric] for x in rows])
                    prefix = f'external.{dataset}.{arm}.{metric}'
                    self.put(prefix, calc)
                    old = block['models'][arm][metric]
                    for k, oldk in [('mean', 'mean'), ('sd', 'sample_sd'), ('se', 'se')]:
                        self.check('external_saved_statistics', prefix + '.' + k, calc[k], old[oldk])
                diffs = [x['models']['P25'][metric] - x['models']['A'][metric] for x in rows]
                calc = summarize(diffs)
                prefix = f'external.{dataset}.difference.{metric}'
                self.put(prefix, calc)
                old = block['P25_minus_A'][metric]
                q = t_quantile(.9875, 4)
                simultaneous = [calc['mean'] - q * calc['se'], calc['mean'] + q * calc['se']]
                self.metrics[prefix + '.bonferroni.lower'], self.metrics[prefix + '.bonferroni.upper'] = simultaneous
                for k, oldk in [('mean', 'mean'), ('sd', 'sample_sd'), ('se', 'se')]:
                    self.check('external_saved_statistics', prefix + '.' + k, calc[k], old[oldk])
                self.check('external_saved_statistics', prefix + '.ci95', calc['ci95'], [old['ci95']['lower'], old['ci95']['upper']])
                oldci = old['bonferroni_two_datasets_ci97_5']
                self.check('external_saved_statistics', prefix + '.bonferroni', simultaneous, [oldci['lower'], oldci['upper']])
            for row in rows:
                seed = row['seed']
                for arm in ('A', 'P25'):
                    self.metrics[f'external.seed.{dataset}.{seed}.{arm}'] = row['models'][arm]['acc']
                self.metrics[f'external.seed.{dataset}.{seed}.difference'] = row['models']['P25']['acc'] - row['models']['A']['acc']
        self.coverage['external_seed_statistics'] = dict(level='recomputed_from_saved_per_seed_metrics',
            n=5, paired_ci='Student t, df=4', simultaneous_ci='Bonferroni over two datasets, individual 97.5%',
            limit='Queries/classes are not treated as additional training seeds.')

    def prediction_rows(self):
        import numpy as np
        all_saved = self.load(EXTERNAL)
        count = 0
        for dataset, exp in [('k49', 'exp_043_kuzushiji49_generalization'), ('cifar100', 'exp_044_cifar100_generalization')]:
            for seed in ('s18', 's19', 's20', 's21', 's22'):
                old = next(x for x in all_saved['datasets'][dataset]['per_seed'] if x['seed'] == seed)
                pair = {}
                for arm in ('A', 'P25'):
                    rel = f'experiments/{exp}/multiseed_10M_20261022/{seed}/outputs/{arm}_rows.npz'
                    with np.load(self.path(rel), allow_pickle=False) as f:
                        pair[arm] = {k: f[k].copy() for k in ('full_pred', 'pair_pred', 'target', 'ce', 'query_classes', 'image_ids')}
                    a = pair[arm]
                    calc = dict(acc=float(np.mean(a['full_pred'] == a['target'])),
                        ce=float(np.mean(a['ce'], dtype=np.float64)),
                        pair_acc=float(np.mean(a['pair_pred'] == a['target'])))
                    for metric, value in calc.items():
                        self.check('external_prediction_rows', f'{dataset}.{seed}.{arm}.{metric}', value, old['models'][arm][metric])
                    count += len(a['target'])
                for k in ('target', 'query_classes', 'image_ids'):
                    assert np.array_equal(pair['A'][k], pair['P25'][k]), (dataset, seed, k)
        rel = 'experiments/exp_041_emnist_generalization/outputs/'
        saved = self.load(rel + 'analysis.json')
        pairs = {}
        for arm in ('A', 'P25'):
            with np.load(self.path(rel + arm + '_rows.npz'), allow_pickle=False) as f:
                a = {k: f[k].copy() for k in ('full_pred', 'pair_pred', 'target', 'ce', 'query_classes', 'image_ids')}
            pairs[arm] = dict(acc=np.asarray(a['full_pred'] == a['target'], np.float64),
                ce=np.asarray(a['ce'], np.float64), pair_acc=np.asarray(a['pair_pred'] == a['target'], np.float64),
                classes=a['query_classes'], targets=a['target'], image_ids=a['image_ids'])
            for metric in ('acc', 'ce', 'pair_acc'):
                per_class = [float(pairs[arm][metric][a['query_classes'] == g].mean()) for g in np.unique(a['query_classes'])]
                calc = summarize(per_class)
                self.put(f'emnist.{arm}.{metric}', calc)
                old = saved['models'][arm][metric]
                self.check('emnist_saved_prediction_rows', arm + '.' + metric + '.mean', calc['mean'], old['mean'])
                self.check('emnist_saved_prediction_rows', arm + '.' + metric + '.ci95', calc['ci95'], [old['lower95'], old['upper95']])
        assert np.array_equal(pairs['A']['image_ids'], pairs['P25']['image_ids'])
        assert np.array_equal(pairs['A']['classes'], pairs['P25']['classes'])
        assert np.array_equal(pairs['A']['targets'], pairs['P25']['targets'])
        for metric in ('acc', 'ce', 'pair_acc'):
            diff = pairs['P25'][metric] - pairs['A'][metric]
            classes = pairs['A']['classes']
            calc = summarize([float(diff[classes == g].mean()) for g in np.unique(classes)])
            self.put(f'emnist.difference.{metric}', calc)
            old = saved['P25_minus_A'][metric]
            self.check('emnist_saved_prediction_rows', metric + '.mean', calc['mean'], old['mean'])
            self.check('emnist_saved_prediction_rows', metric + '.ci95', calc['ci95'], [old['lower95'], old['upper95']])
        self.coverage['prediction_rows'] = dict(level='recomputed_from_saved_predictions_and_per_query_CE',
            external_prediction_rows=count, emnist_queries=len(pairs['A']['classes']), emnist_query_classes=47,
            limit='CE is a mean of saved per-query CE. Logits, encoders and model forward passes are not recomputed. EMNIST remains one training trajectory.')

    def cost(self):
        saved = self.load(COST + '/analysis.json')
        all_seconds, raw_count = 0.0, 0
        values = {}
        for row in saved['rows']:
            arm, times = row['arm'], []
            assert len(row['runs']) == 4
            for old in row['runs']:
                raw = self.load(COST + '/outputs/' + old['tag'] + '/result.json', optional=True)
                source = raw if raw is not None else old
                raw_count += raw is not None
                windows = source['windows']
                assert len(windows) == 10
                seconds = sum(w['seconds'] for w in windows)
                steps = sum(w['steps'] for w in windows)
                assert steps == 10000
                ms = 1000 * seconds / steps
                self.check('cost_saved_windows', old['tag'], ms, old['ms_per_step'])
                self.metrics[f'cost.round.{int(old["round"])+1}.{arm}'] = ms
                times.append(ms)
                all_seconds += seconds
            mean, sd = stats.mean(times), stats.stdev(times)
            self.metrics[f'cost.{arm}.mean'], self.metrics[f'cost.{arm}.sd'] = mean, sd
            self.metrics[f'cost.{arm}.memory_saved_peak'] = row['sampled_device_peak_MiB']
            values[arm] = mean
            self.check('cost_repeats', arm + '.mean', mean, row['ms_per_step_mean'])
            self.check('cost_repeats', arm + '.sd', sd, row['ms_per_step_sd'])
        for arm, mean in values.items():
            self.metrics[f'cost.{arm}.relative_A'] = mean / values['A']
        reduction = 100 * (1 - values['P25'] / values['C25'])
        self.metrics['cost.P25.reduction_vs_C25_pct'] = reduction
        self.check('cost_repeats', 'reduction', reduction, saved['P25_step_time_reduction_vs_C25_pct'])
        self.check('cost_repeats', 'total_seconds', all_seconds, saved['total_measured_seconds'])
        if raw_count != 16:
            self.failures.append(dict(group='cost_required_inputs', issue='incomplete_process_result_inputs',
                expected=16, actual=raw_count,
                detail='Fallback summaries were calculated but do not satisfy the required saved-process-window coverage.'))
        self.coverage['cost'] = dict(level='recomputed_from_saved_timing_windows', process_repeats=16,
            windows=160, saved_process_result_json_count=raw_count, analysis_embedded_window_fallback_count=16-raw_count,
            limit='Process result files may be projected to manuscript-required windows; complete per-step logs are not inferred. Timing arithmetic is recomputed. GPU performance and memory occupancy are not remeasured; repeats are not training seeds.')

    def descriptive_readback(self):
        mapping = [('main', 'experiments/exp_031_path_only_protection/outputs/analysis.json'),
            ('main_random', 'experiments/exp_033_random_path_only_protection/outputs/analysis.json'),
            ('main_replay', 'experiments/exp_034_icl_replay/outputs/analysis.json'),
            ('main_mean', 'experiments/exp_046_single_constraint_task_loss/outputs/analysis.json'),
            ('strength', 'experiments/exp_038_protection_strength/outputs/analysis.json'),
            ('s7', 'experiments/exp_036_multiseed_path_protection/outputs/paper_s7_summary.json')]
        absent = []
        for group, rel in mapping:
            d = self.load(rel, optional=True)
            if d is None:
                absent.append(rel)
                continue
            for arm, tasks in d['descriptive'].items():
                for task in TASKS:
                    for metric in ('acc', 'loss'):
                        value = tasks[task][metric]['mean']
                        assert math.isfinite(value)
                        self.metrics[f'{group}.{arm}.{task}.{metric}'] = value
        self.coverage['main_strength_mechanism'] = dict(level='saved_summary_numeric_readback', missing=absent,
            limit='Except the separately rechecked s7 arrays, these summaries do not independently re-estimate row-level CIs, intervention effects, or update diagnostics.')
        frozen = self.load('figures/paper_figures/minimal_revision_20261023/data_frozen.json')
        def count_numeric(x):
            if isinstance(x, dict):
                return sum(count_numeric(v) for v in x.values())
            if isinstance(x, list):
                return sum(count_numeric(v) for v in x)
            return int(isinstance(x, (int, float)) and not isinstance(x, bool))
        self.coverage['mechanism_frozen_inputs'] = dict(level='saved_numeric_inputs_readback_only',
            sections=[k for k in frozen if k != 'sources'], numerical_scalars=count_numeric(frozen),
            limit='Frozen plotting numbers are readable and hash-bound. Intervention computation and its sampling uncertainty are not independently recomputed by this entry.')
        for group, arms in [('main', ('A', 'B', 'C25', 'P25')), ('main_random', ('PR25',)),
                            ('main_replay', ('REPLAY', 'L2')), ('main_mean', ('MEAN25',))]:
            for arm in arms:
                for task in TASKS:
                    key = f'{group}.{arm}.{task}.acc'
                    if key in self.metrics:
                        self.metrics[f'main_difference.{arm}.{task}.acc'] = self.metrics[key] - self.metrics[f'main.A.{task}.acc']

    def mean_task_rows(self):
        import numpy as np
        saved = self.load('experiments/exp_046_single_constraint_task_loss/outputs/analysis.json')
        rel = 'experiments/exp_046_single_constraint_task_loss/outputs/s5/paired_window_metrics.npz'
        count = 0
        with np.load(self.path(rel), allow_pickle=False) as arrays:
            for arm in ('MEAN25', 'A', 'C25', 'P25'):
                for task in TASKS:
                    for metric in ('acc', 'loss'):
                        values = np.asarray(arrays[f'{arm}/{task}/{metric}'], np.float64)
                        assert values.shape == (5000,) and np.isfinite(values).all()
                        actual = float(values.mean())
                        expected = saved['descriptive'][arm][task][metric]['mean']
                        self.check('mean_task_saved_rows', f'{arm}.{task}.{metric}', actual, expected)
                        self.metrics[f'main_mean.{arm}.{task}.{metric}'] = actual
                        count += 1
        self.coverage['mean_task_rows'] = dict(level='recomputed_from_saved_checkpoint_averaged_scores',
            scalar_checks=count, query_rows_per_task=5000, seed='init5/train2',
            limit='Arrays were already averaged over eleven checkpoints. Their means are reaggregated; original per-checkpoint logits, query-class intervals, and training are not independently rerun.')

    def s7_rows(self):
        import numpy as np
        old = self.load('experiments/exp_036_multiseed_path_protection/outputs/paper_s7_summary.json', optional=True)
        if old is None:
            raise FileNotFoundError('Required manuscript s7 summary missing')
        count = 0
        for arm in ('A', 'P25', 'PR25'):
            rel = f'experiments/exp_036_multiseed_path_protection/outputs/main_metric_delivery/s7_{arm}.npz'
            p = self.path(rel, optional=True)
            if p is None:
                continue
            with np.load(p, allow_pickle=False) as f:
                for task in TASKS:
                    classes = np.asarray(f[task + '/identities'][:, -1, 0])
                    unique, inverse = np.unique(classes, return_inverse=True)
                    for metric in ('acc', 'loss'):
                        a = np.asarray(f[task + '/' + metric], np.float64)
                        assert a.shape == (11, 5000)
                        reduced = a.mean(axis=0)
                        mean = float(reduced.mean())
                        sums = np.bincount(inverse, weights=reduced-mean, minlength=len(unique))
                        se = math.sqrt(len(unique)/(len(unique)-1)*float(sums@sums)/len(reduced)**2)
                        saved = old['descriptive'][arm][task][metric]
                        self.check('s7_saved_window_rows', arm + '.' + task + '.' + metric + '.mean', mean, saved['mean'])
                        self.check('s7_saved_window_rows', arm + '.' + task + '.' + metric + '.se', se, saved['se'])
                        self.metrics[f's7.{arm}.{task}.{metric}'] = mean
            count += 1
        if count != 3:
            self.failures.append(dict(group='s7_required_inputs', issue='incomplete_saved_arrays', expected=3, actual=count))
        self.coverage['s7_rows'] = dict(level='recomputed_from_saved_11_checkpoint_rows', arrays=count,
            limit='One init7/train4 trajectory only. Saving/reducing rows does not reproduce model forwards or add independent training seeds.')

    def paper(self, expected):
        spec = json.loads(expected.read_text())
        for item in spec['checks']:
            key = item['metric_key']
            if key not in self.metrics:
                self.failures.append(dict(group='manuscript_display', name=key, issue='required metric unavailable', table=item['table']))
                continue
            tolerance = .5 * 10 ** (-item['decimals']) + 1e-9
            self.check('manuscript_display', key, self.metrics[key] * item.get('scale', 1), item['expected'], tolerance)
        self.coverage['manuscript_binding'] = dict(source_file=spec['source_file'], source_sha256=spec['source_sha256'],
            checks=len(spec['checks']), tables=sorted({x['table'] for x in spec['checks']}),
            limit='Bindings cover the explicitly listed table cells and narrative values, not every manuscript statement or every figure pixel.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--supplements', type=Path)
    parser.add_argument('--expected', type=Path, default=Path(__file__).with_name('paper_expected_values.json'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    r = Recheck(args.root.resolve(), args.supplements.resolve() if args.supplements else None)
    for name in ('scale', 'horizon', 'external', 'prediction_rows', 'cost', 'descriptive_readback', 's7_rows', 'mean_task_rows'):
        try:
            getattr(r, name)()
        except Exception as e:
            r.failures.append(dict(group=name, issue=type(e).__name__, detail=str(e)))
    r.paper(args.expected)
    report = dict(schema='saved_paper_results_recheck_v1', passed=not r.failures,
        elapsed_seconds=time.monotonic()-started, no_model_execution=True,
        groups=r.groups, failures=r.failures, coverage=r.coverage,
        metrics=r.metrics, inputs=r.inputs,
        claim_boundary='Saved-result arithmetic and statistical reaggregation only; no independent training, model forward, logits, checkpoints or hardware rerun.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps(dict(passed=report['passed'], seconds=report['elapsed_seconds'],
        numeric_checks=sum(x['checked'] for x in r.groups.values()), failures=len(r.failures)), ensure_ascii=False), file=sys.stderr)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Independent NumPy/HDF5 reader; no scientific runtime import or model call.

The original frozen reader provides identity, tensor algebra and mechanism
checks. New strength-grid statistics below are separately implemented.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np
from reference_reader import (Reader, TASKS, SCALARS, verify_identity,
    feature_lookup, verify_datasets, verify_metric_group, clustered,
    gain_test, adjust_holm, independent_ni, digest_file, check_completion,
    verify_mechanism_gate, file_records, resolve, is_sha)

WORK = Path(__file__).resolve().parent
NEW = [k+s for s in ('10', '50', '75') for k in ('P', 'C', 'D')]
ALL = ['A']+[k+s for s in ('10', '25', '50', '75') for k in ('C', 'P', 'D')]


def frozen_design(r, s):
    r.require('scope/arms', s['arms'] == NEW)
    r.require('scope/seed', s['training_seeds'] == [{'id': 's5', 'init': 5, 'train': 2}])
    r.require('scope/grid', s['strength_grid'] == [.1, .25, .5, .75])
    r.require('scope/rhos', s['relaxation'] == {a: int(a[1:])/100 for a in NEW})
    r.require('scope/endpoints', (s['start_stage'], s['end_stage'], s['n'], s['train_bs'], s['train_microbs']) == (4400000, 10000000, 5000, 32, 32))
    r.require('scope/window', s['confirm_stages'] == list(range(9000000, 10000001, 100000)))
    r.require('scope/metrics_tasks', s['fresh_tasks'] == list(TASKS) and s['primary_tasks'] == ['icl', 'iwl_copy_avail', 'train_eval'])
    r.require('scope/families', s['primary_joint_family_size'] == s['primary_gain_family_size'] == 9)
    r.require('scope/thresholds', (s['alpha'], s['min_icl_gain'], s['ni_acc_tolerance'], s['ni_ce_tolerance']) == (.05, .05, .02, .1))
    r.require('scope/historical_boundary', s['original_Flip_joint_criteria_unchanged'] is True and s['confirmation_data_previously_observed'] is True)
    r.require('scope/constraint_definition', s['protection_mode'] == 'C_two_native_CE_P_single_counterfactual_D_all_three')


def references(r, s):
    manifest = json.loads((WORK/'references_manifest.json').read_text())
    for name, sha in manifest.items(): r.file('reference/receipt/'+name, WORK/'references'/name, sha)
    versions = {}
    for exp, root in [('exp030', Path(s['reference_work'])), ('exp031', Path(s['reference_p25_work']))]:
        v = json.loads((WORK/'references'/exp/'verification.json').read_text())
        r.require('reference/'+exp+'/passed', v.get('passed') is True and not v.get('failures'))
        settings = json.loads((WORK/'references'/exp/'settings.json').read_text())
        for key in ['training_seeds', 'start_stage', 'end_stage', 'train_bs', 'train_microbs', 'confirm_stages', 'data_sets', 'preserved_data_sha256', 'protection_bs_per_task', 'candidate_path_head']:
            r.require('reference/'+exp+'/matched/'+key, settings[key] == s[key])
        # Both old verifiers bind the original shared exp030 confirmation
        # dataset. P25 raw predictions still reside under exp031 below.
        path = Path(s['reference_work'])/'outputs/data/confirm.h5'
        expected = v['verified_file_sha256'].get(str(path.resolve()))
        r.require('reference/'+exp+'/data_registered', expected == s['preserved_data_sha256']['confirm.h5'])
        r.file('reference/'+exp+'/confirm_data', path, expected)
        versions[exp] = v
    return versions


def dataset_checks(r, s, inputs):
    import h5py
    # Metadata was created by the frozen exp030 generator, not by exp038.
    # The old reader's pilot branch encodes the approved unused-discovery rule.
    creation = r.file('identity/frozen_data_generator', WORK/'data_creation_manifest.json', s['data_generator_manifest_sha256'])
    r.data_generator_manifest_sha256 = creation
    adapted = dict(s, analysis_scope='single_seed_pilot')
    for name, digest in s['preserved_data_sha256'].items():
        r.file('data/preserved/'+name, r.out/'data'/name, digest)
    labels, query = verify_datasets(r, adapted, feature_lookup(r, inputs))
    with h5py.File(r.out/'data/confirm.h5', 'r') as f:
        identities = {task: f[task+'/identities'][:] for task in TASKS}
    audit_path = r.out/'data/identity_audit.json'
    r.file('data/identity_audit', audit_path)
    audit = json.loads(audit_path.read_text())
    r.require('data/identity_audit/passed', audit['passed'] is True and audit['resampling'] is False)
    for name in s['data_sets']:
        path = str((r.out/'data'/f'{name}.h5').resolve())
        r.require('data/identity_audit/hash/'+name, audit['data'][name]['sha256'] == r.new_inputs.get(path) == s['preserved_data_sha256'][name+'.h5'])
    return labels, query, identities


def preflight_checks(r, s, inputs, source, code):
    path = r.out/'preflight.json'; r.file('preflight/receipt', path)
    pre = json.loads(path.read_text())
    r.require('preflight/passed', pre.get('passed') is True)
    r.require('preflight/code', pre.get('code_manifest_sha256') == code)
    r.require('preflight/gpu', pre.get('model_forward_and_gradient_gpu_verified') is True)
    r.require('preflight/frozen_limits', pre.get('limits') == s)
    for section, expected, base in [('inputs', inputs, r.assets_root), ('source', source, r.assets_root/'source')]:
        r.require('preflight/'+section, file_records(base, pre.get(section)) == expected)
    checks = pre['checks']
    for kind, count in [('C', 2), ('P', 1), ('D', 3)]:
        r.require('preflight/'+kind+'/constraint_count', checks[kind+'_constraint_count'] == count)
        r.require('preflight/'+kind+'/rho1_adam_exact', checks[kind+'_rho1_adam_exact'] is True)
        model_error = checks[kind+'_rho1_model_max_abs']
        r.require('preflight/'+kind+'/rho1_model', 0 <= model_error <= s['numeric_model_adam_max_abs'])
        for suffix, limit in [('numpy_max_abs', s['numeric_projection_reference_max_abs']),
                              ('numpy_relative', s['numeric_projection_reference_max_relative']),
                              ('relative_violation', s['numeric_projection_relative_violation_max']),
                              ('jit_relative_violation', s['numeric_projection_relative_violation_max'])]:
            key = 'projection_'+kind+'_'+suffix
            value = checks[key]
            r.require('preflight/'+key, isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= limit)
        grid = pre['grid_checks'][kind]
        r.require('preflight/'+kind+'/exact_grid', set(grid) == {'10', '25', '50', '75'})
        for suffix in ['10', '25', '50', '75']:
            detail = grid[suffix]
            r.require('preflight/'+kind+'/'+suffix+'/identity', detail['passed'] is True and detail['rho'] == int(suffix)/100)
            for field, limit in [('numpy_max_abs', s['numeric_projection_reference_max_abs']),
                                 ('numpy_relative', s['numeric_projection_reference_max_relative']),
                                 ('relative_violation', s['numeric_projection_relative_violation_max']),
                                 ('jit_relative_violation', s['numeric_projection_relative_violation_max']),
                                 ('production_model_max_abs', s['numeric_model_adam_max_abs'])]:
                value = detail[field]
                r.require('preflight/'+kind+'/'+suffix+'/'+field, isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= limit)
    for case in ['first_count', 'zero_gradient_nonzero_moments']:
        for field, limit in [('max_abs', s['numeric_projection_reference_max_abs']), ('relative', s['numeric_projection_reference_max_relative'])]:
            value = checks[case][field]
            r.require('preflight/optimizer/'+case+'/'+field, isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= limit)
    r.require('preflight/evaluation_scope', pre['evaluation_parity_rows_per_task'] == 128)
    for task in (*TASKS, 'pure_iwl'):
        for field, limit in [('eval_ce_max_mean', s['numeric_eval_mean_ce_max_abs']), ('eval_acc_rows', s['numeric_eval_acc_max_rows'])]:
            v = checks[task+'_'+field]
            r.require('preflight/'+task+'/'+field, isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= limit)
    derivative = checks['functional_derivative']
    norm, fd = derivative['autodiff'], derivative['finite_difference']
    error = abs(norm-fd)
    r.require('preflight/functional_derivative_nonzero', math.isfinite(norm) and norm > 0)
    r.close('preflight/functional_derivative_error', derivative['abs_error'], error, 1e-10)
    r.require('preflight/functional_derivative_gate', error <= s['numeric_functional_grad_absolute_tolerance']+s['numeric_functional_grad_relative_tolerance']*norm)
    path = r.out/'performance_validation.json'; r.file('performance/receipt', path)
    perf = json.loads(path.read_text())
    r.require('performance/passed', perf['passed'] is True)
    r.require('performance/code', perf['code_manifest_sha256'] == code)
    for kind in ('C', 'P', 'D'):
        for field in ['model_exact', 'adam_exact', 'prng_exact', 'status_exact', 'resume_exact']:
            r.require('performance/'+kind+'/'+field, perf['metrics'][kind][field] is True)
        r.require('performance/'+kind+'/fixed_steps', perf['metrics'][kind]['steps'] == 64)
        for field in ['model_max_abs', 'adam_max_abs']:
            r.require('performance/'+kind+'/'+field, perf['metrics'][kind][field] == 0.)
        timing = perf['timings'][kind]
        for field in ['strict_seconds', 'chunk_seconds']:
            value = timing[field]
            r.require('performance/'+kind+'/'+field, isinstance(value, (int, float)) and math.isfinite(value) and value > 0)
        selected = perf['selected'][kind]
        r.require('performance/'+kind+'/selected_boolean', isinstance(selected, bool))
        r.require('performance/'+kind+'/reason', isinstance(perf['reason'][kind], str) and bool(perf['reason'][kind].strip()))
        r.require('performance/'+kind+'/enable_only_if_faster', not selected or timing['chunk_seconds'] < timing['strict_seconds'])
        speedup = timing['strict_seconds']/timing['chunk_seconds']
        r.close('performance/'+kind+'/speedup', timing['speedup'], speedup, 1e-10)
        r.require('performance/'+kind+'/selection_threshold', selected == (speedup > 1.03))
    r.require('performance/scientific_batch_preserved', perf['scientific_batch_size'] == s['train_bs'] and perf['committed_training_steps'] == 0 and perf['chunk_size'] == s['chunk_size'] == 16)
    return pre, perf


def read_window(r, s, parent, arm, labels, identities, verification=None):
    rows = {task: {metric: [] for metric in SCALARS} for task in TASKS}
    for stage in s['confirm_stages']:
        path = parent/'outputs/s5/confirm'/f'{arm}_{stage:011d}.npz'
        expected = None
        if verification is not None:
            expected = verification['verified_file_sha256'].get(str(path.resolve()))
            r.require('reference/raw_registered/'+str(path), is_sha(expected))
        r.file('raw/'+arm+'/'+str(stage), path, expected)
        with np.load(path, allow_pickle=False) as raw:
            for task in TASKS:
                r.close(f'raw/{arm}/{stage}/{task}/identities', raw[task+'/identities'], identities[task], 0.)
                metrics = verify_metric_group(r, raw, task, labels[task], s['n'], f'raw/{arm}/{stage}/{task}')
                for metric, value in metrics.items(): rows[task][metric].append(value)
    return {t: {m: np.sum(np.asarray(values), axis=0, dtype=np.float64)/len(s['confirm_stages']) for m, values in ms.items()} for t, ms in rows.items()}


def prng_hash(record):
    return hashlib.sha256(b''.join(np.asarray(record[key], dtype=np.uint32).tobytes() for key in ['train_data_seed', 'train_model_seed'])).hexdigest()


def training_checks(r, s, inputs, source, code, versions):
    expected_stages = list(range(s['start_stage'], s['end_stage']+1, s['eval_every']))
    markers, completions = {}, {}
    for arm in NEW:
        kind, rho, count = arm[0], s['relaxation'][arm], {'C': 2, 'P': 1, 'D': 3}[arm[0]]
        receipt = check_completion(r, 's5', arm, s, inputs, source, code)
        completions[arm] = receipt
        r.require('completion/'+arm+'/kind_rho', receipt['kind'] == kind and receipt['rho'] == rho)
        r.require('completion/'+arm+'/constraint_count', receipt['constraint_count'] == count)
        steps = receipt.get('steps', receipt.get('update_statistics_sum', {}).get('steps'))
        r.require('completion/'+arm+'/steps', steps == 175000)
        markers[arm] = {}
        files = {p.name for p in (r.out/'s5/evaluated').glob(arm+'_*.json')}
        r.require('checkpoint/'+arm+'/exact_stages', files == {f'{arm}_{stage:011d}.json' for stage in expected_stages})
        for stage in expected_stages:
            marker_path = r.out/'s5/evaluated'/f'{arm}_{stage:011d}.json'; r.file('marker/'+arm+'/'+str(stage), marker_path)
            marker = json.loads(marker_path.read_text()); markers[arm][stage] = marker
            r.require(f'checkpoint/{arm}/{stage}/identity', marker['iter'] == stage and marker['code_manifest_sha256'] == code and marker['kind'] == kind and marker['rho'] == rho)
            r.require(f'checkpoint/{arm}/{stage}/adam_count', marker['adam_count'] == stage//s['train_bs'])
            for key in ['train_data_seed', 'train_model_seed', 'eval_model_seed']:
                a = np.asarray(marker[key])
                r.require(f'checkpoint/{arm}/{stage}/{key}', a.shape == (2,) and np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a < 2**32)))
            ck = r.out/'s5/checkpoints'/f'{arm}_{stage:011d}.eqx'
            r.file(f'checkpoint/{arm}/{stage}/bytes', ck, marker['checkpoint_sha256'])
        final_prng = receipt['training_prng']
        for key in ['train_data_seed', 'train_model_seed', 'eval_model_seed']:
            r.close('completion/'+arm+'/'+key, final_prng[key], markers[arm][s['end_stage']][key], 0.)
        r.require('completion/'+arm+'/PRNG_digest', receipt['training_prng_sha256'] == prng_hash(final_prng))
        for key in ['mean_update_norm_ratio', 'mean_correction_norm_ratio', 'conflict_rate', 'near_zero_rate']:
            value = receipt['update_statistics'][key]
            r.require('completion/'+arm+'/update/'+key, isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 and (key not in ['conflict_rate', 'near_zero_rate'] or value <= 1))
        raw_sum = receipt['update_statistics_sum']
        for summary, total in [('mean_update_norm_ratio', 'update_norm_ratio'), ('mean_correction_norm_ratio', 'correction_norm_ratio'), ('conflict_rate', 'conflicts'), ('near_zero_rate', 'near_zero_steps')]:
            r.close('completion/'+arm+'/sum/'+summary, receipt['update_statistics'][summary], raw_sum[total]/175000, 1e-10)
        audit_files = sorted((r.out/'s5/update_audit').glob(arm+'_*.json'))
        audit_stages = {s['start_stage']+offset*s['train_bs'] for offset in s['diagnostic_update_offsets']}
        r.require('audit/'+arm+'/five_fixed_stages', {p.name for p in audit_files} == {f'{arm}_{stage:011d}.json' for stage in audit_stages})
        for path in audit_files:
            r.file('audit/'+path.name, path); a = json.loads(path.read_text())
            r.require('audit/'+path.name+'/identity', a['arm'] == arm and a['kind'] == kind and a['head'] == 0 and a['code_manifest_sha256'] == code and a['adjusted_parameters'] is False and a['stage'] in audit_stages)
            r.require('audit/'+path.name+'/batch_names', set(a['results']) == {'actual_protection_batch', 'independent_monitor_batch'})
            for batch, fields in a['results'].items():
                required = {'loss_before', 'actual_raw_change', 'actual_new_change', 'predicted_optimizer_raw', 'predicted_raw_applied', 'predicted_new_applied'}
                r.require('audit/'+path.name+'/'+batch+'/fields', set(fields) == required)
                for name, value in fields.items():
                    v = np.asarray(value)
                    r.require('audit/'+path.name+'/'+batch+'/'+name, v.shape == (count,) and np.isfinite(v).all())
                # First-order feasibility is checked only on the actual
                # constraint batch; held-out nonlinear changes are diagnostic.
                if batch == 'actual_protection_batch':
                    before = np.asarray(fields['predicted_optimizer_raw'], np.float64)
                    after = np.asarray(fields['predicted_new_applied'], np.float64)
                    bound = rho*np.maximum(before, 0.)
                    normalized = float(np.max(np.maximum(after-bound, 0.)) / max(float(np.max(np.abs(before))), 1e-20))
                    # FP32 application rounding may exceed the optimizer-space
                    # projection tolerance. Record, do not retune or reject.
                    r.maximum_errors['applied_audit_relative_violation'] = max(normalized, r.maximum_errors.get('applied_audit_relative_violation', 0.))
    for arm in NEW[1:]:
        for stage in expected_stages:
            for key in ['train_data_seed', 'train_model_seed', 'eval_model_seed']:
                r.close(f'paired_training/{arm}/{stage}/{key}', markers[arm][stage][key], markers[NEW[0]][stage][key], 0.)
    path = r.out/'s5/reference_training_prng.json'; r.file('paired_training/historical_receipt', path)
    reference = json.loads(path.read_text())
    source_path = Path(reference['source_path'])
    expected = versions['exp030']['verified_file_sha256'].get(str(source_path.resolve()))
    r.require('paired_training/reference_source_registered', is_sha(expected) and expected == reference['source_sha256'])
    r.file('paired_training/reference_source', source_path, expected)
    keys = reference.get('training_prng', reference)
    indices = reference.get('seed_leaf_indices', reference.get('numpy_stream_indices'))
    if indices is not None:
        r.require('paired_training/reference_three_leaf_indices', set(indices) == {'train_data_seed', 'train_model_seed', 'eval_model_seed'} and len(set(indices.values())) == 3 and all(isinstance(i, int) and i >= 0 for i in indices.values()))
        selected = {}
        with source_path.open('rb') as stream:
            for index in range(max(indices.values())+1):
                value = np.load(stream, allow_pickle=False)
                for key, slot in indices.items():
                    if slot == index: selected[key] = value
        for key in ['train_data_seed', 'train_model_seed', 'eval_model_seed']:
            r.require('paired_training/reference_independent_leaf/'+key+'/uint32', selected[key].shape == (2,) and selected[key].dtype == np.uint32)
            r.close('paired_training/reference_independent_leaf/'+key, keys[key], selected[key], 0.)
        # Check every marker against the immutable NumPy leaf stream rather
        # than trusting the JSON PRNG fields written by the training process.
        for arm in NEW:
            for stage in expected_stages:
                checkpoint = r.out/'s5/checkpoints'/f'{arm}_{stage:011d}.eqx'
                found = {}
                with checkpoint.open('rb') as stream:
                    for index in range(max(indices.values())+1):
                        value = np.load(stream, allow_pickle=False)
                        for key, slot in indices.items():
                            if slot == index: found[key] = value
                for key in ['train_data_seed', 'train_model_seed', 'eval_model_seed']:
                    value = found[key]
                    r.require(f'paired_training/checkpoint/{arm}/{stage}/{key}/uint32_leaf', value.shape == (2,) and value.dtype == np.uint32)
                    r.close(f'paired_training/checkpoint/{arm}/{stage}/{key}/marker', value, markers[arm][stage][key], 0.)
    else:
        r.require('paired_training/reference_independent_leaf_indices_required', False)
    r.require('paired_training/reference_hash', reference['training_prng_sha256'] == prng_hash(keys))
    for arm in NEW:
        for key in ['train_data_seed', 'train_model_seed']:
            r.close('paired_training/historical_final/'+arm+'/'+key, completions[arm]['training_prng'][key], keys[key], 0.)
    return completions


def independent_statistics(data, query, s):
    desc = {arm: {task: {m: clustered(value, query[task]) for m, value in ms.items()}
                         for task, ms in ts.items()} for arm, ts in data.items()}
    pairs = [(arm, 'A') for arm in ALL if arm != 'A']
    pairs += [(l+suf, rr+suf) for suf in ['10', '25', '50', '75'] for l, rr in [('P', 'C'), ('P', 'D'), ('D', 'C')]]
    differences = {a+'-'+b: {task: {metric: clustered(data[a][task][metric]-data[b][task][metric], query[task]) for metric in SCALARS}
                                  for task in TASKS} for a, b in pairs}
    gain_tests, joint_tests, gates = [], [], {}
    for arm in NEW:
        name = arm+'-A'; delta = differences[name]
        gain = gain_test(delta['icl']['acc'], name+'/icl/acc', s['min_icl_gain'])
        gain_tests.append(gain); components, tests = [gain], {}
        for task in ('iwl_copy_avail', 'train_eval'):
            for metric in ('acc', 'loss'):
                key = task+'/'+metric
                tests[key] = independent_ni(delta[task][metric], metric, s)
                if metric == 'acc': z, threshold, suffix = delta[task][metric], -s['ni_acc_tolerance'], ''
                else:
                    z = clustered(data['A'][task][metric]-data[arm][task][metric], query[task])
                    threshold, suffix = -s['ni_ce_tolerance'], '/negative_delta'
                components.append(gain_test(z, name+'/'+key+suffix, threshold))
        gates[name] = {'tests': tests, 'all_passed': all(item['passed'] for item in tests.values())}
        joint_tests.append({'name': name+'/ordinary_ICL_and_capability_joint', 'threshold': 0.,
                            'mean': min(item['mean']-item['threshold'] for item in components),
                            'valid': all(item['valid'] for item in components), 'p_raw': max(item['p_raw'] for item in components),
                            'components': components})
    gains, joint = adjust_holm(gain_tests, s['alpha']), adjust_holm(joint_tests, s['alpha'])
    decisions = {arm: {'gain_Holm9_passed': gains['tests'][i]['passed'], 'all_four_noninferiority_passed': gates[arm+'-A']['all_passed'],
                       'joint_Holm9_passed': joint['tests'][i]['passed']} for i, arm in enumerate(NEW)}
    points = {arm: tuple(desc[arm][task]['acc']['mean'] for task in ['icl', 'iwl_copy_avail', 'train_eval']) for arm in ALL}
    dominators = {arm: [other for other in ALL if other != arm and all(x >= y for x, y in zip(points[other], points[arm]))
                        and any(x > y for x, y in zip(points[other], points[arm]))] for arm in ALL}
    return dict(descriptive=desc, paired_comparisons=differences, ordinary_ICL_gain_Holm9=gains, primary_joint_Holm9=joint,
                noninferiority=gates, arm_decisions=decisions, descriptive_accuracy_frontier={'non_dominated': [a for a in ALL if not dominators[a]], 'dominated_by': dominators})


def verify():
    r, independent, decisions = Reader(WORK), {}, {}
    try:
        s = json.loads((WORK/'settings.json').read_text()); frozen_design(r, s)
        provenance, inputs, source, code = verify_identity(r)
        r.require('identity/CUDA_backend', bool(provenance.get('device')) and all('cuda' in str(d).lower() or 'gpu' in str(d).lower() for d in provenance['device']))
        r.require('identity/author_precision', provenance.get('enable_x64') is False and provenance.get('default_matmul_precision') is None)
        versions = references(r, s)
        labels, query, ids = dataset_checks(r, s, inputs)
        preflight_checks(r, s, inputs, source, code)
        path = r.out/'s5/mechanism_confirmation.json'; r.file('mechanism/receipt', path)
        record = json.loads(path.read_text())
        passed = verify_mechanism_gate(r, s['training_seeds'][0], dict(s, analysis_scope='single_seed_pilot'), record, inputs, source, code)
        r.require('mechanism/required_gate', passed is True)
        training_checks(r, s, inputs, source, code, versions)
        expected_names = {f'{a}_{stage:011d}.npz' for a in NEW for stage in s['confirm_stages']}
        r.require('raw/exact_new_file_set', {p.name for p in (r.out/'s5/confirm').glob('*.npz')} == expected_names)
        data = {a: read_window(r, s, WORK, a, labels, ids) for a in NEW}
        for a in ['A', 'C25', 'D25']:
            data[a] = read_window(r, s, Path(s['reference_work']), a, labels, ids, versions['exp030'])
        data['P25'] = read_window(r, s, Path(s['reference_p25_work']), 'P25', labels, ids, versions['exp031'])
        saved = json.loads((r.out/'analysis.json').read_text())
        for key, path in [('code_manifest_sha256', WORK/'code_manifest.json'), ('settings_sha256', WORK/'settings.json'),
                          ('protocol_sha256', WORK/'protocol.md'), ('references_manifest_sha256', WORK/'references_manifest.json'),
                          ('confirm_data_sha256', r.out/'data/confirm.h5')]:
            r.require('analysis/binding/'+key, saved[key] == digest_file(path))
        r.require('analysis/rho25_descriptive', saved['historical_rho25_role'] == 'descriptive_only_not_in_new_Holm_family')
        r.require('analysis/original_Flip_unchanged', saved['original_Flip_joint_criteria_unchanged'] is True and saved['confirmation_data_previously_observed'] is True)
        independent = independent_statistics(data, query, s)
        for key, value in independent.items(): r.subset('analysis/'+key, saved[key], value)
        decisions = independent['arm_decisions']
        for name in ['report.md', 'summary.tsv', 'parameter_curves.png', 'icl_ciwl_scatter.png']:
            r.file('analysis/artifact/'+name, r.out/name)
        import csv
        with (r.out/'summary.tsv').open() as stream: table = list(csv.DictReader(stream, delimiter='\t'))
        r.require('analysis/summary_complete', len(table) == len(ALL)*len(TASKS)*len(SCALARS))
        seen = set()
        for row in table:
            key = (row['arm'], row['task'], row['metric']); seen.add(key)
            stat = independent['descriptive'][key[0]][key[1]][key[2]]
            for field in ['mean', 'se', 'ci95_lower', 'ci95_upper']:
                observed = None if row[field] == '' else float(row[field])
                r.subset('analysis/table/'+'/'.join(key)+'/'+field, observed, stat[field])
        r.require('analysis/summary_unique', len(seen) == len(table))
    except Exception as error:
        r.require('reader/no_exception', False, dict(type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
    passed = not r.failures
    result = dict(passed=passed, status='verified' if passed else 'failed', checks=r.checks,
                  failures=r.failures, counts=dict(r.counts), maximum_errors=r.maximum_errors,
                  verified_file_sha256=r.hashes, independent_statistics=independent,
                  scientific_support={a: bool(passed and d['joint_Holm9_passed']) for a, d in decisions.items()},
                  scope='Independent saved tensor/metric algebra, immutable data and source hashes, PRNG leaves and fixed-window statistics; no independent training rerun or full forward reproduction.',
                  completed_at_utc=datetime.now(timezone.utc).isoformat())
    r.out.mkdir(exist_ok=True)
    (r.out/'verification.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(passed=passed, checks=r.checks, failures=r.failures[:3], scientific_support=result['scientific_support']), ensure_ascii=False), flush=True)
    return passed


if __name__ == '__main__':
    raise SystemExit(0 if verify() else 1)

"""Run the two frozen inference experiments; every checkpoint/group is resumable."""
import argparse
import os
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--phase', choices=['preflight', 'window', 'path'], required=True)
args = ap.parse_args()
began = time.perf_counter()
selection = json.loads((O / 'time_selection.json').read_text())
assert selection['frozen_before_fresh_inputs']
assert selection['settings_sha256'] == sha(W / 'settings.json')
repair = json.loads((W / 'execution_repair.json').read_text()) if (W / 'execution_repair.json').exists() else None
assert selection['protocol_sha256'] == (repair['original_protocol_sha256'] if repair else sha(W / 'protocol.md'))
assert selection['code_manifest_sha256'] == (repair['original_code_manifest_sha256'] if repair else sha(W / 'code_manifest.json'))
inputs = verify_inputs(selection['window_stages'])
old, discovery, confirmation = load_inputs()
template = get_template()
author_native = make_author_reference(False)
author_block = make_author_reference(True)
provenance = dict(version=SETTINGS['version'], settings=SETTINGS,
                  input_identity=inputs, confirmation_sha256=sha(O / 'confirmation_data.h5'),
                  discovery_sha256=sha(O / 'discovery_data.h5'),
                  devices=[str(d) for d in jax.devices()], jax=jax.__version__, equinox=eqx.__version__,
                  dtype='float32', matmul_precision='float32 evaluation arithmetic; original training/default logs retained',
                  no_training=True, scientific_background='native checkpoint, no cross-stage mixing')
for output in [O, PO]:
    dest = output / 'provenance.json'
    if dest.exists():
        previous=json.loads(dest.read_text())
        if previous != provenance and (O/'reused_baseline_groups.json').exists():
            assert previous['matmul_precision'].startswith('float32')
            assert previous['confirmation_sha256']==provenance['confirmation_sha256']
            assert previous['discovery_sha256']==provenance['discovery_sha256']
            write_json(output/'provenance_before_validation_amendment.json',previous)
            write_json(dest,provenance)
        else:
            assert previous == provenance, 'Resume provenance mismatch'
    else:
        write_json(dest, provenance)


def budget_check():
    start = float(os.environ.get('EXPERIMENT_STARTED_EPOCH', time.time()))
    if time.time() - start >= SETTINGS['execution']['safe_budget_seconds']:
        write_json(O / 'budget_stop.json', dict(phase=args.phase, time=time.time(),
                    completed_groups_preserved=True, continuation_required=True))
        raise RuntimeError('Safe 2.5-hour boundary reached; completed groups preserved')


def has_group(f, dataset, stage, code, name):
    key = f'{dataset}/{stage}/{code}/{name}'
    return key in f and bool(f[key].attrs.get('completed', False))


def save_group(f, dataset, stage, code, name, chunks):
    key = f'{dataset}/{stage}/{code}/{name}'
    if key in f:
        assert not f[key].attrs.get('completed', False)
        del f[key]
    g = f.create_group(key)
    summary = dict(stage=stage, actual_iter=checkpoint_path(stage)[1], dataset=dataset,
                   code=code, evaluator=name, count=N)
    for metric, arrays0 in chunks.items():
        value = np.concatenate(arrays0, axis=0)
        assert value.shape[0] == N and np.isfinite(value).all(), (key, metric)
        g.create_dataset(metric, data=value, compression='gzip', compression_opts=1)
        mean = value.mean(axis=0, dtype=np.float64)
        summary[metric] = float(mean) if mean.ndim == 0 else mean.tolist()
    g.attrs['summary'] = json.dumps(summary, sort_keys=True)
    g.attrs['matrix_precision']='float32'
    g.attrs['producer_code_manifest_sha256']=sha(W/'code_manifest.json')
    g.attrs['completed'] = True
    f.flush()
    return summary


def append_chunks(chunks, value):
    for k, v in value.items():
        chunks.setdefault(k, []).append(v)


def err(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def final_equivalence(a, b):
    z, w = np.asarray(a['out'][:, -1]), np.asarray(b['out'][:, -1])
    assert np.array_equal(z.argmax(axis=1), w.argmax(axis=1))
    delta = err(z, w)
    assert delta <= SETTINGS['audit']['logit_equivalence_tolerance'], delta
    return delta


def structure(model, x, y):
    rn = native(model, x, y)
    rs = suppress(model, x, y, rn, 'P1')
    r_author_native = author_native(model, x, y)
    r_author_block = author_block(model, x, y)
    r_pb = suppress(model, x, y, rn, 'PB')
    k = path_result(model, x, y, rn, rs, 'K')
    v = path_result(model, x, y, rn, rs, 'V')
    kb = path_result(model, x, y, rn, rs, 'KB')
    vb = path_result(model, x, y, rn, rs, 'VB')
    kv = path_result(model, x, y, rn, rs, 'KV')
    an, ass = attn(rn, 0), attn(rs, 0)
    bn, bs = attn(rn, 1), attn(rs, 1)
    nonlabel = jnp.array([0, 2, 4])
    checks = {}
    checks['unmodified_author_native'] = final_equivalence(rn, r_author_native)
    checks['author_opto_PB_rebuild'] = final_equivalence(r_pb, r_author_block)
    def exact(name, a, b):
        checks[name] = err(a, b)
        assert checks[name] == 0, (name, checks[name])
    exact('L1_Q', an['q'], ass['q'])
    exact('L1_K', an['k'], ass['k'])
    exact('L1_V', an['v'], ass['v'])
    exact('nonlabel_L1_output', rn['transformer_output']['block_outputs'][0]['out'][:, nonlabel],
          rs['transformer_output']['block_outputs'][0]['out'][:, nonlabel])
    exact('query_L2Q', bn['q'][:, :, :, 4], bs['q'][:, :, :, 4])
    exact('query_direct_residual', rn['transformer_output']['block_outputs'][1]['x'][:, 4],
          rs['transformer_output']['block_outputs'][1]['x'][:, 4])
    exact('nonlabel_L2K', bn['k'][:, :, :, nonlabel], bs['k'][:, :, :, nonlabel])
    exact('nonlabel_L2V', bn['v'][:, :, :, nonlabel], bs['v'][:, :, :, nonlabel])
    sn, ss = np.asarray(an['attn_pre_softmax']), np.asarray(ass['attn_pre_softmax'])
    exact('non_target_score', np.where(PREV, 0, sn), np.where(PREV, 0, ss))
    exact('target_score', np.where(PREV, sn-np.float32(1), 0), np.where(PREV, ss, 0))
    aa = np.asarray(ass['attn_scores'])
    checks['future_attention'] = float(np.max(np.where(VALID, 0, aa)))
    assert checks['future_attention'] == 0
    checks['rowsum'] = float(np.max(np.abs(aa.sum(axis=-1)-1)))
    assert checks['rowsum'] <= SETTINGS['audit']['attention_rowsum_tolerance']
    exact('K_only_V_preserved', attn(k, 1)['v'], bn['v'])
    exact('K_only_query_Q_preserved', attn(k, 1)['q'][:, :, :, 4], bn['q'][:, :, :, 4])
    exact('K_only_nonlabel_K', attn(k, 1)['k'][:, :, :, nonlabel], bn['k'][:, :, :, nonlabel])
    exact('K_only_label_K', attn(k, 1)['k'][:, :, :, jnp.array([1, 3])],
          bs['k'][:, :, :, jnp.array([1, 3])])
    exact('V_only_K_preserved', attn(v, 1)['k'], bn['k'])
    exact('V_only_attention_preserved', attn(v, 1)['attn_scores'], bn['attn_scores'])
    rebuilt = (attn(k, 1)['q'] @ jnp.swapaxes(attn(k, 1)['k'], -1, -2)) * model.transformer.blocks[1].attn.scale
    actual = attn(k, 1)['attn_pre_softmax']
    checks['K_QK_rebuild'] = err(rebuilt[:, :, :, 4], actual[:, :, :, 4])
    assert checks['K_QK_rebuild'] <= 1e-4
    for name, a, b in [('KB_equals_V', kb, v), ('VB_equals_K', vb, k), ('KV_equals_S', kv, rs)]:
        checks[name] = final_equivalence(a, b)
        ma, mb = metrics_numpy(a, y, rn, model), metrics_numpy(b, y, rn, model)
        assert np.array_equal(ma['pair_pred'], mb['pair_pred'])
    zs, zk = jnp.asarray(ZS), jnp.asarray(ZKV)
    sham = full_fn(model, x, y, an['attn_pre_softmax'], jnp.asarray(LABEL_ROWS),
                   bn['k'], zk, bn['v'], zk)
    row_shift = full_fn(model, x, y, an['attn_pre_softmax']-1, jnp.asarray(LABEL_ROWS),
                   bn['k'], zk, bn['v'], zk)
    checks['score_self_fill'] = final_equivalence(sham, rn)
    checks['row_constant_sham'] = final_equivalence(row_shift, rn)
    # Relative attention control: check the requested relative mass directly.
    rh = suppress(model, x, y, rn, 'H6R')
    ah = np.asarray(attn(rh, 0)['attn_scores'])
    ns = np.asarray(an['attn_scores'])
    rs_attn = np.asarray(ass['attn_scores'])
    for row, col in [(1, 0), (3, 2)]:
        p0 = ns[:, 0, 0, row, col].astype(np.float64)
        ratio = np.exp(-1)/(1-p0+p0*np.exp(-1))
        target = ratio * ns[:, 0, 6, row, col]
        checks[f'H6R_mass_row{row}'] = float(np.max(np.abs(ah[:, 0, 6, row, col]-target)))
        assert checks[f'H6R_mass_row{row}'] <= 1e-6
        checks[f'H6R_target_clipping_row{row}'] = int(np.sum(target > 1-1e-7))
        checks[f'h0_expected_mass_row{row}'] = float(np.max(np.abs(rs_attn[:, 0, 0, row, col]-ratio*p0)))
        assert checks[f'h0_expected_mass_row{row}'] <= 1e-6
    return checks


def evaluate_native(f, dataset, stages, data, sets):
    for stage in stages:
        budget_check()
        model = load_checkpoint(stage, template)
        before = digest_arrays(model)
        for name in sets:
            if has_group(f, dataset, stage, 'N', name):
                continue
            chunks = {}
            x, y = data[name]
            for offset in range(0, N, BS):
                xx, yy = jnp.asarray(x[offset:offset+BS]), jnp.asarray(y[offset:offset+BS])
                rn = native(model, xx, yy)
                append_chunks(chunks, metrics_numpy(rn, yy, rn, model))
            row = save_group(f, dataset, stage, 'N', name, chunks)
            print('NATIVE', dataset, stage, name, row['full_loss'], row['pair_acc'], flush=True)
        assert digest_arrays(model) == before


def main_preflight():
    audits = {}
    for stage in sorted(set([0, selection['peak_stage'], 4400000, 20000000])):
        model = load_checkpoint(stage, template)
        before = digest_arrays(model)
        audits[str(stage)] = {}
        for name in SETTINGS['eval_sets']:
            x, y = confirmation[name]
            audits[str(stage)][name] = structure(model, jnp.asarray(x[:BS]), jnp.asarray(y[:BS]))
        assert digest_arrays(model) == before
        print('STRUCTURE_PASSED', stage, flush=True)
    write_json(O / 'structure_audit.json', dict(passed=True, checks=audits,
                count_per_task=BS, all_non_target_exact=True, path_protocol_sha256=sha(P/'protocol.md')))
    write_json(PO / 'structure_audit.json', json.loads((O / 'structure_audit.json').read_text()))
    with h5py.File(O/'per_sequence.h5', 'a') as f:
        f.attrs['provenance'] = json.dumps(provenance, sort_keys=True)
        evaluate_native(f, 'original', [0, 2000000, 4400000, 8000000, 20000000], old, SETTINGS['eval_sets'])
        evaluate_native(f, 'discovery', sorted(set([selection['left_stage'], selection['peak_stage'],
                                                  selection['right_stage']])), discovery, ['iwl_copy_avail'])
        bridge = {}
        for folder in ['main', 'main_resume_00002950016']:
            with h5py.File(ROOT/'exp_002_main_reproduction/runs'/folder/'log.h5', 'r') as ref:
                its = np.asarray(ref['eval_iter'], np.int64)
                for stage in [0, 2000000, 4400000, 8000000, 20000000]:
                    if checkpoint_path(stage)[1] not in its:
                        continue
                    idx = int(np.flatnonzero(its == checkpoint_path(stage)[1])[0])
                    for name in SETTINGS['eval_sets']:
                        g = f[f'original/{stage}/N/{name}']
                        mean_loss = float(g['full_loss'][:].mean(dtype=np.float64))
                        mean_acc = float(g['pair_acc'][:].mean(dtype=np.float64))
                        dl = abs(mean_loss-float(np.asarray(ref[name+'/loss'][idx]).mean(dtype=np.float64)))
                        da = abs(mean_acc-float(np.asarray(ref[name+'/in_context_acc'][idx]).mean(dtype=np.float64)))
                        bridge[f'{stage}/{name}'] = dict(loss_abs_diff=dl, pair_acc_abs_diff=da,
                          passed_original_gate=bool(dl<=SETTINGS['audit']['historical_mean_loss_tolerance'] and
                                                    da<=SETTINGS['audit']['historical_mean_accuracy_tolerance']))
    assert len(bridge) == 5 * len(SETTINGS['eval_sets']), ('Missing historical bridge groups', len(bridge))
    write_json(O/'cross_precision_drift.json', dict(groups=bridge,
             passed_original_gate=all(v['passed_original_gate'] for v in bridge.values()),
             original_loss_tolerance=SETTINGS['audit']['historical_mean_loss_tolerance'],
             original_acc_tolerance=SETTINGS['audit']['historical_mean_accuracy_tolerance'],
             same_precision_source_validation_required=True))
    # Check every fixed locator point against the original default-precision log.
    precision_bridge = {}
    with h5py.File(O/'precision_bridge.h5', 'w') as pf:
        for stage in sorted(set([selection['left_stage'], selection['peak_stage'], selection['right_stage']])):
            model = load_checkpoint(stage, template)
            before = digest_arrays(model)
            x, y = old['iwl_copy_avail']
            chunks = {}
            for offset in range(0, N, BS):
                xx, yy = jnp.asarray(x[offset:offset+BS]), jnp.asarray(y[offset:offset+BS])
                rn = native(model, xx, yy)
                m = metrics_numpy(rn, yy, rn, model)
                append_chunks(chunks, {k:m[k] for k in ['full_loss','pair_acc']})
            group = pf.create_group(str(stage))
            means = {}
            for metric, parts in chunks.items():
                value = np.concatenate(parts)
                group.create_dataset(metric, data=value)
                means[metric] = float(value.mean(dtype=np.float64))
            old_row = next(r for r in selection['historical_curve'] if r['stage'] == stage)
            dl, da = abs(means['full_loss']-old_row['full_loss']), abs(means['pair_acc']-old_row['pair_acc'])
            precision_bridge[str(stage)] = dict(count=N, means=means,
                    loss_abs_diff=dl, pair_acc_abs_diff=da,
                    passed_original_gate=bool(dl<=SETTINGS['audit']['historical_mean_loss_tolerance'] and
                                              da<=SETTINGS['audit']['historical_mean_accuracy_tolerance']))
            assert digest_arrays(model) == before
    write_json(O/'precision_bridge.json', dict(groups=precision_bridge,
                passed_original_gate=all(v['passed_original_gate'] for v in precision_bridge.values()),
                old_precision='None (historical)',new_precision='float32',frozen_times_unchanged=True,
                comparison_role='cross-precision diagnostic; original failures retained, not source identity gate'))
    write_json(PO/'precision_bridge.json', json.loads((O/'precision_bridge.json').read_text()))


def main_sweep(path_phase):
    output = PO if path_phase else O
    stages = selection['path_stages'] if path_phase else selection['window_stages']
    codes = SETTINGS['path_codes'] if path_phase else SETTINGS['window_codes']
    completed = 0
    with h5py.File(output/'per_sequence.h5', 'a') as f:
        if 'provenance' in f.attrs:
            assert json.loads(f.attrs['provenance']) == provenance
        else:
            f.attrs['provenance'] = json.dumps(provenance, sort_keys=True)
        for i, stage in enumerate(stages):
            budget_check()
            model = load_checkpoint(stage, template)
            before = digest_arrays(model)
            sets = ['iwl_copy_avail']
            if stage in selection['diagnostic_stages']:
                sets += ['train_eval', 'icl', 'flip_icl']
            for name in sets:
                missing = [c for c in codes if not has_group(f, 'confirmation', stage, c, name)]
                if not missing:
                    completed += len(codes)
                    continue
                chunks = {c: {} for c in missing}
                x, y = confirmation[name]
                max_equivalence = {}
                for offset in range(0, N, BS):
                    xx, yy = jnp.asarray(x[offset:offset+BS]), jnp.asarray(y[offset:offset+BS])
                    rn = native(model, xx, yy)
                    rs = suppress(model, xx, yy, rn, 'P1') if path_phase else None
                    final_logits = {}
                    for code in codes:
                        if path_phase:
                            r = path_result(model, xx, yy, rn, rs, code)
                        else:
                            r = rn if code == 'N' else suppress(model, xx, yy, rn, code)
                        if code in missing:
                            append_chunks(chunks[code], metrics_numpy(r, yy, rn, model))
                        if path_phase:
                            final_logits[code] = np.asarray(r['out'][:, -1])
                    if path_phase:
                        for a, b in [('KB', 'V'), ('VB', 'K'), ('KV', 'S')]:
                            delta = err(final_logits[a], final_logits[b])
                            assert delta <= 1e-4 and np.array_equal(final_logits[a].argmax(1), final_logits[b].argmax(1))
                            max_equivalence[a+'='+b] = max(max_equivalence.get(a+'='+b, 0.), delta)
                for code in missing:
                    row = save_group(f, 'confirmation', stage, code, name, chunks[code])
                    print('GROUP', args.phase, stage, name, code,
                          'loss', round(row['full_loss'], 6), 'pair', round(row['pair_acc'], 6), flush=True)
                if path_phase:
                    audit_path = output/'full_row_path_equivalence.json'
                    existing = json.loads(audit_path.read_text()) if audit_path.exists() else {}
                    existing[f'{stage}/{name}'] = dict(count=N, max_logit_error=max_equivalence, passed=True)
                    write_json(audit_path, existing)
                completed += len(codes)
            assert digest_arrays(model) == before, ('Model mutated', stage)
            progress = dict(phase=args.phase, checkpoints_finished=i+1, checkpoints_total=len(stages),
                    last_stage=stage, completed_groups=completed, elapsed_seconds=time.perf_counter()-began,
                    projected_phase_seconds=(time.perf_counter()-began)*len(stages)/(i+1),
                    no_training=True, state='running')
            write_json(output/'progress.json', progress)
            print('PROGRESS', json.dumps(progress), flush=True)
    write_json(output/'phase_finished.json', dict(phase=args.phase, checkpoints=len(stages),
                    groups=completed, elapsed_seconds=time.perf_counter()-began, finished=True, no_training=True))


if args.phase == 'preflight':
    main_preflight()
else:
    main_sweep(args.phase == 'path')
after = verify_inputs(selection['window_stages'])
assert after == inputs
for output in [O, PO]:
    write_json(output/'input_after.json', dict(passed=True, identity=after))
print('PHASE_COMPLETED', args.phase, 'seconds', time.perf_counter()-began, flush=True)

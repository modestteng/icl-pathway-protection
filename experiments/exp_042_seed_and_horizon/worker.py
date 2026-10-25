"""One static head or one training trajectory per CUDA process. No self-dispatch."""
from common import *
import science as R
import engine
import argparse
import math
import signal
import traceback
from datetime import datetime, timezone

STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def now():
    return datetime.now(timezone.utc).isoformat()


def state_record(ck, path, evaluated_stage):
    return dict(iter=int(ck['iter']), checkpoint_path=str(Path(path).relative_to(OUT)),
        checkpoint_sha256=sha(path), evaluated_stage=evaluated_stage,
        protection_offset=protection_offset(int(ck['iter'])) if int(ck['iter']) >= SETTINGS['start_stage'] else None,
        model_sha256=state_digest(ck['model']), adam_sha256=state_digest(ck['opt_state']),
        adam_count=int(np.asarray(ck['opt_state'][0].count)),
        prng={k: np.asarray(v).tolist() for k, v in ck['seeds'].items()},
        code_manifest_sha256=code_identity())


def restore(record, seed):
    assert record['code_manifest_sha256'] == code_identity()
    path = OUT / record['checkpoint_path']
    assert sha(path) == record['checkpoint_sha256']
    ck = load_checkpoint(path, seed)
    assert int(ck['iter']) == record['iter']
    assert state_digest(ck['model']) == record['model_sha256']
    assert state_digest(ck['opt_state']) == record['adam_sha256']
    assert {k: np.asarray(v).tolist() for k, v in ck['seeds'].items()} == record['prng']
    if record['protection_offset'] is not None:
        assert record['protection_offset'] == protection_offset(int(ck['iter']))
    return ck


def progress(job_dir, **values):
    write_json(job_dir / 'progress.json', dict(utc=now(), pid=os.getpid(), **values))


def finish_on_stop():
    if STOP or (OUT / 'STOP').exists():
        raise SystemExit(3)


def require_validation():
    for arm in SETTINGS['arms']:
        result = read_json(OUT / 'validation' / (arm + '.json'))
        assert result['passed'] and result['code_manifest_sha256'] == code_identity()


def chosen(seed):
    record = read_json(OUT / seed['id'] / 'path_selection.json')
    assert record['code_manifest_sha256'] == code_identity()
    assert record['checkpoint_sha256'] == sha(OUT / seed['id'] / 'prefix/00004400000.eqx')
    assert record['discovery_sha256'] == sha(DATA / 'mechanism_discovery.h5')
    return int(record['candidate_head'])


def validate_queues(seed, base, arm, head, destination, fork_stage):
    protect, _ = load_dataset('protection')
    cache = R.cache_protection(protect) if arm == 'P25' else []
    sample, (_, optimizer) = make_sampler(seed), template(seed)
    kind = 'A' if arm == 'A' else 'P'
    started = time.monotonic()
    serial = engine.serial_steps(base, 64, kind, head, sample, optimizer, protect, fork_stage)
    sync(serial)
    serial_seconds = time.monotonic() - started
    started = time.monotonic()
    queued, _ = engine.queue_steps(base, 64, kind, head, sample, optimizer, cache, fork_stage)
    queue_seconds = time.monotonic() - started
    mid, _ = engine.queue_steps(base, 32, kind, head, sample, optimizer, cache, fork_stage)
    path = destination.with_suffix('.eqx')
    save_checkpoint(path, mid)
    reloaded = load_checkpoint(path, seed)
    resumed, _ = engine.queue_steps(reloaded, 32, kind, head, sample, optimizer, cache, fork_stage)
    checks = dict(model_exact=exact(serial['model'], queued['model']),
        adam_exact=exact(serial['opt_state'], queued['opt_state']), prng_exact=exact(serial['seeds'], queued['seeds']),
        reload_exact=exact(mid, reloaded), resumed_model_exact=exact(serial['model'], resumed['model']),
        resumed_adam_exact=exact(serial['opt_state'], resumed['opt_state']),
        resumed_prng_exact=exact(serial['seeds'], resumed['seeds']),
        iter_exact=int(serial['iter']) == int(resumed['iter']) == int(queued['iter']))
    assert all(checks.values()), checks
    record = dict(passed=True, checks=checks, seed=seed, arm=arm, static_head=head,
                  fork_stage=fork_stage, start=int(base['iter']), code_manifest_sha256=code_identity(),
                  serial_seconds=serial_seconds, queued_seconds=queue_seconds,
                  queued_sequences_per_second=64 * BS / max(queue_seconds, 1e-12))
    write_json(destination, record)
    return record


def validate(arm):
    seed = SETTINGS['training_seeds'][0]
    base = initial_checkpoint(seed)
    # Warm real author moments without changing any production checkpoint.
    sample, (_, optimizer) = make_sampler(seed), template(seed)
    base = engine.serial_steps(base, 8, 'A', 0, sample, optimizer, {}, 0)
    return validate_queues(seed, base, arm, 0, OUT / 'validation' / (arm + '.json'), 0)


def prefix(seed, job_dir):
    require_validation()
    directory = OUT / seed['id'] / 'prefix'
    pointer = directory / 'resume.json'
    if pointer.exists():
        prior = read_json(pointer)
        ck = restore(prior, seed)
        evaluated = prior['evaluated_stage']
    else:
        ck, evaluated = initial_checkpoint(seed), -1
        save_checkpoint(directory / '00000000000.eqx', ck)
    sample, (_, optimizer) = make_sampler(seed), template(seed)
    monitor, mids = load_dataset('monitor')
    end, every = SETTINGS['start_stage'], SETTINGS['eval_every']
    while True:
        stage = int(ck['iter'])
        if stage % every == 0 and evaluated != stage:
            ck['seeds']['eval_model_seed'], _ = jax.random.split(ck['seeds']['eval_model_seed'])
            if stage % SETTINGS['monitor_every'] == 0 or stage == end:
                evaluate_stage(ck, seed, 'A_prefix', stage, monitor, mids,
                               directory / 'monitor' / f'{stage:011d}.npz')
            evaluated = stage
        if stage % every == 0:
            path = directory / f'{stage:011d}.eqx'
            save_checkpoint(path, ck)
            rec = state_record(ck, path, evaluated)
            write_json(pointer, rec)
            progress(job_dir, phase='prefix', seed=seed, iter=stage, end=end, checkpoint_sha256=rec['checkpoint_sha256'])
            if stage == end:
                _, expected = jax.random.split(jax.random.PRNGKey(load_opts(seed).eval_seed))
                for _ in range(end // every + 1):
                    expected, _ = jax.random.split(expected)
                assert np.array_equal(np.asarray(expected), np.asarray(ck['seeds']['eval_model_seed']))
                write_json(directory / 'complete.json', dict(passed=True, **rec, evaluation_key_splits=45))
                return rec
            finish_on_stop()
            # The rolling checkpoint is sufficient between scheduled snapshots.
            for older in directory.glob('*.eqx'):
                if older != path and older.name != '00000000000.eqx':
                    older.unlink()
        next_boundary = min(end, (stage // every + 1) * every)
        count = min(SETTINGS['queue_chunk'], (next_boundary - stage) // BS)
        ck, _ = engine.queue_steps(ck, count, 'A', 0, sample, optimizer, [], 0)


def discover(seed, head):
    require_validation()
    model = base_checkpoint(seed)['model']
    data, ids = load_dataset('mechanism_discovery')
    record, audit = R.mechanism_evaluate(model, data, ids, head)
    assert R.structure_passed(audit), 'discovery structure failed'
    path = OUT / seed['id'] / 'discovery' / f'head{head}.npz'
    save_npz(path, record)
    score = float(np.mean([np.mean(np.asarray(record[t + '/native/donor_ce'], np.float64)
        - np.asarray(record[t + '/path/donor_ce'], np.float64)) for t in SETTINGS['protection_tasks']]))
    result = dict(head=head, score=score, raw_sha256=sha(path), structure_passed=True,
                  code_manifest_sha256=code_identity())
    write_json(path.with_suffix('.json'), result)
    return result


def selection(seed):
    scores, identities = {}, {}
    for head in range(8):
        path = OUT / seed['id'] / 'discovery' / f'head{head}.json'
        value = read_json(path)
        assert value['code_manifest_sha256'] == code_identity()
        assert sha(path.with_suffix('.npz')) == value['raw_sha256']
        scores[str(head)], identities[str(head)] = value['score'], value['raw_sha256']
    selected = min(range(8), key=lambda h: (-scores[str(h)], h))
    result = dict(seed=seed, candidate_head=selected, discovery_scores=scores, raw_sha256=identities,
        checkpoint_sha256=sha(OUT / seed['id'] / 'prefix/00004400000.eqx'),
        discovery_sha256=sha(DATA / 'mechanism_discovery.h5'), selected_before_confirm=True,
        selected_before_protection=True, code_manifest_sha256=code_identity())
    path = OUT / seed['id'] / 'path_selection.json'
    if path.exists():
        assert read_json(path) == result
    else:
        write_json(path, result)
    return result


def confirmation_summary(record, ids, audit, head):
    checks, floors = {}, {}
    for task in SETTINGS['protection_tasks']:
        cls = ids[task][:, -1, 0]
        ce = np.asarray(record[task + '/native/donor_ce'], np.float64)
        donor = ce - np.asarray(record[task + '/donor/donor_ce'], np.float64)
        k = ce - np.asarray(record[task + '/path/donor_ce'], np.float64)
        v = ce - np.asarray(record[task + '/v_path/donor_ce'], np.float64)
        contrasts = [('donor_ce', donor, SETTINGS['mechanism_donor_ce_min']),
            ('path_ce', k, SETTINGS['mechanism_path_ce_min']),
            ('retention', k - SETTINGS['mechanism_retention_min'] * donor, 0.),
            ('k_vs_v', k - v, SETTINGS['mechanism_k_vs_v_min']),
            ('pair_margin', np.asarray(record[task + '/path/donor_margin'], np.float64)
                - np.asarray(record[task + '/native/donor_margin'], np.float64), SETTINGS['mechanism_pair_margin_min'])]
        for name, values, threshold in contrasts:
            checks[task + '/' + name] = R.cluster_test(values, cls, threshold)
        for field, floor in [('native/native_pair_acc', SETTINGS['mechanism_pair_acc_floor']),
                             ('donor/donor_pair_acc', SETTINGS['mechanism_pair_acc_floor']),
                             ('native/native_full_acc', SETTINGS['mechanism_full_acc_floor'])]:
            value = R.cluster_test(record[task + '/' + field], cls, floor)
            value['passed'] = bool(math.isfinite(value['se']) and value['se'] > 1e-15 and value['lower95'] > floor)
            floors[task + '/' + field] = value
        assert bool(record[task + '/pair_valid'].all())
    for name, p in zip(checks, R.holm([v['p'] for v in checks.values()])):
        checks[name].update(p_holm=p, passed=bool(p < SETTINGS['alpha'] and checks[name]['mean'] > checks[name]['threshold']))
    structural = R.structure_passed(audit)
    for task in SETTINGS['protection_tasks']:
        for pred in ['full_pred', 'pair_pred']:
            structural &= bool(np.array_equal(record[task + '/self/' + pred], record[task + '/native/' + pred]))
            structural &= bool(np.array_equal(record[task + '/query_label_control/' + pred], record[task + '/native/' + pred]))
    return dict(scientific_passed=bool(structural and all(v['passed'] for v in checks.values())
                and all(v['passed'] for v in floors.values())), structure_passed=bool(structural),
                checks=checks, floors=floors, candidate_head=head, gate_controls_execution=False)


def confirm(seed):
    head = chosen(seed)
    data, ids = load_dataset('mechanism_confirm')
    record, audit = R.mechanism_evaluate(base_checkpoint(seed)['model'], data, ids, head)
    path = OUT / seed['id'] / 'mechanism_confirm.npz'
    save_npz(path, record)
    result = confirmation_summary(record, ids, audit, head)
    result.update(code_manifest_sha256=code_identity(), raw_sha256=sha(path))
    write_json(path.with_suffix('.json'), result)
    assert result['structure_passed'], 'structural failure blocks training; statistical gate does not'
    return result


def preflight(seed):
    head, base = chosen(seed), base_checkpoint(seed)
    _, optimizer = template(seed)
    sample = make_sampler(seed)
    data, _ = load_dataset('protection')
    px, py = R.protect_batch(data, 0)
    _, dk = jax.random.split(base['seeds']['train_data_seed'])
    _, mk = jax.random.split(base['seeds']['train_model_seed'])
    batch = sync(sample(dk))
    assert exact(batch, sample.native(dk, sample.data))
    reference = sync(main.train_step(model=base['model'], fwd_fn=R.FWD, optimizer=optimizer,
        opt_state=base['opt_state'], microbs=BS, weight_decay=0., x=batch['examples'], y=batch['labels'], key=mk))
    result = sync(R.modified_step(base['model'], base['opt_state'], batch['examples'], batch['labels'],
                                 mk, px, py, 'P', 1., head, optimizer))
    assert len(result[4]) == 1 and exact(result[2], reference[2])
    assert maxdiff(result[1], reference[1]) <= SETTINGS['numeric_model_adam_max_abs']
    assert maxdiff(eqx.apply_updates(base['model'], result[3]), reference[1]) <= SETTINGS['numeric_model_adam_max_abs']
    projected, diag = R.project_tree(result[3], result[4], .25)
    vector, _ = R.project_numpy(R.flat(result[3]), np.stack([R.flat(g) for g in result[4]]), .25)
    error = float(np.max(np.abs(R.flat(projected) - vector)))
    relative = float(np.linalg.norm(R.flat(projected).astype(np.float64) - vector)
                     / max(np.linalg.norm(R.flat(result[3]).astype(np.float64)), 1e-30))
    assert error <= SETTINGS['numeric_projection_reference_max_abs']
    assert relative <= SETTINGS['numeric_projection_reference_max_relative']
    assert bool(np.asarray(diag['success']))
    # First count and nonzero-moment/zero-gradient bridges use official Optax.
    for gradient, state in [(result[4][0], optimizer.init(eqx.filter(base['model'], eqx.is_array))),
                            (jax.tree_util.tree_map(jnp.zeros_like, result[4][0]), base['opt_state'])]:
        raw_ref, new_state = sync(R.official_optimizer_update(gradient, state, optimizer))
        rebuilt = sync(R.adam_update_from_state(new_state))
        difference = R.flat(rebuilt).astype(np.float64) - R.flat(raw_ref).astype(np.float64)
        assert np.max(np.abs(difference)) <= SETTINGS['numeric_projection_reference_max_abs']
        assert np.linalg.norm(difference) / max(np.linalg.norm(R.flat(raw_ref).astype(np.float64)), 1e-30) <= SETTINGS['numeric_projection_reference_max_relative']
    xx, yy = px.reshape(-1, 3, 512), py.reshape(-1, 3)
    keys = jax.random.split(KEY, len(xx))
    _, grad = R.functional_objective(base['model'], xx, yy, keys, head)
    norm = float(np.asarray(R.tree_norm(grad)))
    assert math.isfinite(norm) and norm > 0
    step = SETTINGS['numeric_functional_grad_fd_step']
    plus = eqx.apply_updates(base['model'], jax.tree_util.tree_map(lambda v: step * v / norm, grad))
    minus = eqx.apply_updates(base['model'], jax.tree_util.tree_map(lambda v: -step * v / norm, grad))
    lp, _ = R.functional_objective(plus, xx, yy, keys, head)
    lm, _ = R.functional_objective(minus, xx, yy, keys, head)
    fd = float(np.asarray(lp - lm)) / (2 * step)
    assert abs(fd - norm) <= SETTINGS['numeric_functional_grad_absolute_tolerance'] + SETTINGS['numeric_functional_grad_relative_tolerance'] * norm
    mechanism_xy, mechanism_ids = load_dataset('mechanism_confirm')
    rows = {t: (x[:128], y[:128]) for t, (x, y) in mechanism_xy.items()}
    row_ids = {t: v[:128] for t, v in mechanism_ids.items()}
    mr, audit = R.mechanism_evaluate(base['model'], rows, row_ids, head)
    assert R.structure_passed(audit)
    for task in SETTINGS['protection_tasks']:
        f_loss, _ = R.functional_objective(base['model'], jnp.asarray(rows[task][0]),
            jnp.asarray(rows[task][1]), jax.random.split(KEY, len(rows[task][0])), head)
        assert abs(float(np.asarray(f_loss)) - float(mr[task + '/path/donor_ce'].mean())) <= SETTINGS['numeric_eval_mean_ce_max_abs']
    monitor, _ = load_dataset('monitor')
    xy = {t: (x[:128], y[:128]) for t, (x, y) in monitor.items() if t in SETTINGS['reported_tasks']}
    native = main.evaluate(model=base['model'], fwd_fn=R.FWD, key=KEY,
        eval_data={t: dict(examples=jnp.asarray(x), labels=jnp.asarray(y)) for t, (x, y) in xy.items()},
        eval_batch_size=SETTINGS['eval_bs'])
    adapted = R.evaluate(base['model'], xy)
    for task in xy:
        assert abs(float(np.asarray(native[task]['loss']).mean()) - float(adapted[task + '/loss'].mean())) <= SETTINGS['numeric_eval_mean_ce_max_abs']
        assert np.count_nonzero(np.asarray(native[task]['acc']) != adapted[task + '/acc']) <= SETTINGS['numeric_eval_acc_max_rows']
    record = dict(passed=True, head=head, projection_abs=error, projection_relative=relative,
        functional_gradient_norm=norm, functional_finite_difference=fd, author_evaluator_rows=128,
        code_manifest_sha256=code_identity())
    write_json(OUT / seed['id'] / 'preflight.json', record)
    return record


def evaluate_stage(ck, seed, arm, stage, xy, ids, destination):
    record = R.evaluate(ck['model'], {t: xy[t] for t in SETTINGS['reported_tasks']}, ids)
    record.update(stage=np.asarray(stage), seed_id=np.asarray(seed['id']), arm=np.asarray(arm),
                  code_manifest_sha256=np.asarray(code_identity()))
    save_npz(destination, record)
    return {t: dict(acc=float(record[t + '/acc'].mean()), ce=float(record[t + '/loss'].mean()))
            for t in SETTINGS['reported_tasks']}


def capture_update(ck, next_ck, seed, head, sample, optimizer, protection):
    _, dk = jax.random.split(ck['seeds']['train_data_seed'])
    _, mk = jax.random.split(ck['seeds']['train_model_seed'])
    batch = sample(dk)
    px, py = R.protect_batch(protection, protection_offset(int(ck['iter'])))
    _, model, state, raw, gradients = R.modified_step(ck['model'], ck['opt_state'], batch['examples'],
        batch['labels'], mk, px, py, 'P', .25, head, optimizer)
    assert exact(model, next_ck['model']) and exact(state, next_ck['opt_state'])
    applied = jax.tree_util.tree_map(lambda a, b: a - b, eqx.filter(model, eqx.is_array), eqx.filter(ck['model'], eqx.is_array))
    projected, _ = R.project_tree(raw, gradients, .25)
    path = OUT / seed['id'] / 'update_audit' / f'P25_{int(ck["iter"]):011d}.npz'
    save_npz(path, dict(raw=R.flat(raw), projected=R.flat(projected), applied=R.flat(applied),
        gradient=R.flat(gradients[0]), stage=np.asarray(ck['iter']), head=np.asarray(head)))
    monitor, _ = load_dataset('monitor')
    R.audit_update(seed, 'P25', int(ck['iter']), ck['model'], model, raw, gradients, px, py, monitor, head, 'P')


def train(seed, arm, budget, job_dir):
    require_validation()
    pre = read_json(OUT / seed['id'] / 'preflight.json')
    assert pre['passed'] and pre['code_manifest_sha256'] == code_identity()
    assert budget in (10000000, 40000000)
    if budget == 40000000:
        assert seed['id'] in SETTINGS['long_seeds']
        assert read_json(OUT / 'short_complete.json')['code_manifest_sha256'] == code_identity()
    head = chosen(seed)
    directory = OUT / seed['id'] / arm
    pointer = directory / 'resume.json'
    if pointer.exists():
        prior = read_json(pointer)
        ck, evaluated = restore(prior, seed), prior['evaluated_stage']
    else:
        assert budget == 10000000
        ck, evaluated = base_checkpoint(seed), -1
    start_record = state_record(ck, OUT / (read_json(pointer)['checkpoint_path'] if pointer.exists()
        else f'{seed["id"]}/prefix/00004400000.eqx'), evaluated)
    begin = int(ck['iter'])
    assert SETTINGS['start_stage'] <= begin <= budget, 'resume state is outside this budget'
    if budget == 40000000:
        assert begin >= 10000000
        boundary = read_json(directory / 'complete_10000000.json')
        if begin == 10000000:
            assert start_record['checkpoint_sha256'] == boundary['checkpoint_sha256']
            assert evaluated == 10000000
            origin_path = directory / 'long_origin.json'
            origin = dict(source_10M_sha256=boundary['checkpoint_sha256'], state=start_record,
                          code_manifest_sha256=code_identity())
            if origin_path.exists():
                assert read_json(origin_path) == origin
            else:
                write_json(origin_path, origin)
            validate_queues(seed, ck, arm, head, OUT / seed['id'] / 'resume_acceptance' / (arm + '.json'), SETTINGS['start_stage'])
        else:
            assert read_json(directory / 'long_origin.json')['source_10M_sha256'] == boundary['checkpoint_sha256']
    monitor, mids = load_dataset('monitor')
    confirm_xy, cids = load_dataset('confirm')
    protection, _ = load_dataset('protection')
    cache = R.cache_protection(protection) if arm == 'P25' else []
    sample, (_, optimizer) = make_sampler(seed), template(seed)
    every = SETTINGS['eval_every']
    confirms = set(v for b in budgets(seed['id']) for v in window(b))
    audit_offsets = set(SETTINGS['diagnostic_update_offsets'])
    telemetry = np.zeros(5, np.float64)
    started = time.monotonic()
    while True:
        stage = int(ck['iter'])
        if stage % every == 0 and evaluated != stage:
            ck['seeds']['eval_model_seed'], _ = jax.random.split(ck['seeds']['eval_model_seed'])
            if stage % SETTINGS['monitor_every'] == 0 or stage == SETTINGS['start_stage']:
                evaluate_stage(ck, seed, arm, stage, monitor, mids, directory / 'monitor' / f'{stage:011d}.npz')
            if stage in confirms:
                evaluate_stage(ck, seed, arm, stage, confirm_xy, cids, directory / 'confirm' / f'{stage:011d}.npz')
            evaluated = stage
        if stage % every == 0:
            retain = (stage % SETTINGS['checkpoint_every'] == 0 or stage in confirms
                      or stage in [SETTINGS['start_stage'], 10000000, budget])
            path = directory / 'checkpoints' / (f'{stage:011d}.eqx' if retain else 'rolling.eqx')
            save_checkpoint(path, ck)
            rec = state_record(ck, path, evaluated)
            write_json(pointer, rec)
            write_json(directory / 'states' / f'{stage:011d}.json', rec)
            progress(job_dir, phase='train', seed=seed, arm=arm, iter=stage, end=budget,
                     protection_offset=rec['protection_offset'], checkpoint_sha256=rec['checkpoint_sha256'])
            write_json(directory / 'training_progress.json', dict(iter=stage, budget=budget,
                launch_start=begin, elapsed_seconds=time.monotonic() - started,
                steps=int(telemetry[0]), mean_update_norm_ratio=float(telemetry[1] / max(1, telemetry[0])),
                mean_correction_norm_ratio=float(telemetry[2] / max(1, telemetry[0])),
                projected_steps=int(telemetry[3]), max_relative_violation=float(telemetry[4])))
            if stage == budget:
                result = dict(passed=True, **rec, seed=seed, arm=arm, budget=budget,
                    launch_start=begin, launch_state=start_record, protected_head=head,
                    fork_stage=SETTINGS['start_stage'], elapsed_seconds=time.monotonic() - started)
                write_json(directory / f'complete_{budget}.json', result)
                return result
            finish_on_stop()
        next_boundary = min(budget, (stage // every + 1) * every)
        offset = protection_offset(stage)
        count = min(SETTINGS['queue_chunk'], (next_boundary - stage) // BS)
        upcoming = [v for v in audit_offsets if offset < v < offset + count]
        if upcoming:
            count = min(upcoming) - offset
        audit = arm == 'P25' and offset in audit_offsets
        if audit:
            count = 1
        old = ck
        ck, status = engine.queue_steps(ck, count, 'A' if arm == 'A' else 'P', head,
                                       sample, optimizer, cache, SETTINGS['start_stage'])
        telemetry += [count, status[:, 3].sum(), status[:, 4].sum(), status[:, 5].sum(), 0]
        telemetry[4] = max(telemetry[4], float(status[:, 2].max()))
        if audit:
            capture_update(old, ck, seed, head, sample, optimizer, protection)


def diagnostic(seed, arm, budget):
    head = chosen(seed)
    receipt = read_json(OUT / seed['id'] / arm / f'complete_{budget}.json')
    ck = restore(receipt, seed)
    data, ids = load_dataset('mechanism_confirm')
    record, audit = R.mechanism_evaluate(ck['model'], data, ids, head)
    assert R.structure_passed(audit)
    path = OUT / seed['id'] / arm / 'mechanism_late' / f'{budget:011d}.npz'
    save_npz(path, record)
    result = confirmation_summary(record, ids, audit, head)
    result.update(raw_sha256=sha(path), checkpoint_sha256=receipt['checkpoint_sha256'],
                  code_manifest_sha256=code_identity())
    write_json(path.with_suffix('.json'), result)
    return result


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    job = next(v for v in graph() if v['name'] == args.job)
    directory = OUT / 'jobs' / job['name']
    directory.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()
    try:
        environment = verify_inputs(require_gpu=job['kind'] not in ['data', 'select', 'short_complete'])
        write_json(directory / 'input_environment.json', environment)
        if job['kind'] == 'data':
            import data_identity
            result = data_identity.prepare()
        elif job['kind'] == 'validate':
            result = validate(job['arm'])
        elif job['kind'] == 'short_complete':
            result = dict(passed=True, training_seeds=SETTINGS['training_seeds'], long_seeds=SETTINGS['long_seeds'],
                          code_manifest_sha256=code_identity())
            write_json(OUT / 'short_complete.json', result)
        else:
            seed = seed_spec(job['seed'])
            if job['kind'] == 'prefix':
                result = prefix(seed, directory)
            elif job['kind'] == 'discover':
                result = discover(seed, job['head'])
            elif job['kind'] == 'select':
                result = selection(seed)
            elif job['kind'] == 'confirm':
                result = confirm(seed)
            elif job['kind'] == 'preflight':
                result = preflight(seed)
            elif job['kind'] == 'train':
                result = train(seed, job['arm'], job['budget'], directory)
            elif job['kind'] == 'diagnostic':
                result = diagnostic(seed, job['arm'], job['budget'])
            else:
                raise ValueError(job)
        write_json(directory / 'complete.json', dict(passed=True, job=job, result=result,
            code_manifest_sha256=code_identity(), elapsed_seconds=time.monotonic() - started, utc=now()))
    except SystemExit as exc:
        write_json(directory / 'stopped.json', dict(exit_code=exc.code, utc=now(), checkpoint_preserved=True))
        raise
    except Exception:
        write_json(directory / 'failure.json', dict(traceback=traceback.format_exc(), utc=now(), automatic_retry=False))
        raise


if __name__ == '__main__':
    main_cli()

"""Exp040 GPU job runner. One job per process; resumable at every 100k boundary.

Jobs (dependencies enforced by scheduler.py and re-checked here):
  data                       fresh five-purpose inputs (architecture independent)
  validate   --arch          implementation/numerical checks on the CUDA host
  normal     --arch --seed   author run A from 0; monitor every 100k; timing lock;
                             A confirm evaluations in the locked window
  mechanism  --arch --seed   discovery at t0, path-set selection, random set,
                             confirmation, A-fork exactness, seed preflight,
                             shared start diagnostics and A end diagnostics
  arm        --arch --seed --arm   protected continuation t0 -> end
"""
from common import *
import science as R
import engine
import path_set as PS
import timing as T
import argparse
import signal
import traceback
from datetime import datetime, timezone

STOP = False
STARTED = time.monotonic()
JOB_DIR = None
LAST_HEARTBEAT = 0.0


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def stop_signal(signum, frame):
    global STOP
    STOP = True


def stop_reason():
    if STOP:
        return 'signal_requested'
    if (OUTPUTS / 'STOP').exists():
        return 'stop_marker'
    return None


def heartbeat(**values):
    global LAST_HEARTBEAT
    if JOB_DIR is not None:
        write_json(JOB_DIR / 'progress.json', dict(pid=os.getpid(), utc=utc_now(), arch=ARCH,
                   elapsed_seconds=time.monotonic() - STARTED, **values))
    LAST_HEARTBEAT = time.monotonic()


def require_running(phase):
    reason = stop_reason()
    if reason:
        write_json(JOB_DIR / 'stopped.json', dict(reason=reason, phase=phase, pid=os.getpid(),
                   utc=utc_now(), checkpoint_retained=True))
        raise SystemExit(3)


def staged(name):
    data, ids = load_dataset(name)
    return R.stage_on_gpu({t: data[t] for t in TASKS}), {t: ids[t] for t in TASKS}


def write_eval(path, model, xy, ids=None, **meta):
    record = R.evaluate(model, xy, ids)
    record.update({k: np.asarray(v) for k, v in meta.items()})
    save_npz(path, record)
    return R.summary_of(record)


def validation_seed():
    return seed_spec(SETTINGS['training_seeds'][0]['id'])


def test_set():
    return (0, 9) if ARCHITECTURE['depth'] == 4 else (0, 1)


# ------------------------------------------------------------------ validate
def validate():
    p = verify_inputs()
    seed = validation_seed()
    base = initial_checkpoint(seed)
    sync(base)
    protect, _ = load_dataset('protection')
    import mechanism_checks
    import set_checks
    import copy

    def factory(width, depth):
        opts = copy.deepcopy(load_opts(seed))
        opts.d_model, opts.depth = int(width), int(depth)
        return main_utils.get_model_from_opts(opts, (512,))
    width, depth = int(ARCHITECTURE['d_model']), int(ARCHITECTURE['depth'])
    architectures = {(64, 2): ((64, 2),), (128, 2): ((64, 2), (128, 2)), (128, 4): ((128, 4),)}[(width, depth)]
    heartbeat(phase='validate_mechanism_checks')
    single = mechanism_checks.run_checks(factory, protect['icl'][0][0], protect['icl'][1][0],
                                         architectures=architectures, include_gradient=True)
    assert single['passed']
    write_json(OUT / 'validation' / 'mechanism_single_path.json', single)
    heartbeat(phase='validate_set_checks')
    sets = set_checks.run_set_checks(base['model'], protect['icl'][0][0], protect['icl'][1][0], SETTINGS)
    assert sets['passed']
    write_json(OUT / 'validation' / 'mechanism_path_set.json', sets)
    heartbeat(phase='validate_queue')
    sample, (_, optimizer) = make_sampler(seed), template(seed)
    cache = R.cache_protection(protect)
    results = {}
    for kind, rho, ids in [('A', 1.0, ()), ('C', .75, ()), ('P', .25, test_set())]:
        started = time.monotonic()
        serial, ss = engine.serial_steps(base, 64, kind, rho, ids, sample, optimizer, protect, 0)
        serial_seconds = time.monotonic() - started
        started = time.monotonic()
        queued, qs, _ = engine.queue_steps(base, 64, kind, rho, ids, sample, optimizer, cache, 0)
        queue_seconds = time.monotonic() - started
        mid, _, _ = engine.queue_steps(base, 32, kind, rho, ids, sample, optimizer, cache, 0)
        path = OUT / 'validation' / (kind + '_32steps.eqx')
        digest = save_checkpoint(path, mid)
        reloaded = load_checkpoint(path, seed)
        resumed, _, _ = engine.queue_steps(reloaded, 32, kind, rho, ids, sample, optimizer, cache, 0)
        same = dict(model_exact=exact(serial['model'], queued['model']),
                    adam_exact=exact(serial['opt_state'], queued['opt_state']),
                    prng_exact=exact(serial['seeds'], queued['seeds']),
                    status_exact=bool(np.array_equal(ss, qs)),
                    reload_exact=exact(mid, reloaded),
                    resumed_model_exact=exact(serial['model'], resumed['model']),
                    resumed_adam_exact=exact(serial['opt_state'], resumed['opt_state']),
                    resumed_prng_exact=exact(serial['seeds'], resumed['seeds']))
        assert all(same.values()), ('queue/reload changed scientific state', kind, same)
        results[kind] = dict(passed=True, rho=rho, path_set=list(ids), **same,
                             serial_seconds=serial_seconds, queued_seconds=queue_seconds,
                             queued_sequences_per_second=64 * BS / max(queue_seconds, 1e-12),
                             checkpoint_sha256=digest)
        heartbeat(phase='validate_queue', kind=kind, result=results[kind])
    write_json(OUT / 'validation.json', dict(passed=True, architecture_id=ARCH, architecture=ARCHITECTURE,
        inputs=p['inputs'], versions=p['versions'], device_kind=p['device_kind'],
        code_manifest_sha256=code_identity(), queued_serial_and_resume=results,
        single_path_receipt_sha256=sha(OUT / 'validation' / 'mechanism_single_path.json'),
        path_set_receipt_sha256=sha(OUT / 'validation' / 'mechanism_path_set.json'),
        elapsed_seconds=time.monotonic() - STARTED))
    print('VALIDATION_PASSED', ARCH, json.dumps(results), flush=True)


def require_validation():
    check = read_json(OUT / 'validation.json')
    assert check['passed'] and check['code_manifest_sha256'] == code_identity()


# -------------------------------------------------------------------- normal
def monitor_curve(directory):
    path = directory / 'curve.json'
    return read_json(path) if path.exists() else dict(points=[])


def normal(seed):
    """Author run A; every 100k: checkpoint + monitor evaluation; then lock timing."""
    require_validation()
    sd = OUT / seed['id']
    directory = normal_dir(seed)
    cfg = SETTINGS['timing']
    pointer = directory / 'resume.json'
    if (sd / 'timing.json').exists() and (directory / 'complete.json').exists():
        print('NORMAL_ALREADY_COMPLETE', seed['id'], flush=True)
        return
    if pointer.exists():
        receipt = read_json(pointer)
        assert receipt['code_manifest_sha256'] == code_identity() and receipt['seed'] == seed
        assert sha(OUTPUTS / receipt['checkpoint_path']) == receipt['checkpoint_sha256']
        ck = load_checkpoint(OUTPUTS / receipt['checkpoint_path'], seed)
        last_eval = int(receipt['last_evaluated_iter'])
        target = int(receipt['target_end'])
    else:
        ck = initial_checkpoint(seed)
        last_eval, target = -1, cfg['first_horizon']
    curve = monitor_curve(directory)
    monitor, mids = staged('monitor')
    sample, (_, optimizer) = make_sampler(seed), template(seed)

    def save_resume():
        path = normal_checkpoint_path(seed, int(ck['iter']))
        digest = sha(path) if path.exists() else save_checkpoint(path, ck)
        write_json(pointer, dict(seed=seed, iter=int(ck['iter']), last_evaluated_iter=last_eval,
                   target_end=target, checkpoint_path=rel(path), checkpoint_sha256=digest,
                   code_manifest_sha256=code_identity()))

    while True:
        stage = int(ck['iter'])
        if stage % EVERY == 0 and last_eval != stage:
            ck['seeds']['eval_model_seed'], _ = jax.random.split(ck['seeds']['eval_model_seed'])
            path = normal_checkpoint_path(seed, stage)
            digest = save_checkpoint(path, ck)
            meta = dict(stage=stage, seed_id=seed['id'], arm='A', checkpoint_sha256=digest,
                        code_manifest_sha256=code_identity())
            summary = write_eval(directory / 'monitor' / f'{stage:011d}.npz', ck['model'], monitor, mids, **meta)
            curve['points'] = sorted([v for v in curve['points'] if v['stage'] != stage]
                                     + [dict(stage=stage, checkpoint_sha256=digest, monitor=summary)],
                                     key=lambda v: v['stage'])
            write_json(directory / 'curve.json', curve)
            last_eval = stage
            assert int(np.asarray(ck['opt_state'][0].count)) == stage // BS
            if stage == target:
                accuracy = {v['stage']: v['monitor']['icl']['acc'] for v in curve['points']}
                lock = T.decide(accuracy, cfg)
                need = T.required_end(stage, cfg, lock)
                if need is not None and need > target:
                    target = int(need)
                    write_json(directory / 'extensions' / f'{stage:011d}.json', dict(
                        from_end=stage, to_end=target, reason=lock['reason'] if not lock['defined'] else 'window_end',
                        lock_preview=lock, code_manifest_sha256=code_identity()))
            save_resume()
            heartbeat(phase='normal', seed=seed, iter=stage, target=target)
            print('NORMAL_PROGRESS', ARCH, seed['id'], stage, target, flush=True)
        if stop_reason():
            save_resume()
            require_running('normal_after_checkpoint')
        if stage >= target:
            break
        next_boundary = (stage // EVERY + 1) * EVERY
        count = min(int(SETTINGS['queue_chunk']), (next_boundary - stage) // BS)
        ck, _, _ = engine.queue_steps(ck, count, 'A', 1.0, (), sample, optimizer, [], 0)
        if time.monotonic() - LAST_HEARTBEAT > 30:
            heartbeat(phase='normal', seed=seed, iter=int(ck['iter']), target=target)
    accuracy = {v['stage']: v['monitor']['icl']['acc'] for v in curve['points']}
    lock = T.decide(accuracy, cfg)
    app = T.applicability(accuracy, lock, cfg)
    assert app['applicable'] is not None, ('window not trained', app)
    record = dict(seed=seed, architecture_id=ARCH, applicable=bool(app['applicable']),
                  applicability=app, rule=cfg['rule'], timing_settings=cfg, lock=lock,
                  curve_sha256=sha(directory / 'curve.json'), normal_end=int(ck['iter']),
                  selected_without_protected_results=True, code_manifest_sha256=code_identity(),
                  utc=utc_now())
    if app['applicable']:
        record.update(start_stage=int(lock['start_stage']), end_stage=int(lock['end_stage']),
                      window_stages=lock['window_stages'],
                      start_checkpoint_sha256=sha(normal_checkpoint_path(seed, lock['start_stage'])))
    if (sd / 'timing.json').exists():
        prior = read_json(sd / 'timing.json')
        for key in ('applicable', 'lock', 'curve_sha256', 'start_checkpoint_sha256'):
            assert prior.get(key) == record.get(key), 'timing lock would change on resume: ' + key
    else:
        write_json(sd / 'timing.json', record)
    print('TIMING_LOCK', ARCH, seed['id'], json.dumps({k: record.get(k) for k in ('applicable', 'start_stage', 'end_stage')}), flush=True)
    if app['applicable']:
        confirm, cids = staged('confirm')
        for stage in lock['window_stages']:
            path = sd / 'A' / 'confirm' / f'A_{stage:011d}.npz'
            if path.exists():
                continue
            cpath = normal_checkpoint_path(seed, stage)
            model = load_checkpoint(cpath, seed)['model']
            write_eval(path, model, confirm, cids, stage=stage, seed_id=seed['id'], arm='A',
                       checkpoint_sha256=sha(cpath), code_manifest_sha256=code_identity())
            heartbeat(phase='A_confirm', seed=seed, stage=stage)
    write_json(directory / 'complete.json', dict(passed=True, seed=seed, end=int(ck['iter']),
               timing_sha256=sha(sd / 'timing.json'), code_manifest_sha256=code_identity(),
               elapsed_seconds=time.monotonic() - STARTED))


# ----------------------------------------------------------------- mechanism
def late_record(model, path_ids, label, stage, sd, seed):
    """Same confirm inputs at start/end for one model and one path set."""
    data, ids = load_dataset('confirm')
    tag = '-'.join(map(str, path_ids))
    path = sd / 'mechanism_late' / f'{label}_{stage:011d}_set{tag}.npz'
    marker = path.with_suffix('.json')
    if marker.exists():
        info = read_json(marker)
        assert info['code_manifest_sha256'] == code_identity() and sha(path) == info['sha256']
        return info
    raw, audit = R.mechanism_evaluate(model, data, ids, path_ids)
    assert R.structure_passed(audit), 'late mechanism structure failed'
    summary = R.mechanism_summary(raw, ids, audit, path_ids)
    raw.update(stage=np.asarray(stage), seed_id=np.asarray(seed['id']), arm=np.asarray(label))
    save_npz(path, raw)
    gains = {t: dict(path_gain=float(R.gain(raw, t, 'path').mean()), donor_gain=float(R.gain(raw, t, 'donor').mean()))
             for t in SETTINGS['protection_tasks']}
    info = dict(label=label, stage=stage, path_set=list(path_ids), path=rel(path), sha256=sha(path),
                structure_passed=True, gains=gains, coverage=summary['coverage'],
                gate_passed=summary['passed'], code_manifest_sha256=code_identity())
    write_json(marker, info)
    return info


def discover(seed, model, sd):
    data, ids = load_dataset('mechanism_discovery')
    candidates = list(M_candidates(model))
    scores, raw_sha = {}, {}
    for p in candidates:
        path = sd / 'discovery' / f'single{p}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as rec:
                result = {k: rec[k] for k in rec.files}
        else:
            result, audit = R.mechanism_evaluate(model, data, ids, (p,), audit_logits=False)
            if not R.structure_passed(audit):
                raise RuntimeError('Discovery structure failed for path ' + str(p))
            save_npz(path, result)
        scores[str(p)] = float(np.mean([R.gain(result, t, 'path').mean() for t in SETTINGS['protection_tasks']]))
        raw_sha[str(p)] = sha(path)
        heartbeat(phase='discovery_single', seed=seed, path=p)
    ranking = PS.rank_paths(scores)
    cfg = SETTINGS['path_set']
    prefixes, coverages, prefix_sha = [], [], {}
    for k in range(1, min(cfg['k_max'], len(ranking)) + 1):
        ids_k = tuple(sorted(ranking[:k]))
        path = sd / 'discovery' / f'prefix{k}.npz'
        if k == 1:
            path = sd / 'discovery' / f'single{ids_k[0]}.npz'
            with np.load(path, allow_pickle=False) as rec:
                result = {kk: rec[kk] for kk in rec.files}
        elif path.exists():
            with np.load(path, allow_pickle=False) as rec:
                result = {kk: rec[kk] for kk in rec.files}
        else:
            result, audit = R.mechanism_evaluate(model, data, ids, ids_k, audit_logits=False)
            if not R.structure_passed(audit):
                raise RuntimeError('Discovery structure failed for prefix ' + str(ids_k))
            save_npz(path, result)
        cov, by_task = R.coverage(result)
        prefixes.append(list(ids_k))
        coverages.append(cov)
        prefix_sha[str(k)] = sha(path)
        heartbeat(phase='discovery_prefix', seed=seed, k=k, coverage=cov)
    k_star, reason = PS.choose_prefix(coverages, cfg['coverage_target'])
    selected = tuple(prefixes[k_star - 1])
    random_set = PS.random_control_set(seed['init'], seed['train'], selected, SETTINGS['random_path_seed'])
    return dict(seed=seed, candidate_domain=candidates, discovery_scores=scores, discovery_raw_sha256=raw_sha,
                ranking=ranking, prefixes=prefixes, prefix_coverage=coverages, prefix_raw_sha256=prefix_sha,
                k=k_star, selection_reason=reason, selected=list(selected),
                selected_description=PS.describe(selected), random_set=list(random_set),
                random_description=PS.describe(random_set), rules=cfg,
                discovery_sha256=sha(DATA / 'mechanism_discovery.h5'))


def M_candidates(model):
    import mechanism as M
    return M.candidate_path_ids(model)


def a_fork_exactness(seed, base, lock, sd):
    """A continued from the fork through the arm machinery must equal the normal run."""
    receipt_path = sd / 'a_fork_exactness.json'
    if receipt_path.exists():
        return read_json(receipt_path)
    sample, (_, optimizer) = make_sampler(seed), template(seed)
    ck = dict(base, seeds=dict(base['seeds']))
    ck['seeds']['eval_model_seed'], _ = jax.random.split(ck['seeds']['eval_model_seed'])
    stage = lock['start_stage'] + EVERY
    while int(ck['iter']) < stage:
        count = min(int(SETTINGS['queue_chunk']), (stage - int(ck['iter'])) // BS)
        ck, _, _ = engine.queue_steps(ck, count, 'A', 1.0, (), sample, optimizer, [], lock['start_stage'])
    assert int(ck['iter']) == stage
    reference = load_checkpoint(normal_checkpoint_path(seed, stage), seed)
    result = dict(stage=stage, model_exact=exact(ck['model'], reference['model']),
                  adam_exact=exact(ck['opt_state'], reference['opt_state']),
                  train_prng_exact=exact((ck['seeds']['train_data_seed'], ck['seeds']['train_model_seed']),
                                         (reference['seeds']['train_data_seed'], reference['seeds']['train_model_seed'])),
                  model_max_abs=maxdiff(ck['model'], reference['model']),
                  adam_max_abs=maxdiff(ck['opt_state'], reference['opt_state']),
                  code_manifest_sha256=code_identity())
    result['passed'] = bool(result['train_prng_exact'] and result['model_max_abs'] <= SETTINGS['numeric_model_adam_max_abs']
                            and result['adam_max_abs'] <= SETTINGS['numeric_model_adam_max_abs'])
    write_json(receipt_path, result)
    assert result['passed'], ('A fork differs from the normal run', result)
    return result


def mechanism(seed):
    require_validation()
    sd = OUT / seed['id']
    done = sd / 'mechanism_complete.json'
    if done.exists():
        print('MECHANISM_ALREADY_COMPLETE', seed['id'], flush=True)
        return
    base, lock = fork_checkpoint(seed)
    model = base['model']
    selection_path = sd / 'path_selection.json'
    if selection_path.exists():
        selection = read_json(selection_path)
        assert selection['code_manifest_sha256'] == code_identity()
    else:
        selection = discover(seed, model, sd)
        selection.update(start_stage=lock['start_stage'], start_checkpoint_sha256=lock['start_checkpoint_sha256'],
                         code_manifest_sha256=code_identity(), selected_before_confirm=True,
                         selected_before_any_protected_update=True, utc=utc_now())
        write_json(selection_path, selection)
    selected, random_set = tuple(selection['selected']), tuple(selection['random_set'])
    print('PATH_SET', ARCH, seed['id'], selected, random_set, flush=True)
    data, ids = load_dataset('mechanism_confirm')
    confirmations = {}
    for role, path_ids in (('selected', selected), ('random', random_set)):
        out = sd / f'mechanism_confirmation_{role}.json'
        if not out.exists():
            record, audit = R.mechanism_evaluate(model, data, ids, path_ids)
            save_npz(sd / f'mechanism_confirm_{role}.npz', record)
            both = R.mechanism_summary(record, ids, audit, path_ids)
            icl_only = R.mechanism_summary(record, ids, audit, path_ids, tasks=('icl',))
            write_json(out, dict(role=role, seed=seed, gate_both_settings=both,
                                 gate_standard_icl_only=dict(passed=icl_only['passed'], checks=icl_only['checks'],
                                                            floors=icl_only['floors']),
                                 raw_sha256=sha(sd / f'mechanism_confirm_{role}.npz'),
                                 start_stage=lock['start_stage'], code_manifest_sha256=code_identity()))
        confirmations[role] = read_json(out)
        heartbeat(phase='confirmation', seed=seed, role=role)
    exactness = a_fork_exactness(seed, base, lock, sd)
    preflight = sd / 'preflight.json'
    if not preflight.exists():
        R.seed_preflight(seed, base, selected, random_set, preflight)
    late = {}
    for role, path_ids in (('selected', selected), ('random', random_set)):
        late['start_' + role] = late_record(model, path_ids, 'START', lock['start_stage'], sd, seed)
        end_model = load_checkpoint(normal_checkpoint_path(seed, lock['end_stage']), seed)['model']
        late['A_end_' + role] = late_record(end_model, path_ids, 'A', lock['end_stage'], sd, seed)
    write_json(done, dict(passed=True, seed=seed, selected=list(selected), random_set=list(random_set),
        k=selection['k'], selection_reason=selection['selection_reason'],
        gate_passed=confirmations['selected']['gate_both_settings']['passed'],
        gate_standard_icl_only_passed=confirmations['selected']['gate_standard_icl_only']['passed'],
        confirm_coverage=confirmations['selected']['gate_both_settings']['coverage'],
        a_fork_exactness=exactness, preflight_sha256=sha(preflight), late=late,
        code_manifest_sha256=code_identity(), elapsed_seconds=time.monotonic() - STARTED))
    print('MECHANISM_COMPLETE', ARCH, seed['id'], flush=True)


# ---------------------------------------------------------------------- arms
def retained(stage, lock):
    return stage % 500000 == 0 or stage in lock['window_stages'] or stage in (lock['start_stage'], lock['end_stage'])


def train_arm(seed, arm):
    require_validation()
    sd = OUT / seed['id']
    mech = read_json(sd / 'mechanism_complete.json')
    assert mech['passed'] and mech['code_manifest_sha256'] == code_identity()
    complete = sd / arm / 'complete.json'
    if complete.exists():
        print('ARM_ALREADY_COMPLETE', seed['id'], arm, flush=True)
        return
    definition = SETTINGS['arm_definitions'][arm]
    kind, rho = definition['kind'], float(definition['rho'])
    path_ids = {'selected': tuple(mech['selected']), 'random': tuple(mech['random_set']), 'none': ()}[definition['path_set']]
    base, lock = fork_checkpoint(seed)
    start, end = int(lock['start_stage']), int(lock['end_stage'])
    ad = sd / arm
    pointer = ad / 'resume.json'
    stats = dict(steps=0, update_norm_ratio=0., correction_norm_ratio=0., conflicts=0., near_zero_steps=0,
                 max_projection_relative_violation=0., constraint_count=2 if kind == 'C' else 1)
    last_eval = -1
    previous = [None]
    if pointer.exists():
        receipt = read_json(pointer)
        assert receipt['code_manifest_sha256'] == code_identity() and receipt['arm'] == arm
        assert receipt['start_checkpoint_sha256'] == lock['start_checkpoint_sha256']
        assert sha(OUTPUTS / receipt['checkpoint_path']) == receipt['checkpoint_sha256']
        ck = load_checkpoint(OUTPUTS / receipt['checkpoint_path'], seed)
        stats, last_eval = receipt['stats'], int(receipt['last_evaluated_iter'])
        previous[0] = int(receipt['iter'])
    else:
        ck = base
    protect, _ = load_dataset('protection')
    cache = R.cache_protection(protect)
    monitor, mids = staged('monitor')
    monitor_all, _ = load_dataset('monitor')
    confirm, cids = staged('confirm')
    sample, (_, optimizer) = make_sampler(seed), template(seed)

    def save_resume():
        stage = int(ck['iter'])
        path = ad / 'checkpoints' / f'{stage:011d}.eqx'
        digest = save_checkpoint(path, ck)
        write_json(pointer, dict(iter=stage, seed=seed, arm=arm, stats=stats, last_evaluated_iter=last_eval,
                   start_checkpoint_sha256=lock['start_checkpoint_sha256'], checkpoint_path=rel(path),
                   checkpoint_sha256=digest, code_manifest_sha256=code_identity()))
        old = previous[0]
        if old is not None and old != stage and not retained(old, lock):
            (ad / 'checkpoints' / f'{old:011d}.eqx').unlink(missing_ok=True)
        previous[0] = stage
        return path, digest

    while True:
        stage = int(ck['iter'])
        if stage % EVERY == 0 and last_eval != stage:
            # As in exp039: the fork snapshot is already after its evaluation; the
            # first branch evaluation consumes one new key (dropout is 0).
            ck['seeds']['eval_model_seed'], _ = jax.random.split(ck['seeds']['eval_model_seed'])
            meta = dict(stage=stage, arm=arm, seed_id=seed['id'], code_manifest_sha256=code_identity(),
                        start_checkpoint_sha256=lock['start_checkpoint_sha256'])
            if stage % SETTINGS['arm_monitor_every'] == 0 or stage == start:
                write_eval(ad / 'monitor' / f'{arm}_{stage:011d}.npz', ck['model'], monitor, mids, **meta)
            if stage in lock['window_stages']:
                write_eval(ad / 'confirm' / f'{arm}_{stage:011d}.npz', ck['model'], confirm, cids, **meta)
            assert int(np.asarray(ck['opt_state'][0].count)) == stage // BS
            assert bool(np.asarray(R.finite_candidate(ck['model'], ck['opt_state'])))
            last_eval = stage
            save_resume()
            heartbeat(phase='arm', seed=seed, arm=arm, iter=stage, start=start, end=end, stats=stats,
                      path_set=list(path_ids), kind=kind, rho=rho)
            print('ARM_PROGRESS', ARCH, seed['id'], arm, stage, end, flush=True)
        if stop_reason():
            save_resume()
            require_running('arm_after_checkpoint')
        if stage == end:
            break
        next_boundary = min(end, (stage // EVERY + 1) * EVERY)
        offset = (stage - start) // BS
        count = min(int(SETTINGS['queue_chunk']), (next_boundary - stage) // BS)
        audits = SETTINGS['diagnostic_update_offsets']
        future = [v - offset for v in audits if v > offset]
        if future:
            count = min(count, min(future))
        if offset in audits:
            count = 1
        candidate, status, audit = engine.queue_steps(ck, count, kind, rho, path_ids, sample, optimizer,
                                                      cache, start, tuple(audits))
        if audit is not None:
            i, old, new, raw, gradients, px, py = audit
            R.audit_update(ad, seed, arm, i, old, new, raw, gradients, px, py, monitor_all, path_ids, kind)
            if offset == 0:
                projected, _ = R.project_tree(raw, gradients, rho)
                vector, _ = R.project_numpy(R.flat(raw), np.stack([R.flat(g) for g in gradients]), rho)
                delta = R.flat(projected).astype(np.float64) - vector
                absolute = float(np.max(np.abs(delta)))
                relative = float(np.linalg.norm(delta) / max(np.linalg.norm(R.flat(raw).astype(np.float64)), 1e-30))
                assert absolute <= SETTINGS['numeric_projection_reference_max_abs']
                assert relative <= SETTINGS['numeric_projection_reference_max_relative']
                _, dk, _, mk = engine.split_keys(ck['seeds']['train_data_seed'], ck['seeds']['train_model_seed'])
                batch = sample(dk)
                author = sync(main.train_step(model=ck['model'], fwd_fn=R.FWD, optimizer=optimizer,
                    opt_state=ck['opt_state'], microbs=BS, weight_decay=0., x=batch['examples'], y=batch['labels'], key=mk))
                assert exact(candidate['opt_state'], author[2]), 'first real-step Adam differs from author'
                write_json(ad / 'arm_preflight.json', dict(passed=True, path_set=list(path_ids), kind=kind, rho=rho,
                    constraint_count=len(gradients), adam_state_exact=True, numpy_projection_max_abs=absolute,
                    numpy_projection_relative=relative, code_manifest_sha256=code_identity()))
        stats['steps'] += count
        stats['update_norm_ratio'] += float(status[:, 3].astype(np.float64).sum())
        stats['correction_norm_ratio'] += float(status[:, 4].astype(np.float64).sum())
        stats['conflicts'] += float(status[:, 5].astype(np.float64).sum())
        stats['near_zero_steps'] += int((status[:, 3] < .01).sum())
        stats['max_projection_relative_violation'] = max(stats['max_projection_relative_violation'], float(status[:, 2].max()))
        ck = candidate
        if time.monotonic() - LAST_HEARTBEAT > 30:
            heartbeat(phase='arm', seed=seed, arm=arm, iter=int(ck['iter']), start=start, end=end, stats=stats)
    final = ad / 'checkpoints' / f'{end:011d}.eqx'
    late = {}
    for role in ('selected', 'random'):
        ids_role = tuple(mech['selected'] if role == 'selected' else mech['random_set'])
        late['end_' + role] = late_record(ck['model'], ids_role, arm, end, sd, seed)
    n = max(stats['steps'], 1)
    bound = {rel(p): sha(p) for folder in ('confirm', 'monitor', 'update_audit') for p in (ad / folder).glob('*') if p.is_file()}
    bound[rel(final)] = sha(final)
    write_json(complete, dict(passed=True, seed=seed, arm=arm, kind=kind, rho=rho, path_set=list(path_ids),
        start_stage=start, end_stage=end, final_checkpoint_path=rel(final), final_checkpoint_sha256=sha(final),
        start_checkpoint_sha256=lock['start_checkpoint_sha256'], update_statistics=dict(
            mean_update_norm_ratio=stats['update_norm_ratio'] / n,
            mean_correction_norm_ratio=stats['correction_norm_ratio'] / n,
            conflict_rate=stats['conflicts'] / n, near_zero_rate=stats['near_zero_steps'] / n),
        update_statistics_sum=stats, late_diagnostics=late, outputs_file_sha256=bound,
        code_manifest_sha256=code_identity(), elapsed_seconds=time.monotonic() - STARTED))
    print('ARM_COMPLETE', ARCH, seed['id'], arm, flush=True)


# ----------------------------------------------------------------------- cli
def main_cli():
    global JOB_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True, choices=['data', 'validate', 'normal', 'mechanism', 'arm'])
    parser.add_argument('--seed')
    parser.add_argument('--arm', choices=SETTINGS['arms'])
    args = parser.parse_args()
    name = '_'.join(v for v in [args.job, ARCH if args.job != 'data' else None, args.seed, args.arm] if v)
    JOB_DIR = OUTPUTS / 'jobs' / name
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop_signal)
    write_json(JOB_DIR / 'started.json', dict(pid=os.getpid(), utc=utc_now(), args=vars(args), arch=ARCH,
               code_manifest_sha256=code_identity(),
               cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
               mem_fraction=os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')))
    heartbeat(phase='setup', job=args.job)
    try:
        if args.job == 'data':
            import datasets
            datasets.prepare()
        else:
            verify_inputs()
            if args.job == 'validate':
                validate()
            else:
                seed = seed_spec(args.seed)
                if args.job == 'normal':
                    normal(seed)
                elif args.job == 'mechanism':
                    mechanism(seed)
                else:
                    assert args.arm
                    train_arm(seed, args.arm)
        write_json(JOB_DIR / 'complete.json', dict(passed=True, pid=os.getpid(), utc=utc_now(),
                   elapsed_seconds=time.monotonic() - STARTED, code_manifest_sha256=code_identity()))
    except BaseException as error:
        if not (isinstance(error, SystemExit) and error.code == 3):
            write_json(JOB_DIR / 'failure.json', dict(error_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc(), utc=utc_now(), pid=os.getpid(),
                       elapsed_seconds=time.monotonic() - STARTED, automatic_retry=False))
        raise
    finally:
        write_json(JOB_DIR / 'exit.json', dict(pid=os.getpid(), utc=utc_now(),
                   elapsed_seconds=time.monotonic() - STARTED))


if __name__ == '__main__':
    main_cli()

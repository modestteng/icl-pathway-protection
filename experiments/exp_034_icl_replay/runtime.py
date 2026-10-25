"""Independent resumable REPLAY/L2 processes, sharing only immutable inputs.

Run ``runtime.py prepare-data``, ``runtime.py preflight`` once before launching
``runtime.py run --arm REPLAY`` and ``runtime.py run --arm L2`` concurrently.
The launcher sets each process's GPU allocation and the shared wall deadline.
"""
from common import *
from objectives import FWD, baseline_step, packed_status
from datasets import prepare as prepare_data
import argparse
import signal
import traceback

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def coefficient_for(arm):
    if arm not in SETTINGS['arms'] or arm not in ('REPLAY', 'L2'):
        raise ValueError('unregistered arm: ' + arm)
    value = SETTINGS['replay_coefficient' if arm == 'REPLAY' else 'l2_coefficient']
    if value is None or not np.isfinite(value) or value <= 0:
        raise ValueError('positive coefficient must be frozen before execution')
    if arm == 'REPLAY':
        assert value == 2.0
        assert SETTINGS['protection_tasks'] == ['icl', 'flip_icl']
    else:
        assert SETTINGS['l2_form'] == 'sum_of_squared_all_trainable_array_entries_without_half'
        assert value == 1e-4
    return float(value)


def protect_indices(step):
    n = SETTINGS['data_sets']['protection']['n']
    size = SETTINGS['protection_bs_per_task']
    return (np.arange(size, dtype=np.int64) + step * size) % n


def protect_batch(data, step):
    if isinstance(data, list):
        return data[step % len(data)]
    take = protect_indices(step)
    xy = [data[t] for t in SETTINGS['protection_tasks']]
    return (jnp.asarray(np.stack([x[take] for x, _ in xy])),
            jnp.asarray(np.stack([y[take] for _, y in xy])))


def cache_protection(data):
    count, remainder = divmod(SETTINGS['data_sets']['protection']['n'],
                             SETTINGS['protection_bs_per_task'])
    assert remainder == 0
    cached = [protect_batch(data, step) for step in range(count)]
    sync(cached)
    return cached


def stage_on_gpu(data):
    return {task: tuple(jax.device_put(v) for v in xy) for task, xy in data.items()}


def audit_indices(n):
    return np.asarray([i for i in SETTINGS['audit_rows'] if i < n], np.int32)


@eqx.filter_jit
def eval_batch(model, x, y, pure=False):
    raw = jax.vmap(partial(FWD, model=model))(x=x, y=y,
        key=jax.random.split(KEY, len(x)))['out'][:, -1, :]
    logp = jax.nn.log_softmax(raw, axis=-1)
    target, row = y[:, -1], jnp.arange(len(x))
    out = dict(loss=main.ce(raw, target), acc=jnp.argmax(raw, 1) == target,
               full_pred=jnp.argmax(raw, 1), target=target)
    if not pure:
        candidates = raw[row[:, None], y[:, :2]]
        ti = jnp.argmax(y[:, :2] == target[:, None], 1)
        other = 1 - ti
        pred = jnp.where(candidates[:, 0] > candidates[:, 1], y[:, 0],
            jnp.where(candidates[:, 0] < candidates[:, 1], y[:, 1],
                      jnp.minimum(y[:, 0], y[:, 1])))
        mass = jax.scipy.special.logsumexp(logp[row[:, None], y[:, :2]], axis=1)
        out.update(in_context_acc=pred == target, pair_pred=pred,
            pair_loss=jax.scipy.special.logsumexp(candidates, axis=1) - candidates[row, ti],
            pair_margin=candidates[row, ti] - candidates[row, other],
            context_mass=jnp.exp(mass), log_context_mass=mass)
    return out, raw, logp


def evaluate(model, data, ids=None):
    record = {}
    for task, (x, y) in data.items():
        collected, raw_rows, logp_rows = {}, [], []
        ar = audit_indices(len(x))
        for lo in range(0, len(x), SETTINGS['eval_bs']):
            hi = min(lo + SETTINGS['eval_bs'], len(x))
            out, raw, logp = eval_batch(model, x[lo:hi], y[lo:hi], pure=task == 'pure_iwl')
            # Transfer only the small metric vectors plus prespecified logit rows.
            take = ar[(ar >= lo) & (ar < hi)] - lo
            out, audit_raw, audit_lp = jax.device_get((out, raw[take], logp[take]))
            for key, value in out.items():
                arr = np.asarray(value)
                if not np.isfinite(arr).all():
                    raise RuntimeError('nonfinite evaluation: ' + task + '/' + key)
                collected.setdefault(key, []).append(arr)
            raw_rows.extend(audit_raw)
            logp_rows.extend(audit_lp)
        for key, values in collected.items():
            record[task + '/' + key] = np.concatenate(values)
        record[task + '/audit_rows'] = ar
        record[task + '/audit_logits'] = np.asarray(raw_rows)
        record[task + '/audit_log_probs'] = np.asarray(logp_rows)
        record[task + '/audit_labels'] = np.asarray(y[ar], np.int32)
        record[task + '/audit_examples_sha256'] = np.asarray([
            hashlib.sha256(np.asarray(v, np.float32).tobytes()).hexdigest()
            for v in np.asarray(x[ar])])
        if ids is not None:
            record[task + '/identities'] = ids[task]
    return record


def original_data():
    with h5py.File(ASSETS / 'data/eval_data.h5', 'r') as f:
        return {t: (np.asarray(f[t]['examples'], np.float32),
                    np.asarray(f[t]['labels'], np.int32)) for t in (*TASKS, 'pure_iwl')}


def attach_meta(record, seed, arm, stage, data_name, checkpoint_path, checkpoint_sha):
    data_path = ASSETS / 'data/eval_data.h5' if data_name == 'original' else OUT / 'data' / (data_name + '.h5')
    metadata = dict(arm=arm, stage=int(stage), seed_id=seed['id'], init_seed=seed['init'],
        train_seed=seed['train'], code_manifest_sha256=code_identity(),
        settings_sha256=sha(WORK / 'settings.json'), data_sha256=sha(data_path),
        checkpoint_sha256=checkpoint_sha, checkpoint_path=str(checkpoint_path.relative_to(WORK)))
    for name, value in metadata.items():
        record['_meta/' + name] = np.asarray(value, dtype=np.int64 if isinstance(value, int) else None)
    return record


def key_record(seeds):
    return {k: np.asarray(v, np.uint32).tolist() for k, v in seeds.items()}


def metrics_record(metrics):
    selected = {k: metrics[k] for k in ('loss','grad_norm','grad_batch_stddev',
        'train_ce','replay_ce','replay_task_ce','squared_l2','regularization_loss') if k in metrics}
    return {k: np.asarray(v).tolist() for k,v in jax.device_get(selected).items()}


def batch_digest(x, y):
    return hashlib.sha256(np.asarray(x, np.float32).tobytes()
                          + np.asarray(y, np.int32).tobytes()).hexdigest()


def audit_update(sd, seed, arm, stage, offset, ck, next_ck, metrics, batch, px, py,
                 train_data_key, train_model_key, coefficient):
    paths = {}
    for label, state in [('before', ck), ('after', next_ck)]:
        path = sd / 'update_audit' / f'{arm}_{stage:011d}_{label}.eqx'
        paths[label] = dict(path=str(path.relative_to(WORK)), sha256=save_checkpoint(path, state))
    actual_path = sd / 'update_audit' / f'{arm}_{stage:011d}_actual_step.npz'
    actual = {'g_used/' + f'{idx:04d}': np.asarray(value)
              for idx,value in enumerate(arrays(metrics['g_used']))}
    actual.update({'actual/' + k: np.asarray(v) for k,v in metrics['actual'].items()})
    actual.update({'metrics/' + k: np.asarray(v) for k,v in metrics_record(metrics).items()})
    actual.update({'actual/train_examples': np.asarray(batch['examples'], np.float32),
        'actual/train_labels': np.asarray(batch['labels'], np.int32),
        'actual/train_targets': np.asarray(batch['labels'][:, -1], np.int32),
        'actual/protection_examples': np.asarray(px, np.float32),
        'actual/protection_labels': np.asarray(py, np.int32),
        'actual/protection_targets': np.asarray(py[:, :, -1], np.int32)})
    save_npz(actual_path, actual)
    gradient_leaves = [dict(path=jax.tree_util.keystr(path), shape=list(value.shape),
        dtype=str(value.dtype), npz_key=f'g_used/{idx:04d}')
        for idx,(path,value) in enumerate(jax.tree_util.tree_flatten_with_path(metrics['g_used'])[0])]
    write_json(sd / 'update_audit' / f'{arm}_{stage:011d}.json', dict(
        seed=seed, arm=arm, stage=int(stage), update_offset=int(offset), coefficient=coefficient,
        normal_batch_size=BS, protection_tasks=SETTINGS['protection_tasks'],
        protection_indices=protect_indices(offset).tolist(),
        protection_used_for_gradient=arm == 'REPLAY',
        normal_batch_sha256=batch_digest(batch['examples'], batch['labels']),
        protection_batch_sha256={t: batch_digest(px[i], py[i])
            for i, t in enumerate(SETTINGS['protection_tasks'])},
        training_prng_before=key_record(ck['seeds']), training_prng_after=key_record(next_ck['seeds']),
        train_data_key_used=np.asarray(train_data_key, np.uint32).tolist(),
        train_model_key_used=np.asarray(train_model_key, np.uint32).tolist(),
        metrics_before=metrics_record(metrics),
        actual_step_npz_path=str(actual_path.relative_to(WORK)), actual_step_npz_sha256=sha(actual_path),
        gradient_leaves=gradient_leaves, gradient_source='actual gradient used by this formal Adam update',
        production_capture_interface_fixed=True,
        checkpoints=paths, code_manifest_sha256=code_identity(),
        settings_sha256=sha(WORK / 'settings.json'),
        protection_data_sha256=sha(OUT / 'data/protection.h5')))


@jax.jit
def expected_training_keys(data_key, model_key, steps):
    def advance(_, keys):
        return tuple(jax.random.split(k)[0] for k in keys)
    return jax.lax.fori_loop(0, steps, advance, (data_key, model_key))


def stopped(start, previous):
    deadline = float(os.environ.get('ICL034_DEADLINE_EPOCH', 'inf'))
    return (STOP or (WORK / 'STOP').exists() or time.time() >= deadline
            or previous + time.monotonic() - start >= SETTINGS['max_compute_seconds'])


def train_arm(seed, arm):
    coefficient = coefficient_for(arm)
    sd = OUT / seed['id'] / arm
    sd.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cost_path = sd / 'compute_seconds.json'
    previous = json.loads(cost_path.read_text())['compute_seconds'] if cost_path.exists() else 0.0
    def record_cost(phase):
        write_json(cost_path, dict(compute_seconds=previous + time.monotonic() - started,
                                  phase=phase, arm=arm, code_manifest_sha256=code_identity()))
    complete = sd / 'training_complete.json'
    if complete.exists():
        receipt = json.loads(complete.read_text())
        assert receipt['passed'] and receipt['end_stage'] == SETTINGS['end_stage']
        assert receipt['code_manifest_sha256'] == code_identity()
        for name, digest in receipt['output_sha256'].items():
            assert sha(WORK / name) == digest
        return receipt
    base = base_checkpoint(seed)
    control = sd / 'resume.json'
    sums = dict(steps=0, loss=0.0, train_ce=0.0, replay_ce=0.0, squared_l2=0.0,
                regularization_loss=0.0, grad_norm=0.0)
    last_evaluated = -1
    if control.exists():
        receipt = json.loads(control.read_text())
        assert receipt['code_manifest_sha256'] == code_identity()
        checkpoint = WORK / receipt['checkpoint_path']
        assert sha(checkpoint) == receipt['checkpoint_sha256']
        ck = load_checkpoint(checkpoint, seed)
        assert int(ck['iter']) == receipt['iter']
        sums, last_evaluated = receipt['statistics_sum'], receipt['last_evaluated_iter']
    else:
        ck = dict(base, seeds=dict(base['seeds']))
    assert (int(ck['iter']) - SETTINGS['start_stage']) // BS == sums['steps']
    protect, _ = load_dataset('protection')
    protect = cache_protection(protect)
    monitor, mids = load_dataset('monitor')
    confirm, cids = load_dataset('confirm')
    original, monitor, confirm = [stage_on_gpu(v) for v in (original_data(), monitor, confirm)]
    sample = make_sampler(seed)
    _, optimizer = template(seed)
    def save_resume(known_path=None, known_sha=None):
        path = known_path or sd / 'checkpoints' / f'{arm}_resume_{int(ck["iter"]):011d}.eqx'
        digest = known_sha or save_checkpoint(path, ck)
        write_json(control, dict(iter=int(ck['iter']), checkpoint_path=str(path.relative_to(WORK)),
            checkpoint_sha256=digest, last_evaluated_iter=last_evaluated,
            statistics_sum=sums, seed=seed, arm=arm, coefficient=coefficient,
            code_manifest_sha256=code_identity(), training_prng=key_record(ck['seeds'])))
    try:
        while True:
            i = int(ck['iter'])
            if i % SETTINGS['eval_every'] == 0 and last_evaluated != i:
                # Match author resume semantics: consume one eval seed per scheduled
                # evaluation; no train data/model seed is used by any evaluator.
                ck['seeds']['eval_model_seed'], _ = jax.random.split(ck['seeds']['eval_model_seed'])
                checkpoint = sd / 'checkpoints' / f'{arm}_{i:011d}.eqx'
                digest = save_checkpoint(checkpoint, ck)
                eval_outputs = {}
                groups = [('original', original, None)]
                if i % SETTINGS['monitor_every'] == 0 or i == SETTINGS['start_stage']:
                    groups.append(('monitor', monitor, mids))
                if i in SETTINGS['confirm_stages']:
                    groups.append(('confirm', confirm, cids))
                for name, data, ids in groups:
                    path = sd / name / f'{arm}_{i:011d}.npz'
                    result = attach_meta(evaluate(ck['model'], data, ids), seed, arm, i, name, checkpoint, digest)
                    save_npz(path, result)
                    eval_outputs[str(path.relative_to(WORK))] = sha(path)
                assert int(np.asarray(ck['opt_state'][0].count)) == i // BS
                last_evaluated = i
                write_json(sd / 'evaluated' / f'{arm}_{i:011d}.json', dict(iter=i,
                    checkpoint_sha256=digest, output_sha256=eval_outputs,
                    code_manifest_sha256=code_identity()))
                save_resume(checkpoint, digest)
                record_cost('training')
                write_json(sd / 'progress.json', dict(seed=seed, arm=arm, iter=i,
                    end=SETTINGS['end_stage'], steps=sums['steps'],
                    compute_seconds=previous + time.monotonic() - started,
                    code_manifest_sha256=code_identity()))
                print('PROGRESS', seed['id'], arm, i, flush=True)
            if stopped(started, previous):
                save_resume()
                record_cost('safe_stop')
                raise SystemExit(3)
            if i == SETTINGS['end_stage']:
                break
            if i > SETTINGS['end_stage']:
                raise RuntimeError('checkpoint exceeds frozen training endpoint')
            dk_next, dk = jax.random.split(ck['seeds']['train_data_seed'])
            mk_next, mk = jax.random.split(ck['seeds']['train_model_seed'])
            batch = sample(dk)
            offset = (i - SETTINGS['start_stage']) // BS
            px, py = protect_batch(protect, offset)
            metrics, model, state = baseline_step(ck['model'], ck['opt_state'],
                batch['examples'], batch['labels'], mk, px, py, arm, coefficient, optimizer)
            status = np.asarray(packed_status(model, state, metrics))
            if not bool(status[0]):
                save_resume()
                write_json(sd / 'failure.json', dict(stage=i, reason='nonfinite candidate rejected',
                    model_adam_prng_committed=False, code_manifest_sha256=code_identity()))
                raise RuntimeError('nonfinite candidate rejected before model/Adam/PRNG commit')
            next_ck = dict(iter=i + BS, model=model, opt_state=state,
                seeds=dict(ck['seeds'], train_data_seed=dk_next, train_model_seed=mk_next))
            if offset in SETTINGS['diagnostic_update_offsets']:
                audit_update(sd, seed, arm, i, offset, ck, next_ck, metrics, batch, px, py, dk, mk, coefficient)
            sums['steps'] += 1
            for name, value in zip(('loss','train_ce','replay_ce','squared_l2','regularization_loss','grad_norm'), status[1:]):
                sums[name] += float(value)
            ck = next_ck
            if int(ck['iter']) % SETTINGS['resume_every'] == 0 and int(ck['iter']) % SETTINGS['eval_every']:
                save_resume()
                record_cost('resume_checkpoint')
        expected_keys = sync(expected_training_keys(base['seeds']['train_data_seed'],
            base['seeds']['train_model_seed'], sums['steps']))
        actual_keys = (ck['seeds']['train_data_seed'], ck['seeds']['train_model_seed'])
        assert exact(expected_keys, actual_keys)
        final = sd / 'checkpoints' / f'{arm}_{int(ck["iter"]):011d}.eqx'
        prng_path = sd / 'training_prng.json'
        write_json(prng_path, dict(passed=True, steps=sums['steps'],
            start=key_record(base['seeds']), end=key_record(ck['seeds']),
            expected_train_data_seed=np.asarray(expected_keys[0], np.uint32).tolist(),
            expected_train_model_seed=np.asarray(expected_keys[1], np.uint32).tolist(),
            train_data_and_model_streams_exact=True,
            protection_consumes_training_stream=False, code_manifest_sha256=code_identity()))
        outputs = {}
        for group, pattern in [('original','*.npz'),('monitor','*.npz'),('confirm','*.npz'),
                               ('checkpoints','*.eqx'),('update_audit','*'),('evaluated','*.json')]:
            outputs.update({str(path.relative_to(WORK)): sha(path)
                            for path in sorted((sd / group).glob(pattern)) if path.is_file()})
        outputs[str(prng_path.relative_to(WORK))] = sha(prng_path)
        result = dict(passed=True, arm=arm, seed=seed, coefficient=coefficient,
            start_stage=SETTINGS['start_stage'], end_stage=int(ck['iter']), steps=sums['steps'],
            base_checkpoint_sha256=sha(ASSETS / 'checkpoints/00004400000.eqx'),
            final_checkpoint_path=str(final.relative_to(WORK)), final_checkpoint_sha256=sha(final),
            model_and_adam_finite=True, adam_count=int(np.asarray(ck['opt_state'][0].count)),
            training_prng_exact=True, training_prng_sha256=sha(prng_path),
            statistics_sum=sums, statistics_mean={k:v/sums['steps'] for k,v in sums.items() if k != 'steps'},
            output_sha256=outputs, code_manifest_sha256=code_identity(),
            settings_sha256=sha(WORK / 'settings.json'), inputs=provenance()['inputs'],
            new_inputs=provenance().get('new_inputs', {}), source=provenance()['source'],
            compute_seconds=previous + time.monotonic() - started,
            mechanism_diagnostics=False, projection=False)
        write_json(complete, result)
        record_cost('completed')
        print('TRAINING_COMPLETE', seed['id'], arm, result['compute_seconds'], flush=True)
        return result
    except BaseException as error:
        if not isinstance(error, SystemExit):
            save_resume()
            write_json(sd / f'exception_{time.time_ns()}.json', dict(error_type=type(error).__name__,
                error=str(error), traceback=traceback.format_exc(), code_manifest_sha256=code_identity(),
                stage=int(ck['iter']), automatic_retry=False, scientific_conditions_changed=False))
        raise
    finally:
        record_cost('runner_exit')


def run(arm, seeds):
    verify_inputs()
    audit = json.loads((OUT / 'data/identity_audit.json').read_text())
    assert audit['passed']
    p = provenance()
    for name, digest in SETTINGS['preserved_data_sha256'].items():
        path = OUT / 'data' / name
        assert sha(path) == digest
        p.setdefault('new_inputs', {})[str(path.relative_to(WORK))] = digest
    write_json(OUT / 'provenance.json', p)
    prepath = OUT / ('preflight_' + arm + '.json')
    if not prepath.exists():
        from gpu_preflight import preflight
        preflight(arm)
    pre = json.loads(prepath.read_text())
    assert pre['passed'] and pre['code_manifest_sha256'] == code_identity()
    for seed in SETTINGS['training_seeds']:
        if seed['id'] in seeds:
            train_arm(seed, arm)


if __name__ == '__main__':
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare-data','preflight','benchmark','run'])
    parser.add_argument('--arm', choices=['REPLAY','L2'])
    parser.add_argument('--seeds', default=','.join(s['id'] for s in SETTINGS['training_seeds']))
    parser.add_argument('--benchmark-steps', type=int, default=64)
    args = parser.parse_args()
    if args.mode == 'prepare-data':
        prepare_data()
    elif args.mode == 'preflight':
        from gpu_preflight import preflight
        preflight()
    elif args.mode == 'benchmark':
        from gpu_preflight import benchmark
        benchmark(args.arm, args.benchmark_steps)
    else:
        if args.arm is None:
            parser.error('--arm is required for run')
        run(args.arm, args.seeds.split(','))

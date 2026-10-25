"""One new scientific condition; immutable exp036/author kernels remain unchanged."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'settings.json').read_text())
OUT = ROOT / 'outputs'
STOP = False


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def request_stop(*_):
    global STOP
    STOP = True


def runtime():
    base = Path(os.environ.get('ICL_EXPERIMENT_ROOT', str(ROOT.parent)))
    source = base / CFG['kernel_experiment']
    for name, digest in CFG['kernel_files'].items():
        assert sha(source / name) == digest, ('kernel_identity', name)
    assets = base / 'exp_030_functional_update_constraints/assets'
    for name, digest in CFG['asset_files'].items():
        assert sha(assets / name) == digest, ('asset_identity', name)
    for name, digest in CFG['data_files'].items():
        assert sha(base / 'exp_030_functional_update_constraints/outputs/data' / name) == digest
    for rec in CFG['start_checkpoints'].values():
        assert sha(base / rec['path']) == rec['sha256']
    refs = json.loads((ROOT / 'reference_manifest.json').read_text())
    for arms in refs.values():
        for records in arms.values():
            for name, digest in records.items():
                assert sha(base / name) == digest, ('reference_identity', name)
    os.environ['ICL036_ASSETS'] = str(assets)
    sys.path.insert(0, str(source))
    import runtime as r
    from importlib.metadata import version
    versions = {k: version(k) for k in CFG['versions']}
    assert versions == CFG['versions'], versions
    assert r.jax.default_backend() == 'gpu'
    assert not r.jax.config.jax_enable_x64
    assert r.jax.config.jax_default_matmul_precision is None
    return r, base, versions


def train(seed):
    import numpy as np
    import h5py
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()
    r, base, versions = runtime()
    sd = OUT / seed['id']
    sd.mkdir(parents=True, exist_ok=True)
    manifest_sha = sha(ROOT / 'code_manifest.json')
    for name, digest in json.loads((ROOT / 'code_manifest.json').read_text()).items():
        assert sha(ROOT / name) == digest, ('sealed_code', name)
    assert not (sd / 'complete.json').exists(), 'Do not repeat completed training'
    initial = CFG['start_checkpoints'][seed['id']]
    checkpoint = r.load_checkpoint(base / initial['path'], seed)
    assert int(checkpoint['iter']) == CFG['start_stage']
    initial_prng = {k: np.asarray(v).tolist() for k, v in checkpoint['seeds'].items()}
    _, optimizer = r.template(seed)
    sample = r.make_sampler(seed)
    data_root = base / 'exp_030_functional_update_constraints/outputs/data'

    def dataset(name, protection=False):
        tasks = r.SETTINGS['protection_tasks'] if protection else CFG['eval_tasks']
        with h5py.File(data_root / (name + '.h5'), 'r') as f:
            xy = {t: (np.asarray(f[t]['examples'], np.float32),
                      np.asarray(f[t]['labels'], np.int32)) for t in tasks}
            ids = {t: np.asarray(f[t]['identities'], np.int32) for t in tasks}
        return xy, ids

    protection, _ = dataset('protection', True)
    protection = r.cache_protection(protection)
    monitor, mids = dataset('monitor')
    confirm, cids = dataset('confirm')
    original = {k: v for k, v in r.original_data().items() if k in CFG['eval_tasks']}
    monitor, confirm, original = [r.stage_on_gpu(d) for d in (monitor, confirm, original)]

    @r.eqx.filter_jit
    def mean_tree(a, b):
        return r.jax.tree_util.tree_map(lambda x, y: (x + y) * .5, a, b)

    def step(ck, batch, key, offset, rho):
        met, _, state = r.main.train_step(model=ck['model'], fwd_fn=r.FWD,
            optimizer=optimizer, opt_state=ck['opt_state'], microbs=r.BS,
            weight_decay=0., x=batch['examples'], y=batch['labels'], key=key)
        raw = r.adam_update_from_state(state)
        px, py = r.protect_batch(protection, offset)
        losses, gradients = [], []
        for i in range(2):
            keys = r.jax.random.split(r.jax.random.fold_in(r.KEY, i), px.shape[1])
            loss, grad = r.task_objective(ck['model'], px[i], py[i], keys)
            losses.append(loss)
            gradients.append(grad)
        g = mean_tree(*gradients)
        loss = (losses[0] + losses[1]) * .5
        model, diag = r.constrain_update(ck['model'], raw, (g,), 'MEAN', rho)
        status = np.asarray(r.packed_status(model, state, (met['loss'], loss), diag))
        assert bool(status[0]) and bool(status[1]), ('nonfinite_or_projection', status)
        assert float(status[2]) <= 2e-6, ('projection_violation', status)
        return model, state, raw, gradients, g, met, losses, status

    pointer = sd / 'resume.json'
    stats = dict(steps=0, correction_sum=0., norm_sum=0., projected_steps=0,
                 near_zero_steps=0, max_projection_violation=0.)
    last_eval = -1
    previous = 0.
    if pointer.exists():
        receipt = json.loads(pointer.read_text())
        assert receipt['code_manifest_sha256'] == manifest_sha
        assert receipt['initial_sha256'] == initial['sha256']
        assert sha(ROOT / receipt['checkpoint']) == receipt['sha256']
        checkpoint = r.load_checkpoint(ROOT / receipt['checkpoint'], seed)
        stats, last_eval, previous = receipt['stats'], receipt['last_eval'], receipt['compute_seconds']

    def save_resume():
        path = sd / 'checkpoints' / f'MEAN25_resume_{int(checkpoint["iter"]):011d}.eqx'
        digest = r.save_checkpoint(path, checkpoint)
        write(pointer, dict(seed=seed, iter=int(checkpoint['iter']), checkpoint=str(path.relative_to(ROOT)),
            sha256=digest, stats=stats, last_eval=last_eval, initial_sha256=initial['sha256'],
            code_manifest_sha256=manifest_sha, compute_seconds=previous + time.monotonic() - started))

    def evaluate(path, data, ids=None):
        result = r.evaluate(checkpoint['model'], data, ids)
        result.update(stage=np.asarray(checkpoint['iter']), arm=np.asarray(CFG['arm']),
            seed_id=np.asarray(seed['id']), initial_sha256=np.asarray(initial['sha256']),
            code_manifest_sha256=np.asarray(manifest_sha))
        r.save_npz(path, result)
        return result

    provenance = dict(seed=seed, versions=versions,
        initial_checkpoint=initial, code_manifest_sha256=manifest_sha,
        device=[str(d) for d in r.jax.devices()], config=CFG,
        initial_prng=initial_prng)
    if (sd / 'provenance.json').exists():
        assert json.loads((sd / 'provenance.json').read_text()) == provenance
    else:
        write(sd / 'provenance.json', provenance)
    try:
        while True:
            i = int(checkpoint['iter'])
            if i % CFG['eval_every'] == 0 and last_eval != i:
                next_eval_key, _ = r.jax.random.split(checkpoint['seeds']['eval_model_seed'])
                path = sd / 'checkpoints' / f'MEAN25_{i:011d}.eqx'
                evaluated_checkpoint = dict(checkpoint, seeds=dict(checkpoint['seeds'], eval_model_seed=next_eval_key))
                r.save_checkpoint(path, evaluated_checkpoint)
                evaluate(sd / 'original' / f'MEAN25_{i:011d}.npz', original)
                if i % CFG['monitor_every'] == 0 or i == CFG['start_stage']:
                    result = evaluate(sd / 'monitor' / f'MEAN25_{i:011d}.npz', monitor, mids)
                    if i == CFG['start_stage']:
                        exp = 'exp_031_path_only_protection' if seed['id'] == 's5' else CFG['kernel_experiment']
                        ref = base / exp / 'outputs' / seed['id'] / 'monitor' / f'P25_{i:011d}.npz'
                        with np.load(ref) as z:
                            start_checks = {t: dict(ce_error=abs(float(np.mean(result[t+'/loss']) - np.mean(z[t+'/loss']))),
                                acc_row_mismatches=int(np.sum(result[t+'/acc'] != z[t+'/acc'])),
                                identities_equal=bool(np.array_equal(result[t+'/identities'], z[t+'/identities']))) for t in CFG['eval_tasks']}
                        assert all(v['ce_error'] <= .001 and v['acc_row_mismatches'] <= 5 and v['identities_equal'] for v in start_checks.values()), start_checks
                        write(sd / 'start_evaluation_check.json', dict(passed=True, checks=start_checks))
                if i in CFG['confirm_stages']:
                    result = evaluate(sd / 'confirm' / f'MEAN25_{i:011d}.npz', confirm, cids)
                    summary = {t: dict(acc=float(result[t+'/acc'].mean()), ce=float(result[t+'/loss'].mean())) for t in CFG['eval_tasks']}
                    print('CONFIRM', seed['id'], i, json.dumps(summary), flush=True)
                assert int(np.asarray(checkpoint['opt_state'][0].count)) == i // r.BS
                checkpoint['seeds']['eval_model_seed'] = next_eval_key
                last_eval = i
                save_resume()
                elapsed = previous + time.monotonic() - started
                write(sd / 'progress.json', dict(seed=seed, stage=i, end_stage=CFG['end_stage'],
                    compute_seconds=elapsed, stats=stats, code_manifest_sha256=manifest_sha))
                print('PROGRESS', seed['id'], i, 'seconds', round(elapsed, 2), flush=True)
            if STOP:
                save_resume()
                write(sd / 'stopped.json', dict(stage=i, reason='supervisor_or_user_stop', checkpoint_retained=True))
                return 3
            if i == CFG['end_stage']:
                break
            dn, dk = r.jax.random.split(checkpoint['seeds']['train_data_seed'])
            mn, mk = r.jax.random.split(checkpoint['seeds']['train_model_seed'])
            batch = sample(dk)
            offset = (i - CFG['start_stage']) // r.BS
            result = step(checkpoint, batch, mk, offset, CFG['rho'])
            model, state, raw, gradients, g, met, losses, status = result
            if offset in r.SETTINGS['diagnostic_update_offsets']:
                projected, _ = r.project_tree(raw, (g,), CFG['rho'])
                audit = dict(raw_update=r.flat(raw), component_gradients=np.stack([r.flat(v) for v in gradients]),
                    mean_gradient=r.flat(g), projected_update=r.flat(projected), component_losses=np.asarray(losses),
                    rho=np.asarray(CFG['rho']), stage=np.asarray(i), seed_id=np.asarray(seed['id']))
                r.save_npz(sd / 'update_audit' / f'MEAN25_{i:011d}.npz', audit)
                expected, _ = r.project_numpy(audit['raw_update'], audit['mean_gradient'][None, :], CFG['rho'])
                err = np.max(np.abs(expected - audit['projected_update']))
                relative = np.linalg.norm(expected.astype(np.float64) - audit['projected_update']) / max(np.linalg.norm(audit['raw_update'].astype(np.float64)), 1e-30)
                assert err <= 2e-5 and relative <= 2e-5, (err, relative)
                mean_ref = audit['component_gradients'].astype(np.float64).mean(axis=0)
                mean_err = float(np.max(np.abs(mean_ref - audit['mean_gradient'])))
                assert np.linalg.norm(mean_ref - audit['mean_gradient']) <= 2e-7 * max(np.linalg.norm(mean_ref), 1e-30)
                if offset == 0:
                    author = r.sync(r.main.train_step(model=checkpoint['model'], fwd_fn=r.FWD,
                        optimizer=optimizer, opt_state=checkpoint['opt_state'], microbs=r.BS, weight_decay=0.,
                        x=batch['examples'], y=batch['labels'], key=mk))
                    assert r.exact(state, author[2]) and r.exact(met['loss'], author[0]['loss'])
                    raw_err = r.maxdiff(r.eqx.apply_updates(checkpoint['model'], raw), author[1])
                    assert raw_err <= 1e-4
                    rho1, _ = r.project_tree(raw, (g,), 1.)
                    assert r.exact(rho1, raw)
                    write(sd / 'preflight.json', dict(passed=True, constraint_count=1,
                        main_adam_exact=True, main_loss_exact=True, rho1_raw_exact=True,
                        raw_reconstruction_error=raw_err, numpy_projection_max_abs=float(err),
                        numpy_projection_relative=float(relative), mean_gradient_max_abs=mean_err))
                    print('PREFLIGHT_PASSED', seed['id'], flush=True)
            stats['steps'] += 1
            stats['norm_sum'] += float(status[3])
            stats['correction_sum'] += float(status[4])
            stats['projected_steps'] += int(status[5])
            stats['near_zero_steps'] += int(status[3] < .01)
            stats['max_projection_violation'] = max(stats['max_projection_violation'], float(status[2]))
            checkpoint.update(model=model, opt_state=state, iter=i + r.BS)
            checkpoint['seeds']['train_data_seed'], checkpoint['seeds']['train_model_seed'] = dn, mn
        assert stats['steps'] == 175000
        assert int(np.asarray(checkpoint['opt_state'][0].count)) == 312500
        files = {str(p.relative_to(ROOT)): sha(p) for folder in ['confirm', 'update_audit'] for p in (sd / folder).glob('*.npz')}
        assert len(list((sd / 'confirm').glob('*.npz'))) == 11
        final = sd / 'checkpoints' / 'MEAN25_00010000000.eqx'
        write(sd / 'complete.json', dict(passed=True, seed=seed, end_stage=10000000,
            final_checkpoint=str(final.relative_to(ROOT)), final_sha256=sha(final), stats=stats,
            final_prng={k: np.asarray(v).tolist() for k, v in checkpoint['seeds'].items()},
            code_manifest_sha256=manifest_sha, files=files,
            compute_seconds=previous + time.monotonic() - started))
        print('COMPLETE', seed['id'], flush=True)
        return 0
    except BaseException as e:
        save_resume()
        write(sd / 'failure.json', dict(stage=int(checkpoint['iter']), error=repr(e), checkpoint_retained=True))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', choices=[s['id'] for s in CFG['training_seeds']], required=True)
    args = p.parse_args()
    seed = next(s for s in CFG['training_seeds'] if s['id'] == args.seed)
    sys.exit(train(seed))

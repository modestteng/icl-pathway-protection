"""Short matched cost measurement; import and call sealed scientific kernels."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'settings.json').read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('arm', choices=CFG['arms'])
    p.add_argument('--validate', action='store_true')
    p.add_argument('--tag', required=True)
    p.add_argument('--round', type=int, default=-1)
    args = p.parse_args()
    out = ROOT / 'outputs' / args.tag
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base = Path(os.environ.get('ICL_EXPERIMENT_ROOT', str(ROOT.parent)))
    source = base / 'exp_033_random_path_only_protection'
    for name, digest in CFG['sealed_kernel_files'].items():
        assert sha(source / name) == digest, name
    assets = base / 'exp_030_functional_update_constraints/assets'
    os.environ['ICL033_ASSETS'] = str(assets)
    # The frozen module's env name is verified against its source before launch.
    os.environ['ICL031_ASSETS'] = str(assets)
    for name, digest in CFG['asset_files'].items():
        assert sha(assets / name) == digest, name
    protection_path = base / 'exp_030_functional_update_constraints/outputs/data/protection.h5'
    assert sha(protection_path) == CFG['protection_sha256']
    sys.path.insert(0, str(source))
    import runtime as r
    import numpy as np
    import jax
    import jax.numpy as jnp
    import h5py
    from importlib.metadata import version
    versions = {k: version(k) for k in CFG['versions']}
    assert versions == CFG['versions'], versions
    assert jax.default_backend() == 'gpu'
    assert not jax.config.jax_enable_x64
    assert jax.config.jax_default_matmul_precision is None
    seed = {'id': 's5', 'init': 5, 'train': 2}
    ck = r.base_checkpoint(seed)
    _, optimizer = r.template(seed)
    sample = r.make_sampler(seed)
    protect = None
    if args.arm != 'A':
        with h5py.File(protection_path, 'r') as f:
            protect = {t: (np.asarray(f[t]['examples'], np.float32),
                           np.asarray(f[t]['labels'], np.int32))
                       for t in r.SETTINGS['protection_tasks']}
        protect = r.cache_protection(protect)
    kind = 'C' if args.arm == 'C25' else 'P'
    head = 1 if args.arm == 'PR25' else 0
    status_rows = []

    def step(offset, record=True):
        dk_next, dk = jax.random.split(ck['seeds']['train_data_seed'])
        mk_next, mk = jax.random.split(ck['seeds']['train_model_seed'])
        batch = sample(dk)
        if args.arm == 'A':
            met, model, state = r.main.train_step(model=ck['model'], fwd_fn=r.FWD,
                optimizer=optimizer, opt_state=ck['opt_state'], microbs=r.BS,
                weight_decay=0.0, x=batch['examples'], y=batch['labels'], key=mk)
            finite = r.finite_candidate(model, (state, met['loss']))
            assert bool(np.asarray(finite)), 'nonfinite normal step'
            extra = None
        else:
            px, py = r.protect_batch(protect, offset)
            met, model, state, raw, gradients = r.modified_step(
                ck['model'], ck['opt_state'], batch['examples'], batch['labels'],
                mk, px, py, kind, .25, head, optimizer)
            status = np.asarray(r.packed_status(model, state,
                (met['loss'], met['protection_loss']), met['projection']))
            assert bool(status[0]) and bool(status[1]), status
            assert float(status[2]) <= r.SETTINGS['numeric_projection_relative_violation_max']
            if record:
                status_rows.append(status.tolist())
            extra = (raw, gradients)
        # Finite/status reductions synchronize the full updated model and Adam.
        old = (ck['model'], ck['opt_state'])
        ck.update(model=model, opt_state=state, iter=int(ck['iter']) + r.BS)
        ck['seeds']['train_data_seed'] = dk_next
        ck['seeds']['train_model_seed'] = mk_next
        return old, (met, model, state), batch, mk, extra

    if args.validate:
        old, result, batch, mk, extra = step(0, False)
        reference = r.sync(r.main.train_step(model=old[0], fwd_fn=r.FWD,
            optimizer=optimizer, opt_state=old[1], microbs=r.BS,
            weight_decay=0.0, x=batch['examples'], y=batch['labels'], key=mk))
        checks = {'adam_exact': r.exact(result[2], reference[2]),
                  'loss_exact': r.exact(result[0]['loss'], reference[0]['loss'])}
        assert all(checks.values()), checks
        if extra is None:
            checks['model_exact'] = r.exact(result[1], reference[1])
            assert checks['model_exact']
        else:
            raw, gradients = extra
            flat_raw = r.flat(raw)
            expected, diag = r.project_numpy(flat_raw,
                np.stack([r.flat(g) for g in gradients]), .25)
            # Compare the projected model directly with the NumPy update reference.
            expected_model = (r.flat(old[0]) + expected).astype(np.float32)
            err = float(np.max(np.abs(expected_model - r.flat(result[1]))))
            raw_err = float(np.max(np.abs((r.flat(old[0]) + flat_raw).astype(np.float32)
                                         - r.flat(reference[1]))))
            checks.update(projection_model_max_abs=err, raw_model_max_abs=raw_err,
                          constraint_count=len(gradients), head=head)
            assert err <= 2e-5 and raw_err <= 1e-4, checks
        write(out / 'validation.json',
              dict(passed=True, arm=args.arm, checks=checks, versions=versions,
                   elapsed_seconds=time.perf_counter()-started))
        print('VALIDATED', args.arm, checks, flush=True)
        return

    memory_before_warmup = jax.devices()[0].memory_stats()
    setup_seconds = time.perf_counter() - started
    warm_start = time.perf_counter()
    for i in range(CFG['warmup_steps']):
        step(i, False)
    warm_seconds = time.perf_counter() - warm_start
    print('WARM', args.arm, warm_seconds, flush=True)
    memory_after_warmup = jax.devices()[0].memory_stats()
    windows = []
    for w in range(CFG['windows']):
        t = time.perf_counter()
        for k in range(CFG['steps_per_window']):
            step(CFG['warmup_steps'] + w*CFG['steps_per_window'] + k)
        elapsed = time.perf_counter() - t
        windows.append(dict(window=w, steps=CFG['steps_per_window'],
                            seconds=elapsed, ms_per_step=elapsed*1000/CFG['steps_per_window']))
        write(out / 'progress.json', dict(arm=args.arm, round=args.round, windows=windows))
        if (w+1) % 5 == 0:
            r.save_checkpoint(out / 'latest_state.eqx', ck)
        print('WINDOW', args.arm, w, elapsed, flush=True)
    memory_end = jax.devices()[0].memory_stats()
    assert int(np.asarray(ck['opt_state'][0].count)) == int(ck['iter']) // r.BS
    measured = sum(x['seconds'] for x in windows)
    final_sha = r.save_checkpoint(out / 'final_state.eqx', ck)
    result = dict(passed=True, arm=args.arm, round=args.round, tag=args.tag, seed=seed, start_stage=4400000,
        final_checkpoint_sha256=final_sha,
        final_prng={k: np.asarray(v).tolist() for k,v in ck['seeds'].items()},
        memory_before_warmup=memory_before_warmup,
        end_stage=int(ck['iter']), versions=versions,
        device=str(jax.devices()[0]), setup_seconds=setup_seconds,
        warmup_seconds=warm_seconds, measured_seconds=measured, windows=windows,
        ms_per_step=measured*1000/(CFG['windows']*CFG['steps_per_window']),
        memory_after_warmup=memory_after_warmup, memory_end=memory_end,
        status_rows=status_rows, elapsed_seconds=time.perf_counter()-started,
        kernel_source='unchanged exp033 runtime.modified_step and author main.train_step',
        timing_includes='PRNG, real data sampler, method update, finite/projection checks and synchronization',
        timing_excludes='setup, warmup/JIT, evaluation, checkpoints, path discovery and confirmation',
        environment={k:os.environ.get(k) for k in ['XLA_PYTHON_CLIENT_PREALLOCATE',
          'XLA_PYTHON_CLIENT_ALLOCATOR', 'JAX_PLATFORMS','OMP_NUM_THREADS']})
    write(out / 'result.json', result)
    print('DONE', args.arm, result['ms_per_step'], memory_end, flush=True)


if __name__ == '__main__':
    main()

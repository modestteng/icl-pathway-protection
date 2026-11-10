"""Queue unchanged author kernels; commit only after the full chunk is checked.

exp039 engine.py with two explicit arguments instead of globals: the arm's own
fork stage (offset of the protection batch cycle) and the protected path set.
"""
import science as R
from common import *


@eqx.filter_jit
def split_keys(data_key, model_key):
    dn, dk = jax.random.split(data_key)
    mn, mk = jax.random.split(model_key)
    return dn, dk, mn, mk


@eqx.filter_jit
def packed_status(model, state, metrics, kind):
    losses = (metrics['loss'], metrics.get('protection_loss', jnp.asarray([])))
    finite = R.finite_candidate(model, (state, losses)).astype(jnp.float32)
    if kind == 'A':
        return jnp.stack((finite, jnp.float32(1), jnp.float32(0), jnp.float32(1), jnp.float32(0), jnp.float32(0)))
    d = metrics['projection']
    return jnp.stack((finite, d['success'].astype(jnp.float32), d['relative_constraint_violation'],
                      d['update_norm_ratio'], d['correction_norm_ratio'], d['projected'].astype(jnp.float32)))


def queue_steps(ck, count, kind, rho, path_ids, sample, optimizer, cache, start_stage, audit_offsets=()):
    """Run ``count`` author steps (A) or protected steps (C/P); validate before commit."""
    current = dict(ck, seeds=dict(ck['seeds']))
    pending, audit = [], None
    for _ in range(count):
        old = current
        dn, dk, mn, mk = split_keys(old['seeds']['train_data_seed'], old['seeds']['train_model_seed'])
        batch = sample(dk)
        offset = (int(old['iter']) - int(start_stage)) // BS
        if kind == 'A':
            metrics, model, state = main.train_step(model=old['model'], fwd_fn=R.FWD, optimizer=optimizer,
                opt_state=old['opt_state'], microbs=BS, weight_decay=0., x=batch['examples'],
                y=batch['labels'], key=mk)
        else:
            px, py = cache[offset % len(cache)]
            metrics, model, state, raw, gradients = R.modified_step(old['model'], old['opt_state'],
                batch['examples'], batch['labels'], mk, px, py, kind, rho, path_ids, optimizer)
            if offset in audit_offsets:
                assert count == 1, 'audit updates remain individual transaction boundaries'
                audit = (int(old['iter']), old['model'], model, raw, gradients, px, py)
        pending.append(packed_status(model, state, metrics, kind))
        keys = dict(old['seeds'], train_data_seed=dn, train_model_seed=mn)
        current = dict(old, model=model, opt_state=state, iter=int(old['iter']) + BS, seeds=keys)
    status = np.asarray(jax.device_get(jnp.stack(pending)))
    valid = ((status[:, 0] == 1) & (status[:, 1] == 1) & np.isfinite(status).all(axis=1)
             & (status[:, 2] <= SETTINGS['numeric_projection_relative_violation_max']))
    if not valid.all():
        raise RuntimeError('Uncommitted chunk rejected; first invalid offset=' + str(int(np.flatnonzero(~valid)[0]))
                           + ' status=' + repr(status.tolist()))
    return current, status, audit


def serial_steps(ck, count, kind, rho, path_ids, sample, optimizer, protect, start_stage):
    """Reference one-step-at-a-time loop with host readback after every step."""
    current = dict(ck, seeds=dict(ck['seeds']))
    statuses = []
    for _ in range(count):
        dn, dk = jax.random.split(current['seeds']['train_data_seed'])
        mn, mk = jax.random.split(current['seeds']['train_model_seed'])
        batch = sample(dk)
        offset = (int(current['iter']) - int(start_stage)) // BS
        if kind == 'A':
            metrics, model, state = main.train_step(model=current['model'], fwd_fn=R.FWD, optimizer=optimizer,
                opt_state=current['opt_state'], microbs=BS, weight_decay=0., x=batch['examples'],
                y=batch['labels'], key=mk)
            row = [1., 1., 0., 1., 0., 0.]
            finite = bool(np.asarray(R.finite_candidate(model, (state, (metrics['loss'], jnp.asarray([]))))))
        else:
            px, py = R.protect_batch(protect, offset)
            metrics, model, state, raw, grads = R.modified_step(current['model'], current['opt_state'],
                batch['examples'], batch['labels'], mk, px, py, kind, rho, path_ids, optimizer)
            d = metrics['projection']
            finite = bool(np.asarray(R.finite_candidate(model, (state, (metrics['loss'], metrics['protection_loss'])))))
            row = [1., float(np.asarray(d['success'])), float(np.asarray(d['relative_constraint_violation'])),
                   float(np.asarray(d['update_norm_ratio'])), float(np.asarray(d['correction_norm_ratio'])),
                   float(np.asarray(d['projected']))]
        row[0] = float(finite)
        assert row[0] and row[1] and row[2] <= SETTINGS['numeric_projection_relative_violation_max']
        statuses.append(row)
        current = dict(current, model=model, opt_state=state, iter=int(current['iter']) + BS,
                       seeds=dict(current['seeds'], train_data_seed=dn, train_model_seed=mn))
    return current, np.asarray(statuses, np.float32)

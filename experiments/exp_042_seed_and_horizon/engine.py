"""Bounded queues of unchanged author/exp036 steps; original fork-relative indexing."""
from common import *
import science as R


@eqx.filter_jit
def split_keys(data_key, model_key):
    dn, dk = jax.random.split(data_key)
    mn, mk = jax.random.split(model_key)
    return dn, dk, mn, mk


def queue_steps(ck, count, kind, head, sample, optimizer, protection, fork_stage):
    current = dict(ck, seeds=dict(ck['seeds']))
    pending = []
    for _ in range(count):
        dn, dk, mn, mk = split_keys(current['seeds']['train_data_seed'], current['seeds']['train_model_seed'])
        batch = sample(dk)
        if kind == 'A':
            metrics, model, state = main.train_step(model=current['model'], fwd_fn=R.FWD,
                optimizer=optimizer, opt_state=current['opt_state'], microbs=BS, weight_decay=0.,
                x=batch['examples'], y=batch['labels'], key=mk)
            status = jnp.stack((R.finite_candidate(model, (state, metrics['loss'])).astype(jnp.float32),
                               jnp.float32(1), jnp.float32(0), jnp.float32(1), jnp.float32(0), jnp.float32(0)))
        elif kind == 'P':
            offset = (int(current['iter']) - fork_stage) // BS
            px, py = protection[offset % len(protection)]
            metrics, model, state, _, _ = R.modified_step(current['model'], current['opt_state'],
                batch['examples'], batch['labels'], mk, px, py, 'P', .25, int(head), optimizer)
            status = R.packed_status(model, state, (metrics['loss'], metrics['protection_loss']), metrics['projection'])
        else:
            raise ValueError('only A and P are permitted')
        pending.append(status)
        current = dict(current, model=model, opt_state=state, iter=int(current['iter']) + BS,
                       seeds=dict(current['seeds'], train_data_seed=dn, train_model_seed=mn))
    values = np.asarray(jax.device_get(jnp.stack(pending)))
    if not (np.isfinite(values).all() and np.all(values[:, 0] == 1) and np.all(values[:, 1] == 1)
            and np.all(values[:, 2] <= SETTINGS['numeric_projection_relative_violation_max'])):
        raise RuntimeError('uncommitted queue rejected: nonfinite or numerical constraint failure')
    return current, values


def serial_steps(ck, count, kind, head, sample, optimizer, protection, fork_stage):
    current = dict(ck, seeds=dict(ck['seeds']))
    for _ in range(count):
        dn, dk = jax.random.split(current['seeds']['train_data_seed'])
        mn, mk = jax.random.split(current['seeds']['train_model_seed'])
        batch = sample(dk)
        if kind == 'A':
            metrics, model, state = main.train_step(model=current['model'], fwd_fn=R.FWD,
                optimizer=optimizer, opt_state=current['opt_state'], microbs=BS, weight_decay=0.,
                x=batch['examples'], y=batch['labels'], key=mk)
            assert bool(np.asarray(R.finite_candidate(model, (state, metrics['loss']))))
        else:
            offset = (int(current['iter']) - fork_stage) // BS
            px, py = R.protect_batch(protection, offset)
            metrics, model, state, _, _ = R.modified_step(current['model'], current['opt_state'],
                batch['examples'], batch['labels'], mk, px, py, 'P', .25, int(head), optimizer)
            assert bool(np.asarray(metrics['projection']['success']))
            assert float(np.asarray(metrics['projection']['relative_constraint_violation'])) <= SETTINGS['numeric_projection_relative_violation_max']
            assert bool(np.asarray(R.finite_candidate(model, (state, metrics['loss'], metrics['protection_loss']))))
        current = dict(current, model=model, opt_state=state, iter=int(current['iter']) + BS,
                       seeds=dict(current['seeds'], train_data_seed=dn, train_model_seed=mn))
    return current

"""Author-structure coupled baselines; fixed capture interface on every update.

Production keeps the original FP32/default matmul semantics.  Mathematical
cross-graph verification uses a separate process and explicit float32 matmul.
"""
from common import *

FWD = opto.make_fn_from_opts(load_opts())


def tree_dot(a, b):
    return sum(jnp.vdot(x, y).real for x, y in zip(arrays(a), arrays(b)))


def tree_norm(a):
    return jnp.sqrt(tree_dot(a, a))


def squared_l2(model):
    # The present author Transformer has no BatchNorm.  Do not silently include
    # BatchNorm variables if a different architecture is supplied later.
    for leaf in jax.tree_util.tree_leaves(model, is_leaf=lambda v: isinstance(v, eqx.nn.BatchNorm)):
        if isinstance(leaf, eqx.nn.BatchNorm):
            raise RuntimeError('BatchNorm exclusion requires a separately reviewed model adapter')
    return sum(jnp.sum(v * v) for v in arrays(eqx.filter(model, eqx.is_inexact_array)))


def query_ce(model, x, y, keys):
    pred = jax.vmap(partial(FWD, model=model))(x=x, y=y, key=keys)['out']
    return main.ce(pred[:, -1, :], y[:, -1]).mean()


def replay_keys():
    # Same fixed, per-task independent protection keys as exp030/031.  Dropout is
    # zero, but retaining the explicit split avoids hidden PRNG changes.
    return jnp.stack([jax.random.split(jax.random.fold_in(KEY, i),
                                      SETTINGS['protection_bs_per_task'])
                      for i in range(len(SETTINGS['protection_tasks']))])


def replay_ce(model, px, py):
    keys = replay_keys()
    losses = jnp.stack([query_ce(model, px[i], py[i], keys[i])
                        for i in range(len(SETTINGS['protection_tasks']))])
    return losses.mean(), losses


def _loss_components(model, fwd_fn, weight_decay, x, y, keys, px, py, arm, coefficient):
    pred_y = jax.vmap(partial(fwd_fn, model=model))(x=x, y=y, key=keys)['out']
    query_ce = main.ce(pred_y[:, -1, :], y[:, -1])
    # Verbatim author's expression, including its zero-weight term.
    weight_norm = jnp.sum(jnp.stack(jax.tree_map(lambda x: jnp.linalg.norm(jnp.where(x != 0, x, 0)), jax.tree_leaves(eqx.filter(model, eqx.is_array)))))
    original = query_ce.mean() + weight_decay * weight_norm
    if coefficient == 0.0:
        return original, dict(train_ce=query_ce.mean(), train_logits=pred_y[:, -1, :],
                              train_rows=query_ce, squared_l2=jnp.asarray(0.0))
    if arm == 'REPLAY':
        protection_losses, logits, row_losses = [], [], []
        pkeys = replay_keys()
        for idx in range(2):
            output = jax.vmap(partial(fwd_fn, model=model))(
                x=px[idx], y=py[idx], key=pkeys[idx])['out'][:, -1, :]
            rows = main.ce(output, py[idx, :, -1])
            protection_losses.append(rows.mean())
            logits.append(output)
            row_losses.append(rows)
        # Scaling is inside AD for every term; do not scale an already-computed
        # low-precision gradient as an alleged numerically identical reference.
        total = (original + .5*coefficient*protection_losses[0]
                          + .5*coefficient*protection_losses[1])/(1.0+coefficient)
        return total, dict(train_ce=query_ce.mean(), train_logits=pred_y[:, -1, :],
            train_rows=query_ce, protection_ce=jnp.stack(protection_losses),
            protection_logits=jnp.stack(logits), protection_rows=jnp.stack(row_losses),
            squared_l2=jnp.asarray(0.0))
    if arm == 'L2':
        norm = squared_l2(model)
        return original+coefficient*norm, dict(train_ce=query_ce.mean(),
            train_logits=pred_y[:, -1, :], train_rows=query_ce, squared_l2=norm)
    raise ValueError('unsupported candidate arm: '+arm)


@eqx.filter_value_and_grad
def compute_loss(model, fwd_fn, weight_decay, x, y, keys, px, py, arm, coefficient):
    loss, _ = _loss_components(model, fwd_fn, weight_decay, x, y, keys, px, py, arm, coefficient)
    return loss


@eqx.filter_value_and_grad(has_aux=True)
def compute_loss_diagnostics(model, fwd_fn, weight_decay, x, y, keys, px, py, arm, coefficient):
    return _loss_components(model, fwd_fn, weight_decay, x, y, keys, px, py, arm, coefficient)


@eqx.filter_jit
def native_train_step(model, fwd_fn, optimizer, opt_state, microbs, weight_decay,
               x, y, key, px, py, arm, coefficient, include_diagnostics=False):
    # The following training structure and original three metrics are copied
    # from frozen main.train_step, with only its scalar-loss call adapted.
    keys = jax.random.split(key, x.shape[0])

    losses = []
    grads = []
    lengths = []
    diagnostic_aux = []
    for i in range(0, x.shape[0], microbs):
        use_x = x[i:i+microbs]
        use_y = y[i:i+microbs]
        use_keys = keys[i:i+microbs]
        if include_diagnostics:
            (mini_loss, aux), mini_grad = compute_loss_diagnostics(model, fwd_fn,
                weight_decay, use_x, use_y, use_keys, px, py, arm, coefficient)
            diagnostic_aux.append(aux)
        else:
            mini_loss, mini_grad = compute_loss(model, fwd_fn, weight_decay,
                use_x, use_y, use_keys, px, py, arm, coefficient)
        losses.append(mini_loss)
        grads.append(mini_grad)

    loss = jnp.mean(jnp.array(losses))
    avg_fn = lambda *args: jnp.mean(jnp.array(args),axis=0)
    norm_fn = lambda model_like: jnp.sqrt(jnp.sum(jnp.array([jnp.sum(arr) for arr in jax.tree_util.tree_flatten(model_like)[0]])))

    grad = jax.tree_map(avg_fn, *grads)
    grad_norm = norm_fn(jax.tree_map(lambda x: x**2, grad))
    grads_dev = [jax.tree_map(lambda x,y: (x-y)**2, g, grad) for g in grads]
    grad_batch_var = jax.tree_map(avg_fn, *grads_dev)
    grad_batch_stddev = norm_fn(grad_batch_var)

    update, opt_state = optimizer.update(grad, opt_state)
    model = eqx.apply_updates(model, update)
    metrics = {'loss': loss, 'grad_norm': grad_norm, 'grad_batch_stddev': grad_batch_stddev}
    if include_diagnostics:
        return metrics, model, opt_state, dict(gradient=grad, update=update,
                                               microbatches=tuple(diagnostic_aux))
    return metrics, model, opt_state



def baseline_step(model, state, x, y, key, px, py, arm, coefficient, optimizer):
    if not isinstance(coefficient, (float, int)) or coefficient < 0:
        raise ValueError('sealed nonnegative coefficient required')
    # Every formal update uses this exact capture interface; no auditing graph.
    metrics, new_model, new_state, captured = native_train_step(model, FWD, optimizer,
        state, BS, 0.0, x, y, key, px, py, arm, float(coefficient), True)
    assert len(captured['microbatches']) == 1
    actual = captured['microbatches'][0]
    train = actual['train_ce']
    zero = jnp.asarray(0.0, dtype=train.dtype)
    if arm == 'REPLAY' and coefficient != 0.0:
        per_task = actual['protection_ce']
        replay = per_task.mean()
        penalty = coefficient*replay/(1.0+coefficient)
    else:
        per_task = jnp.zeros(2, dtype=train.dtype)
        replay = zero
        penalty = coefficient*actual['squared_l2'] if arm == 'L2' else zero
    metrics = dict(metrics, train_ce=train, replay_ce=replay,
        replay_task_ce=per_task, squared_l2=actual['squared_l2'],
        regularization_loss=penalty, g_used=captured['gradient'], actual=actual)
    return metrics, new_model, new_state


@eqx.filter_jit
def packed_status(model, state, metrics):
    scalars = tuple(metrics[k] for k in ('loss','train_ce','replay_ce','squared_l2',
                                        'regularization_loss','grad_norm'))
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(v))
        for v in arrays((model, state, metrics['g_used'], scalars))]))
    return jnp.stack([finite.astype(jnp.float32), *scalars])

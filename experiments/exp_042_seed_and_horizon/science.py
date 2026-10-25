"""Inherited exp036 scientific functions. Orchestration is in worker.py.
Only A and P25 are scheduled; unused legacy numerical branches do not create arms.
"""
from common import *
from projection import project_tree, project_numpy
from mechanism import path_loss, patch_example, random_control_head
from statistics import NormalDist
import math
FWD = opto.make_fn_from_opts(load_opts())

def audit_indices(n):
    return np.asarray([v for v in SETTINGS['audit_rows'] if v < n], np.int32)

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
            out, raw, logp = eval_batch(model, jnp.asarray(x[lo:hi]), jnp.asarray(y[lo:hi]),
                                       pure=task == 'pure_iwl')
            for key, value in out.items():
                arr = np.asarray(value)
                if not np.isfinite(arr).all():
                    raise RuntimeError('nonfinite evaluation')
                collected.setdefault(key, []).append(arr)
            take = ar[(ar >= lo) & (ar < hi)] - lo
            if len(take):
                raw_rows.extend(np.asarray(raw)[take])
                logp_rows.extend(np.asarray(logp)[take])
        for key, value in collected.items():
            record[task + '/' + key] = np.concatenate(value)
        record[task + '/labels'] = np.asarray(y, np.int32)
        record[task + '/audit_rows'] = ar
        record[task + '/audit_logits'] = np.asarray(raw_rows)
        record[task + '/audit_log_probs'] = np.asarray(logp_rows)
        if ids is not None:
            record[task + '/identities'] = ids[task]
    return record

@eqx.filter_jit
@eqx.filter_value_and_grad
def functional_objective(model, x, y, keys, head):
    with jax.default_matmul_precision('float32'):
        return path_loss(model, x, y, keys, head)

def tree_norm(tree):
    return jnp.sqrt(sum(jnp.vdot(v, v).real for v in arrays(tree)))

@eqx.filter_jit
def finite_candidate(model, state):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in arrays((model,state))]))

@eqx.filter_jit
def task_objective(model, x, y, keys):
    return main.compute_loss(model, FWD, 0.0, x, y, keys)

@eqx.filter_jit
def adam_update_from_state(new_state):
    # Pinned Optax0.2.2 pure default Adam; original moments/count remain intact.
    from optax._src import transform
    adam = new_state[0]
    mu_hat = transform.bias_correction(adam.mu, 0.9, adam.count)
    nu_hat = transform.bias_correction(adam.nu, 0.999, adam.count)
    normalized = jax.tree_util.tree_map(
        lambda m, v: m / (jnp.sqrt(v + 0.0) + 1e-8), mu_hat, nu_hat)
    raw, _ = optax.scale(-load_opts().lr).update(normalized, new_state[1])
    return raw

@eqx.filter_jit
def official_optimizer_update(grad, state, optimizer):
    return optimizer.update(grad, state)

@eqx.filter_jit
def constrain_update(model, raw, gradients, kind, rho):
    if kind == 'B':
        new = eqx.tree_at(embedding_leaves, raw,
                         tuple(jnp.zeros_like(v) for v in embedding_leaves(raw)))
        correction = jax.tree_util.tree_map(lambda a,b: a-b, new, raw)
        denom = jnp.maximum(tree_norm(raw), jnp.finfo(jnp.float32).tiny)
        diag = dict(success=jnp.asarray(True), update_norm_ratio=tree_norm(new)/denom,
                    correction_norm_ratio=tree_norm(correction)/denom, projected=jnp.asarray(False),
                    normalized_constraint_violation=jnp.asarray(0.0), relative_constraint_violation=jnp.asarray(0.0), constraint_violation=jnp.asarray(0.0))
    else:
        new, diag = project_tree(raw, gradients, rho)
    new_model = eqx.apply_updates(model, new)
    effective = jax.tree_util.tree_map(lambda a,b: a-b, eqx.filter(new_model, eqx.is_array),
                                       eqx.filter(model, eqx.is_array))
    diag['applied_embedding_update_norm'] = tree_norm(embedding_leaves(effective))
    other = eqx.tree_at(embedding_leaves, effective,
                       tuple(jnp.zeros_like(v) for v in embedding_leaves(effective)))
    diag['applied_other_update_norm'] = tree_norm(other)
    return new_model, diag

def modified_step(model, state, x, y, key, px, py, kind, rho, head, optimizer):
    # Host orchestration is intentional: never fuse protection with author Adam.
    metrics, author_model, new_state = main.train_step(model=model, fwd_fn=FWD,
        optimizer=optimizer, opt_state=state, microbs=BS, weight_decay=0.0,
        x=x, y=y, key=key)
    raw = adam_update_from_state(new_state)
    losses, gradients = [], []
    if kind in ('C','D','R'):
        for i in range(2):
            keys = jax.random.split(jax.random.fold_in(KEY, i), px.shape[1])
            loss, grad = task_objective(model, px[i], py[i], keys)
            losses.append(loss)
            gradients.append(grad)
    if kind in ('D','R','P'):
        xx, yy = px.reshape(-1,3,512), py.reshape(-1,3)
        loss, grad = functional_objective(model, xx, yy,
            jax.random.split(KEY, len(xx)), head)
        losses.append(loss)
        gradients.append(grad)
    if kind == 'P':
        assert len(gradients) == len(losses) == 1
    new_model, diag = constrain_update(model, raw, tuple(gradients), kind, rho)
    metrics = dict(metrics, projection=diag, protection_loss=jnp.asarray(losses))
    return metrics, new_model, new_state, raw, tuple(gradients)

def protect_batch(data, step):
    if isinstance(data, list):
        return data[step % len(data)]
    take = (np.arange(SETTINGS['protection_bs_per_task'])
            + step * SETTINGS['protection_bs_per_task']) % SETTINGS['data_sets']['protection']['n']
    xy = [data[t] for t in SETTINGS['protection_tasks']]
    return (jnp.asarray(np.stack([x[take] for x, _ in xy])),
            jnp.asarray(np.stack([y[take] for _, y in xy])))

@eqx.filter_jit
def packed_status(model, state, losses, diag):
    return jnp.stack((finite_candidate(model, (state, losses)).astype(jnp.float32),
        diag['success'].astype(jnp.float32), diag['relative_constraint_violation'],
        diag['update_norm_ratio'], diag['correction_norm_ratio'],
        diag['projected'].astype(jnp.float32)))

def stage_on_gpu(data):
    return {task: tuple(jax.device_put(v) for v in xy) for task, xy in data.items()}

def cache_protection(data):
    count = SETTINGS['data_sets']['protection']['n']//SETTINGS['protection_bs_per_task']
    cached = [protect_batch(data, step) for step in range(count)]
    sync(cached)
    return cached

def flat(tree):
    return np.concatenate([np.asarray(a).ravel() for a in arrays(tree)])

def cluster_test(values, classes, threshold=0.0):
    v, cls = np.asarray(values, np.float64), np.asarray(classes)
    mean = float(v.mean())
    unique, inverse = np.unique(cls, return_inverse=True)
    g = len(unique)
    residual = np.bincount(inverse, weights=v - mean, minlength=g)
    se = math.sqrt(g / (g - 1) * float(residual @ residual) / len(v) ** 2) if g > 1 else float('inf')
    p = .5 * math.erfc((mean - threshold) / (se * math.sqrt(2))) if se > 0 and math.isfinite(se) else 1.0
    valid = math.isfinite(se) and se > 1e-15
    return dict(mean=mean, se=se, lower95=mean - NormalDist().inv_cdf(.95) * se if valid else None,
                upper95=mean + NormalDist().inv_cdf(.95) * se if valid else None, p=p, threshold=threshold,
                classes=g, rows=len(v))

def holm(values):
    order = sorted(range(len(values)), key=lambda k: values[k])
    result, current = [1.0] * len(values), 0.0
    for rank, k in enumerate(order):
        current = max(current, min(1.0, (len(values) - rank) * values[k]))
        result[k] = current
    return result

@eqx.filter_jit
def mechanism_batch(model, x, y, head, include_logits, include_cache):
    with jax.default_matmul_precision('float32'):
        return mechanism_batch_inner(model,x,y,head,include_logits,include_cache)

def mechanism_batch_inner(model, x, y, head, include_logits, include_cache):
    out = jax.vmap(partial(patch_example, model, include_cache=include_cache), in_axes=(0, 0, 0, None))(
        x, y, jax.random.split(KEY, len(x)), head)
    dt, nt, rows = out['donor_target'], out['native_target'], jnp.arange(len(x))
    metrics = dict(pair_valid=out['pair_valid'], native_target=nt, donor_target=dt)
    for name, z in out.items():
        if not name.endswith('_logits'):
            continue
        lp = jax.nn.log_softmax(z, axis=-1)
        candidates = z[rows[:, None], out['context_labels']]
        pair_pred = jnp.where(candidates[:, 0] > candidates[:, 1], out['context_labels'][:, 0],
            jnp.where(candidates[:, 0] < candidates[:, 1], out['context_labels'][:, 1],
                      jnp.minimum(out['context_labels'][:, 0], out['context_labels'][:, 1])))
        stem = name[:-7]
        metrics[stem + '/donor_ce'] = -lp[rows, dt]
        metrics[stem + '/native_ce'] = -lp[rows, nt]
        metrics[stem + '/donor_margin'] = z[rows, dt] - z[rows, nt]
        metrics[stem + '/donor_pair_acc'] = pair_pred == dt
        metrics[stem + '/native_pair_acc'] = pair_pred == nt
        metrics[stem + '/native_full_acc'] = jnp.argmax(z, axis=1) == nt
        metrics[stem + '/donor_full_acc'] = jnp.argmax(z, axis=1) == dt
        metrics[stem + '/full_pred'] = jnp.argmax(z, axis=1)
        metrics[stem + '/pair_pred'] = pair_pred
    for branch in ['audit', 'reverse_audit']:
        for name, value in out[branch].items():
            metrics[branch + '/' + name] = value
    logits = {k: v for k, v in out.items() if k.endswith('_logits')} if include_logits else {}
    return metrics, logits, out.get('audit_cache', {})

def mechanism_evaluate(model, data, ids, head):
    record, max_audit = {}, {}
    for task in SETTINGS['protection_tasks']:
        x, y = data[task]
        collect, saved_logits, saved_cache = {}, {}, {}
        ar = audit_indices(len(x))
        for lo in range(0, len(x), SETTINGS['mechanism_bs']):
            hi = min(lo + SETTINGS['mechanism_bs'], len(x))
            take = ar[(ar >= lo) & (ar < hi)] - lo
            met, logits, cache = mechanism_batch(model, jnp.asarray(x[lo:hi]), jnp.asarray(y[lo:hi]),
                                                  head, bool(len(take)), lo < 128)
            for name, values in met.items():
                values = np.asarray(values)
                assert np.isfinite(values).all()
                collect.setdefault(name, []).append(values)
                if name.startswith(('audit/', 'reverse_audit/')):
                    max_audit[name] = max(max_audit.get(name, 0.0), float(values.max()))
            for name, values in logits.items():
                saved_logits.setdefault(name, []).extend(np.asarray(values)[take])
            for name, values in cache.items():
                saved_cache.setdefault(name, []).append(np.asarray(values)[:max(0,128-lo)])
        for name, values in collect.items():
            record[task + '/' + name] = np.concatenate(values)
        for name, values in saved_logits.items():
            record[task + '/audit_' + name] = np.asarray(values)
        for name, values in saved_cache.items():
            record[task + '/cache/' + name] = np.concatenate(values)
        record[task + '/cache_rows'] = np.arange(min(128,len(x)), dtype=np.int32)
        record[task + '/audit_rows'] = ar
        record[task + '/identities'] = ids[task]
    return record, max_audit

def structure_passed(audit):
    exact_keys = {'embedding_max_abs','l1_upstream_max_abs','l1_values_pre_proj_max_abs',
        'l1_values_post_proj_max_abs','l2_norm_input_max_abs','l2_raw_qkv_max_abs','q_max_abs',
        'other_channel_max_abs','nonlabel_channel_max_abs','target_channel_max_abs',
        'hybrid_nonlabel_residual_max_abs','query_input_max_abs'}
    special = {'qk_recompute_max_abs': SETTINGS['numeric_path_qk_max_abs'],
        'future_attention_max': SETTINGS['numeric_path_future_attention_max'],
        'attention_row_sum_max_abs': SETTINGS['numeric_path_rowsum_max_abs']}
    for name, value in audit.items():
        base = name.split('/')[-1]
        if not math.isfinite(value):
            return False
        if base in exact_keys:
            if value != 0.0:
                return False
        elif value > special.get(base, SETTINGS['numeric_sham_logit_max_abs']):
            return False
    return True

def protection_gradients(model, px, py, kind, head):
    losses, gradients = [], []
    if kind == 'C':
        for k in range(2):
            value, grad = task_objective(model,px[k],py[k],
                jax.random.split(jax.random.fold_in(KEY,k),px.shape[1]))
            losses.append(float(np.asarray(value))); gradients.append(grad)
    elif kind == 'P':
        xx,yy=px.reshape(-1,3,512),py.reshape(-1,3)
        value,grad=functional_objective(model,xx,yy,jax.random.split(KEY,len(xx)),head)
        losses.append(float(np.asarray(value))); gradients.append(grad)
    else:
        raise ValueError('Only C or P gradients are defined')
    return np.asarray(losses), gradients

def audit_update(seed,arm,stage,old,new,raw,gradients,px,py,monitor,head,kind):
    """Prespecified loss-change diagnostic; never changes accepted update/rho."""
    raw_model=eqx.apply_updates(old,raw)
    actual_update=jax.tree_util.tree_map(lambda a,b:a-b,eqx.filter(new,eqx.is_array),
                                         eqx.filter(old,eqx.is_array))
    raw_applied=jax.tree_util.tree_map(lambda a,b:a-b,eqx.filter(raw_model,eqx.is_array),
                                      eqx.filter(old,eqx.is_array))
    results={}
    mx=jnp.asarray(np.stack([monitor[t][0][:SETTINGS['protection_bs_per_task']]
                            for t in SETTINGS['protection_tasks']]))
    my=jnp.asarray(np.stack([monitor[t][1][:SETTINGS['protection_bs_per_task']]
                            for t in SETTINGS['protection_tasks']]))
    for name,xx,yy in [('actual_protection_batch',px,py),('independent_monitor_batch',mx,my)]:
        before,gs=protection_gradients(old,xx,yy,kind,head)
        after_raw,_=protection_gradients(raw_model,xx,yy,kind,head)
        after_new,_=protection_gradients(new,xx,yy,kind,head)
        results[name]=dict(loss_before=before.tolist(),actual_raw_change=(after_raw-before).tolist(),
            actual_new_change=(after_new-before).tolist(),
            predicted_optimizer_raw=[float(flat(g).astype(np.float64)@flat(raw).astype(np.float64)) for g in gs],
            predicted_raw_applied=[float(flat(g).astype(np.float64)@flat(raw_applied).astype(np.float64)) for g in gs],
            predicted_new_applied=[float(flat(g).astype(np.float64)@flat(actual_update).astype(np.float64)) for g in gs])
    write_json(OUT/seed['id']/'update_audit'/f'{arm}_{stage:011d}.json',dict(
        seed=seed,arm=arm,stage=stage,head=head,kind=kind,results=results,
        code_manifest_sha256=code_identity(),adjusted_parameters=False))

"""Exp040 scientific operations: exp039 kernels with label-K path sets.

The author Adam graph, the projection algebra (projection.py, byte-identical to
exp036-exp039) and the mechanism confirmation thresholds are unchanged. The
only scientific generalisation is the protected object: a ranked set of
adjacent-layer source heads (path_set.py) instead of exactly one head. Nothing
executes on import except building the author forward function.
"""
from common import *
from projection import project_tree, project_numpy
import mechanism as M
import path_set as PS
from statistics import NormalDist
import inspect
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
    """Per-example full-label metrics for the three reported behaviour tasks."""
    record = {}
    for task, (x, y) in data.items():
        if task not in TASKS:
            continue
        collected, raw_rows, logp_rows = {}, [], []
        ar = audit_indices(len(x))
        for lo in range(0, len(x), SETTINGS['eval_bs']):
            hi = min(lo + SETTINGS['eval_bs'], len(x))
            out, raw, logp = eval_batch(model, jnp.asarray(x[lo:hi]), jnp.asarray(y[lo:hi]))
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


def summary_of(record):
    return {t: {'acc': float(np.asarray(record[t + '/acc'], np.float64).mean()),
                'loss': float(np.asarray(record[t + '/loss'], np.float64).mean()),
                'in_context_acc': float(np.asarray(record[t + '/in_context_acc'], np.float64).mean())}
            for t in TASKS}


@eqx.filter_jit
@eqx.filter_value_and_grad
def functional_objective(model, x, y, keys, path_ids):
    with jax.default_matmul_precision('float32'):
        return PS.set_loss(model, x, y, keys, path_ids)


def tree_norm(tree):
    return jnp.sqrt(sum(jnp.vdot(v, v).real for v in arrays(tree)))


@eqx.filter_jit
def finite_candidate(model, state):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in arrays((model, state))]))


@eqx.filter_jit
def task_objective(model, x, y, keys):
    return main.compute_loss(model, FWD, 0.0, x, y, keys)


@eqx.filter_jit
def adam_update_from_state(new_state):
    # Pinned Optax 0.2.2 default Adam; original moments/count remain intact.
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
def constrain_update(model, raw, gradients, rho):
    new, diag = project_tree(raw, gradients, rho)
    new_model = eqx.apply_updates(model, new)
    effective = jax.tree_util.tree_map(lambda a, b: a - b, eqx.filter(new_model, eqx.is_array),
                                       eqx.filter(model, eqx.is_array))
    diag['applied_embedding_update_norm'] = tree_norm(embedding_leaves(effective))
    other = eqx.tree_at(embedding_leaves, effective,
                       tuple(jnp.zeros_like(v) for v in embedding_leaves(effective)))
    diag['applied_other_update_norm'] = tree_norm(other)
    return new_model, diag


def modified_step(model, state, x, y, key, px, py, kind, rho, path_ids, optimizer):
    # Host orchestration is intentional: never fuse protection with author Adam.
    metrics, author_model, new_state = main.train_step(model=model, fwd_fn=FWD,
        optimizer=optimizer, opt_state=state, microbs=BS, weight_decay=0.0,
        x=x, y=y, key=key)
    raw = adam_update_from_state(new_state)
    losses, gradients = [], []
    if kind == 'C':
        for i in range(2):
            keys = jax.random.split(jax.random.fold_in(KEY, i), px.shape[1])
            loss, grad = task_objective(model, px[i], py[i], keys)
            losses.append(loss)
            gradients.append(grad)
    elif kind == 'P':
        xx, yy = px.reshape(-1, 3, 512), py.reshape(-1, 3)
        loss, grad = functional_objective(model, xx, yy, jax.random.split(KEY, len(xx)), tuple(path_ids))
        losses.append(loss)
        gradients.append(grad)
    else:
        raise ValueError('kind must be C or P')
    new_model, diag = constrain_update(model, raw, tuple(gradients), rho)
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


def cache_protection(data):
    count = SETTINGS['data_sets']['protection']['n'] // SETTINGS['protection_bs_per_task']
    cached = [protect_batch(data, step) for step in range(count)]
    sync(cached)
    return cached


def stage_on_gpu(data):
    return {task: tuple(jax.device_put(v) for v in xy) for task, xy in data.items()}


def flat(tree):
    return np.concatenate([np.asarray(a).ravel() for a in arrays(tree)])


# ----------------------------------------------------------------- statistics
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


# ------------------------------------------------------------------ mechanism
@eqx.filter_jit
def mechanism_batch(model, x, y, path_ids, include_logits, include_cache):
    with jax.default_matmul_precision('float32'):
        return mechanism_batch_inner(model, x, y, path_ids, include_logits, include_cache)


def mechanism_batch_inner(model, x, y, path_ids, include_logits, include_cache):
    out = jax.vmap(lambda a, b, k: PS.set_patch_example(model, a, b, k, path_ids, include_cache=include_cache))(
        x, y, jax.random.split(KEY, len(x)))
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


def _pad(array, size):
    """Repeat the last row up to a fixed batch size (rows are independent under vmap)."""
    array = np.asarray(array)
    if len(array) == size:
        return array
    return np.concatenate([array, np.repeat(array[-1:], size - len(array), axis=0)])


def mechanism_evaluate(model, data, ids, path_ids, tasks=None, audit_logits=True):
    """Per-row mechanism metrics. Fixed batch shape -> one compilation per path set.

    Differs from exp039 only in execution: the final partial batch is padded with
    copies of its last row (padding rows are discarded), and full logits of the
    prespecified audit rows come from one separate padded call. No cache arrays.
    """
    path_ids = tuple(int(p) for p in path_ids)
    bs = SETTINGS['mechanism_bs']
    record, max_audit = {}, {}
    for task in (tasks or SETTINGS['protection_tasks']):
        x, y = data[task]
        collect = {}
        for lo in range(0, len(x), bs):
            hi = min(lo + bs, len(x))
            met, _, _ = mechanism_batch(model, jnp.asarray(_pad(x[lo:hi], bs)), jnp.asarray(_pad(y[lo:hi], bs)),
                                        path_ids, False, False)
            for name, values in met.items():
                values = np.asarray(values)[:hi - lo]
                assert np.isfinite(values).all()
                collect.setdefault(name, []).append(values)
                if name.startswith(('audit/', 'reverse_audit/')):
                    max_audit[name] = max(max_audit.get(name, 0.0), float(values.max()))
        for name, values in collect.items():
            record[task + '/' + name] = np.concatenate(values)
        ar = audit_indices(len(x))
        if audit_logits and len(ar):
            _, logits, _ = mechanism_batch(model, jnp.asarray(_pad(x[ar], bs)), jnp.asarray(_pad(y[ar], bs)),
                                           path_ids, True, False)
            for name, values in logits.items():
                record[task + '/audit_' + name] = np.asarray(values)[:len(ar)]
        record[task + '/audit_rows'] = ar
        record[task + '/identities'] = ids[task]
    record['path_set'] = np.asarray(path_ids, np.int32)
    return record, max_audit


EXACT_KEYS = {'embedding_max_abs', 'l1_upstream_max_abs', 'l1_values_pre_proj_max_abs',
              'l1_values_post_proj_max_abs', 'l2_norm_input_max_abs', 'l2_raw_qkv_max_abs', 'q_max_abs',
              'other_channel_max_abs', 'nonlabel_channel_max_abs', 'target_channel_max_abs',
              'hybrid_nonlabel_residual_max_abs', 'query_input_max_abs'}


def structure_passed(audit):
    special = {'qk_recompute_max_abs': SETTINGS['numeric_path_qk_max_abs'],
               'future_attention_max': SETTINGS['numeric_path_future_attention_max'],
               'attention_row_sum_max_abs': SETTINGS['numeric_path_rowsum_max_abs']}
    for name, value in audit.items():
        base = name.split('/')[-1]
        if not math.isfinite(value):
            return False
        if base in EXACT_KEYS:
            if value != 0.0:
                return False
        elif value > special.get(base, SETTINGS['numeric_sham_logit_max_abs']):
            return False
    return True


def gain(record, task, stem):
    native = np.asarray(record[task + '/native/donor_ce'], np.float64)
    return native - np.asarray(record[task + '/' + stem + '/donor_ce'], np.float64)


def coverage(record):
    """min over protection tasks of mean(set gain)/mean(full donor gain)."""
    ratios = {}
    for t in SETTINGS['protection_tasks']:
        donor = float(gain(record, t, 'donor').mean())
        path = float(gain(record, t, 'path').mean())
        ratios[t] = path / donor if donor > 0 else None
    finite = [v for v in ratios.values() if v is not None]
    return (min(finite) if len(finite) == len(ratios) else None), ratios


def mechanism_summary(record, ids, audit, path_ids, tasks=None):
    """Unchanged exp036/exp039 confirmation contrasts, floors and Holm family."""
    tasks = tuple(tasks or SETTINGS['protection_tasks'])
    checks, floors = {}, {}
    for t in tasks:
        c = ids[t][:, -1, 0]
        donor, k, v = gain(record, t, 'donor'), gain(record, t, 'path'), gain(record, t, 'v_path')
        contrasts = [('donor_ce', donor, SETTINGS['mechanism_donor_ce_min']),
            ('path_ce', k, SETTINGS['mechanism_path_ce_min']),
            ('retention', k - SETTINGS['mechanism_retention_min'] * donor, 0.0),
            ('k_vs_v', k - v, SETTINGS['mechanism_k_vs_v_min']),
            ('pair_margin', np.asarray(record[t + '/path/donor_margin'], np.float64)
                - np.asarray(record[t + '/native/donor_margin'], np.float64),
                SETTINGS['mechanism_pair_margin_min'])]
        for name, value, threshold in contrasts:
            checks[t + '/' + name] = cluster_test(value, c, threshold)
        for field, floor in [('native/native_pair_acc', SETTINGS['mechanism_pair_acc_floor']),
                ('donor/donor_pair_acc', SETTINGS['mechanism_pair_acc_floor']),
                ('native/native_full_acc', SETTINGS['mechanism_full_acc_floor'])]:
            result = cluster_test(record[t + '/' + field], c, floor)
            result['passed'] = bool(math.isfinite(result['se']) and result['se'] > 1e-15
                                    and result['lower95'] > floor)
            floors[t + '/' + field] = result
        assert bool(record[t + '/pair_valid'].all())
        assert np.all(record[t + '/native_target'] != record[t + '/donor_target'])
    for name, p in zip(checks, holm([v['p'] for v in checks.values()])):
        checks[name].update(p_holm=p, passed=bool(p < SETTINGS['alpha']
            and checks[name]['mean'] > checks[name]['threshold']))
    structural = structure_passed(audit)
    for t in tasks:
        for pred in ['full_pred', 'pair_pred']:
            structural = structural and np.array_equal(record[t + '/self/' + pred], record[t + '/native/' + pred])
            structural = structural and np.array_equal(record[t + '/query_label_control/' + pred], record[t + '/native/' + pred])
    cov, ratios = coverage(record)
    return dict(passed=bool(structural and all(v['passed'] for v in checks.values())
                            and all(v['passed'] for v in floors.values())),
                path_set=[int(p) for p in path_ids], path_set_description=PS.describe(path_ids),
                checks=checks, floors=floors, structure_passed=bool(structural), audit_max=audit,
                coverage=cov, coverage_by_task=ratios, tasks=list(tasks))


def protection_gradients(model, px, py, kind, path_ids):
    losses, gradients = [], []
    if kind == 'C':
        for k in range(2):
            value, grad = task_objective(model, px[k], py[k],
                jax.random.split(jax.random.fold_in(KEY, k), px.shape[1]))
            losses.append(float(np.asarray(value)))
            gradients.append(grad)
    elif kind == 'P':
        xx, yy = px.reshape(-1, 3, 512), py.reshape(-1, 3)
        value, grad = functional_objective(model, xx, yy, jax.random.split(KEY, len(xx)), tuple(path_ids))
        losses.append(float(np.asarray(value)))
        gradients.append(grad)
    else:
        raise ValueError('Only C or P gradients are defined')
    return np.asarray(losses), gradients


def audit_update(directory, seed, arm, stage, old, new, raw, gradients, px, py, monitor, path_ids, kind):
    """Prespecified loss-change diagnostic; never changes the accepted update."""
    raw_model = eqx.apply_updates(old, raw)
    actual_update = jax.tree_util.tree_map(lambda a, b: a - b, eqx.filter(new, eqx.is_array),
                                           eqx.filter(old, eqx.is_array))
    raw_applied = jax.tree_util.tree_map(lambda a, b: a - b, eqx.filter(raw_model, eqx.is_array),
                                         eqx.filter(old, eqx.is_array))
    results = {}
    mx = jnp.asarray(np.stack([monitor[t][0][:SETTINGS['protection_bs_per_task']]
                               for t in SETTINGS['protection_tasks']]))
    my = jnp.asarray(np.stack([monitor[t][1][:SETTINGS['protection_bs_per_task']]
                               for t in SETTINGS['protection_tasks']]))
    for name, xx, yy in [('actual_protection_batch', px, py), ('independent_monitor_batch', mx, my)]:
        before, gs = protection_gradients(old, xx, yy, kind, path_ids)
        after_raw, _ = protection_gradients(raw_model, xx, yy, kind, path_ids)
        after_new, _ = protection_gradients(new, xx, yy, kind, path_ids)
        results[name] = dict(loss_before=before.tolist(), actual_raw_change=(after_raw - before).tolist(),
            actual_new_change=(after_new - before).tolist(),
            predicted_optimizer_raw=[float(flat(g).astype(np.float64) @ flat(raw).astype(np.float64)) for g in gs],
            predicted_raw_applied=[float(flat(g).astype(np.float64) @ flat(raw_applied).astype(np.float64)) for g in gs],
            predicted_new_applied=[float(flat(g).astype(np.float64) @ flat(actual_update).astype(np.float64)) for g in gs])
    write_json(directory / 'update_audit' / f'{arm}_{stage:011d}.json', dict(
        seed=seed, arm=arm, stage=stage, path_set=list(path_ids), kind=kind, results=results,
        code_manifest_sha256=code_identity(), adjusted_parameters=False))


# ------------------------------------------------------------------ preflight
def seed_preflight(seed, base, selected, random_set, receipt_path):
    """Per-seed numerical checks at the fork state before any protected update."""
    verify_inputs()
    data, _ = load_dataset('protection')
    _, optimizer = template(seed)
    sample = make_sampler(seed)
    _, dk = jax.random.split(base['seeds']['train_data_seed'])
    _, mk = jax.random.split(base['seeds']['train_model_seed'])
    batch = sync(sample(dk))
    assert exact(batch, sample.native(dk, sample.data))
    px, py = protect_batch(data, 0)
    reference = sync(main.train_step(model=base['model'], fwd_fn=FWD, optimizer=optimizer,
        opt_state=base['opt_state'], microbs=BS, weight_decay=0.0,
        x=batch['examples'], y=batch['labels'], key=mk))
    checks = {}
    rhos = sorted({d['rho'] for d in SETTINGS['arm_definitions'].values()})
    for kind, ids in [('C', ()), ('P', tuple(selected)), ('P', tuple(random_set))]:
        tag = kind + ('' if kind == 'C' else '_' + '-'.join(map(str, ids)))
        result = sync(modified_step(base['model'], base['opt_state'], batch['examples'],
            batch['labels'], mk, px, py, kind, 1.0, ids, optimizer))
        checks[tag + '_adam_state_exact'] = exact(result[2], reference[2])
        assert checks[tag + '_adam_state_exact']
        assert len(result[4]) == (2 if kind == 'C' else 1)
        error = maxdiff(eqx.apply_updates(base['model'], result[3]), reference[1])
        checks[tag + '_raw_adam_reconstructed_model_max_abs'] = error
        assert error <= SETTINGS['numeric_model_adam_max_abs']
        error = maxdiff(result[1], reference[1])
        checks[tag + '_rho1_model_max_abs'] = error
        assert error <= SETTINGS['numeric_model_adam_max_abs']
        assert bool(np.asarray(result[0]['projection']['success']))
        raw_ref, state_ref = sync(official_optimizer_update(result[4][0], base['opt_state'], optimizer))
        raw_rebuilt = sync(adam_update_from_state(state_ref))
        difference = flat(raw_rebuilt).astype(np.float64) - flat(raw_ref).astype(np.float64)
        checks[tag + '_official_raw_reconstruction_max_abs'] = float(np.max(np.abs(difference)))
        assert checks[tag + '_official_raw_reconstruction_max_abs'] <= SETTINGS['numeric_projection_reference_max_abs']
        for rho in rhos:
            projected, diag = project_tree(result[3], result[4], rho)
            vector, _ = project_numpy(flat(result[3]), np.stack([flat(g) for g in result[4]]), rho)
            diff = float(np.max(np.abs(flat(projected) - vector)))
            relative = float(np.linalg.norm(flat(projected).astype(np.float64) - vector)
                             / max(np.linalg.norm(flat(result[3]).astype(np.float64)), 1e-30))
            checks[f'{tag}_rho{rho}_numpy_projection_max_abs'] = diff
            checks[f'{tag}_rho{rho}_numpy_projection_relative'] = relative
            assert diff <= SETTINGS['numeric_projection_reference_max_abs']
            assert relative <= SETTINGS['numeric_projection_reference_max_relative']
            assert bool(np.asarray(diag['success']))
            constrained = sync(modified_step(base['model'], base['opt_state'], batch['examples'],
                batch['labels'], mk, px, py, kind, rho, ids, optimizer))
            assert maxdiff(constrained[1], eqx.apply_updates(base['model'], projected)) <= SETTINGS['numeric_model_adam_max_abs']
            assert exact(constrained[2], reference[2])
            violation = float(np.asarray(constrained[0]['projection']['relative_constraint_violation']))
            checks[f'{tag}_rho{rho}_jit_relative_violation'] = violation
            assert violation <= SETTINGS['numeric_projection_relative_violation_max']
    xx, yy = px.reshape(-1, 3, 512), py.reshape(-1, 3)
    fk = jax.random.split(KEY, len(xx))
    for ids in (tuple(selected), tuple(random_set)):
        loss, grad = functional_objective(base['model'], xx, yy, fk, ids)
        norm = float(np.asarray(tree_norm(grad)))
        assert np.isfinite(norm) and norm > 0
        direction = jax.tree_util.tree_map(lambda v: v / norm, grad)
        step = SETTINGS['numeric_functional_grad_fd_step']
        plus = eqx.apply_updates(base['model'], jax.tree_util.tree_map(lambda v: step * v, direction))
        minus = eqx.apply_updates(base['model'], jax.tree_util.tree_map(lambda v: -step * v, direction))
        lp, _ = functional_objective(plus, xx, yy, fk, ids)
        lm, _ = functional_objective(minus, xx, yy, fk, ids)
        fd = float(np.asarray(lp - lm)) / (2 * step)
        error = abs(fd - norm)
        assert error <= SETTINGS['numeric_functional_grad_absolute_tolerance'] + SETTINGS['numeric_functional_grad_relative_tolerance'] * norm
        checks['functional_derivative_' + '-'.join(map(str, ids))] = dict(autodiff=norm, finite_difference=fd, abs_error=error)
    confirm, cids = load_dataset('mechanism_confirm')
    rows = {t: (x[:128], y[:128]) for t, (x, y) in confirm.items()}
    rids = {t: v[:128] for t, v in cids.items()}
    for ids in (tuple(selected), tuple(random_set)):
        mr, audits = mechanism_evaluate(base['model'], rows, rids, ids)
        assert structure_passed(audits)
        for t in SETTINGS['protection_tasks']:
            f_loss, _ = functional_objective(base['model'], jnp.asarray(rows[t][0]), jnp.asarray(rows[t][1]),
                                             jax.random.split(KEY, len(rows[t][0])), ids)
            error = abs(float(np.asarray(f_loss)) - float(mr[t + '/path/donor_ce'].mean()))
            assert error <= SETTINGS['numeric_eval_mean_ce_max_abs']
            checks[f'functional_loss_vs_mechanism_{t}_' + '-'.join(map(str, ids))] = error
    monitor, _ = load_dataset('monitor')
    xy = {t: monitor[t] for t in TASKS}
    native = main.evaluate(model=base['model'], fwd_fn=FWD, key=KEY,
                           eval_data={k: dict(examples=jnp.asarray(x), labels=jnp.asarray(y)) for k, (x, y) in xy.items()},
                           eval_batch_size=SETTINGS['eval_bs'])
    adapted = evaluate(base['model'], xy)
    for t in xy:
        d = abs(float(np.asarray(native[t]['loss']).mean()) - float(adapted[t + '/loss'].mean()))
        rows_diff = int(np.count_nonzero(np.asarray(native[t]['acc']) != adapted[t + '/acc']))
        assert d <= SETTINGS['numeric_eval_mean_ce_max_abs'] and rows_diff <= SETTINGS['numeric_eval_acc_max_rows']
        checks[t + '_author_evaluator_ce_max_mean'] = d
        checks[t + '_author_evaluator_acc_rows'] = rows_diff
    write_json(receipt_path, dict(passed=True, checks=checks, seed=seed, selected=list(selected),
        random_set=list(random_set), fork_stage=int(base['iter']),
        code_manifest_sha256=code_identity(), model_forward_and_gradient_gpu_verified=True))
    print('SEED_PREFLIGHT_PASSED', seed['id'], flush=True)

"""Resumable single-GPU execution; no automatic dependency installation.

Entry: python runtime.py run. Every scientific phase requires sealed inputs,
CUDA and unchanged author numerical semantics. No work executes on import.
"""
from common import *
from projection import project_tree, project_numpy
from mechanism import path_loss, patch_example, random_control_head
from datasets import prepare as prepare_data
from functools import partial
from statistics import NormalDist
import argparse
import inspect
import math
import signal
import shutil
import traceback
from datetime import datetime, timezone

AUTHOR_STEP = inspect.unwrap(main.train_step)
FWD = opto.make_fn_from_opts(load_opts())
STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


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
        record[task + '/audit_rows'] = ar
        record[task + '/audit_logits'] = np.asarray(raw_rows)
        record[task + '/audit_log_probs'] = np.asarray(logp_rows)
        if ids is not None:
            record[task + '/identities'] = ids[task]
    return record


def original_data():
    with h5py.File(ASSETS / 'data/eval_data.h5', 'r') as f:
        return {t: (np.asarray(f[t]['examples'], np.float32),
                    np.asarray(f[t]['labels'], np.int32)) for t in (*TASKS, 'pure_iwl')}


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
    take = (np.arange(SETTINGS['protection_bs_per_task'])
            + step * SETTINGS['protection_bs_per_task']) % SETTINGS['data_sets']['protection']['n']
    xy = [data[t] for t in SETTINGS['protection_tasks']]
    return (jnp.asarray(np.stack([x[take] for x, _ in xy])),
            jnp.asarray(np.stack([y[take] for _, y in xy])))


def flat(tree):
    return np.concatenate([np.asarray(a).ravel() for a in arrays(tree)])


def preflight():
    p = verify_inputs()
    data, _ = load_dataset('protection')
    seed = SETTINGS['training_seeds'][0]
    base = base_checkpoint(seed)
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
    for kind in ['P']:
        print('PREFLIGHT_CHECK', kind, flush=True)
        head = 0 if kind != 'R' else 6
        result = sync(modified_step(base['model'], base['opt_state'], batch['examples'],
            batch['labels'], mk, px, py, kind, 1.0, head, optimizer))
        checks[kind + '_adam_max_abs'] = maxdiff(result[2], reference[2])
        checks[kind + '_adam_state_exact'] = exact(result[2], reference[2])
        checks[kind + '_constraint_count'] = len(result[4])
        print('PREFLIGHT_ADAM_ERROR', kind, checks[kind + '_adam_max_abs'], flush=True)
        assert checks[kind + '_adam_max_abs'] <= SETTINGS['numeric_model_adam_max_abs']
        assert exact(result[2], reference[2])
        reconstructed_model = eqx.apply_updates(base['model'], result[3])
        reconstruction_error = maxdiff(reconstructed_model, reference[1])
        checks[kind + '_raw_adam_reconstructed_model_max_abs'] = reconstruction_error
        assert reconstruction_error <= SETTINGS['numeric_model_adam_max_abs']
        if kind == 'B':
            assert exact(embedding_leaves(result[1]), embedding_leaves(base['model']))
            expected = eqx.tree_at(embedding_leaves, reference[1], embedding_leaves(base['model']))
            error = maxdiff(result[1], expected)
        else:
            error = maxdiff(result[1], reference[1])
        checks[kind + '_model_max_abs'] = error
        assert error <= SETTINGS['numeric_model_adam_max_abs']
        assert bool(np.asarray(result[0]['projection']['success']))
        if kind != 'B':
            raw_ref, state_ref = sync(official_optimizer_update(result[4][0], base['opt_state'], optimizer))
            raw_rebuilt = sync(adam_update_from_state(state_ref))
            raw_difference = flat(raw_rebuilt).astype(np.float64) - flat(raw_ref).astype(np.float64)
            raw_max_abs = float(np.max(np.abs(raw_difference)))
            raw_relative = float(np.linalg.norm(raw_difference) / max(np.linalg.norm(flat(raw_ref).astype(np.float64)), 1e-30))
            checks[kind + '_official_raw_reconstruction_max_abs'] = raw_max_abs
            checks[kind + '_official_raw_reconstruction_relative'] = raw_relative
            assert raw_max_abs <= SETTINGS['numeric_projection_reference_max_abs']
            assert raw_relative <= SETTINGS['numeric_projection_reference_max_relative']
            if kind == 'P':
                initial_state = optimizer.init(eqx.filter(base['model'], eqx.is_array))
                zero_grad = jax.tree_util.tree_map(jnp.zeros_like, result[4][0])
                for probe, gradient, source_state in [
                        ('first_count', result[4][0], initial_state),
                        ('zero_gradient_nonzero_moments', zero_grad, base['opt_state'])]:
                    expected_raw, after_state = sync(official_optimizer_update(gradient, source_state, optimizer))
                    rebuilt = sync(adam_update_from_state(after_state))
                    difference = flat(rebuilt).astype(np.float64) - flat(expected_raw).astype(np.float64)
                    absolute = float(np.max(np.abs(difference)))
                    relative_raw = float(np.linalg.norm(difference) / max(np.linalg.norm(flat(expected_raw).astype(np.float64)), 1e-30))
                    checks['raw_bridge_' + probe] = dict(max_abs=absolute, relative_l2=relative_raw)
                    assert absolute <= SETTINGS['numeric_projection_reference_max_abs']
                    assert relative_raw <= SETTINGS['numeric_projection_reference_max_relative']
            projected, diag = project_tree(result[3], result[4], .25)
            vector, nd = project_numpy(flat(result[3]), np.stack([flat(g) for g in result[4]]), .25)
            diff = float(np.max(np.abs(flat(projected) - vector)))
            relative = float(np.linalg.norm(flat(projected).astype(np.float64) - vector)
                             / max(np.linalg.norm(flat(result[3]).astype(np.float64)), 1e-30))
            checks[kind + '_numpy_projection_max_abs'] = diff
            checks[kind + '_numpy_projection_relative'] = relative
            assert diff <= SETTINGS['numeric_projection_reference_max_abs']
            assert relative <= SETTINGS['numeric_projection_reference_max_relative']
            assert bool(np.asarray(diag['success']))
            checks[kind + '_projection_relative_violation'] = float(np.asarray(diag['relative_constraint_violation']))
            assert checks[kind + '_projection_relative_violation'] <= SETTINGS['numeric_projection_relative_violation_max']
            constrained = sync(modified_step(base['model'], base['opt_state'], batch['examples'],
                batch['labels'], mk, px, py, kind, .25, head, optimizer))
            expected_model = eqx.apply_updates(base['model'], projected)
            assert maxdiff(constrained[1], expected_model) <= SETTINGS['numeric_model_adam_max_abs']
            assert maxdiff(constrained[2], reference[2]) <= SETTINGS['numeric_model_adam_max_abs']
            assert bool(np.asarray(constrained[0]['projection']['success']))
            checks[kind + '_jit_projection_relative_violation'] = float(np.asarray(constrained[0]['projection']['relative_constraint_violation']))
            assert checks[kind + '_jit_projection_relative_violation'] <= SETTINGS['numeric_projection_relative_violation_max']
    # Check the total functional derivative in an a priori gradient direction.
    xx, yy = px.reshape(-1,3,512), py.reshape(-1,3)
    fk = jax.random.split(KEY, len(xx))
    loss, grad = functional_objective(base['model'], xx, yy, fk, 0)
    norm = float(np.asarray(tree_norm(grad)))
    assert np.isfinite(norm) and norm > 0
    direction = jax.tree_util.tree_map(lambda v: v/norm, grad)
    step = SETTINGS['numeric_functional_grad_fd_step']
    plus = eqx.apply_updates(base['model'], jax.tree_util.tree_map(lambda v: step*v, direction))
    minus = eqx.apply_updates(base['model'], jax.tree_util.tree_map(lambda v: -step*v, direction))
    lp, _ = functional_objective(plus, xx, yy, fk, 0)
    lm, _ = functional_objective(minus, xx, yy, fk, 0)
    fd = float(np.asarray(lp-lm))/(2*step)
    error = abs(fd-norm)
    assert error <= SETTINGS['numeric_functional_grad_absolute_tolerance'] + SETTINGS['numeric_functional_grad_relative_tolerance']*norm
    checks['functional_derivative'] = dict(autodiff=norm, finite_difference=fd, abs_error=error)
    confirm, cids = load_dataset('mechanism_confirm')
    sample_rows = {t:(x[:128], y[:128]) for t,(x,y) in confirm.items()}
    sample_ids = {t:v[:128] for t,v in cids.items()}
    for head in [0, random_control_head(init_seed=5,train_seed=2,causal_head=0,
                                      random_path_seed=SETTINGS['random_path_seed'])]:
        mr, audits = mechanism_evaluate(base['model'], sample_rows, sample_ids, head)
        assert structure_passed(audits)
        for t in SETTINGS['protection_tasks']:
            f_loss, _ = functional_objective(base['model'], jnp.asarray(sample_rows[t][0]),
                jnp.asarray(sample_rows[t][1]), jax.random.split(KEY, len(sample_rows[t][0])), head)
            error = abs(float(np.asarray(f_loss)) - float(mr[t+'/path/donor_ce'].mean()))
            assert error <= SETTINGS['numeric_eval_mean_ce_max_abs']
            assert np.array_equal(mr[t+'/self/full_pred'], mr[t+'/native/full_pred'])
            assert np.array_equal(mr[t+'/query_label_control/full_pred'], mr[t+'/native/full_pred'])
            assert np.array_equal(mr[t+'/self/pair_pred'], mr[t+'/native/pair_pred'])
            checks[f'functional_loss_head{head}_{t}_max_mean'] = error
    # New evaluator must reproduce the unmodified author's complete metrics.
    xy = original_data()
    native_xy = {k:dict(examples=jnp.asarray(x),labels=jnp.asarray(y)) for k,(x,y) in xy.items()}
    native = main.evaluate(model=base['model'], fwd_fn=FWD, key=KEY, eval_data=native_xy,
                           eval_batch_size=SETTINGS['eval_bs'])
    adapted = evaluate(base['model'], xy)
    for t in xy:
        d = abs(float(np.asarray(native[t]['loss']).mean()) - float(adapted[t + '/loss'].mean()))
        rows = int(np.count_nonzero(np.asarray(native[t]['acc']) != adapted[t + '/acc']))
        assert d <= SETTINGS['numeric_eval_mean_ce_max_abs'] and rows <= SETTINGS['numeric_eval_acc_max_rows']
        checks[t + '_eval_ce_max_mean'] = d
        checks[t + '_eval_acc_rows'] = rows
    write_json(OUT / 'preflight.json', dict(passed=True, checks=checks,
        code_manifest_sha256=code_identity(), inputs=p['inputs'], source=p['source'],
        limits=SETTINGS, model_forward_and_gradient_gpu_verified=True))
    print('PREFLIGHT_PASSED', json.dumps(checks), flush=True)


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


def validate_mechanism(seed):
    model = base_checkpoint(seed)['model']
    sd = OUT / seed['id']
    sd.mkdir(exist_ok=True)
    if seed['id'] == 's5':
        head, scores = 0, {}
    else:
        discovery, dids = load_dataset('mechanism_discovery')
        scores = {}
        for h in range(8):
            result, _ = mechanism_evaluate(model, discovery, dids, h)
            save_npz(sd / 'mechanism_discovery' / f'head{h}.npz', result)
            scores[str(h)] = float(np.mean([np.mean(np.asarray(result[t + '/native/donor_ce'],np.float64)
                - np.asarray(result[t + '/path/donor_ce'],np.float64)) for t in SETTINGS['protection_tasks']]))
        head = min(range(8), key=lambda h: (-scores[str(h)], h))
    random_head = random_control_head(init_seed=seed['init'], train_seed=seed['train'],
        causal_head=head, random_path_seed=SETTINGS['random_path_seed'])
    data, ids = load_dataset('mechanism_confirm')
    record, audit = mechanism_evaluate(model, data, ids, head)
    save_npz(sd / 'mechanism_confirm.npz', record)
    checks, floors = {}, {}
    for t in SETTINGS['protection_tasks']:
        c = ids[t][:, -1, 0]
        native_ce=np.asarray(record[t + '/native/donor_ce'],np.float64)
        donor = native_ce - np.asarray(record[t + '/donor/donor_ce'],np.float64)
        k = native_ce - np.asarray(record[t + '/path/donor_ce'],np.float64)
        v = native_ce - np.asarray(record[t + '/v_path/donor_ce'],np.float64)
        contrasts = [ ('donor_ce', donor, SETTINGS['mechanism_donor_ce_min']),
            ('path_ce', k, SETTINGS['mechanism_path_ce_min']),
            ('retention', k - SETTINGS['mechanism_retention_min'] * donor, 0.0),
            ('k_vs_v', k - v, SETTINGS['mechanism_k_vs_v_min']),
            ('pair_margin', np.asarray(record[t + '/path/donor_margin'],np.float64) - np.asarray(record[t + '/native/donor_margin'],np.float64),
             SETTINGS['mechanism_pair_margin_min']) ]
        for name, value, threshold in contrasts:
            checks[t + '/' + name] = cluster_test(value, c, threshold)
        for field, floor in [('native/native_pair_acc', SETTINGS['mechanism_pair_acc_floor']),
                             ('donor/donor_pair_acc', SETTINGS['mechanism_pair_acc_floor']),
                             ('native/native_full_acc', SETTINGS['mechanism_full_acc_floor'])]:
            result = cluster_test(record[t + '/' + field], c, floor)
            result['passed'] = math.isfinite(result['se']) and result['se'] > 1e-15 and result['lower95'] > floor
            floors[t + '/' + field] = result
        assert bool(record[t + '/pair_valid'].all())
        assert np.all(record[t + '/native_target'] != record[t + '/donor_target'])
    for name, p in zip(checks, holm([v['p'] for v in checks.values()])):
        checks[name].update(p_holm=p, passed=p < SETTINGS['alpha'] and checks[name]['mean'] > checks[name]['threshold'])
    structural = structure_passed(audit)
    for t in SETTINGS['protection_tasks']:
        for pred in ['full_pred','pair_pred']:
            structural = structural and np.array_equal(record[t+'/self/'+pred],record[t+'/native/'+pred])
            structural = structural and np.array_equal(record[t+'/query_label_control/'+pred],record[t+'/native/'+pred])
    passed = structural and all(v['passed'] for v in checks.values()) and all(v['passed'] for v in floors.values())
    result = dict(passed=passed, candidate_head=head, random_head=int(random_head),
        discovery_scores=scores, checks=checks, floors=floors, structure_passed=structural,
        audit_max=audit, code_manifest_sha256=code_identity(), inputs=provenance()['inputs'],
        new_inputs=provenance().get('new_inputs', {}), source=provenance()['source'],
        checkpoint_sha256=sha(ASSETS / 'checkpoints/00004400000.eqx') if seed['id']=='s5'
                          else sha(sd / 'prefix/00004400000.eqx'),
        confirm_raw_sha256=sha(sd / 'mechanism_confirm.npz'))
    write_json(sd / 'mechanism_confirmation.json', result)
    print('MECHANISM_CONFIRM', seed['id'], head, passed, flush=True)
    return result


def protection_gradients(model, px, py, kind, head):
    if kind != 'P':
        raise ValueError('exp031 permits only the single path objective')
    xx, yy = px.reshape(-1,3,512), py.reshape(-1,3)
    value, grad = functional_objective(model, xx, yy, jax.random.split(KEY,len(xx)), head)
    return np.asarray([float(np.asarray(value))]), [grad]


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


def late_diagnostics(seed,mechanism,start,previous):
    """Read saved terminal states; use 4.4M-selected head on every arm."""
    data,ids=load_dataset('confirm')
    for arm in SETTINGS['arms']:
        receipt_path=OUT/seed['id']/'training_complete'/(arm+'.json')
        if not receipt_path.exists():
            continue
        receipt=json.loads(receipt_path.read_text())
        path=OUT/seed['id']/'mechanism_late'/f'{arm}_{SETTINGS["end_stage"]:011d}.npz'
        if receipt.get('late_mechanism_state')=='completed':
            assert receipt['late_mechanism_sha256']==sha(path)
            assert receipt['late_candidate_head']==mechanism['candidate_head']
            continue
        ck=load_checkpoint(WORK/receipt['final_checkpoint_path'],seed)
        result,audit=mechanism_evaluate(ck['model'],data,ids,int(mechanism['candidate_head']))
        save_npz(path,result)
        receipt.update(late_mechanism_state='completed',late_mechanism_sha256=sha(path),
            late_candidate_head=int(mechanism['candidate_head']),late_protected_head=(
                int(mechanism['random_head']) if arm.startswith('R') else int(mechanism['candidate_head'])),
            late_audit_max=audit,late_structure_passed=structure_passed(audit))
        write_json(receipt_path,receipt)
        record_cost(start,previous,'late_diagnostics')
        if check_budget(start,previous):
            raise SystemExit(3)


def cumulative_seconds(start, previous):
    return previous + time.monotonic() - start


def check_budget(start, previous):
    return STOP or cumulative_seconds(start, previous) >= SETTINGS['max_compute_seconds']


def record_cost(start, previous, status):
    write_json(OUT / 'compute_seconds.json', dict(compute_seconds=cumulative_seconds(start, previous),
        status=status, includes='all scientific phases including failed/resumed work in this runner',
        excludes='agent design, environment setup, upload and queue', code_manifest_sha256=code_identity()))


def prefix(seed, start, previous):
    sd = OUT / seed['id'] / 'prefix'
    sd.mkdir(parents=True, exist_ok=True)
    final = sd / '00004400000.eqx'
    if final.exists():
        ck = load_checkpoint(final, seed)
        assert int(ck['iter']) == SETTINGS['start_stage']
        return
    pointer = sd / 'resume.json'
    if pointer.exists():
        receipt = json.loads(pointer.read_text())
        assert receipt['code_manifest_sha256']==code_identity()
        resume = WORK / receipt['checkpoint_path']
        assert sha(resume)==receipt['checkpoint_sha256']
        ck = load_checkpoint(resume, seed)
    else:
        ck = initial_checkpoint(seed)
    _, optimizer = template(seed)
    sample = make_sampler(seed)
    while int(ck['iter']) < SETTINGS['start_stage']:
        i = int(ck['iter'])
        if i % SETTINGS['eval_every'] == 0:
            ck['seeds']['eval_model_seed'], _ = jax.random.split(ck['seeds']['eval_model_seed'])
        ck['seeds']['train_data_seed'], dk = jax.random.split(ck['seeds']['train_data_seed'])
        ck['seeds']['train_model_seed'], mk = jax.random.split(ck['seeds']['train_model_seed'])
        batch = sample(dk)
        _, ck['model'], ck['opt_state'] = main.train_step(model=ck['model'], fwd_fn=FWD,
            optimizer=optimizer, opt_state=ck['opt_state'], microbs=BS, weight_decay=0.0,
            x=batch['examples'], y=batch['labels'], key=mk)
        if not bool(np.asarray(finite_candidate(ck['model'],ck['opt_state']))):
            raise RuntimeError('prefix nonfinite model/Adam')
        ck['iter'] = i + BS
        if ck['iter'] % SETTINGS['resume_every'] == 0 or check_budget(start, previous):
            resume = sd / f'resume_{ck["iter"]:011d}.eqx'
            digest = save_checkpoint(resume, ck)
            write_json(pointer,dict(checkpoint_path=str(resume.relative_to(WORK)),
                checkpoint_sha256=digest,code_manifest_sha256=code_identity(),iter=ck['iter']))
            write_json(sd / 'progress.json', dict(iter=ck['iter'], seed=seed, code_manifest_sha256=code_identity()))
            record_cost(start, previous, 'prefix')
            if check_budget(start, previous):
                raise SystemExit(3)
    # Author snapshots at a scheduled stage are after that stage's evaluation.
    # The prefix loop stopped just before the 4.4M evaluation, so consume its
    # key once before sealing the branch point. Continuation later re-evaluates
    # the start stage just as author run_with_opts(load_from_ckpt=...) does.
    ck['seeds']['eval_model_seed'], _ = jax.random.split(ck['seeds']['eval_model_seed'])
    reference_eval_seed = base_checkpoint(SETTINGS['training_seeds'][0])['seeds']['eval_model_seed']
    assert exact(ck['seeds']['eval_model_seed'],reference_eval_seed)
    digest = save_checkpoint(final, ck)
    write_json(sd / 'complete.json', dict(passed=True, seed=seed, end_stage=ck['iter'], sha256=digest,
               code_manifest_sha256=code_identity(), inputs=provenance()['inputs']))


def train_arm(seed, arm, mechanism, start, previous):
    sd = OUT / seed['id']
    complete = sd / 'training_complete' / (arm + '.json')
    if complete.exists():
        receipt = json.loads(complete.read_text())
        if receipt['code_manifest_sha256'] != code_identity() or receipt['end_stage'] != SETTINGS['end_stage']:
            raise RuntimeError('completed branch identity changed')
        return
    if arm.startswith(('D','R','P')) and not mechanism['passed']:
        write_json(sd / 'not_run' / (arm + '.json'), dict(reason='prespecified mechanism gate failed',
                   code_manifest_sha256=code_identity(), seed=seed, arm=arm))
        return
    kind = arm[0]
    rho = SETTINGS['relaxation'][arm] if kind in ('C','D','R','P') else 1.0
    head = int(mechanism['random_head'] if kind=='R' else mechanism['candidate_head'])
    control = sd / 'checkpoints' / (arm + '_resume.json')
    stats = dict(steps=0, update_norm_ratio=0.0, correction_norm_ratio=0.0, conflicts=0.0, near_zero_steps=0)
    last_evaluated_iter = -1
    if control.exists():
        receipt = json.loads(control.read_text())
        assert receipt['code_manifest_sha256'] == code_identity()
        checkpoint = WORK / receipt['checkpoint_path']
        assert receipt['checkpoint_sha256'] == sha(checkpoint)
        ck = load_checkpoint(checkpoint,seed)
        stats = receipt['stats']
        last_evaluated_iter = receipt['last_evaluated_iter']
    else:
        ck = base_checkpoint(seed)
    # Restores the same exact frozen parameters after any resume.
    frozen = [np.asarray(v).copy() for v in embedding_leaves(base_checkpoint(seed)['model'])]
    protect, _ = load_dataset('protection')
    mon, mids = load_dataset('monitor')
    confirm, cids = load_dataset('confirm')
    original = original_data()
    sample = make_sampler(seed)
    _, optimizer = template(seed)
    def save_resume():
        checkpoint = sd / 'checkpoints' / f'{arm}_resume_{int(ck["iter"]):011d}.eqx'
        digest = save_checkpoint(checkpoint, ck)
        write_json(control, dict(iter=int(ck['iter']), stats=stats,
            checkpoint_sha256=digest,checkpoint_path=str(checkpoint.relative_to(WORK)),
            last_evaluated_iter=last_evaluated_iter,code_manifest_sha256=code_identity(),seed=seed,arm=arm))
    def require_finite_candidate(model, state, metrics):
        losses = (metrics['loss'], metrics.get('protection_loss', jnp.asarray([])))
        if not bool(np.asarray(finite_candidate(model, (state, losses)))):
            save_resume()
            write_json(sd/'failure'/(arm+'.json'),dict(iter=int(ck['iter']),
                reason='nonfinite candidate model, Adam or loss',code_manifest_sha256=code_identity()))
            raise RuntimeError('nonfinite candidate rejected before statistics/model/Adam/PRNG commit')
    while True:
        i = int(ck['iter'])
        if i % SETTINGS['eval_every'] == 0:
            marker = sd / 'evaluated' / f'{arm}_{i:011d}.json'
            if last_evaluated_iter != i:
                ck['seeds']['eval_model_seed'], _ = jax.random.split(ck['seeds']['eval_model_seed'])
                save_npz(sd / 'original' / f'{arm}_{i:011d}.npz', evaluate(ck['model'], original))
                if i % SETTINGS['monitor_every'] == 0 or i == SETTINGS['start_stage']:
                    save_npz(sd / 'monitor' / f'{arm}_{i:011d}.npz', evaluate(ck['model'], mon, mids))
                if i in SETTINGS['confirm_stages']:
                    save_npz(sd / 'confirm' / f'{arm}_{i:011d}.npz', evaluate(ck['model'], confirm, cids))
                assert int(np.asarray(ck['opt_state'][0].count)) == i // BS
                assert all(np.isfinite(np.asarray(v)).all() for v in arrays(ck))
                if kind == 'B':
                    assert all(np.array_equal(np.asarray(v), r) for v, r in zip(embedding_leaves(ck['model']), frozen))
                digest = save_checkpoint(sd / 'checkpoints' / f'{arm}_{i:011d}.eqx', ck)
                write_json(marker, dict(checkpoint_sha256=digest, iter=i, code_manifest_sha256=code_identity()))
                last_evaluated_iter = i
                save_resume()
                record_cost(start, previous, 'training')
                write_json(sd / 'progress.json', dict(seed=seed, arm=arm, iter=i, end=SETTINGS['end_stage'],
                    compute_seconds=cumulative_seconds(start, previous), code_manifest_sha256=code_identity()))
                print('PROGRESS', seed['id'], arm, i, flush=True)
        if check_budget(start, previous):
            save_resume()
            record_cost(start, previous, 'safe_stop')
            raise SystemExit(3)
        if i == SETTINGS['end_stage']:
            break
        dk_next, dk = jax.random.split(ck['seeds']['train_data_seed'])
        mk_next, mk = jax.random.split(ck['seeds']['train_model_seed'])
        batch = sample(dk)
        offset = (i - SETTINGS['start_stage']) // BS
        if kind == 'A':
            met, model, state = main.train_step(model=ck['model'], fwd_fn=FWD, optimizer=optimizer,
                opt_state=ck['opt_state'], microbs=BS, weight_decay=0.0,
                x=batch['examples'], y=batch['labels'], key=mk)
            require_finite_candidate(model, state, met)
            stats['update_norm_ratio'] += 1.0
        else:
            px, py = protect_batch(protect, offset)
            met, model, state, raw, gradients = modified_step(ck['model'], ck['opt_state'],
                batch['examples'], batch['labels'], mk, px, py, kind, rho, head, optimizer)
            require_finite_candidate(model, state, met)
            diag = met['projection']
            if not bool(np.asarray(diag['success'])) or float(np.asarray(diag['relative_constraint_violation'])) > SETTINGS['numeric_projection_relative_violation_max']:
                save_resume()
                write_json(sd / 'failure' / (arm + '.json'), dict(iter=i, reason='projection failed',
                    diagnostics={k: np.asarray(v).tolist() for k,v in diag.items()}, code_manifest_sha256=code_identity()))
                raise RuntimeError('rejected update; model/Adam not committed')
            stats['update_norm_ratio'] += float(np.asarray(diag['update_norm_ratio']))
            stats['correction_norm_ratio'] += float(np.asarray(diag['correction_norm_ratio']))
            stats['conflicts'] += float(np.asarray(diag['projected']))
            stats['near_zero_steps'] += int(float(np.asarray(diag['update_norm_ratio'])) < .01)
            if kind in ('C','D','R','P') and offset in SETTINGS['diagnostic_update_offsets']:
                audit_update(seed,arm,i,ck['model'],model,raw,gradients,px,py,mon,head,kind)
        stats['steps'] += 1
        ck.update(model=model, opt_state=state, iter=i+BS)
        ck['seeds']['train_data_seed'], ck['seeds']['train_model_seed'] = dk_next, mk_next
        if seed['id']=='s5' and arm=='A' and ck['iter']==4450016:
            reference = load_checkpoint(ASSETS / 'checkpoints/00004450016.eqx', seed)
            md, od = maxdiff(ck['model'], reference['model']), maxdiff(ck['opt_state'], reference['opt_state'])
            keys_ok = all(exact(ck['seeds'][k], reference['seeds'][k]) for k in ['train_data_seed','train_model_seed'])
            passed = md<=SETTINGS['numeric_model_adam_max_abs'] and od<=SETTINGS['numeric_model_adam_max_abs'] and keys_ok
            write_json(sd / 'replay.json', dict(passed=passed, model_error=md, adam_error=od, training_prng_exact=keys_ok))
            if not passed:
                save_resume()
                raise RuntimeError('s5 natural replay failed')
    final = sd / 'checkpoints' / f'{arm}_{i:011d}.eqx'
    denominator = max(stats['steps'], 1)
    summary = dict(mean_update_norm_ratio=stats['update_norm_ratio']/denominator,
                   mean_correction_norm_ratio=stats['correction_norm_ratio']/denominator,
                   conflict_rate=stats['conflicts']/denominator,
                   near_zero_rate=stats['near_zero_steps']/denominator)
    write_json(complete, dict(passed=True, end_stage=i, seed=seed, arm=arm,
        final_checkpoint_path=str(final.relative_to(WORK)), final_checkpoint_sha256=sha(final),
        code_manifest_sha256=code_identity(), inputs=provenance()['inputs'],
        new_inputs=provenance().get('new_inputs', {}), source=provenance()['source'],
        update_statistics=summary, update_statistics_sum=stats, late_mechanism_state='pending_selected_head',
        compute_seconds=cumulative_seconds(start, previous)))


def enforce_budget(start, previous, phase):
    if check_budget(start, previous):
        record_cost(start, previous, 'safe_stop_'+phase)
        write_json(OUT/'budget_stop.json',dict(phase=phase,compute_seconds=cumulative_seconds(start,previous),
            max_compute_seconds=SETTINGS['max_compute_seconds'],requested_stop=STOP,
            checkpoint_retained=True,code_manifest_sha256=code_identity()))
        raise SystemExit(3)


def run_postprocessor(name, start, previous):
    import subprocess
    enforce_budget(start,previous,'before_'+name)
    remaining=SETTINGS['max_compute_seconds']-cumulative_seconds(start,previous)
    try:
        subprocess.run([sys.executable,str(WORK/name)],check=True,timeout=remaining)
    except subprocess.TimeoutExpired:
        write_json(OUT/'budget_stop.json',dict(phase=name,reason='postprocessor_compute_budget',
            compute_seconds=cumulative_seconds(start,previous),max_compute_seconds=SETTINGS['max_compute_seconds'],
            checkpoint_retained=True,code_manifest_sha256=code_identity()))
        raise SystemExit(3)
    enforce_budget(start,previous,'after_'+name)


def run(seeds):
    previous_file = OUT / 'compute_seconds.json'
    previous = json.loads(previous_file.read_text())['compute_seconds'] if previous_file.exists() else 0.0
    start = time.monotonic()
    try:
        enforce_budget(start,previous,'before_inputs')
        verify_inputs()
        enforce_budget(start,previous,'after_inputs')
        print('INPUTS_VERIFIED', code_identity(), flush=True)
        enforce_budget(start,previous,'before_data')
        prepare_data()
        enforce_budget(start,previous,'after_data')
        print('IMMUTABLE_EXP030_DATA_REUSED', flush=True)
        # Bind all fresh data bytes before model-dependent steps.
        p = provenance()
        p['new_inputs'] = {str(f.relative_to(WORK)): sha(f) for f in sorted((OUT/'data').glob('*.h5'))}
        write_json(OUT / 'provenance.json', p)
        pre = OUT / 'preflight.json'
        enforce_budget(start,previous,'before_preflight')
        if not pre.exists():
            preflight()
        enforce_budget(start,previous,'after_preflight')
        assert json.loads(pre.read_text())['passed'] and json.loads(pre.read_text())['code_manifest_sha256']==code_identity()
        for seed in SETTINGS['training_seeds']:
            if seed['id'] not in seeds:
                continue
            if seed['id']!='s5':
                prefix(seed, start, previous)
            mf = OUT / seed['id'] / 'mechanism_confirmation.json'
            enforce_budget(start,previous,'before_mechanism')
            mechanism = json.loads(mf.read_text()) if mf.exists() else validate_mechanism(seed)
            enforce_budget(start,previous,'after_mechanism')
            assert mechanism['code_manifest_sha256'] == code_identity()
            if not mechanism['passed'] or mechanism['candidate_head'] != SETTINGS['candidate_path_head']:
                raise RuntimeError('Prespecified head0 functional gate failed; no path-only training')
            for arm in SETTINGS['arms']:
                assert arm == 'P25'
                train_arm(seed, arm, mechanism, start, previous)
            enforce_budget(start,previous,'before_late')
            late_diagnostics(seed,mechanism,start,previous)
            enforce_budget(start,previous,'after_late')
        record_cost(start, previous, 'scientific_phases_complete')
        run_postprocessor('analyze.py',start,previous)
        run_postprocessor('verify_results.py',start,previous)
        verification=json.loads((OUT/'verification.json').read_text())
        assert verification['passed'] and not verification['failures']
        report_path=OUT/'report.md'
        report_path.write_text(report_path.read_text().replace('统计结果待独立读取器核验。','统计结果已通过独立读取器核验。'))
        write_json(OUT/'runtime_complete.json', dict(passed=True, seeds=seeds,
            code_manifest_sha256=code_identity(), compute_seconds=cumulative_seconds(start, previous),
            verification_sha256=sha(OUT/'verification.json'),analysis_sha256=sha(OUT/'analysis.json'),
            verification_checks=verification['checks'],scientific_support=verification['scientific_support']))
    except Exception as error:
        attempt=WORK/'attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        attempt.mkdir(parents=True)
        manifest=json.loads((WORK/'code_manifest.json').read_text())
        for name in manifest:
            (attempt/name).parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(WORK/name,attempt/name)
        shutil.copyfile(WORK/'code_manifest.json',attempt/'code_manifest.json')
        write_json(attempt/'failure.json',dict(error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),code_manifest_sha256=code_identity(),
            compute_seconds=cumulative_seconds(start,previous),
            scientific_conditions_changed=False,automatic_retry=False))
        raise
    finally:
        record_cost(start, previous, 'runner_exit')


if __name__=='__main__':
    for signum in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(signum, request_stop)
    parser=argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run','preflight','prepare-data'])
    parser.add_argument('--seeds', default=','.join(s['id'] for s in SETTINGS['training_seeds']))
    args=parser.parse_args()
    if args.mode=='preflight':
        preflight()
    elif args.mode=='prepare-data':
        prepare_data()
    else:
        run(args.seeds.split(','))

"""CUDA startup checks for path sets (run after exp039 ``mechanism_checks.run_checks``).

1. Every single path: ``path_set`` with one id reproduces ``mechanism.py`` field
   by field with max |difference| = 0 (eager), and the lean losses agree.
2. Multi-head sets (same layer and, for depth 4, cross-layer): exact structural
   audits, sham ~ native, lean loss == audited CE, unchanged prefix below the
   lowest sink, natural continuation above the highest sink, finite-difference
   parameter gradient, live recipient and donor input gradients.
3. JIT: the compiled set objective for one id matches the compiled exp039 path
   objective within ``numeric_set_jit_loss_max_abs``.
No threshold is relaxed relative to exp039; failures raise.
"""


def run_set_checks(model, x, y, settings):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import mechanism as M
    import path_set as PS

    if jax.default_backend() != 'gpu':
        raise RuntimeError('set checks require the CUDA GPU')
    x, y = jnp.asarray(x), jnp.asarray(y)
    if x.ndim == 3:
        x, y = x[0], y[0]
    key = jax.random.PRNGKey(20261040)
    keys = jnp.expand_dims(key, 0)
    depth = len(model.transformer.blocks)
    exact_keys = {'embedding_max_abs', 'l1_upstream_max_abs', 'l1_values_pre_proj_max_abs',
                  'l1_values_post_proj_max_abs', 'l2_norm_input_max_abs', 'l2_raw_qkv_max_abs', 'q_max_abs',
                  'other_channel_max_abs', 'nonlabel_channel_max_abs', 'target_channel_max_abs',
                  'hybrid_nonlabel_residual_max_abs', 'query_input_max_abs'}
    special = {'qk_recompute_max_abs': settings['numeric_path_qk_max_abs'],
               'future_attention_max': settings['numeric_path_future_attention_max'],
               'attention_row_sum_max_abs': settings['numeric_path_rowsum_max_abs']}

    def max_abs(a, b):
        return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))

    def audit_check(audit):
        values = {k: float(np.asarray(v)) for k, v in audit.items()}
        for name, value in values.items():
            base = name.split('/')[-1]
            limit = 0.0 if base in exact_keys else special.get(base, settings['numeric_sham_logit_max_abs'])
            if not np.isfinite(value) or value > limit:
                raise AssertionError(f'set structural audit {name}: {value} > {limit}')
        return values

    def tree_norm(tree):
        return jnp.sqrt(sum(jnp.sum(jnp.square(v)) for v in jax.tree_util.tree_leaves(tree) if eqx.is_array(v)))

    receipt = dict(passed=True, depth=depth, width=int(model.embed_dim), single=dict(), sets=dict())
    with jax.default_matmul_precision('float32'):
        # 1. single-path identity with exp039
        for p in M.candidate_path_ids(model):
            ref = M.patch_example(model, x, y, key, p, include_cache=False)
            new = PS.set_patch_example(model, x, y, key, (p,), include_cache=False)
            diffs = {}
            for name, value in ref.items():
                if isinstance(value, dict):
                    if set(value) != set(new[name]):
                        raise AssertionError('audit fields differ for single path ' + str(p))
                    for field, v in value.items():
                        diffs[name + '/' + field] = max_abs(v, new[name][field])
                else:
                    diffs[name] = max_abs(value, new[name])
            diffs['path_loss'] = max_abs(M.path_loss(model, x[None], y[None], keys, p),
                                         PS.set_loss(model, x[None], y[None], keys, (p,)))
            if max(diffs.values()) != 0.0:
                bad = {k: v for k, v in diffs.items() if v != 0.0}
                raise AssertionError(f'single-path set differs from exp039 for {p}: {bad}')
            receipt['single'][str(p)] = dict(fields=len(diffs), max_abs=0.0)
        # 2. multi-head sets
        sets = [(0, 1), (0, 3, 5, 7)]
        if depth == 4:
            sets += [(0, 9), (2, 9, 18), (0, 1, 9, 17), (16, 23)]
        for ids in sets:
            ids = PS.normalize(ids, depth)
            result = PS.set_patch_example(model, x, y, key, ids)
            forward = audit_check(result['audit'])
            reverse = audit_check(result['reverse_audit'])
            sham = max_abs(result['self_logits'], result['native_logits'])
            if sham > settings['numeric_sham_logit_max_abs']:
                raise AssertionError('set sham changed logits')
            loss = PS.set_loss(model, x[None], y[None], keys, ids)
            ce = M.complete_ce(result['path_logits'], result['donor_target'])
            if max_abs(loss, ce) > 1e-4:
                raise AssertionError('lean set loss differs from audited set path')
            recipient, donor, _ = M._pair_caches(model, x, y, key)
            hybrids = PS._hybrid_set(model, recipient, donor, key, ids)
            patched = PS._set_forward(model, x, y, key, recipient, PS._channels(hybrids), 'K')
            sinks = sorted(hybrids)
            prefix = max([max_abs(M._blocks(patched)[l]['out'], M._blocks(recipient)[l]['out'])
                          for l in range(sinks[0])] + [max_abs(patched['embedding'], recipient['embedding'])])
            if prefix != 0.0:
                raise AssertionError('recipient prefix below the lowest sink changed')
            h = M._blocks(patched)[sinks[-1]]['out']
            block_keys = jax.random.split(jax.random.split(key, 2)[1], depth)
            for layer in range(sinks[-1] + 1, depth):
                h = model.transformer.blocks[layer].call_with_all_aux(h, key=block_keys[layer])['out']
            h = jax.vmap(model.transformer.norm)(h)
            h = jax.vmap(model.transformer.unembed)(h)
            continuation = max_abs(jax.vmap(model.unembed)(h)[0::2], patched['out'])
            if continuation != 0.0:
                raise AssertionError('layers above the highest sink were not recomputed naturally')
            # same-layer deltas add; label K at every sink equals the cached hybrid K
            for s in sinks:
                pk = M._attention(patched, s)['k'][..., 1::2, :]
                hk = hybrids[s][0]['k'][..., 1::2, :]
                if max_abs(pk, hk) != 0.0:
                    raise AssertionError('label K at sink %d not replaced' % s)

            def objective(m):
                return PS.set_loss(m, x[None], y[None], keys, ids)
            value, grad = eqx.filter_jit(eqx.filter_value_and_grad(objective))(model)
            norm = float(np.asarray(tree_norm(grad)))
            if not np.isfinite(norm) or norm <= 0:
                raise AssertionError('nonfinite or zero set gradient')
            direction = jax.tree_util.tree_map(lambda v: None if v is None else v / norm, grad)
            step = settings['numeric_functional_grad_fd_step']
            plus = eqx.apply_updates(model, jax.tree_util.tree_map(lambda v: None if v is None else step * v, direction))
            minus = eqx.apply_updates(model, jax.tree_util.tree_map(lambda v: None if v is None else -step * v, direction))
            f = eqx.filter_jit(objective)
            fd = float(np.asarray(f(plus) - f(minus))) / (2 * step)
            allowed = settings['numeric_functional_grad_absolute_tolerance'] + settings['numeric_functional_grad_relative_tolerance'] * norm
            if not np.isfinite(fd) or abs(fd - norm) > allowed:
                raise AssertionError('set loss finite-difference gradient failed')
            donor_x = x[jnp.array([1, 0, 2])]
            target = M._targets(y)[1]

            def input_loss(rec_x, don_x):
                rec = model.call_with_all_aux(examples=rec_x, labels=y, key=key)
                don = model.call_with_all_aux(examples=don_x, labels=y, key=key)
                hy = PS._hybrid_set(model, rec, don, key, ids)
                out = PS._set_forward(model, rec_x, y, key, rec, PS._channels(hy), 'K')
                return M.complete_ce(out['out'][-1], target)
            rg, dg = jax.jit(jax.grad(input_loss, argnums=(0, 1)))(x, donor_x)
            norms = [float(np.linalg.norm(np.asarray(g, np.float64))) for g in (rg, dg)]
            if not all(np.isfinite(n) and n > 0 for n in norms):
                raise AssertionError('recipient/donor branch detached for set')
            receipt['sets']['-'.join(map(str, ids))] = dict(
                path_set=list(ids), sinks=sinks, forward=forward, reverse=reverse, sham_max_abs=sham,
                prefix_max_abs=prefix, continuation_max_abs=continuation,
                gradient=dict(autodiff=norm, finite_difference=fd, allowed=allowed),
                recipient_input_grad_norm=norms[0], donor_input_grad_norm=norms[1])
    # 3. compiled single-id objective vs compiled exp039 objective
    batch_x, batch_y = jnp.stack([x, x]), jnp.stack([y, y])
    batch_keys = jax.random.split(key, 2)

    @eqx.filter_jit
    def old_loss(m, p):
        with jax.default_matmul_precision('float32'):
            return M.path_loss(m, batch_x, batch_y, batch_keys, p)

    @eqx.filter_jit
    def new_loss(m, ids):
        with jax.default_matmul_precision('float32'):
            return PS.set_loss(m, batch_x, batch_y, batch_keys, ids)
    jit_diffs = {}
    for p in M.candidate_path_ids(model):
        jit_diffs[str(p)] = max_abs(old_loss(model, p), new_loss(model, (p,)))
        if jit_diffs[str(p)] > settings['numeric_set_jit_loss_max_abs']:
            raise AssertionError('compiled set objective differs from exp039 for ' + str(p))
    receipt['jit_single_loss_max_abs'] = jit_diffs
    return receipt

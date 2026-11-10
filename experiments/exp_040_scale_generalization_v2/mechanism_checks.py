"""CUDA-only preflight checks for the minimally generalized path definition.

Importing this module does not import JAX, create arrays or run an experiment.
``run_checks`` is an explicit startup validation entry point. Its model factory
must use the frozen author initialization/configuration and accept two integers
``(embed_dim, depth)``; it must return a model, not a checkpoint tuple.

All candidate paths receive sham, mask/channel, source-head reconstruction and
downstream-continuation checks. One head in each adjacent pair additionally
receives a parameter-gradient finite-difference check and live-recipient/donor
input-gradient check. The unchanged 64-dimensional two-layer definition is
compared to the supplied frozen exp036 module for all eight original heads.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


DEFAULT_TOLERANCES = {
    # These are the frozen exp036 settings; no scientific threshold is relaxed.
    "numeric_sham_logit_max_abs": 1e-4,
    "numeric_path_qk_max_abs": 1e-5,
    "numeric_path_future_attention_max": 1e-7,
    "numeric_path_rowsum_max_abs": 1e-6,
    "numeric_functional_grad_relative_tolerance": 0.05,
    "numeric_functional_grad_absolute_tolerance": 0.005,
    "numeric_functional_grad_fd_step": 0.001,
}

EXACT_KEYS = frozenset({
    "embedding_max_abs", "l1_upstream_max_abs", "l1_values_pre_proj_max_abs",
    "l1_values_post_proj_max_abs", "l2_norm_input_max_abs",
    "l2_raw_qkv_max_abs", "q_max_abs", "other_channel_max_abs",
    "nonlabel_channel_max_abs", "target_channel_max_abs",
    "hybrid_nonlabel_residual_max_abs", "query_input_max_abs",
})


def _load_original(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Frozen equivalence reference missing: " + str(path))
    spec = spec_from_file_location("exp039_frozen_original_mechanism64", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_checks(model_factory, x, y, *, original_module_path=None,
               architectures=((64, 2), (128, 2), (128, 4)),
               tolerances=None, include_gradient=True):
    """Fail closed on any structural/numerical failure; return a JSON receipt.

    ``x``/``y`` may contain a batch; only its first valid row is used. This is a
    correctness check, not training, discovery or a performance estimate.
    ``architectures`` may select the current architecture plus (64,2) to avoid
    duplicate checks in independently launched width/depth jobs. The full
    default checks both requested scales. Callers should persist this receipt
    before the first training update.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import mechanism as mech

    if jax.default_backend() != "gpu":
        raise RuntimeError("Mechanism numerical validation requires the approved CUDA GPU")
    if jax.config.jax_enable_x64:
        raise RuntimeError("The frozen author model requires float32 parameter/data semantics")
    tol = dict(DEFAULT_TOLERANCES)
    if tolerances is not None:
        for name in tol:
            if name in tolerances:
                requested = float(tolerances[name])
                if requested != tol[name]:
                    raise ValueError("Changed frozen numerical threshold: " + name)
    x, y = jnp.asarray(x), jnp.asarray(y)
    if x.ndim == 3:
        x, y = x[0], y[0]
    if x.shape != (3, 512) or y.shape != (3,):
        raise ValueError("Expected x=(3,512), y=(3,) or aligned batches")
    if x.dtype != jnp.float32:
        raise TypeError("Validation features must have dtype float32")
    if not bool(np.asarray(mech._targets(y)[2])):
        raise ValueError("Validation row violates pairing target contract")
    key = jax.random.PRNGKey(20261039)
    keys = jnp.expand_dims(key, 0)
    special = {
        "qk_recompute_max_abs": tol["numeric_path_qk_max_abs"],
        "future_attention_max": tol["numeric_path_future_attention_max"],
        "attention_row_sum_max_abs": tol["numeric_path_rowsum_max_abs"],
    }

    def max_abs(a, b):
        return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))

    def audit_check(audit):
        values = {name: float(np.asarray(value)) for name, value in audit.items()}
        for name, value in values.items():
            if not np.isfinite(value):
                raise AssertionError("Nonfinite structural audit: " + name)
            limit = 0.0 if name in EXACT_KEYS else special.get(
                name, tol["numeric_sham_logit_max_abs"])
            if value > limit:
                raise AssertionError(f"Structural audit {name}: {value} > {limit}")
        return values

    def tree_norm(tree):
        values = [value for value in jax.tree_util.tree_leaves(tree) if eqx.is_array(value)]
        return jnp.sqrt(sum(jnp.sum(jnp.square(value)) for value in values))

    def downstream_logits(model, patched, sink):
        # Independently continue from the changed sink; no other block is pinned.
        h = mech._blocks(patched)[sink]["out"]
        tr_key = jax.random.split(key, 2)[1]
        block_keys = jax.random.split(tr_key, len(model.transformer.blocks))
        for layer in range(sink + 1, len(model.transformer.blocks)):
            h = model.transformer.blocks[layer].call_with_all_aux(
                h, key=block_keys[layer])["out"]
        h = jax.vmap(model.transformer.norm)(h)
        h = jax.vmap(model.transformer.unembed)(h)
        return jax.vmap(model.unembed)(h)[0::2]

    def gradient_check(model, path_id):
        def loss(m):
            return mech.path_loss(m, x[None], y[None], keys, path_id)
        objective = eqx.filter_jit(eqx.filter_value_and_grad(loss))
        value, grad = objective(model)
        norm = float(np.asarray(tree_norm(grad)))
        if not np.isfinite(float(value)) or not np.isfinite(norm) or norm <= 0:
            raise AssertionError("Nonfinite or zero path gradient")
        direction = jax.tree_util.tree_map(lambda v: None if v is None else v / norm, grad)
        step = tol["numeric_functional_grad_fd_step"]
        plus = eqx.apply_updates(model, jax.tree_util.tree_map(
            lambda v: None if v is None else step * v, direction))
        minus = eqx.apply_updates(model, jax.tree_util.tree_map(
            lambda v: None if v is None else -step * v, direction))
        # The JIT/precision context is identical to the gradient objective.
        evaluate_loss = eqx.filter_jit(loss)
        lp = evaluate_loss(plus)
        lm = evaluate_loss(minus)
        fd = float(np.asarray(lp - lm)) / (2 * step)
        error = abs(fd - norm)
        allowed = tol["numeric_functional_grad_absolute_tolerance"] + \
            tol["numeric_functional_grad_relative_tolerance"] * norm
        if not np.isfinite(fd) or error > allowed:
            raise AssertionError(f"Functional gradient FD failed: {error} > {allowed}")

        # Separate differentiable inputs verify both current-model branches are
        # live, including retained recipient Q/V/K (they are never stop_gradient).
        donor_x = x[jnp.array([1, 0, 2])]
        target = mech._targets(y)[1]

        def input_loss(rec_x, don_x):
            rec = model.call_with_all_aux(examples=rec_x, labels=y, key=key)
            don = model.call_with_all_aux(examples=don_x, labels=y, key=key)
            channels, _ = mech._hybrid_attention(model, rec, don, key, path_id)
            result = mech._channel_forward(model, rec_x, y, key, rec, channels, "K", path_id)
            return mech.complete_ce(result["out"][-1], target)

        rec_grad, don_grad = jax.jit(jax.grad(input_loss, argnums=(0, 1)))(x, donor_x)
        recipient_norm = float(np.linalg.norm(np.asarray(rec_grad, np.float64)))
        donor_norm = float(np.linalg.norm(np.asarray(don_grad, np.float64)))
        if not all(np.isfinite(n) and n > 0 for n in (recipient_norm, donor_norm)):
            raise AssertionError("Recipient/donor branch gradient is nonfinite or detached")

        # Isolate the retained Q/V and non-label K arrays themselves. A live
        # residual could otherwise make recipient input gradients nonzero even
        # if these cached recipient channels had accidentally been detached.
        rec = model.call_with_all_aux(examples=x, labels=y, key=key)
        don = model.call_with_all_aux(examples=donor_x, labels=y, key=key)
        channels, _ = mech._hybrid_attention(model, rec, don, key, path_id)
        sink = mech.path_spec(model, path_id)["sink_layer"]
        native = mech._attention(rec, sink)

        def retained_loss(q, k, v):
            modified = dict(rec)
            tr = dict(rec["transformer_output"])
            blocks = list(tr["block_outputs"])
            block = dict(blocks[sink])
            attention = dict(block["attn_output"])
            attention.update(q=q, k=k, v=v)
            block["attn_output"] = attention
            blocks[sink] = block
            tr["block_outputs"] = blocks
            modified["transformer_output"] = tr
            result = mech._channel_forward(model, x, y, key, modified, channels, "K", path_id)
            return mech.complete_ce(result["out"][-1], target)

        channel_grads = jax.jit(jax.grad(retained_loss, argnums=(0, 1, 2)))(
            native["q"], native["k"], native["v"])
        channel_norms = {name: float(np.linalg.norm(np.asarray(value, np.float64)))
                         for name, value in zip(("q", "nonlabel_k", "v"), channel_grads)}
        if not all(np.isfinite(n) and n > 0 for n in channel_norms.values()):
            raise AssertionError("A retained recipient Q/K/V channel gradient was detached")
        k_label_grad = np.asarray(channel_grads[1])[..., [1, 3], :]
        if np.any(k_label_grad != 0):
            raise AssertionError("Recipient label K has a direct retained-channel gradient")
        return dict(autodiff=norm, finite_difference=fd, abs_error=error,
                    allowed_error=allowed, step=step, recipient_input_grad_norm=recipient_norm,
                    donor_input_grad_norm=donor_norm, path_id=path_id,
                    retained_channel_gradient_norms=channel_norms,
                    replaced_label_k_direct_gradient_max_abs=float(np.max(np.abs(k_label_grad))),
                    **{k: v for k, v in mech.path_spec(model, path_id).items() if k != "path_id"})

    receipt = dict(passed=True, backend=jax.default_backend(), architectures={},
                   numerical_tolerances=tol, equivalence64=None)
    original = None
    # Keep architecture inputs/results alive throughout validation. The pinned
    # CUDA runtime aborted during manual architecture cleanup; leave executable
    # caches to normal process lifetime instead of destroying them here.
    validation_keepalive = []
    print("MECHANISM_CHECKS_BEGIN", list(architectures), flush=True)
    with jax.default_matmul_precision("float32"):
        for width, depth in architectures:
            print("MECHANISM_ARCH_BEGIN", width, depth, flush=True)
            model = model_factory(int(width), int(depth))
            if model.embed_dim != width or len(model.transformer.blocks) != depth:
                raise ValueError("Model factory ignored the requested architecture")
            mech._check_contract(model, x, y, 0, "K")
            model_result = dict(all_path_ids=list(mech.candidate_path_ids(model)),
                                structural_by_path={}, gradient_by_source={})
            recipient, donor, _ = mech._pair_caches(model, x, y, key)
            if (width, depth) == (64, 2):
                print("MECHANISM_EQUIVALENCE64_BEGIN", flush=True)
                path = original_module_path or Path(__file__).with_name("original_mechanism64.py")
                original = _load_original(path)
                receipt["equivalence64"] = dict(passed=True, paths={}, reference_path=str(path))
            for path_id in mech.candidate_path_ids(model):
                print("MECHANISM_PATH_BEGIN", width, depth, path_id, flush=True)
                spec = mech.path_spec(model, path_id)
                result = mech.patch_example(model, x, y, key, path_id, include_cache=True)
                forward = audit_check(result["audit"])
                reverse = audit_check(result["reverse_audit"])
                if not bool(np.asarray(result["pair_valid"])):
                    raise AssertionError("Lost valid pairing contract")
                for name, coordinate in spec.items():
                    if int(np.asarray(result[name])) != coordinate or \
                       int(np.asarray(result["audit_cache"][name])) != coordinate:
                        raise AssertionError("Path metadata does not match actual construction")
                expected_head_mask = np.arange(width) // (width // 8) == spec["local_head"]
                if not np.array_equal(result["audit_cache"]["head_dim_mask"], expected_head_mask):
                    raise AssertionError("Head mask does not match width/local head")
                if not np.array_equal(result["audit_cache"]["label_token_mask"], [False, True, False, True, False]):
                    raise AssertionError("Label token mask changed")
                channels, hybrid = mech._hybrid_attention(model, recipient, donor, key, path_id)
                patched = mech._channel_forward(model, x, y, key, recipient, channels, "K", path_id)
                continuation_delta = max_abs(downstream_logits(model, patched, spec["sink_layer"]), patched["out"])
                if continuation_delta != 0:
                    raise AssertionError("Downstream layers were pinned or recomputed incorrectly")
                upstream_delta = max(max_abs(mech._blocks(patched)[layer]["out"],
                                             mech._blocks(recipient)[layer]["out"])
                                     for layer in range(spec["sink_layer"]))
                if upstream_delta != 0:
                    raise AssertionError("Recipient prefix/residual changed")
                # Verify V-control independently; patch_example's main audit is K.
                v_patched = mech._channel_forward(model, x, y, key, recipient, channels, "V", path_id)
                v_audit = audit_check(mech._path_audit(model, recipient, v_patched, channels, hybrid, "V", path_id))
                loss = mech.path_loss(model, x[None], y[None], keys, path_id)
                ce = mech.complete_ce(result["path_logits"], result["donor_target"])
                loss_delta = max_abs(loss, ce)
                if loss_delta > 1e-4:
                    raise AssertionError("Lean training path differs from audited evaluation path")
                model_result["structural_by_path"][str(path_id)] = dict(
                    **spec, forward=forward, reverse=reverse, v_control=v_audit,
                    upstream_prefix_max_abs=upstream_delta,
                    downstream_continuation_max_abs=continuation_delta,
                    path_loss_ce_max_abs=loss_delta)
                if (width, depth) == (64, 2):
                    print("MECHANISM_EQUIVALENCE64_PATH_BEGIN", path_id, flush=True)
                    reference = original.patch_example(model, x, y, key, path_id, include_cache=True)
                    differences = {}
                    for name, before in reference.items():
                        if isinstance(before, dict):
                            for field, value in before.items():
                                differences[name + "/" + field] = max_abs(value, result[name][field])
                        else:
                            differences[name] = max_abs(before, result[name])
                    reference_loss = original.path_loss(model, x[None], y[None], keys, path_id)
                    differences["path_loss"] = max_abs(reference_loss, loss)
                    if max(differences.values()) != 0:
                        raise AssertionError("64/2 original numerical equivalence failed")
                    expected_random = original.random_control_head(6, 3, path_id)
                    if mech.random_control_head(6, 3, path_id) != expected_random:
                        raise AssertionError("Original deterministic random control changed")
                    receipt["equivalence64"]["paths"][str(path_id)] = dict(
                        max_abs=0.0, checked_fields=len(differences), random_control=expected_random)
                    print("MECHANISM_EQUIVALENCE64_PATH_PASSED", path_id, flush=True)
                random_path = mech.random_control_head(6, 3, path_id)
                if random_path == path_id or random_path // 8 != spec["source_layer"]:
                    raise AssertionError("Random control does not match the adjacent pair")
                print("MECHANISM_PATH_PASSED", width, depth, path_id, flush=True)
            if (width, depth) == (64, 2):
                print("MECHANISM_EQUIVALENCE64_PASSED", flush=True)
            if include_gradient:
                for source in range(depth - 1):
                    print("MECHANISM_GRADIENT_BEGIN", width, depth, source * 8, flush=True)
                    model_result["gradient_by_source"][str(source)] = gradient_check(model, source * 8)
                    print("MECHANISM_GRADIENT_PASSED", width, depth, source * 8, flush=True)
            receipt["architectures"][f"w{width}_l{depth}"] = model_result
            validation_keepalive.append((model, recipient, donor, result,
                                         patched, v_patched, channels, hybrid))
            print("MECHANISM_ARCH_PASSED", width, depth,
                  "cache_cleanup=deferred_to_process_lifetime", flush=True)
    print("MECHANISM_CHECKS_PASSED", flush=True)
    return receipt

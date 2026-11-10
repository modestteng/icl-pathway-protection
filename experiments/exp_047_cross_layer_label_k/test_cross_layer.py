"""Remote CUDA structural/gradient checks; --topology-only is dependency-free.

The caller supplies its actual sealed model, protection row and exp040 numeric
settings to run_checks(). These checks do not replace independent functional
confirmation, natural-ICL update diagnostics, or paired continuation.
"""

import json
import cross_layer as C


def topology_checks():
    assert len(C.candidate_paths(2)) == 8
    assert len(C.candidate_paths(4)) == 48
    assert len(C.by_sink(C.candidate_paths(4))[3]) == 24
    assert C.normalize(((0, 3, 2), (0, 3, 2), (1, 2, 1)), 4) == ((0, 3, 2), (1, 2, 1))
    for invalid in [(0, 0, 0), (2, 1, 0), (0, 4, 0), (0, 2, 8)]:
        try:
            C.normalize((invalid,), 4)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted invalid path: " + str(invalid))
    for invalid in [(0, 1), (0, 1, 2)]:
        try:
            C.normalize(invalid, 4)
        except TypeError:
            pass
        else:
            raise AssertionError("Legacy integer ids accepted ambiguously")
    return dict(passed=True, depth2_candidates=8, depth4_candidates=48,
                scientific_execution=False)


def run_checks(model, x, y, settings):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    if jax.default_backend() != "gpu":
        raise RuntimeError("Scientific operator checks require CUDA GPU")
    M, PS = C.legacy_modules()
    x, y = jnp.asarray(x), jnp.asarray(y)
    if x.ndim == 3:
        x, y = x[0], y[0]
    depth = len(model.transformer.blocks)
    key = jax.random.PRNGKey(20261047)
    keys = key[None]
    exact = {"embedding_max_abs", "l1_upstream_max_abs", "l1_values_pre_proj_max_abs",
             "l1_values_post_proj_max_abs", "l2_norm_input_max_abs", "l2_raw_qkv_max_abs",
             "q_max_abs", "other_channel_max_abs", "nonlabel_channel_max_abs",
             "target_channel_max_abs", "hybrid_nonlabel_residual_max_abs", "query_input_max_abs"}
    special = {"qk_recompute_max_abs": settings["numeric_path_qk_max_abs"],
               "future_attention_max": settings["numeric_path_future_attention_max"],
               "attention_row_sum_max_abs": settings["numeric_path_rowsum_max_abs"]}

    def difference(a, b):
        return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))

    def audit_check(audit):
        report = {}
        for name, value in audit.items():
            value = float(np.asarray(value))
            base = name.split("/")[-1]
            limit = 0.0 if base in exact else special.get(base, settings["numeric_sham_logit_max_abs"])
            if not np.isfinite(value) or value > limit:
                raise AssertionError(f"Structural audit {name}: {value} > {limit}")
            report[name] = value
        return report

    def tree_norm(tree):
        return jnp.sqrt(sum(jnp.vdot(v, v).real for v in jax.tree_util.tree_leaves(tree) if eqx.is_array(v)))

    report = dict(passed=True, backend=jax.default_backend(), depth=depth,
                  topology=topology_checks(), adjacent_hybrids={}, closure={}, sets={})
    with jax.default_matmul_precision("float32"):
        C._check(model, x, y, ((0, 1, 0),), "K")
        recipient, donor, _ = M._pair_caches(model, x, y, key)
        label_mask = np.arange(M.SEQUENCE_LENGTH) % 2 == 1
        if difference(recipient["embedding"][label_mask], donor["embedding"][label_mask]) != 0:
            raise AssertionError("Closure precondition failed: label embeddings differ")
        # Every old adjacent operator must retain bit-identical hybrid residual
        # and K/V, not just a matching output score.
        for legacy_id in M.candidate_path_ids(model):
            triples = C.adjacent_paths((legacy_id,))
            sink = triples[0][1]
            old = PS._hybrid_set(model, recipient, donor, key, (legacy_id,))[sink]
            new = C._hybrid_set(model, recipient, donor, key, triples)[sink]
            error = max(difference(old[1], new[1]), difference(old[0]["k"], new[0]["k"]),
                        difference(old[0]["v"], new[0]["v"]))
            if error != 0:
                raise AssertionError("Adjacent operator changed: " + str(legacy_id))
            report["adjacent_hybrids"][str(legacy_id)] = error
        # Selecting every upstream head must recover the complete donor label
        # residual/K. This is the new independent residual-accounting invariant.
        for sink in range(1, depth):
            triples = tuple((source, sink, head) for source in range(sink) for head in range(C.HEADS))
            channels, hybrid = C._hybrid_set(model, recipient, donor, key, triples)[sink]
            residual_error = difference(hybrid[label_mask], M._blocks(donor)[sink - 1]["out"][label_mask])
            key_error = difference(channels["k"][..., 1::2, :], M._attention(donor, sink)["k"][..., 1::2, :])
            if max(residual_error, key_error) > settings["numeric_path_qk_max_abs"]:
                raise AssertionError(f"All-earlier-heads closure failed at sink {sink}: {residual_error}, {key_error}")
            if difference(hybrid[~label_mask], M._blocks(recipient)[sink - 1]["out"][~label_mask]) != 0:
                raise AssertionError("Closure changed nonlabel residual")
            # Compare full forwards with ONLY label K changed. Q, V, and
            # nonlabel K must remain recipient, even in the all-source control.
            constructed = C._set_forward(model, x, y, key, recipient, {sink: channels}, "K")
            full_k = C._set_forward(model, x, y, key, recipient,
                                    {sink: M._attention(donor, sink)}, "K")
            actual = M._attention(constructed, sink)
            native = M._attention(recipient, sink)
            q_error = difference(actual["q"], native["q"])
            v_error = difference(actual["v"], native["v"])
            nonlabel_k_error = difference(actual["k"][..., 0::2, :], native["k"][..., 0::2, :])
            logit_error = difference(constructed["out"], full_k["out"])
            if max(q_error, v_error, nonlabel_k_error) != 0:
                raise AssertionError("All-source K closure changed recipient Q/V/nonlabel K")
            if logit_error > settings["numeric_sham_logit_max_abs"]:
                raise AssertionError("All-source closure disagrees with full donor label-K control")
            report["closure"][str(sink)] = dict(residual_max_abs=residual_error, k_max_abs=key_error,
                q_max_abs=q_error, v_max_abs=v_error, nonlabel_k_max_abs=nonlabel_k_error,
                complete_donor_k_logit_max_abs=logit_error)
        sets = [((0, 1, 0), (0, 1, 3))]
        if depth == 4:
            sets += [((0, 3, 2),), ((0, 2, 1), (1, 2, 4)),
                     ((0, 1, 0), (0, 3, 2), (1, 3, 5), (2, 3, 7))]
        for triples in sets:
            out = C.set_patch_example(model, x, y, key, triples)
            audits = {name: audit_check(out[name]) for name in ("audit", "reverse_audit")}
            sham_error = difference(out["self_logits"], out["native_logits"])
            if sham_error > settings["numeric_sham_logit_max_abs"]:
                raise AssertionError("Sham changed output")

            def objective(m):
                return C.set_loss(m, x[None], y[None], keys, triples)

            f = eqx.filter_jit(objective)
            value, grad = eqx.filter_jit(eqx.filter_value_and_grad(objective))(model)
            audited_ce = M.complete_ce(out["path_logits"], out["donor_target"])
            # Same lean-vs-audited CE tolerance as exp040/set_checks.py; the
            # stricter configured tolerance below compares two compiled losses.
            if difference(value, audited_ce) > 1e-4:
                raise AssertionError("Lean loss differs from audited patch")
            norm = float(np.asarray(tree_norm(grad)))
            if not np.isfinite(norm) or norm <= 0:
                raise AssertionError("Invalid functional gradient")
            direction = jax.tree_util.tree_map(lambda v: None if v is None else v / norm, grad)
            step = settings["numeric_functional_grad_fd_step"]
            plus = eqx.apply_updates(model, jax.tree_util.tree_map(lambda v: None if v is None else step * v, direction))
            minus = eqx.apply_updates(model, jax.tree_util.tree_map(lambda v: None if v is None else -step * v, direction))
            fd = float(np.asarray(f(plus) - f(minus))) / (2 * step)
            allowed = settings["numeric_functional_grad_absolute_tolerance"] + settings["numeric_functional_grad_relative_tolerance"] * norm
            if not np.isfinite(fd) or abs(fd - norm) > allowed:
                raise AssertionError("Finite difference gradient check failed")
            donor_x = x[jnp.array([1, 0, 2])]

            def input_loss(rx, dx):
                rec = model.call_with_all_aux(examples=rx, labels=y, key=key)
                don = model.call_with_all_aux(examples=dx, labels=y, key=key)
                hybrid = C._hybrid_set(model, rec, don, key, triples)
                patched = C._set_forward(model, rx, y, key, rec, C._channels(hybrid), "K")
                return M.complete_ce(patched["out"][-1], M._targets(y)[1])

            rg, dg = jax.jit(jax.grad(input_loss, argnums=(0, 1)))(x, donor_x)
            input_norms = [float(np.linalg.norm(np.asarray(g, np.float64))) for g in (rg, dg)]
            if not all(np.isfinite(n) and n > 0 for n in input_norms):
                raise AssertionError("Recipient or donor input branch was detached")
            report["sets"][str(triples)] = dict(audits=audits, sham_max_abs=sham_error,
                gradient_norm=norm, finite_difference=fd, allowed=allowed,
                recipient_input_grad_norm=input_norms[0], donor_input_grad_norm=input_norms[1])
        # Compiled adjacent objectives must still match the old operator.
        for source in range(depth - 1):
            legacy_id = source * C.HEADS + 1
            old = eqx.filter_jit(lambda m: PS.set_loss(m, x[None], y[None], keys, (legacy_id,)))
            new = eqx.filter_jit(lambda m: C.set_loss(m, x[None], y[None], keys, C.adjacent_paths((legacy_id,))))
            error = difference(old(model), new(model))
            if error > settings["numeric_set_jit_loss_max_abs"]:
                raise AssertionError("Adjacent compiled objective changed")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology-only", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(topology_checks(), indent=2))

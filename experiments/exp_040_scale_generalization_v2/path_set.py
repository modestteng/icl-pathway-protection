"""Multi-head label-K path sets: the minimal extension of exp039 ``mechanism.py``.

A path set is a sorted tuple of exp039 path ids ``source_layer * 8 + local_head``
whose sink is ``source_layer + 1``. For every sink layer present in the set, the
donor-minus-recipient projected contributions of the selected source heads are
summed and added to the CLEAN recipient residual at the two label positions.
The sink's own LayerNorm/attention recomputes label K (V for the matched control)
from that temporary state, exactly as exp039 does for one head. A single
recipient forward then replaces only that channel at label positions:

* lowest sink: recipient Q/K/V are cached exactly as in exp039. Everything
  upstream of the lowest sink is unchanged, so these equal the natural values;
* higher sinks: only the label-position channel is replaced. Q, V and non-label
  K follow the actual forward, so effects of a lower sink propagate naturally.

Every block that is not a sink is recomputed normally. For ``len(set) == 1`` all
operations, in the same order, are those of ``mechanism.patch_example`` and
``mechanism.path_loss``; ``set_checks.py`` verifies field-by-field equality.

The donor swaps only the two context examples. Labels, query features and
positions are unchanged; the donor-supported label is the counterfactual target.
Both branches are recomputed from the current model in every call (no frozen
activation, no stop_gradient). Nothing executes at import time.
"""

from collections import Counter
from numbers import Integral

import mechanism as M

HEADS = 8


def normalize(path_ids, n_layers):
    """Sorted unique ids; every id must index an adjacent layer pair."""
    if isinstance(path_ids, Integral):
        path_ids = (path_ids,)
    ids = tuple(sorted({int(p) for p in path_ids}))
    if not ids:
        raise ValueError('empty path set')
    for p in ids:
        if not 0 <= p < (n_layers - 1) * HEADS:
            raise ValueError('path id outside adjacent layer pairs: ' + str(p))
    return ids


def by_sink(path_ids):
    """{sink_layer: tuple(local heads)} in increasing sink order (pure Python)."""
    groups = {}
    for p in path_ids:
        source, head = divmod(int(p), HEADS)
        groups.setdefault(source + 1, []).append(head)
    return {sink: tuple(sorted(heads)) for sink, heads in sorted(groups.items())}


def describe(path_ids):
    return [dict(path_id=int(p), source_layer=int(p) // HEADS, sink_layer=int(p) // HEADS + 1,
                 local_head=int(p) % HEADS) for p in path_ids]


def _check(model, x, y, path_ids, kind):
    ids = normalize(path_ids, len(model.transformer.blocks))
    for p in ids:
        M._check_contract(model, x, y, p, kind)
    return ids


def _hybrid_set(model, recipient, donor, key, path_ids):
    """{sink: (sink attention channels, temporary residual)} from clean states."""
    _, jax, jnp = M._libraries()
    out = {}
    label_sites = (jnp.arange(M.SEQUENCE_LENGTH) % 2 == 1)[:, None]
    for sink, heads in by_sink(path_ids).items():
        source = sink - 1
        native_h = M._blocks(recipient)[source]["out"]
        delta = None
        for head in heads:
            one = M._head_contribution(model, donor, head, source) - M._head_contribution(
                model, recipient, head, source)
            delta = one if delta is None else delta + one
        hybrid_h = native_h + jnp.where(label_sites, delta, 0)
        block = model.transformer.blocks[sink]
        norm = jax.vmap(block.norm1)(hybrid_h)
        channels = block.attn.call_with_all_aux(
            norm, key=M._l2_attention_key(key, len(model.transformer.blocks), sink))
        out[sink] = (channels, hybrid_h)
    return out


def _set_forward(model, x, y, key, recipient, source_channels, kind):
    """Recipient forward replacing only ``kind`` at label positions of each sink."""
    _, _, jnp = M._libraries()
    depth = len(model.transformer.blocks)
    sinks = sorted(source_channels)
    lowest = sinks[0]
    label_sites = (jnp.arange(M.SEQUENCE_LENGTH) % 2 == 1)[None, None, :, None]
    name = kind.lower()
    native_channels = M._attention(recipient, lowest)
    values = {channel: native_channels[channel] for channel in ("q", "k", "v")}
    values[name] = jnp.where(label_sites, source_channels[lowest][name], values[name])
    block_cache = [{} for _ in range(depth)]
    block_masks = [{} for _ in range(depth)]
    block_cache[lowest] = {"attn_output": values}
    block_masks[lowest] = {"attn_output": {
        channel: jnp.ones_like(value, dtype=bool) for channel, value in values.items()}}
    for sink in sinks[1:]:
        value = source_channels[sink][name]
        block_cache[sink] = {"attn_output": {name: value}}
        block_masks[sink] = {"attn_output": {name: jnp.broadcast_to(label_sites, value.shape)}}
    cache = {"transformer_output": {"block_outputs": block_cache}}
    masks = {"transformer_output": {"block_outputs": block_masks}}
    return model.call_with_all_aux(examples=x, labels=y, key=key, cache=cache, cache_mask=masks)


def _channels(hybrids):
    return {sink: channels for sink, (channels, _) in hybrids.items()}


def _set_audit(model, base, patched, hybrids, kind, path_ids):
    """exp039 audit at the lowest sink plus label-channel audits at higher sinks."""
    _, _, jnp = M._libraries()
    sinks = sorted(hybrids)
    lowest = sinks[0]
    first = min(p for p in path_ids if p // HEADS + 1 == lowest)
    channels, hybrid_h = hybrids[lowest]
    audit = M._path_audit(model, base, patched, channels, hybrid_h, kind, first)
    label_mask = (jnp.arange(M.SEQUENCE_LENGTH) % 2 == 1)[None, None, :, None]
    causal = jnp.tril(jnp.ones((M.SEQUENCE_LENGTH, M.SEQUENCE_LENGTH), dtype=bool))
    name = kind.lower()
    for sink in sinks[1:]:
        ch, hh = hybrids[sink]
        p = M._attention(patched, sink)
        scores = (p["q"] @ jnp.swapaxes(p["k"], -1, -2)) * model.transformer.blocks[sink].attn.scale
        prefix = f"sink{sink}/"
        audit[prefix + "target_channel_max_abs"] = M._finite_max_delta(p[name], ch[name], label_mask)
        audit[prefix + "hybrid_nonlabel_residual_max_abs"] = M._finite_max_delta(
            hh, M._blocks(base)[sink - 1]["out"], (jnp.arange(M.SEQUENCE_LENGTH) % 2 == 0)[:, None])
        audit[prefix + "qk_recompute_max_abs"] = M._finite_max_delta(
            scores, p["attn_pre_softmax"], causal[None, None])
        audit[prefix + "future_attention_max"] = jnp.max(jnp.where(~causal[None, None], p["attn_scores"], 0))
        audit[prefix + "attention_row_sum_max_abs"] = jnp.max(jnp.abs(jnp.sum(p["attn_scores"], axis=-1) - 1))
    return audit


def set_patch_example(model, x, y, key, path_ids, kind="K", *, include_cache=False):
    """Set analogue of ``mechanism.patch_example`` with identical output fields.

    ``path_logits``: selected heads -> their sinks' label K (V if kind='V').
    ``v_path_logits``: same temporary states, label V instead (matched control).
    ``complete_k_logits``/``complete_v_logits``: complete donor label channel at
    every sink in the set. ``self_logits``: sham with donor := recipient.
    ``reverse_*``: recipient information sent into the donor, native target.
    Scalar coordinate fields describe the lowest path; ``path_set`` lists all.
    """
    if not isinstance(include_cache, bool):
        raise TypeError("include_cache must be a static Python bool")
    ids = _check(model, x, y, path_ids, kind)
    _, _, jnp = M._libraries()
    first = ids[0]
    source, sink, local_head = first // HEADS, first // HEADS + 1, first % HEADS
    x, y = jnp.asarray(x), jnp.asarray(y)
    native_target, donor_target, valid = M._targets(y)
    recipient, donor, donor_x = M._pair_caches(model, x, y, key)
    hybrids = _hybrid_set(model, recipient, donor, key, ids)
    path = _set_forward(model, x, y, key, recipient, _channels(hybrids), kind)
    v_path = _set_forward(model, x, y, key, recipient, _channels(hybrids), "V")
    donor_channels = {s: M._attention(donor, s) for s in hybrids}
    complete_k = _set_forward(model, x, y, key, recipient, donor_channels, "K")
    complete_v = _set_forward(model, x, y, key, recipient, donor_channels, "V")
    self_hybrids = _hybrid_set(model, recipient, recipient, key, ids)
    self_path = _set_forward(model, x, y, key, recipient, _channels(self_hybrids), kind)
    reverse_hybrids = _hybrid_set(model, donor, recipient, key, ids)
    reverse_path = _set_forward(model, donor_x, y, key, donor, _channels(reverse_hybrids), kind)
    reverse_v = _set_forward(model, donor_x, y, key, donor, _channels(reverse_hybrids), "V")
    query_label_control = model.call_with_all_aux(
        examples=x, labels=y.at[-1].set(donor_target), key=key)
    audit = _set_audit(model, recipient, path, hybrids, kind, ids)
    audit.update({
        "self_logits_max_abs": M._finite_max_delta(self_path["out"], recipient["out"]),
        "invisible_query_label_logits_max_abs": M._finite_max_delta(
            query_label_control["out"], recipient["out"]),
        "query_input_max_abs": M._finite_max_delta(donor_x[-1], x[-1]),
        "l1_native_head_reconstruction_max_abs": M._head_reconstruction_error(model, recipient, source),
        "l1_donor_head_reconstruction_max_abs": M._head_reconstruction_error(model, donor, source),
    })
    for other in sorted({p // HEADS for p in ids} - {source}):
        audit[f"source{other}/native_head_reconstruction_max_abs"] = M._head_reconstruction_error(
            model, recipient, other)
        audit[f"source{other}/donor_head_reconstruction_max_abs"] = M._head_reconstruction_error(
            model, donor, other)
    padded = list(ids) + [-1] * (4 - len(ids)) if len(ids) <= 4 else list(ids)
    result = {
        "native_logits": recipient["out"][-1],
        "donor_logits": donor["out"][-1],
        "path_logits": path["out"][-1],
        "v_path_logits": v_path["out"][-1],
        "complete_k_logits": complete_k["out"][-1],
        "complete_v_logits": complete_v["out"][-1],
        "self_logits": self_path["out"][-1],
        "query_label_control_logits": query_label_control["out"][-1],
        "reverse_path_logits": reverse_path["out"][-1],
        "reverse_v_path_logits": reverse_v["out"][-1],
        "native_target": native_target,
        "donor_target": donor_target,
        "pair_valid": valid,
        "context_labels": y[:M.CONTEXT_PAIRS],
        "head": jnp.asarray(first, jnp.int32),
        "path_id": jnp.asarray(first, jnp.int32),
        "source_layer": jnp.asarray(source, jnp.int32),
        "sink_layer": jnp.asarray(sink, jnp.int32),
        "local_head": jnp.asarray(local_head, jnp.int32),
        "path_set": jnp.asarray(padded, jnp.int32),
        "set_size": jnp.asarray(len(ids), jnp.int32),
        "audit": audit,
        "reverse_audit": _set_audit(model, donor, reverse_path, reverse_hybrids, kind, ids),
    }
    if include_cache:
        result["audit_cache"] = {
            "source_x": x, "source_y": y, "swapped_x": donor_x, "forward_key": key,
            "native_target": native_target, "donor_target": donor_target,
            "path_set": jnp.asarray(padded, jnp.int32),
            "label_token_mask": jnp.arange(M.SEQUENCE_LENGTH) % 2 == 1,
            "recipient_embedding": recipient["embedding"], "patched_embedding": path["embedding"],
        }
        for s, (channels, hybrid_h) in hybrids.items():
            result["audit_cache"][f"sink{s}_hybrid_k"] = channels["k"]
            result["audit_cache"][f"sink{s}_hybrid_v"] = channels["v"]
            result["audit_cache"][f"sink{s}_hybrid_residual"] = hybrid_h
            result["audit_cache"][f"sink{s}_patched_k"] = M._attention(path, s)["k"]
            result["audit_cache"][f"sink{s}_recipient_k"] = M._attention(recipient, s)["k"]
    return result


def set_loss(model, x, y, keys, path_ids):
    """Mean donor-target full-label CE after the set's label-K replacement.

    Lean training objective (no confirmation controls). For one path this is
    operation-for-operation ``mechanism.path_loss``. Invalid rows give NaN so the
    runner fails instead of silently dropping them.
    """
    _, jax, jnp = M._libraries()
    if len(x.shape) != 3 or len(y.shape) != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("Expected aligned batched x=(B,3,512) and y=(B,3)")
    if x.shape[0] < 1 or len(keys.shape) < 1 or keys.shape[0] != x.shape[0]:
        raise ValueError("A nonempty batch and one explicit PRNG key per row are required")
    ids = _check(model, x[0], y[0], path_ids, "K")

    def one(row_x, row_y, row_key):
        _, donor_target, valid = M._targets(row_y)
        recipient, donor, _ = M._pair_caches(model, row_x, row_y, row_key)
        hybrids = _hybrid_set(model, recipient, donor, row_key, ids)
        result = _set_forward(model, row_x, row_y, row_key, recipient, _channels(hybrids), "K")
        loss = M.complete_ce(result["out"][-1], donor_target)
        return jnp.where(valid, loss, jnp.asarray(jnp.nan, dtype=loss.dtype))

    return jnp.mean(jax.vmap(one, in_axes=(0, 0, 0))(x, y, keys))


def rank_paths(scores):
    """Discovery ranking: larger score first, ties smaller path id (exp039 rule)."""
    values = {int(p): float(v) for p, v in scores.items()}
    return sorted(values, key=lambda p: (-values[p], p))


def choose_prefix(coverages, target):
    """coverages: list for k=1..K. Smallest k reaching target, else argmax (ties smaller k)."""
    finite = [(k + 1, c) for k, c in enumerate(coverages) if c is not None and c == c]
    if not finite:
        return 1, 'no_finite_coverage_default_top1'
    reached = [k for k, c in finite if c >= target]
    if reached:
        return min(reached), 'smallest_prefix_reaching_target'
    best = max(c for _, c in finite)
    return min(k for k, c in finite if c == best), 'target_not_reached_max_coverage_prefix'


def random_control_set(init_seed, train_seed, selected, random_path_seed, heads=HEADS):
    """Outcome-blind matched random set: same size and per-source-layer counts.

    Drawn without replacement from non-selected heads of each source layer by a
    SHA-256-seeded ``random.Random``; persisted before any protected update.
    """
    import hashlib
    import random

    selected = tuple(sorted(int(p) for p in selected))
    for value in (init_seed, train_seed, random_path_seed, *selected):
        if not isinstance(value, Integral):
            raise TypeError("Seed/path identifiers must be integers")
    material = (f"exp040/random-set/v1|init={init_seed}|train={train_seed}"
                f"|random_path_seed={random_path_seed}|selected={','.join(map(str, selected))}"
                ).encode("ascii")
    rng = random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))
    chosen = []
    for source, count in sorted(Counter(p // heads for p in selected).items()):
        pool = [source * heads + h for h in range(heads) if source * heads + h not in selected]
        if len(pool) < count:
            raise ValueError("not enough non-selected heads in source layer " + str(source))
        chosen.extend(rng.sample(pool, count))
    return tuple(sorted(chosen))

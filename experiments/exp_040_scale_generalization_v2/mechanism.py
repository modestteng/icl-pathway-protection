"""Architecture-aware sample--label pairing function for exp039.

This is a minimal extension of the frozen exp036 construction. Public ``head``
arguments now encode ``path_id = source_layer * 8 + local_head``. The sink is
always source_layer + 1. For a two-layer model path IDs 0..7 therefore preserve
the original head meaning. The retained ``l1``/``l2`` audit/cache field names
refer to the selected source/sink; explicit layer/head metadata disambiguate
them. Every block after the sink is recomputed normally, without a cache.

This module implements an unvalidated functional loss, not a coordinate anchor.
The caller must first pass the independent mechanism-confirmation gates before
opening arm P. No model, data, experiment, or JIT is created at import time.

For a valid context-pairing row, R=[A,label_a,B,label_b,A_query], the donor swaps
only the two context examples: D=[B,label_a,A,label_b,A_query]. Labels and
query features are unchanged. The donor-supported target is label_b. The
query target supplied to the author forward is invisible to its interleaved
tokens; only the returned loss uses the correct counterfactual target.

The selected source head's donor-minus-recipient projected contribution is added
only at the two recipient label positions. The author sink norm/attention is
then used to recompute K (or V for a control). A final recipient forward caches
only this changed label channel; its Q, other K/V, upstream residual, input,
and parameters remain those of the current recipient model. Those unchanged
cached values retain gradients. Both donor and recipient depend on the same
current model; there is no stop_gradient and no frozen activation reference.

The author cache and full-width head-projection conventions are those in
experiments/reproduction_support/source/models.py and exp014/run_eval.py.
The sealed caller keeps parameter/data dtype float32 and the author training
step's default matrix precision None. Path objectives and mechanism evaluation
are wrapped locally in matrix precision 'float32'; this module changes no global
JAX configuration. Cached fields belong to that path evaluation definition.
Scientific execution belongs on the approved CUDA host, not at import time.
"""

from numbers import Integral


FULL_LABELS = 12800
CONTEXT_PAIRS = 2
EXAMPLE_WIDTH = 512
SEQUENCE_LENGTH = 5


def _libraries():
    # Lazy imports let protocol/static checks import this module without JAX.
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    return eqx, jax, jnp


def path_spec(model, head):
    """Return static zero-based source, sink and local-head coordinates."""
    blocks = model.transformer.blocks
    if not isinstance(head, Integral):
        raise TypeError("head/path_id must be a static integer")
    if not 0 <= int(head) < (len(blocks) - 1) * 8:
        raise ValueError("path_id must index an adjacent layer pair and head")
    source, local_head = divmod(int(head), 8)
    return dict(path_id=int(head), source_layer=source,
                sink_layer=source + 1, local_head=local_head)


def candidate_path_ids(model):
    """Outcome-blind complete discovery grid over adjacent pairs and heads."""
    if len(model.transformer.blocks) not in (2, 4):
        raise ValueError("exp039 supports depth=2 or depth=4")
    return tuple(range((len(model.transformer.blocks) - 1) * 8))


def _check_contract(model, x, y, head, kind):
    """Static author-architecture checks, safe while tracing filtered JIT."""
    eqx, jax, _ = _libraries()
    if tuple(x.shape) != (CONTEXT_PAIRS + 1, EXAMPLE_WIDTH):
        raise ValueError("Expected a single author row x.shape=(3,512)")
    if tuple(y.shape) != (CONTEXT_PAIRS + 1,):
        raise ValueError("Expected context labels and recipient target y.shape=(3,)")
    if not jax.numpy.issubdtype(y.dtype, jax.numpy.integer):
        raise TypeError("Labels must have an integer dtype")
    if model.num_classes != FULL_LABELS or model.embed_dim not in (64, 128):
        raise ValueError("exp039 requires 12800 labels and d_model=64 or 128")
    blocks = model.transformer.blocks
    if len(blocks) not in (2, 4) or any(block.attn.num_heads != 8 for block in blocks):
        raise ValueError("exp039 requires two/four layers and eight heads")
    if any(not isinstance(block.mlp, eqx.nn.Identity) for block in blocks):
        raise ValueError("Head-to-residual replacement is sealed only for the no-MLP model")
    if any(type(block.drop_path2).__name__ != "Zeros" for block in blocks):
        raise ValueError("The frozen author's no-MLP residual convention is required")
    path_spec(model, head)
    if kind not in ("K", "V"):
        raise ValueError("kind must be K (candidate) or V (matched channel control)")
    # An attention-head contribution does not include an independent draw of
    # projection/path dropout. Fail visibly rather than inventing that rule.
    for obj in jax.tree_util.tree_leaves(
        model, is_leaf=lambda value: isinstance(value, eqx.nn.Dropout)
    ):
        if isinstance(obj, eqx.nn.Dropout) and obj.p != 0:
            raise ValueError("This exact path construction requires all author dropout p=0")


def _blocks(result):
    return result["transformer_output"]["block_outputs"]


def _attention(result, layer):
    return _blocks(result)[layer]["attn_output"]


def _targets(y):
    _, _, jnp = _libraries()
    hits = y[:CONTEXT_PAIRS] == y[-1]
    recipient_position = jnp.argmax(hits)
    donor_target = y[1 - recipient_position]
    # Feature/class identities need an independent input audit. Labels alone
    # cannot establish that the query class is present exactly once.
    valid = (
        (jnp.sum(hits) == 1)
        & (y[0] != y[1])
        & jnp.all((y >= 0) & (y < FULL_LABELS))
    )
    return y[-1], donor_target, valid


def _l2_attention_key(key, depth=2, sink_layer=1):
    """Mirror SequenceClassifier -> Transformer -> Block PRNG splitting."""
    _, jax, _ = _libraries()
    transformer_key = jax.random.split(key, 2)[1]
    block_key = jax.random.split(transformer_key, depth)[sink_layer]
    return jax.random.split(block_key, 4)[0]


def _head_contribution(model, result, head, source_layer=0):
    """Source local head's full-width projection, with projection bias removed."""
    _, jax, jnp = _libraries()
    block = model.transformer.blocks[source_layer]
    z = _attention(result, source_layer)["values_pre_proj"]
    width = z.shape[-1]
    head_width = width // block.attn.num_heads
    where = jnp.arange(width) // head_width == head
    # As in exp014, retain the unchanged full-width author projection. A narrow
    # sub-GEMM would introduce a different numerical computation.
    contribution = jax.vmap(block.attn.proj)(z * where[None, :])
    if block.attn.proj.bias is not None:
        contribution = contribution - block.attn.proj.bias
    return contribution


def _hybrid_attention(model, recipient, donor, key, head):
    """Recompute sink channels from label-only source-head contribution swap."""
    _, jax, jnp = _libraries()
    spec = path_spec(model, head)
    source, sink, local_head = spec["source_layer"], spec["sink_layer"], spec["local_head"]
    native_h = _blocks(recipient)[source]["out"]
    delta = _head_contribution(model, donor, local_head, source) - _head_contribution(
        model, recipient, local_head, source
    )
    label_sites = (jnp.arange(SEQUENCE_LENGTH) % 2 == 1)[:, None]
    hybrid_h = native_h + jnp.where(label_sites, delta, 0)
    block = model.transformer.blocks[sink]
    norm = jax.vmap(block.norm1)(hybrid_h)
    channels = block.attn.call_with_all_aux(
        norm, key=_l2_attention_key(key, len(model.transformer.blocks), sink)
    )
    return channels, hybrid_h


def _channel_forward(model, x, y, key, recipient, source_channels, kind, head=0):
    """Recipient forward with only the designated sink label channel changed.

    Native Q/K/V are explicitly cached, matching exp014's cache construction.
    All these arrays remain differentiable functions of the current model.
    Clamping them within this one forward does not freeze their training updates.
    """
    _, _, jnp = _libraries()
    sink = path_spec(model, head)["sink_layer"]
    depth = len(model.transformer.blocks)
    native_channels = _attention(recipient, sink)
    label_sites = (jnp.arange(SEQUENCE_LENGTH) % 2 == 1)[None, None, :, None]
    name = kind.lower()
    values = {channel: native_channels[channel] for channel in ("q", "k", "v")}
    values[name] = jnp.where(label_sites, source_channels[name], values[name])
    block_cache = [{} for _ in range(depth)]
    block_masks = [{} for _ in range(depth)]
    block_cache[sink] = {"attn_output": values}
    block_masks[sink] = {"attn_output": {
        channel: jnp.ones_like(value, dtype=bool)
        for channel, value in values.items()
    }}
    cache = {"transformer_output": {"block_outputs": block_cache}}
    masks = {"transformer_output": {"block_outputs": block_masks}}
    return model.call_with_all_aux(
        examples=x, labels=y, key=key, cache=cache, cache_mask=masks
    )


def _pair_caches(model, x, y, key):
    _, _, jnp = _libraries()
    donor_x = x[jnp.array([1, 0, 2])]
    # The final query label is never interleaved by the author; keeping y
    # unchanged preserves visible context label identity and token positions.
    recipient = model.call_with_all_aux(examples=x, labels=y, key=key)
    donor = model.call_with_all_aux(examples=donor_x, labels=y, key=key)
    return recipient, donor, donor_x


def complete_ce(logits, target):
    """Scalar cross entropy over every one of the 12800 output labels."""
    _, jax, _ = _libraries()
    if tuple(logits.shape) != (FULL_LABELS,):
        raise ValueError("Complete CE requires logits.shape=(12800,), no candidate mask")
    return jax.scipy.special.logsumexp(logits) - logits[target]


def _finite_max_delta(left, right, where=None):
    _, _, jnp = _libraries()
    delta = jnp.abs(left - right)
    if where is not None:
        delta = jnp.where(where, delta, 0)
    return jnp.max(delta)


def _head_reconstruction_error(model, result, source_layer=0):
    _, jax, jnp = _libraries()
    contributions = jax.vmap(
        lambda head: _head_contribution(model, result, head, source_layer)
    )(jnp.arange(model.transformer.blocks[source_layer].attn.num_heads))
    reconstructed = jnp.sum(contributions, axis=0)
    bias = model.transformer.blocks[source_layer].attn.proj.bias
    if bias is not None:
        reconstructed = reconstructed + bias
    return _finite_max_delta(reconstructed, _attention(result, source_layer)["values_post_proj"])


def _path_audit(model, recipient, path, channels, hybrid_h, kind, head=0):
    _, _, jnp = _libraries()
    spec = path_spec(model, head)
    source, sink = spec["source_layer"], spec["sink_layer"]
    native = _attention(recipient, sink)
    patched = _attention(path, sink)
    label_mask = (jnp.arange(SEQUENCE_LENGTH) % 2 == 1)[None, None, :, None]
    causal = jnp.tril(jnp.ones((SEQUENCE_LENGTH, SEQUENCE_LENGTH), dtype=bool))
    name = kind.lower()
    other = "v" if name == "k" else "k"
    valid_scores = (
        patched["q"] @ jnp.swapaxes(patched["k"], -1, -2)
    ) * model.transformer.blocks[sink].attn.scale
    return {
        "embedding_max_abs": _finite_max_delta(path["embedding"], recipient["embedding"]),
        "l1_upstream_max_abs": _finite_max_delta(
            _blocks(path)[source]["out"], _blocks(recipient)[source]["out"]
        ),
        "l1_values_pre_proj_max_abs": _finite_max_delta(
            _attention(path, source)["values_pre_proj"], _attention(recipient, source)["values_pre_proj"]
        ),
        "l1_values_post_proj_max_abs": _finite_max_delta(
            _attention(path, source)["values_post_proj"], _attention(recipient, source)["values_post_proj"]
        ),
        "l2_norm_input_max_abs": _finite_max_delta(
            _blocks(path)[sink]["norm_inp"], _blocks(recipient)[sink]["norm_inp"]
        ),
        "l2_raw_qkv_max_abs": _finite_max_delta(patched["qkv"], native["qkv"]),
        "q_max_abs": _finite_max_delta(patched["q"], native["q"]),
        "other_channel_max_abs": _finite_max_delta(patched[other], native[other]),
        "nonlabel_channel_max_abs": _finite_max_delta(
            patched[name], native[name], ~label_mask
        ),
        "target_channel_max_abs": _finite_max_delta(
            patched[name], channels[name], label_mask
        ),
        "hybrid_nonlabel_residual_max_abs": _finite_max_delta(
            hybrid_h, _blocks(recipient)[source]["out"],
            (jnp.arange(SEQUENCE_LENGTH) % 2 == 0)[:, None],
        ),
        "qk_recompute_max_abs": _finite_max_delta(
            valid_scores, patched["attn_pre_softmax"], causal[None, None]
        ),
        "future_attention_max": jnp.max(
            jnp.where(~causal[None, None], patched["attn_scores"], 0)
        ),
        "attention_row_sum_max_abs": jnp.max(
            jnp.abs(jnp.sum(patched["attn_scores"], axis=-1) - 1)
        ),
    }


def patch_example(model, x, y, key, head: int, kind="K", *, include_cache=False):
    """Return final full logits/targets and exact-control audit for one row.

    ``path_logits`` contains the selected-head -> sink label-K intervention
    (label-V if ``kind='V'``). ``donor_logits`` is the complete donor input;
    ``complete_k_logits``/``complete_v_logits`` patch the complete donor label
    channel, not only the selected-head source. ``v_path_logits`` is the same
    selected source-head -> label-V control. ``self_logits`` repeats the path
    computation with identical donor/recipient caches. Reverse paths send R's
    selected-head information into D and are evaluated against native_target.

    ``include_cache`` is a static Python bool, default False. True adds per-row
    ``audit_cache`` arrays for independent structural readback on the fixed
    first128 audit rows. It returns inputs and activations, never parameters.
    The lean path_loss does not call this optional cache branch.

    ``pair_valid`` checks only label consistency. The caller must independently
    audit actual feature/class identities and exclude CIWL rows where the query
    class is absent; applying this swap's target rule there is scientifically
    invalid. All forwards use the same model and key. No result is validated
    merely by the existence of these returned diagnostics.
    """
    if not isinstance(include_cache, bool):
        raise TypeError("include_cache must be a static Python bool")
    _check_contract(model, x, y, head, kind)
    _, _, jnp = _libraries()
    spec = path_spec(model, head)
    source, sink, local_head = spec["source_layer"], spec["sink_layer"], spec["local_head"]
    x, y = jnp.asarray(x), jnp.asarray(y)
    native_target, donor_target, valid = _targets(y)
    recipient, donor, donor_x = _pair_caches(model, x, y, key)
    channels, hybrid_h = _hybrid_attention(model, recipient, donor, key, head)
    path = _channel_forward(model, x, y, key, recipient, channels, kind, head)
    v_path = _channel_forward(model, x, y, key, recipient, channels, "V", head)
    complete_k = _channel_forward(
        model, x, y, key, recipient, _attention(donor, sink), "K", head
    )
    complete_v = _channel_forward(
        model, x, y, key, recipient, _attention(donor, sink), "V", head
    )
    self_channels, _ = _hybrid_attention(model, recipient, recipient, key, head)
    self_path = _channel_forward(model, x, y, key, recipient, self_channels, kind, head)
    reverse_channels, reverse_h = _hybrid_attention(model, donor, recipient, key, head)
    reverse_path = _channel_forward(
        model, donor_x, y, key, donor, reverse_channels, kind, head
    )
    reverse_v = _channel_forward(
        model, donor_x, y, key, donor, reverse_channels, "V", head
    )
    query_label_control = model.call_with_all_aux(
        examples=x, labels=y.at[-1].set(donor_target), key=key
    )
    audit = _path_audit(model, recipient, path, channels, hybrid_h, kind, head)
    audit.update({
        "self_logits_max_abs": _finite_max_delta(self_path["out"], recipient["out"]),
        "invisible_query_label_logits_max_abs": _finite_max_delta(
            query_label_control["out"], recipient["out"]
        ),
        "query_input_max_abs": _finite_max_delta(donor_x[-1], x[-1]),
        "l1_native_head_reconstruction_max_abs": _head_reconstruction_error(model, recipient, source),
        "l1_donor_head_reconstruction_max_abs": _head_reconstruction_error(model, donor, source),
    })
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
        "context_labels": y[:CONTEXT_PAIRS],
        "head": jnp.asarray(head, jnp.int32),
        "path_id": jnp.asarray(head, jnp.int32),
        "source_layer": jnp.asarray(source, jnp.int32),
        "sink_layer": jnp.asarray(sink, jnp.int32),
        "local_head": jnp.asarray(local_head, jnp.int32),
        "audit": audit,
        "reverse_audit": _path_audit(
            model, donor, reverse_path, reverse_channels, reverse_h, kind, head
        ),
    }
    if include_cache:
        recipient_blocks, donor_blocks, patched_blocks = (
            _blocks(recipient), _blocks(donor), _blocks(path)
        )
        recipient_l1, donor_l1, patched_l1 = (
            _attention(recipient, source), _attention(donor, source), _attention(path, source)
        )
        recipient_l2, patched_l2 = _attention(recipient, sink), _attention(path, sink)
        head_width = model.embed_dim // model.transformer.blocks[source].attn.num_heads
        head_mask = jnp.arange(model.embed_dim) // head_width == local_head
        result["audit_cache"] = {
            "source_x": x,
            "source_y": y,
            "swapped_x": donor_x,
            "forward_key": key,
            "native_target": native_target,
            "donor_target": donor_target,
            "selected_head": jnp.asarray(head, jnp.int32),
            "path_id": jnp.asarray(head, jnp.int32),
            "source_layer": jnp.asarray(source, jnp.int32),
            "sink_layer": jnp.asarray(sink, jnp.int32),
            "local_head": jnp.asarray(local_head, jnp.int32),
            "patched_channel_is_k": jnp.asarray(kind == "K"),
            "label_token_mask": jnp.arange(SEQUENCE_LENGTH) % 2 == 1,
            "head_dim_mask": head_mask,
            "recipient_embedding": recipient["embedding"],
            "patched_embedding": path["embedding"],
            "recipient_l1_out": recipient_blocks[source]["out"],
            "donor_l1_out": donor_blocks[source]["out"],
            "patched_l1_out": patched_blocks[source]["out"],
            "native_l1_residual": recipient_blocks[source]["out"],
            "hybrid_l1_residual": hybrid_h,
            "recipient_l1_residual_post_attn": recipient_blocks[source]["residual_post_attn"],
            "donor_l1_residual_post_attn": donor_blocks[source]["residual_post_attn"],
            "patched_l1_residual_post_attn": patched_blocks[source]["residual_post_attn"],
            "recipient_l1_values_pre_proj": recipient_l1["values_pre_proj"],
            "donor_l1_values_pre_proj": donor_l1["values_pre_proj"],
            "patched_l1_values_pre_proj": patched_l1["values_pre_proj"],
            "recipient_l1_values_post_proj": recipient_l1["values_post_proj"],
            "donor_l1_values_post_proj": donor_l1["values_post_proj"],
            "patched_l1_values_post_proj": patched_l1["values_post_proj"],
            "native_head_values_pre_proj": recipient_l1["values_pre_proj"] * head_mask[None, :],
            "donor_head_values_pre_proj": donor_l1["values_pre_proj"] * head_mask[None, :],
            "native_head_projected": _head_contribution(model, recipient, local_head, source),
            "donor_head_projected": _head_contribution(model, donor, local_head, source),
            "recipient_l2_q": recipient_l2["q"],
            "recipient_l2_k": recipient_l2["k"],
            "recipient_l2_v": recipient_l2["v"],
            "recipient_l2_qkv": recipient_l2["qkv"],
            "patched_l2_q": patched_l2["q"],
            "patched_l2_k": patched_l2["k"],
            "patched_l2_v": patched_l2["v"],
            "patched_l2_qkv": patched_l2["qkv"],
            "hybrid_l2_k": channels["k"],
            "hybrid_l2_v": channels["v"],
            "hybrid_l2_norm_input": channels["x"],
            "recipient_l2_norm_input": recipient_blocks[sink]["norm_inp"],
            "patched_l2_norm_input": patched_blocks[sink]["norm_inp"],
            "recipient_l2_attn_scores": recipient_l2["attn_scores"],
            "patched_l2_attn_scores": patched_l2["attn_scores"],
            "recipient_l2_attn_pre_softmax": recipient_l2["attn_pre_softmax"],
            "patched_l2_attn_pre_softmax": patched_l2["attn_pre_softmax"],
        }
    return result


def path_loss(model, x, y, keys, head: int):
    """Mean selected-head -> label-K donor-target full CE for a valid batch.

    Inputs: x=(B,3,512), y=(B,3), one explicit key per row (legacy keys=(B,2)
    or typed keys=(B,)). ``head`` is static and frozen by the discovery rule.
    The function recomputes both branches under the current model at each call.
    It does not hold K or a donor representation fixed across training steps.
    Invalid label contracts yield NaN; the training runner must fail on any
    nonfinite loss/gradient instead of dropping or silently reinterpreting rows.

    This is deliberately a lean path-only calculation. Confirmation controls
    in patch_example do not add computation to the protected training loss.
    The existing ordinary-ICL and fixed-label-swap protection losses retain
    equal fixed weights. The latter is a training condition, not an additional
    reported task-performance metric.
    """
    _, jax, jnp = _libraries()
    if len(x.shape) != 3 or len(y.shape) != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("Expected aligned batched x=(B,3,512) and y=(B,3)")
    if x.shape[0] < 1 or len(keys.shape) < 1 or keys.shape[0] != x.shape[0]:
        raise ValueError("A nonempty batch and one explicit PRNG key per row are required")
    _check_contract(model, x[0], y[0], head, "K")

    def one(row_x, row_y, row_key):
        _, donor_target, valid = _targets(row_y)
        recipient, donor, _ = _pair_caches(model, row_x, row_y, row_key)
        channels, _ = _hybrid_attention(model, recipient, donor, row_key, head)
        result = _channel_forward(
            model, row_x, row_y, row_key, recipient, channels, "K", head
        )
        loss = complete_ce(result["out"][-1], donor_target)
        return jnp.where(valid, loss, jnp.asarray(jnp.nan, dtype=loss.dtype))

    return jnp.mean(jax.vmap(one, in_axes=(0, 0, 0))(x, y, keys))


def random_control_head(
    init_seed: int,
    train_seed: int,
    causal_head: int,
    random_path_seed: int = 20261030,
):
    """Outcome-blind random path in the selected adjacent pair, different head.

    Persist the resulting ID before any confirmation/training outputs. This
    matched control is unvalidated; it is allowed to have causal effects.
    Loss-gradient norm scaling and the same rho belong to the update solver.
    """
    import hashlib
    import random

    if not all(isinstance(value, Integral) for value in (
        init_seed, train_seed, causal_head, random_path_seed
    )):
        raise TypeError("Seed/head identifiers must be integers")
    if not 0 <= causal_head < 24:
        raise ValueError("causal_head/path_id must be in [0,23]")
    source, local_head = divmod(int(causal_head), 8)
    material = (
        f"exp030/random-head/v1|init={init_seed}|train={train_seed}"
        f"|random_path_seed={random_path_seed}"
    ).encode("ascii")
    seed = int.from_bytes(hashlib.sha256(material).digest(), "big")
    chosen = random.Random(seed).choice([head for head in range(8) if head != local_head])
    return source * 8 + chosen

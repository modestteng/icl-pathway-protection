"""Validated-source candidate operators for exp047.

Paths are explicit ZERO-BASED ``(source_layer, sink_layer, local_head)`` triples.
Every source precedes its sink, but need not be adjacent. At each sink, selected
donor-minus-recipient head contributions are added ONCE to the clean recipient
residual immediately before the sink. Only the two label positions are changed
in that temporary state. The sink's own LayerNorm and attention recompute K;
the recipient forward replaces only label K (V is the matched control).

This extends the source domain, not the optimizer or the scientific gates. The
all-earlier-heads construction closes to donor label residual/K for the sealed
attention-only, zero-dropout architecture with identical label embeddings.
It does not establish that a selected small source set is behaviorally faithful.
Both branches remain differentiable; no activation is frozen across updates.
No model computation or scientific dependency import occurs at import time.
"""

from functools import lru_cache
import importlib.util
from numbers import Integral
from pathlib import Path
import sys

HEADS = 8


@lru_cache(maxsize=1)
def legacy_modules():
    """Load exact exp040 kernels by path, without relying on cwd/import order."""
    root = Path(__file__).resolve().parents[1] / "exp_040_scale_generalization_v2"

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, root / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    mechanism = load("_exp047_mechanism040", "mechanism.py")
    # path_set.py binds its mechanism module at import time. Restore any caller's
    # module immediately, so an exp040 driver never silently changes kernels.
    sentinel = object()
    previous = sys.modules.get("mechanism", sentinel)
    sys.modules["mechanism"] = mechanism
    try:
        path_set = load("_exp047_path_set040", "path_set.py")
    finally:
        if previous is sentinel:
            del sys.modules["mechanism"]
        else:
            sys.modules["mechanism"] = previous
    return mechanism, path_set


def candidate_paths(model_or_depth):
    """Full earlier-source grid: 8 paths at depth 2, 48 at depth 4."""
    depth = (int(model_or_depth) if isinstance(model_or_depth, Integral)
             else len(model_or_depth.transformer.blocks))
    if depth not in (2, 4):
        raise ValueError("The sealed experiment supports depth 2 or 4")
    return tuple((source, sink, head)
                 for source in range(depth - 1)
                 for sink in range(source + 1, depth)
                 for head in range(HEADS))


def normalize(paths, n_layers):
    """Validate and deduplicate triples; no ambiguous legacy integer decoding."""
    if isinstance(paths, Integral):
        raise TypeError("Use explicit (source, sink, head) triples, not exp040 ids")
    values = tuple(paths)
    if any(isinstance(v, Integral) for v in values):
        raise TypeError("Use an outer sequence of triples, e.g. ((0, 3, 2),); legacy id lists are invalid")
    triples = []
    for value in values:
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            raise TypeError("Each path must be a (source, sink, head) triple")
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in value):
            raise TypeError("Path coordinates must be integer layer/head indices")
        source, sink, head = map(int, value)
        if not (0 <= source < sink < n_layers and 0 <= head < HEADS):
            raise ValueError("Invalid earlier-source path: " + str(value))
        triples.append((source, sink, head))
    if not triples:
        raise ValueError("empty path set")
    return tuple(sorted(set(triples)))


def by_sink(paths):
    """Map each sink to sorted unique (source, head) coordinates."""
    groups = {}
    for source, sink, head in paths:
        groups.setdefault(sink, set()).add((source, head))
    return {sink: tuple(sorted(sources)) for sink, sources in sorted(groups.items())}


def describe(paths):
    return [dict(source_layer=int(source), sink_layer=int(sink), local_head=int(head))
            for source, sink, head in paths]


def adjacent_paths(legacy_ids):
    """Explicit adapter used only for exp040 comparison arms and identity checks."""
    return tuple((int(p) // HEADS, int(p) // HEADS + 1, int(p) % HEADS)
                 for p in legacy_ids)


def _check(model, x, y, paths, kind):
    triples = normalize(paths, len(model.transformer.blocks))
    mechanism, _ = legacy_modules()
    # The legacy contract checks the sealed architecture, tensor shapes,
    # dropout, label count and kind. Its adjacent coordinate is used ONLY to
    # exercise that architecture contract; actual sinks remain explicit here.
    source, _, head = triples[0]
    mechanism._check_contract(model, x, y, source * HEADS + head, kind)
    return triples


def _hybrid_set(model, recipient, donor, key, paths):
    """Build sink channels from selected earlier-layer residual contributions."""
    mechanism, _ = legacy_modules()
    _, jax, jnp = mechanism._libraries()
    out = {}
    label_sites = (jnp.arange(mechanism.SEQUENCE_LENGTH) % 2 == 1)[:, None]
    for sink, sources in by_sink(paths).items():
        native_h = mechanism._blocks(recipient)[sink - 1]["out"]
        delta = None
        for source, head in sources:
            one = mechanism._head_contribution(model, donor, head, source) - mechanism._head_contribution(
                model, recipient, head, source)
            delta = one if delta is None else delta + one
        # Do not add donor-minus-recipient full residual as well: the selected
        # earlier head differences already are its explicitly chosen terms.
        hybrid_h = native_h + jnp.where(label_sites, delta, 0)
        block = model.transformer.blocks[sink]
        norm = jax.vmap(block.norm1)(hybrid_h)
        channels = block.attn.call_with_all_aux(
            norm, key=mechanism._l2_attention_key(key, len(model.transformer.blocks), sink))
        out[sink] = (channels, hybrid_h)
    return out


def _set_forward(model, x, y, key, recipient, source_channels, kind):
    return legacy_modules()[1]._set_forward(
        model, x, y, key, recipient, source_channels, kind)


def _channels(hybrids):
    return {sink: channels for sink, (channels, _) in hybrids.items()}


def _set_audit(model, base, patched, hybrids, kind):
    """Retain exp040 invariants at actual sinks, regardless of source distance."""
    mechanism, _ = legacy_modules()
    _, _, jnp = mechanism._libraries()
    lowest = min(hybrids)
    channels, hybrid_h = hybrids[lowest]
    # _path_audit needs coordinates only to identify the sink and the block
    # immediately before it. This is a prefix-preservation audit, not a claim
    # that the selected source is adjacent. Explicit source metadata is below.
    audit = mechanism._path_audit(
        model, base, patched, channels, hybrid_h, kind, (lowest - 1) * HEADS)
    label_mask = (jnp.arange(mechanism.SEQUENCE_LENGTH) % 2 == 1)[None, None, :, None]
    causal = jnp.tril(jnp.ones((mechanism.SEQUENCE_LENGTH, mechanism.SEQUENCE_LENGTH), dtype=bool))
    name = kind.lower()
    for sink in sorted(hybrids):
        if sink == lowest:
            continue
        channels, hybrid_h = hybrids[sink]
        actual = mechanism._attention(patched, sink)
        scores = (actual["q"] @ jnp.swapaxes(actual["k"], -1, -2)) * model.transformer.blocks[sink].attn.scale
        prefix = f"sink{sink}/"
        audit[prefix + "target_channel_max_abs"] = mechanism._finite_max_delta(
            actual[name], channels[name], label_mask)
        audit[prefix + "hybrid_nonlabel_residual_max_abs"] = mechanism._finite_max_delta(
            hybrid_h, mechanism._blocks(base)[sink - 1]["out"],
            (jnp.arange(mechanism.SEQUENCE_LENGTH) % 2 == 0)[:, None])
        audit[prefix + "qk_recompute_max_abs"] = mechanism._finite_max_delta(
            scores, actual["attn_pre_softmax"], causal[None, None])
        audit[prefix + "future_attention_max"] = jnp.max(
            jnp.where(~causal[None, None], actual["attn_scores"], 0))
        audit[prefix + "attention_row_sum_max_abs"] = jnp.max(
            jnp.abs(jnp.sum(actual["attn_scores"], axis=-1) - 1))
    return audit


def set_patch_example(model, x, y, key, paths, kind="K", *, include_cache=False):
    """Audited patch with exp040 logit/control fields and explicit path triples."""
    if not isinstance(include_cache, bool):
        raise TypeError("include_cache must be a static Python bool")
    triples = _check(model, x, y, paths, kind)
    mechanism, _ = legacy_modules()
    _, _, jnp = mechanism._libraries()
    x, y = jnp.asarray(x), jnp.asarray(y)
    native_target, donor_target, valid = mechanism._targets(y)
    recipient, donor, donor_x = mechanism._pair_caches(model, x, y, key)
    hybrids = _hybrid_set(model, recipient, donor, key, triples)
    path = _set_forward(model, x, y, key, recipient, _channels(hybrids), kind)
    v_path = _set_forward(model, x, y, key, recipient, _channels(hybrids), "V")
    donor_channels = {sink: mechanism._attention(donor, sink) for sink in hybrids}
    complete_k = _set_forward(model, x, y, key, recipient, donor_channels, "K")
    complete_v = _set_forward(model, x, y, key, recipient, donor_channels, "V")
    self_hybrids = _hybrid_set(model, recipient, recipient, key, triples)
    self_path = _set_forward(model, x, y, key, recipient, _channels(self_hybrids), kind)
    reverse_hybrids = _hybrid_set(model, donor, recipient, key, triples)
    reverse = _set_forward(model, donor_x, y, key, donor, _channels(reverse_hybrids), kind)
    reverse_v = _set_forward(model, donor_x, y, key, donor, _channels(reverse_hybrids), "V")
    query_control = model.call_with_all_aux(
        examples=x, labels=y.at[-1].set(donor_target), key=key)
    audit = _set_audit(model, recipient, path, hybrids, kind)
    audit.update({
        "self_logits_max_abs": mechanism._finite_max_delta(self_path["out"], recipient["out"]),
        "invisible_query_label_logits_max_abs": mechanism._finite_max_delta(query_control["out"], recipient["out"]),
        "query_input_max_abs": mechanism._finite_max_delta(donor_x[-1], x[-1]),
    })
    for source in sorted({source for source, _, _ in triples}):
        audit[f"source{source}/native_head_reconstruction_max_abs"] = mechanism._head_reconstruction_error(model, recipient, source)
        audit[f"source{source}/donor_head_reconstruction_max_abs"] = mechanism._head_reconstruction_error(model, donor, source)
    first = min(triples, key=lambda p: (p[1], p[0], p[2]))
    source, sink, head = first
    result = {
        "native_logits": recipient["out"][-1], "donor_logits": donor["out"][-1],
        "path_logits": path["out"][-1], "v_path_logits": v_path["out"][-1],
        "complete_k_logits": complete_k["out"][-1], "complete_v_logits": complete_v["out"][-1],
        "self_logits": self_path["out"][-1], "query_label_control_logits": query_control["out"][-1],
        "reverse_path_logits": reverse["out"][-1], "reverse_v_path_logits": reverse_v["out"][-1],
        "native_target": native_target, "donor_target": donor_target, "pair_valid": valid,
        "context_labels": y[:mechanism.CONTEXT_PAIRS],
        "source_layer": jnp.asarray(source, jnp.int32), "sink_layer": jnp.asarray(sink, jnp.int32),
        "local_head": jnp.asarray(head, jnp.int32), "head": jnp.asarray(head, jnp.int32),
        "path_id": jnp.asarray(first, jnp.int32), "path_set": jnp.asarray(triples, jnp.int32),
        "set_size": jnp.asarray(len(triples), jnp.int32), "audit": audit,
        "reverse_audit": _set_audit(model, donor, reverse, reverse_hybrids, kind),
    }
    if include_cache:
        result["audit_cache"] = {
            "source_x": x, "source_y": y, "swapped_x": donor_x, "forward_key": key,
            "native_target": native_target, "donor_target": donor_target,
            "path_set": jnp.asarray(triples, jnp.int32),
            "label_token_mask": jnp.arange(mechanism.SEQUENCE_LENGTH) % 2 == 1,
            "recipient_embedding": recipient["embedding"], "patched_embedding": path["embedding"],
        }
        for actual_sink, (channels, hybrid_h) in hybrids.items():
            result["audit_cache"][f"sink{actual_sink}_hybrid_k"] = channels["k"]
            result["audit_cache"][f"sink{actual_sink}_hybrid_v"] = channels["v"]
            result["audit_cache"][f"sink{actual_sink}_hybrid_residual"] = hybrid_h
            result["audit_cache"][f"sink{actual_sink}_patched_k"] = mechanism._attention(path, actual_sink)["k"]
            result["audit_cache"][f"sink{actual_sink}_recipient_k"] = mechanism._attention(recipient, actual_sink)["k"]
    return result


def set_loss(model, x, y, keys, paths):
    """Mean full-label donor-target CE; invalid pairing rows produce NaN."""
    mechanism, _ = legacy_modules()
    _, jax, jnp = mechanism._libraries()
    if len(x.shape) != 3 or len(y.shape) != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("Expected aligned batched x=(B,3,512) and y=(B,3)")
    if x.shape[0] < 1 or len(keys.shape) < 1 or keys.shape[0] != x.shape[0]:
        raise ValueError("A nonempty batch and one PRNG key per row are required")
    triples = _check(model, x[0], y[0], paths, "K")

    def one(row_x, row_y, row_key):
        _, donor_target, valid = mechanism._targets(row_y)
        recipient, donor, _ = mechanism._pair_caches(model, row_x, row_y, row_key)
        hybrids = _hybrid_set(model, recipient, donor, row_key, triples)
        patched = _set_forward(model, row_x, row_y, row_key, recipient, _channels(hybrids), "K")
        loss = mechanism.complete_ce(patched["out"][-1], donor_target)
        return jnp.where(valid, loss, jnp.asarray(jnp.nan, dtype=loss.dtype))

    return jnp.mean(jax.vmap(one, in_axes=(0, 0, 0))(x, y, keys))

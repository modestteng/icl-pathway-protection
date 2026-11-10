"""Generate a prespecified confirmation set using unmodified author samplers.

Call ONLY after the search candidate and all confirmation conditions are frozen.
This module does not load, evaluate, or select models. The caller must put the
frozen author source on sys.path and verify the parent feature-file identity.
The seed here is the evaluation DATA key itself, not main.py's outer eval_seed.
No overlap, duplicate, exact exemplar match, or result is removed or resampled.
"""

from collections import Counter
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np


AUTHOR_SAMPLER_SHA256 = (
    "d285d7d6796996162c0e445bd4d93cdd54c8e9d400bb69e81d054e1bc7a841da"
)
AUTHOR_ORDER = ("train_eval", "iwl_copy_avail", "flip_icl", "icl", "pure_iwl")
SETS = AUTHOR_ORDER[:4]
CONFIGS = {
    "train_eval": dict(burstiness=1, no_support=False, unique_rest=False,
                       assign_query_label_random=False, fs_relabel=None),
    "iwl_copy_avail": dict(burstiness=0, no_support=True, unique_rest=True,
                           assign_query_label_random=True, fs_relabel=None),
    "flip_icl": dict(burstiness=1, no_support=False, unique_rest=False,
                     assign_query_label_random=False, fs_relabel="flip"),
    "icl": dict(burstiness=1, no_support=False, unique_rest=False,
                assign_query_label_random=False, fs_relabel="01"),
}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(*arrays):
    """Hash shapes, canonical dtypes, and C-order bytes without concatenating."""
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(json.dumps([list(array.shape), array.dtype.str]).encode())
        digest.update(b"\0")
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _row_keys(x, y):
    return [_array_sha(a, b) for a, b in zip(x, y)]


def _feature_lookup(features):
    lookup = {}
    for class_index in range(1600):
        for exemplar_index in range(5):
            key = hashlib.sha256(features[class_index, exemplar_index].tobytes()).digest()
            _require(key not in lookup, "Duplicate feature vectors: identity is ambiguous")
            lookup[key] = (class_index, exemplar_index)
    return lookup


def _validate(name, xy, features, lookup, expected_n=None):
    """Recover exact feature identities and check the four task definitions."""
    x, y = (np.asarray(value) for value in xy)
    _require(x.dtype == np.dtype("float32"), f"{name}: examples must be float32")
    _require(x.ndim == 3 and x.shape[1:] == (3, 512), f"{name}: bad example shape")
    _require(y.shape == x.shape[:2] and np.issubdtype(y.dtype, np.integer),
             f"{name}: bad label shape/dtype")
    _require(expected_n is None or x.shape[0] == expected_n, f"{name}: wrong row count")
    _require(x.shape[0] > 0 and np.isfinite(x).all(), f"{name}: empty or nonfinite examples")
    _require(np.all((0 <= y) & (y < 1600)), f"{name}: labels outside class range")
    x, y = np.ascontiguousarray(x), np.asarray(y, dtype=np.int32, order="C")
    flat_ids = []
    for vector in x.reshape(-1, 512):
        key = hashlib.sha256(vector.tobytes()).digest()
        _require(key in lookup, f"{name}: feature not in approved 1600-by-5 subset")
        identity = lookup[key]
        _require(vector.tobytes() == features[identity].tobytes(),
                 f"{name}: SHA collision or feature mismatch")
        flat_ids.append(identity)
    ids = np.asarray(flat_ids, dtype=np.int32).reshape(x.shape[0], 3, 2)
    classes, exemplars = ids[:, :, 0], ids[:, :, 1]
    qc = classes[:, -1]
    same = classes[:, :2] == qc[:, None]
    _require(np.all(classes[:, 0] != classes[:, 1]), f"{name}: duplicate context classes")
    _require(np.all(y[:, 0] != y[:, 1]), f"{name}: duplicate candidate labels")
    target = y[:, :2] == y[:, -1, None]
    _require(np.all(target.sum(axis=1) == 1), f"{name}: nonunique target candidate")
    if name == "iwl_copy_avail":
        _require(np.all(same.sum(axis=1) == 0), "CIWL query appears in context classes")
        _require(np.array_equal(y[:, -1], qc), "CIWL query label is not its class")
        _require(np.array_equal(y[:, :2][~target], classes[:, :2][~target]),
                 "CIWL unaffected context label changed")
    else:
        _require(np.all(same.sum(axis=1) == 1), f"{name}: query support is not unique")
        _require(np.array_equal(y[:, :2][same], y[:, -1]), f"{name}: wrong support label")
        if name == "train_eval":
            _require(np.array_equal(y, classes), "train_eval labels must equal all classes")
        elif name == "flip_icl":
            _require(np.array_equal(y[:, :2], classes[:, 1::-1]),
                     "Flip context labels are not swapped class labels")
            _require(np.all(y[:, -1] != qc), "Flip target must conflict with learned class")
            _require(np.array_equal(y[:, :2][~same], qc), "Flip opposite label is not query class")
        elif name == "icl":
            _require(np.all(np.sort(y[:, :2], axis=1) == np.array([0, 1])),
                     "ICL candidates must be a permutation of 0 and 1")
        else:
            raise ValueError(f"Unknown evaluator {name}")
    report = {
        "count": int(x.shape[0]),
        "query_classes": int(np.unique(qc).size),
        "identity_verified": True,
        "task_definition_verified": True,
        "array_sha256": _array_sha(x, y),
        "identities_sha256": _array_sha(ids),
        "target_label_position_counts": np.bincount(np.argmax(target, axis=1), minlength=2).tolist(),
        "query_support_exact_exemplar_rows": int(np.any(
            same & (exemplars[:, :2] == exemplars[:, -1, None]), axis=1).sum()),
    }
    return (x, y), ids, report


def _audit_all(data, features, lookup, original_rows, n, stored_ids=None):
    checked, identities, reports, all_rows = {}, {}, {}, []
    original_union = set().union(*(set(rows) for rows in original_rows.values()))
    for name in SETS:
        checked[name], identities[name], reports[name] = _validate(
            name, data[name], features, lookup, expected_n=n)
        if stored_ids is not None:
            _require(np.array_equal(stored_ids[name], identities[name]),
                     f"{name}: persisted identities do not match examples")
        rows = _row_keys(*checked[name])
        counts, old_counts = Counter(rows), Counter(original_rows[name])
        report = reports[name]
        report.update({
            "unique_exact_rows": len(counts),
            "duplicate_rows_beyond_first": len(rows) - len(counts),
            "rows_in_repeated_groups": sum(count for count in counts.values() if count > 1),
            "overlap_rows_with_original_same_task": sum(row in old_counts for row in rows),
            "overlap_unique_rows_with_original_same_task": len(set(rows) & set(old_counts)),
            "overlap_row_pairs_with_original_same_task": sum(count * old_counts[row]
                                                              for row, count in counts.items()),
            "same_row_index_matches_with_original": sum(a == b for a, b in
                                                          zip(rows, original_rows[name])),
            "overlap_rows_with_original_any_task": sum(row in original_union for row in rows),
        })
        all_rows.extend(rows)
    audit = {
        "tasks": reports,
        "all_tasks_duplicate_rows_beyond_first": len(all_rows) - len(set(all_rows)),
        "all_tasks_overlap_rows_with_original_any_task": sum(row in original_union for row in all_rows),
        "row_identity_definition": "Exact ordered float32 examples AND int32 labels, including target",
        "filtering_or_resampling": False,
    }
    return checked, identities, audit


def make_confirmation(feature_data, original_data, output_path, seed=20261013, n=5000):
    """Return ``(data, identities, audit)`` and persist or verify the HDF5.

    ``data[name]`` is ``(examples, labels)``; identities have shape [N,3,2]
    with the final axis (class_index, exemplar_index). Existing outputs are
    accepted only with identical generation/input identities and recomputed
    contents/audits. A failed generation leaves its .partial file for diagnosis;
    it never overwrites an existing destination.
    """
    import jax
    import jax.numpy as jnp
    import jaxlib
    import samplers

    _require(seed in (20261012, 20261013) and n == 5000, "Seeds/N are frozen at 20261012,20261013 / 5000")
    features = np.asarray(feature_data)
    _require(features.dtype == np.dtype("float32") and features.shape == (1600, 5, 512),
             "Expected the approved float32 feature subset with shape (1600,5,512)")
    features = np.ascontiguousarray(features)
    _require(np.isfinite(features).all(), "Nonfinite feature data")
    sampler_sha = _file_sha(samplers.__file__)
    _require(sampler_sha == AUTHOR_SAMPLER_SHA256, "Imported sampler differs from frozen author source")
    _require(set(SETS).issubset(original_data), "Missing an original evaluator")
    lookup = _feature_lookup(features)
    original_reports, original_rows = {}, {}
    for name in SETS:
        xy, _, original_reports[name] = _validate(name, original_data[name], features, lookup,
                                                 expected_n=n)
        original_rows[name] = _row_keys(*xy)
    root_key = jax.random.PRNGKey(seed)
    eval_keys = jax.random.split(root_key, len(AUTHOR_ORDER))
    metadata = {
        "schema": "exp023_fresh_v1",
        "seed": seed,
        "n_per_task": n,
        "root_key_uint32": np.asarray(root_key, dtype=np.uint32).tolist(),
        "key_rule": "jax.random.split(jax.random.PRNGKey(seed), 5); no outer split",
        "author_evaluator_order": list(AUTHOR_ORDER),
        "generated_evaluators": list(SETS),
        "all_five_eval_keys_uint32": np.asarray(eval_keys, dtype=np.uint32).tolist(),
        "common_sampler_parameters": {
            "class_indices": [0, 1599], "class_distribution": "uniform",
            "exemplar_indices": [0, 4], "context_len": 2, "distractor": True,
            "match_query_and_distractors": False, "noise_scale": 0.0,
        },
        "task_sampler_parameters": CONFIGS,
        "feature_subset_sha256": _array_sha(features),
        "original_data_audit": original_reports,
        "samplers_source_sha256": sampler_sha,
        "generator_source_sha256": _file_sha(__file__),
        "versions": {"jax": jax.__version__, "jaxlib": jaxlib.__version__,
                     "numpy": np.__version__, "h5py": h5py.__version__},
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_default_prng_impl": str(jax.config.jax_default_prng_impl),
    }
    path = Path(output_path)
    if path.exists():
        with h5py.File(path, "r") as file:
            _require(json.loads(file.attrs["metadata_json"]) == metadata,
                     "Existing confirmation metadata/input identity differs; refusing reuse")
            _require(set(file.keys()) == set(SETS), "Existing confirmation groups differ")
            data = {name: (file[name]["examples"][:], file[name]["labels"][:]) for name in SETS}
            saved_ids = {name: file[name]["identities"][:] for name in SETS}
            saved_audit = json.loads(file.attrs["audit_json"])
        data, identities, audit = _audit_all(data, features, lookup, original_rows, n, saved_ids)
        _require(audit == saved_audit, "Existing confirmation content/audit mismatch")
        return data, identities, dict(audit, metadata=metadata, reused=True,
                                     output_sha256=_file_sha(path))

    data = {}
    for index, name in enumerate(SETS):
        config = CONFIGS[name]
        class_sampler = partial(
            samplers.get_constant_burst_seq_idxs,
            classes=jnp.arange(1600), class_distr=jnp.ones(1600) / 1600,
            num_seqs=n, context_len=2, burstiness=config["burstiness"],
            distractor=True, no_support=config["no_support"], unique_rest=config["unique_rest"])
        exemplar_sampler = partial(samplers.get_exemplar_inds, allowed_inds=jnp.arange(5),
                                   match_query_and_distractors=False)
        relabel = config["fs_relabel"]
        if relabel == "01":
            relabel = jnp.array([[0, 1]])
        sample = samplers.make_data_sampler(
            class_sampler, exemplar_sampler, fs_relabel=relabel, noise_scale=0.0,
            assign_query_label_random=config["assign_query_label_random"])
        result = sample(eval_keys[index], jnp.asarray(features))
        data[name] = (np.asarray(result["examples"], dtype=np.float32),
                      np.asarray(result["labels"], dtype=np.int32))
    data, identities, audit = _audit_all(data, features, lookup, original_rows, n)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    with h5py.File(temp_path, "w") as file:
        file.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        file.attrs["audit_json"] = json.dumps(audit, sort_keys=True)
        for name in SETS:
            group = file.create_group(name)
            group.create_dataset("examples", data=data[name][0], compression="gzip", compression_opts=1)
            group.create_dataset("labels", data=data[name][1])
            group.create_dataset("identities", data=identities[name])
        file.flush()
    # Atomic, no-clobber publication on the same filesystem. If publication
    # fails (including unsupported hard links), leave .partial and raise.
    os.link(temp_path, path)
    temp_path.unlink()
    return data, identities, dict(audit, metadata=metadata, reused=False,
                                 output_sha256=_file_sha(path))

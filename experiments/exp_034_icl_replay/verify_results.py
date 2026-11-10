"""Independent saved-artifact reader, without producer, JAX or Equinox imports.

Checks identities, all confirmatory row metrics, fixed full-logit audit rows,
NumPy checkpoint streams, author Adam algebra, legacy Threefry keys and paired
statistics. This is not independent training or full model-forward reproduction.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import traceback
from typing import Any

import h5py
import numpy as np

WORK = Path(__file__).resolve().parent
TASKS = ("train_eval", "iwl_copy_avail", "flip_icl", "icl")
SCALARS = ("acc", "loss", "in_context_acc", "pair_loss", "pair_margin",
           "context_mass", "log_context_mass")
ARMS = ("REPLAY", "L2")
REFS = ("A", "C25", "P25")
STAGES = list(range(9000000, 10000001, 100000))
AUDIT = np.asarray([0, 1, 127, 1024, 2499, 4095, 4998, 4999], np.int32)
PARAMETERS = (3, 4, 7, 8, 12, 13, 19, 22, 25, 26, 32, 35, 38, 39, 41, 42)
SEEDS = ("eval_model_seed", "train_data_seed", "train_model_seed")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def at(work: Path, name: str) -> Path:
    value = Path(name)
    return value if value.is_absolute() else work / value


class Reader:
    def __init__(self, work: Path):
        self.work, self.out = work, work / "outputs"
        self.failures: list[dict[str, Any]] = []
        self.checks = 0
        self.counts: Counter = Counter()
        self.hashes: dict[str, str] = {}
        self.maximum_errors: dict[str, float] = {}
        self.preflight_scopes: dict[str, Any] = {}

    def require(self, name: str, passed: Any, detail: Any = None) -> bool:
        self.checks += 1
        valid = bool(passed)
        if not valid:
            self.failures.append({"check": name, "detail": detail})
        return valid

    def close(self, name: str, left: Any, right: Any, atol: float = 2e-5, rtol: float = 0.) -> bool:
        a, b = np.asarray(left), np.asarray(right)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            return self.require(name, False, {"reason": "shape_or_nonfinite", "observed_shape": list(a.shape), "expected_shape": list(b.shape)})
        error = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.
        scale = float(np.max(np.abs(b.astype(np.float64)))) if b.size else 0.
        category = name.rsplit("/", 1)[-1]
        self.maximum_errors[category] = max(error, self.maximum_errors.get(category, 0.))
        return self.require(name, error <= atol + rtol * scale,
                            {"max_abs": error, "absolute_tolerance": atol, "relative_tolerance": rtol, "reference_scale": scale})

    def file(self, name: str, path: Path, expected: str | None = None) -> str | None:
        if not self.require(name + "/exists", path.is_file(), str(path)):
            return None
        key = str(path.resolve())
        if key not in self.hashes:
            self.hashes[key] = digest(path)
            self.counts["hashed_files"] += 1
        value = self.hashes[key]
        if expected is not None:
            self.require(name + "/sha256", is_sha(expected) and value == expected,
                         {"path": str(path), "observed": value, "expected": expected})
        return value

    def subset(self, name: str, observed: Any, expected: Any) -> None:
        if isinstance(expected, dict):
            if not self.require(name + "/mapping", isinstance(observed, dict)):
                return
            for key, value in expected.items():
                if self.require(name + "/key/" + key, key in observed):
                    self.subset(name + "/" + key, observed[key], value)
        elif isinstance(expected, list):
            if self.require(name + "/length", isinstance(observed, list) and len(observed) == len(expected)):
                for slot, (a, b) in enumerate(zip(observed, expected)):
                    self.subset(name + "/" + str(slot), a, b)
        elif expected is None or isinstance(expected, (bool, str)):
            self.require(name, observed == expected, {"observed": observed, "expected": expected})
        elif isinstance(expected, (float, int)):
            self.close(name, observed, expected, 1e-10)
        else:
            self.require(name, observed == expected)


def stat(values: Any, classes: Any) -> dict[str, Any]:
    rows, groups = np.asarray(values, np.float64), np.asarray(classes)
    if rows.ndim != 1 or rows.size == 0 or groups.shape != rows.shape or not np.isfinite(rows).all():
        raise ValueError("Malformed independently paired rows")
    names, membership, counts = np.unique(groups, return_inverse=True, return_counts=True)
    totals = np.zeros(len(names), np.float64)
    np.add.at(totals, membership, rows)
    mean = float(np.sum(rows) / len(rows))
    residual_sums = totals - counts * mean
    variance = float(np.sum(residual_sums**2) * len(names) / (len(names) - 1) / len(rows)**2) if len(names) > 1 else None
    se = math.sqrt(max(variance, 0.)) if variance is not None else None
    valid = se is not None and se > 1e-15
    return {"mean": mean, "se": se, "n": int(len(rows)), "clusters": int(len(names)), "valid": bool(valid),
            "lower_one_sided_95": mean - 1.6448536269514722 * se if valid else None,
            "upper_one_sided_95": mean + 1.6448536269514722 * se if valid else None,
            "ci95_lower": mean - 1.959963984540054 * se if valid else None,
            "ci95_upper": mean + 1.959963984540054 * se if valid else None,
            "estimand": "evaluation_row_weighted_mean", "cluster_unit": "true_query_class",
            "status": "normal_query_class_cluster" if valid else "degenerate_no_confirmatory_support"}


def gain(value: dict[str, Any], name: str, threshold: float) -> dict[str, Any]:
    z = (value["mean"] - threshold) / value["se"] if value["valid"] else None
    p = math.erfc(z / 2**.5) / 2 if z is not None else 1.
    return dict(value, name=name, threshold=float(threshold), alternative="mean > threshold",
                z_or_t=z, p_raw=float(p), p_underflow=bool(z is not None and p == 0.))


def adjusted(items: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    tests = [dict(item) for item in items]
    ranks = np.argsort(np.asarray([item["p_raw"] for item in tests]), kind="stable")
    corrected = np.maximum.accumulate(np.minimum(1., np.asarray([tests[int(slot)]["p_raw"] * (len(tests) - rank)
                                                                for rank, slot in enumerate(ranks)])))
    for slot, p in zip(ranks, corrected):
        item = tests[int(slot)]
        item.update(p_holm=float(p), family_size=len(tests),
                    passed=bool(item["valid"] and p < alpha and item["mean"] > item["threshold"]))
    return {"tests": tests, "alpha": alpha, "family_size": len(tests), "all_passed": all(t["passed"] for t in tests),
            "correction": "Holm", "interval_note": "Reported 95% intervals are ordinary, not Holm simultaneous intervals"}


def ni(value: dict[str, Any], metric: str) -> dict[str, Any]:
    limit = -.02 if metric == "acc" else .1
    bound = value["lower_one_sided_95"] if metric == "acc" else value["upper_one_sided_95"]
    okay = bound is not None and (bound > limit if metric == "acc" else bound < limit)
    return dict(value, threshold=limit, metric=metric, passed=bool(okay),
                alternative="mean > -acc_tolerance" if metric == "acc" else "mean < ce_tolerance",
                multiplicity="intersection_union_all_required_gates")


def recalculate(data: dict[str, Any], classes: dict[str, Any]) -> dict[str, Any]:
    descriptive = {arm: {task: {metric: stat(rows, classes[task]) for metric, rows in group.items()}
                         for task, group in tasks.items()} for arm, tasks in data.items()}
    pairs = [(arm, ref) for arm in ARMS for ref in REFS]
    pairs.extend((ref, arm) for arm in ARMS for ref in ("C25", "P25"))
    pairs.extend((("REPLAY", "L2"), ("L2", "REPLAY")))
    differences = {}
    for arm, ref in pairs:
        differences[arm + "-" + ref] = {task: {metric: stat(np.subtract(data[arm][task][metric], data[ref][task][metric]), classes[task])
                                              for metric in SCALARS} for task in TASKS}
    primary_raw = []
    per_arm = {}
    for arm in ARMS:
        items = [gain(differences[arm + "-A"][t]["acc"], f"{arm}-A/{t}/acc", .05) for t in ("icl", "flip_icl")]
        primary_raw.extend(items)
        per_arm[arm] = adjusted(items, .05)
    primary = adjusted(primary_raw, .05)
    gates = {}
    for arm, ref in pairs:
        name = arm + "-" + ref
        required = ("iwl_copy_avail", "train_eval") if ref == "A" else TASKS
        tests = {task + "/" + metric: ni(differences[name][task][metric], metric) for task in required for metric in ("acc", "loss")}
        gates[name] = {"tests": tests, "all_passed": all(item["passed"] for item in tests.values())}
    forward, reverse, auxiliary = [], [], {}
    for arm in ARMS:
        for pair, family in (("P25-" + arm, forward), (arm + "-P25", reverse)):
            items = [gain(differences[pair][t]["acc"], pair + "/" + t + "/acc", 0.) for t in ("icl", "flip_icl")]
            family.extend(items)
            auxiliary[pair] = adjusted(items, .05)
    values = {"P25_advantage": adjusted(forward, .05), "baseline_advantage": adjusted(reverse, .05)}
    def accepted(family: dict[str, Any], pair: str) -> bool:
        found = {item["name"]: item["passed"] for item in family["tests"]}
        return all(found[pair + "/" + task + "/acc"] for task in ("icl", "flip_icl"))
    methods = {}
    for arm in ARMS:
        benefit, damage = accepted(primary, arm + "-A"), gates[arm + "-A"]["all_passed"]
        forward_gain, reverse_gain = accepted(values["P25_advantage"], "P25-" + arm), accepted(values["baseline_advantage"], arm + "-P25")
        forward_ni, reverse_ni = gates["P25-" + arm]["all_passed"], gates[arm + "-P25"]["all_passed"]
        methods[arm] = {"joint_vs_A": {"gain_Holm4_passed": benefit, "noninferiority_four_passed": damage,
                                       "statistical_supported": benefit and damage, "auxiliary_Holm2_gain_passed": per_arm[arm]["all_passed"]},
                        "noninferior_to_P25": reverse_ni, "noninferior_to_C25": gates[arm + "-C25"]["all_passed"],
                        "P25_extra_value": {"gain_Holm4_passed": forward_gain, "noninferiority_eight_passed": forward_ni,
                                            "statistical_supported": forward_gain and forward_ni},
                        "baseline_extra_value": {"gain_Holm4_passed": reverse_gain, "noninferiority_eight_passed": reverse_ni,
                                                 "statistical_supported": reverse_gain and reverse_ni}}
    return {"descriptive": descriptive, "paired_comparisons": differences, "primary_Holm4": primary,
            "primary_Holm2_auxiliary": per_arm, "noninferiority": gates, "value_Holm4": values,
            "directed_Holm2_zero_auxiliary": auxiliary, "methods": methods}


def split_key(key: Any) -> tuple[tuple[int, int], tuple[int, int]]:
    """Legacy JAX Threefry2x32 split(key, 2), independently in Python integers.

    The four iota counts are split into [0,1] and [2,3], hashed with 20 rounds,
    concatenated then reshaped. Arithmetic is explicitly modulo 2**32.
    """
    mask = 0xffffffff
    k0, k1 = (int(v) for v in key)
    keys = (k0, k1, k0 ^ k1 ^ 0x1bd11bda)
    a, b, c, d = k0, (k1 + 2) & mask, (k0 + 1) & mask, (k1 + 3) & mask
    rotations = (13, 15, 26, 6, 17, 29, 16, 24)
    for round_index in range(20):
        rotation = rotations[round_index % 8]
        a, c = (a + b) & mask, (c + d) & mask
        b = (((b << rotation) & mask) | (b >> (32 - rotation))) ^ a
        d = (((d << rotation) & mask) | (d >> (32 - rotation))) ^ c
        if round_index % 4 == 3:
            injection = round_index // 4 + 1
            a, c = (a + keys[injection % 3]) & mask, (c + keys[injection % 3]) & mask
            b, d = (b + keys[(injection + 1) % 3] + injection) & mask, (d + keys[(injection + 1) % 3] + injection) & mask
    return (a, c), (b, d)


def key_timeline(start: dict[str, Any], wanted: set[int]) -> dict[int, dict[str, Any]]:
    data, model = tuple(start["train_data_seed"]), tuple(start["train_model_seed"])
    result = {}
    for offset in range(max(wanted) + 1):
        if offset in wanted:
            result[offset] = {"train_data_seed": list(data), "train_model_seed": list(model)}
        if offset != max(wanted):
            data, _ = split_key(data)
            model, _ = split_key(model)
    return result


def eqx_stream(path: Path) -> list[np.ndarray]:
    leaves = []
    with path.open("rb") as source:
        size = path.stat().st_size
        while source.tell() < size:
            value = np.load(source, allow_pickle=False)
            if not isinstance(value, np.ndarray) or not np.isfinite(value).all():
                raise ValueError("Nonfinite/nonarray checkpoint stream leaf: " + str(path))
            leaves.append(value)
    if len(leaves) != 80:
        raise ValueError("Expected frozen author's 80-leaf attention-only checkpoint layout")
    return leaves


def checkpoint(r: Reader, path: Path, stage: int, base: list[np.ndarray], expected: dict[str, Any], eval_key: Any, name: str) -> list[np.ndarray]:
    leaves = eqx_stream(path)
    r.require(name + "/iter", leaves[0].shape == () and int(leaves[0]) == stage)
    r.require(name + "/Adam_count", leaves[44].shape == () and leaves[44].dtype == np.int32 and int(leaves[44]) == stage // 32)
    for index, (left, right) in enumerate(zip(leaves, base)):
        r.require(name + "/shape_dtype/" + str(index), left.shape == right.shape and left.dtype == right.dtype)
        if index not in PARAMETERS and index not in range(44, 80) and index != 0:
            r.close(name + "/static_model/" + str(index), left, right, 0.)
    for index in PARAMETERS:
        r.require(name + "/trainable_FP32/" + str(index), leaves[index].dtype == np.float32)
    for field, index in zip(SEEDS, (77, 78, 79)):
        target = eval_key if field == "eval_model_seed" else expected[field]
        r.require(name + "/key_layout/" + field, leaves[index].shape == (2,) and leaves[index].dtype == np.uint32)
        r.close(name + "/PRNG/" + field, leaves[index], target, 0.)
    r.counts["checkpoint_streams"] += 1
    r.counts["finite_checkpoint_leaves"] += len(leaves)
    return leaves


def logsumexp(values: Any) -> np.ndarray:
    a = np.asarray(values, np.float64)
    largest = a.max(axis=1, keepdims=True)
    return largest[:, 0] + np.log(np.exp(a - largest).sum(axis=1))


def metric_group(r: Reader, raw: Any, task: str, y: np.ndarray, ids: np.ndarray,
                 examples: np.ndarray, name: str, new: bool) -> dict[str, np.ndarray]:
    fields = (*SCALARS, "full_pred", "pair_pred", "target")
    values = {}
    for field in fields:
        value = np.asarray(raw[task + "/" + field])
        if not r.require(name + "/shape/" + field, value.shape == (5000,)):
            raise ValueError("Incomplete fixed 5000-row metric")
        r.require(name + "/finite/" + field, np.isfinite(value).all())
        values[field] = value
    target = y[:, -1]
    other = y[:, 0] + y[:, 1] - target
    r.require(name + "/unique_candidates", np.all(y[:, 0] != y[:, 1]))
    r.require(name + "/target_once", np.all((y[:, :2] == target[:, None]).sum(axis=1) == 1))
    r.close(name + "/target_identity", values["target"], target, 0.)
    for field in ("full_pred", "pair_pred", "target"):
        r.require(name + "/integer_labels/" + field, np.issubdtype(values[field].dtype, np.integer))
        r.require(name + "/label_range/" + field, np.all((values[field] >= 0) & (values[field] < 12800)))
    for field in ("acc", "in_context_acc"):
        r.require(name + "/binary/" + field, np.isin(values[field], [0, 1]).all())
    r.close(name + "/full_acc_prediction", values["acc"], values["full_pred"] == target, 0.)
    margin = values["pair_margin"].astype(np.float64)
    predicted = np.where((margin > 0) | ((margin == 0) & (target < other)), target, other)
    r.close(name + "/pair_pred_margin", values["pair_pred"], predicted, 0.)
    r.close(name + "/pair_acc_prediction", values["in_context_acc"], predicted == target, 0.)
    r.close(name + "/pair_CE_margin", values["pair_loss"], np.logaddexp(0., -margin))
    r.close(name + "/full_CE_pair_CE_mass", values["loss"], values["pair_loss"].astype(np.float64) - values["log_context_mass"])
    r.close(name + "/context_mass_exp", values["context_mass"], np.exp(values["log_context_mass"].astype(np.float64)))
    r.require(name + "/nonnegative_CE", np.all(values["loss"] >= -2e-5))
    r.require(name + "/mass_range", np.all((values["context_mass"] >= 0) & (values["context_mass"] <= 1 + 2e-5)) and np.all(values["log_context_mass"] <= 2e-5))
    r.require(name + "/full_correct_implies_pair_correct", np.all(values["acc"] <= values["in_context_acc"]))
    r.close(name + "/query_identities", raw[task + "/identities"], ids, 0.)
    rows = np.asarray(raw[task + "/audit_rows"])
    r.require(name + "/fixed_audit_rows", rows.dtype == np.int32 and np.array_equal(rows, AUDIT))
    logits, lp = np.asarray(raw[task + "/audit_logits"]), np.asarray(raw[task + "/audit_log_probs"])
    if not r.require(name + "/audit_shape", logits.shape == lp.shape == (8, 12800)):
        raise ValueError("Incomplete fixed full-logit audit")
    r.require(name + "/audit_FP32_finite", logits.dtype == lp.dtype == np.float32 and np.isfinite(logits).all() and np.isfinite(lp).all())
    z, logp = logits.astype(np.float64), lp.astype(np.float64)
    norm = logsumexp(z)
    r.close(name + "/audit_log_probs", logp, z - norm[:, None])
    r.close(name + "/audit_normalization", logsumexp(logp), np.zeros(8))
    ix, label, alternative = np.arange(8), target[AUDIT], other[AUDIT]
    gap = z[ix, label] - z[ix, alternative]
    logmass = np.logaddexp(z[ix, label], z[ix, alternative]) - norm
    r.close(name + "/audit_full_CE", values["loss"][AUDIT], norm - z[ix, label])
    r.close(name + "/audit_full_CE_logp", values["loss"][AUDIT], -logp[ix, label])
    r.close(name + "/audit_full_prediction", values["full_pred"][AUDIT], np.argmax(z, axis=1), 0.)
    r.close(name + "/audit_margin", margin[AUDIT], gap)
    r.close(name + "/audit_pair_CE", values["pair_loss"][AUDIT], np.logaddexp(0., -gap))
    r.close(name + "/audit_context_logmass", values["log_context_mass"][AUDIT], logmass)
    if new:
        r.close(name + "/audit_labels", raw[task + "/audit_labels"], y[AUDIT], 0.)
        hashes = np.asarray(raw[task + "/audit_examples_sha256"])
        expected = [hashlib.sha256(np.asarray(row, np.float32).tobytes()).hexdigest() for row in examples]
        r.require(name + "/audit_examples_SHA", hashes.shape == (8,) and hashes.tolist() == expected)
    r.counts["metric_groups"] += 1
    r.counts["raw_summary_rows"] += 5000
    r.counts["fixed_full_logit_rows"] += 8
    return {metric: values[metric].astype(np.float64) for metric in SCALARS}


def identity(r: Reader, s: dict[str, Any]) -> tuple[str, dict[str, Any], Path, list[np.ndarray]]:
    code = r.file("identity/training_manifest", r.work / "code_manifest.json")
    training = read_json(r.work / "code_manifest.json")
    for name, expected in training.items():
        r.file("identity/training_code/" + name, at(r.work, name), expected)
    all_names = set(training)
    post = r.work / "postprocess_manifest.json"
    if post.is_file():
        r.file("identity/postprocess_manifest", post)
        for name, expected in read_json(post).items():
            r.file("identity/postprocess_code/" + name, at(r.work, name), expected)
            all_names.add(name)
    for name in ("settings.json", "protocol.md", "analysis_plan.md", "analyze.py", "verify_results.py", "references_manifest.json", "input_spec.json",
                 "common.py", "runtime.py", "objectives.py", "gpu_preflight.py", "math_preflight.py"):
        r.require("identity/required_sealed_file/" + name, name in all_names)
    design = {"arms": list(ARMS), "reference_arms": list(REFS), "training_seeds": [{"id": "s5", "init": 5, "train": 2}],
              "start_stage": 4400000, "end_stage": 10000000, "confirm_stages": STAGES,
              "late_window": [9000000, 10000000], "fresh_tasks": list(TASKS), "n": 5000,
              "train_bs": 32, "train_microbs": 32, "protection_bs_per_task": 32,
              "protection_tasks": ["icl", "flip_icl"], "min_icl_gain": .05,
              "alpha": .05, "ni_acc_tolerance": .02, "ni_ce_tolerance": .1,
              "primary_gain_family_size": 4, "extra_value_family_size_each_direction": 4,
              "replay_coefficient": 2., "replay_loss_normalizer": 3.,
              "l2_coefficient": 1e-4, "l2_optimizer": "coupled_to_original_adam",
              "l2_form": "sum_of_squared_all_trainable_array_entries_without_half"}
    for key, expected in design.items():
        r.require("design/" + key, s.get(key) == expected)
    r.require("design/protection_pool_8192", s["data_sets"]["protection"]["n"] == 4096)
    p = read_json(r.out / "provenance.json")
    r.file("identity/provenance", r.out / "provenance.json")
    r.require("identity/provenance_code", p.get("code_manifest_sha256") == code)
    r.require("identity/author_float32_precision", p.get("enable_x64") is False and p.get("default_matmul_precision") is None)
    r.require("identity/author_PRNG_implementation", p.get("jax_threefry_partitionable") is False and p.get("jax_default_prng_impl") == "threefry2x32")
    versions = {"jax": "0.4.26", "jaxlib_distribution": "0.4.26+cuda12.cudnn89", "equinox": "0.11.4", "optax": "0.2.2", "numpy": "1.26.4", "h5py": "3.11.0"}
    for key, version in versions.items():
        r.require("identity/version/" + key, p["versions"].get(key) == version)
    assets = at(r.work, p["assets_root"])
    spec = read_json(r.work / "input_spec.json")
    expected_inputs = {name: record["sha256"] for name, record in spec["files"].items()}
    expected_source = {name: expected_inputs["source/" + name] for name in spec["source_files"]}
    r.require("identity/inputs_exact_spec", p.get("inputs") == expected_inputs)
    r.require("identity/source_exact_spec", p.get("source") == expected_source)
    for name, expected in expected_inputs.items():
        r.file("identity/input/" + name, assets / name, expected)
    r.file("identity/export_asset_manifest", assets / "assets_manifest.json", spec["registered_asset_manifest_sha256"])
    r.require("identity/export_manifest_bound", p.get("assets_manifest_sha256") == spec["registered_asset_manifest_sha256"])
    for name, record in read_json(assets / "assets_manifest.json").items():
        r.file("identity/exported_asset/" + name, assets / name, record["sha256"] if isinstance(record, dict) else record)
    r.file("identity/author_replay_checkpoint", assets / spec["replay_checkpoint"]["to"], spec["replay_checkpoint"]["sha256"])
    for name, expected in s["preserved_data_sha256"].items():
        path = r.out / "data" / name
        r.file("identity/preserved_data/" + name, path, expected)
        r.require("identity/data_registered/" + name, p.get("new_inputs", {}).get(str(path.relative_to(r.work))) == expected)
    r.file("identity/data_creation_manifest", r.work / "data_creation_manifest.json", s["data_generator_manifest_sha256"])
    base_path = assets / "checkpoints/00004400000.eqx"
    base = eqx_stream(base_path)
    r.require("identity/base_iter", int(base[0]) == 4400000)
    r.require("identity/base_Adam_count", int(base[44]) == 137500)
    return code, p, assets, base


def fixed_data(r: Reader, s: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    labels, ids, audit_x, query = {}, {}, {}, {}
    with h5py.File(r.out / "data/confirm.h5", "r") as data:
        metadata = json.loads(data.attrs["metadata"])
        r.require("data/confirm_frozen_generator", metadata["code_manifest_sha256"] == s["data_generator_manifest_sha256"])
        r.require("data/confirm_purpose_seed", metadata["name"] == "confirm" and metadata["seed"] == s["data_sets"]["confirm"]["seed"] and metadata["n"] == 5000)
        for task in TASKS:
            labels[task] = np.asarray(data[task + "/labels"])
            ids[task] = np.asarray(data[task + "/identities"])
            audit_x[task] = np.asarray(data[task + "/examples"][AUDIT.tolist()], np.float32)
            r.require("data/confirm_shapes/" + task, data[task + "/examples"].shape == (5000, 3, 512) and labels[task].shape == (5000, 3) and ids[task].shape == (5000, 3, 2))
            r.require("data/confirm_dtypes/" + task, data[task + "/examples"].dtype == np.float32 and labels[task].dtype == ids[task].dtype == np.int32)
            classes = ids[task][:, :, 0]
            query[task] = classes[:, -1]
            target, shared = labels[task][:, -1], classes[:, :2] == classes[:, -1, None]
            r.require("data/confirm_context_classes_unique/" + task, np.all(classes[:, 0] != classes[:, 1]))
            if task == "icl":
                r.require("data/ICL_fresh_relabel", np.all(np.sort(labels[task][:, :2], axis=1) == [0, 1]) and np.all(shared.sum(axis=1) == 1))
            elif task == "flip_icl":
                r.require("data/Flip_identity", np.array_equal(labels[task][:, :2], classes[:, 1::-1]) and np.all(target != classes[:, -1]) and np.all(shared.sum(axis=1) == 1))
            elif task == "train_eval":
                r.require("data/original_identity", np.array_equal(labels[task], classes))
            else:
                r.require("data/CIWL_no_context_match", np.all(shared.sum(axis=1) == 0) and np.array_equal(target, classes[:, -1]))
    receipt = read_json(r.out / "data/identity_audit.json")
    r.file("data/identity_audit", r.out / "data/identity_audit.json")
    r.require("data/identity_audit_passed", receipt.get("passed") is True and receipt.get("resampling") is False)
    active = ["mechanism_confirm", "protection", "monitor", "confirm"]
    r.require("data/active_sets_fixed", receipt.get("active_data_sets") == s.get("active_data_sets") == active)
    for index, left in enumerate(active):
        for right in active[index + 1:]:
            pair = left + "/" + right
            r.require("data/active_exact_independence/" + pair, receipt["exact_sequence_overlap"].get(pair) == 0)
            r.require("data/active_visible_independence/" + pair, receipt["visible_model_input_overlap"].get(pair) == 0)
    for name, value in s["preserved_data_sha256"].items():
        r.require("data/audit_SHA/" + name, receipt["data"][name[:-3]]["sha256"] == value)
    return labels, ids, audit_x, query


def references(r: Reader, s: dict[str, Any], labels: dict[str, Any], ids: dict[str, Any], audit_x: dict[str, Any]) -> dict[str, Any]:
    manifest = read_json(r.work / "references_manifest.json")
    r.require("reference/33_records", len(manifest["raw_records"]) == 33)
    r.require("reference/same_confirm_SHA", manifest["confirm_data_sha256"] == s["preserved_data_sha256"]["confirm.h5"])
    for name, value in manifest["files"].items():
        r.file("reference/sealed/" + name, at(r.work, name), value)
    verifiers = {}
    for name, record in manifest["verifiers"].items():
        path = at(r.work, record["path"])
        r.file("reference/verifier/" + name, path, record["sha256"])
        value = read_json(path)
        r.require("reference/successful_reader/" + name, value.get("passed") is True and value.get("failures") == [])
        verifiers[str(path.resolve())] = value
    for old, completion_name in (("exp030", "completion.json"), ("exp031", "runtime_complete.json")):
        record = manifest["verifiers"][old]
        path = at(r.work, record["path"]).parent / completion_name
        value = read_json(path)
        r.require("reference/successful_completion/" + old, value.get("passed") is True)
        r.require("reference/completion_reader_bound/" + old, value.get("verification_sha256") == record["sha256"])
    registered = {(item["arm"], item["stage"]): item for item in manifest["raw_records"]}
    r.require("reference/exact_arm_stage_grid", set(registered) == {(arm, stage) for arm in REFS for stage in STAGES})
    data = {}
    for arm in REFS:
        rows = {task: {metric: [] for metric in SCALARS} for task in TASKS}
        for stage in STAGES:
            item = registered[arm, stage]
            path, vp = at(r.work, item["path"]), at(r.work, item["verifier_path"])
            r.require("reference/manifest_raw_same/" + path.name, manifest["files"].get(item["path"]) == item["sha256"])
            r.require("reference/verifier_SHA_same/" + path.name, manifest["files"].get(item["verifier_path"]) == item["verifier_sha256"])
            verifier = verifiers[str(vp.resolve())]
            r.require("reference/raw_previously_verified/" + path.name, verifier["verified_file_sha256"].get(str(path.resolve())) == item["sha256"])
            with np.load(path, allow_pickle=False) as raw:
                for task in TASKS:
                    fields = metric_group(r, raw, task, labels[task], ids[task], audit_x[task], "reference/" + arm + "/" + str(stage) + "/" + task, False)
                    for metric, value in fields.items():
                        rows[task][metric].append(value)
        data[arm] = {task: {metric: np.sum(np.asarray(values), axis=0, dtype=np.float64) / 11 for metric, values in item.items()} for task, item in rows.items()}
    return data


def verify_leaf_mask(r: Reader, mask: dict[str, Any], base: list[np.ndarray], name: str) -> None:
    included = mask["included"]
    r.require(name + "/all_16_trainable_leaves", len(included) == mask["included_leaf_count"] == len(PARAMETERS) and mask["excluded"] == [] and mask["excluded_leaf_count"] == 0)
    r.require(name + "/bias_LN_included_no_BN", mask["includes_bias_and_layer_norm"] is True and mask["batch_norm_present"] is False)
    r.require(name + "/total_parameter_count", mask["total_elements"] == sum(base[index].size for index in PARAMETERS))
    expected_norm = sum(float(np.square(base[index].astype(np.float64)).sum()) for index in PARAMETERS)
    r.close(name + "/total_leaf_sum", mask["squared_sum_float64"], expected_norm, 1e-9, 1e-12)
    for record, index in zip(included, PARAMETERS):
        r.require(name + "/leaf_layout/" + str(index), record["shape"] == list(base[index].shape) and record["dtype"] == "float32" and record["size"] == base[index].size and record["included"] is True and isinstance(record["path"], str))
        r.close(name + "/leaf_square/" + str(index), record["squared_sum_float64"], float(np.square(base[index].astype(np.float64)).sum()), 1e-9, 1e-12)


def preflight(r: Reader, s: dict[str, Any], code: str, provenance: dict[str, Any], base: list[np.ndarray]) -> None:
    summary = read_json(r.out / "preflight.json")
    r.file("preflight/summary", r.out / "preflight.json")
    r.require("preflight/all_arms_passed", summary.get("passed") is True and set(summary["arms"]) == set(ARMS))
    r.require("preflight/code", summary.get("code_manifest_sha256") == code)
    r.require("preflight/settings", summary.get("settings_sha256") == digest(r.work / "settings.json"))
    r.require("preflight/legacy_Threefry", summary.get("jax_threefry_partitionable") is False)
    for arm in ARMS:
        info = summary["arms"][arm]
        path = at(r.work, info["path"])
        r.file("preflight/receipt/" + arm, path, info["sha256"])
        value = read_json(path)
        r.require("preflight/passed/" + arm, value.get("passed") is True and info.get("passed") is True)
        r.require("preflight/registered_arm_coefficient/" + arm, value.get("arm") == arm and value.get("coefficient") == (2. if arm == "REPLAY" else 1e-4))
        r.require("preflight/code/" + arm, value.get("code_manifest_sha256") == code)
        r.require("preflight/settings/" + arm, value.get("settings_sha256") == digest(r.work / "settings.json"))
        r.require("preflight/candidates_discarded/" + arm, value.get("all_candidate_states_discarded") is True)
        r.require("preflight/GPU/" + arm, value.get("model_forward_and_gradient_gpu_verified") is True)
        r.require("preflight/inputs/" + arm, value.get("inputs") == provenance["inputs"] and value.get("source") == provenance["source"])
        r.require("preflight/production_default_scope/" + arm, value.get("scope") == "actual_default_production_zero_and_numpy_logits_gradient_adam_and_finite_difference" and "matmul_precision" in value and value["matmul_precision"] is None)
        r.require("preflight/old_default_cross_graph_not_claimed/" + arm, value.get("default_cross_graph_numerical_parity_claimed") is False)
        checks = value["checks"]
        r.require("preflight/summary_checks_same/" + arm, info["checks"] == checks)
        for field in ("sampler_exact", "protection_cache_exact", "production_capture_interface_fixed",
                      "zero_dispatch_four_step_model_exact", "zero_dispatch_four_step_adam_exact", "zero_dispatch_four_step_prng_exact"):
            r.require("preflight/" + arm + "/" + field, checks.get(field) is True)
        r.require("preflight/" + arm + "/nonpartitionable", checks.get("jax_threefry_partitionable") is False)
        r.require("preflight/" + arm + "/exact_four_zero_steps", len(checks["zero_steps"]) == 4)
        for offset, zero in enumerate(checks["zero_steps"]):
            name = "preflight/" + arm + "/zero_step/" + str(offset)
            r.require(name + "/offset", zero.get("step") == offset)
            r.require(name + "/training_PRNG_exact", zero.get("training_prng_exact") is True)
            for field in ("model", "adam"):
                r.require(name + "/" + field + "_bitwise", zero[field]["exact"] is True and zero[field]["max_abs"] == 0. and zero[field]["relative_l2"] == 0.)
            r.require(name + "/loss", math.isfinite(zero["loss_abs"]) and 0 <= zero["loss_abs"] <= .001)
        for field in ("actual_train_ce_rows_max_abs", "actual_total_loss_numpy_abs"):
            r.require("preflight/" + arm + "/" + field, math.isfinite(checks[field]) and 0 <= checks[field] <= .001)
        actual_adam = checks["actual_gradient_numpy_adam"]
        r.require("preflight/" + arm + "/actual_gradient_source", actual_adam.get("gradient_source") == "actual_g_used_returned_by_fixed_production_interface")
        r.require("preflight/" + arm + "/actual_Adam_count_threshold", actual_adam.get("count") == 137501 and actual_adam.get("threshold") == 1e-4)
        for field in ("mu_max_abs", "nu_max_abs", "model_max_abs"):
            r.require("preflight/" + arm + "/actual_Adam/" + field, math.isfinite(actual_adam[field]) and 0 <= actual_adam[field] <= 1e-4)
        fd = checks["objective_directional_derivative"]
        r.close("preflight/" + arm + "/directional_error_algebra", fd["abs_error"], abs(fd["finite_difference"] - fd["autodiff"]), 1e-10)
        r.require("preflight/" + arm + "/directional_original_limits", fd["step"] == .001 and fd["absolute_tolerance"] == .005 and fd["relative_tolerance"] == .05 and fd["source"] == "same fixed production native_train_step interface")
        r.require("preflight/" + arm + "/directional_pass", fd["autodiff"] > 0 and fd["abs_error"] <= fd["absolute_tolerance"] + fd["relative_tolerance"] * fd["autodiff"])
        for task in (*TASKS, "pure_iwl"):
            e = checks["evaluation_" + task]
            r.require("preflight/" + arm + "/evaluation/" + task, e["rows"] == 128 and 0 <= e["mean_ce_abs_error"] <= .001 and 0 <= e["acc_different_rows"] <= 5)
        if arm == "REPLAY":
            r.require("preflight/REPLAY/actual_protection_full_CE", math.isfinite(checks["actual_protection_ce_rows_max_abs"]) and 0 <= checks["actual_protection_ce_rows_max_abs"] <= .001)
        else:
            verify_leaf_mask(r, checks["l2_trainable_leaf_mask"], base, "preflight/L2/default_physical_parameter_mask")
        mathpath = at(r.work, value["mathematical_receipt_path"])
        r.file("math_preflight/receipt/" + arm, mathpath, value["mathematical_receipt_sha256"])
        m = read_json(mathpath)
        r.require("math_preflight/identity/" + arm, m.get("passed") is True and m.get("arm") == arm and m.get("coefficient") == (2. if arm == "REPLAY" else 1e-4) and m.get("code_manifest_sha256") == code and m.get("settings_sha256") == digest(r.work / "settings.json"))
        r.require("math_preflight/inputs/" + arm, m.get("inputs") == provenance["inputs"] and m.get("source") == provenance["source"])
        r.require("math_preflight/separate_mathematical_scope/" + arm, m.get("scope") == "independent_nonzero_formula_gradient_and_adam_math_verification_only" and m.get("matmul_precision") == "float32" and "training_matmul_precision" in m and m["training_matmul_precision"] is None)
        r.require("math_preflight/not_default_parity/" + arm, m.get("default_cross_graph_numerical_parity_claimed") is False and m.get("all_candidate_states_discarded") is True and m.get("weighted_component_scaling_inside_ad") is True)
        mc = m["checks"]
        r.require("math_preflight/original_value_limit/" + arm, math.isfinite(mc["nonzero_value_reference_abs"]) and 0 <= mc["nonzero_value_reference_abs"] <= .001)
        gradient = mc["nonzero_gradient_reference"]
        r.require("math_preflight/original_gradient_limits/" + arm, math.isfinite(gradient["max_abs"]) and math.isfinite(gradient["relative_l2"]) and 0 <= gradient["max_abs"] <= 2e-5 and 0 <= gradient["relative_l2"] <= 2e-5)
        for field in ("nonzero_original_adam_model_reference", "nonzero_original_adam_state_reference"):
            r.require("math_preflight/original_Adam_limit/" + arm + "/" + field, math.isfinite(mc[field]["max_abs"]) and 0 <= mc[field]["max_abs"] <= 1e-4)
        if arm == "L2":
            verify_leaf_mask(r, mc["l2_trainable_leaf_mask"], base, "math_preflight/L2/physical_parameter_mask")
        r.preflight_scopes[arm] = {"production": value["scope"], "production_matmul_precision": None,
                                   "mathematical": m["scope"], "mathematical_matmul_precision": "float32",
                                   "default_cross_graph_numerical_parity_claimed": False}


def audit_objective(r: Reader, metrics: dict[str, Any], arm: str, leaves: list[np.ndarray], name: str) -> None:
    for key, value in metrics.items():
        r.require(name + "/finite/" + key, np.isfinite(np.asarray(value)).all())
    r.require(name + "/CE_nonnegative", metrics["train_ce"] >= -2e-5)
    if arm == "REPLAY":
        tasks = np.asarray(metrics["replay_task_ce"], np.float64)
        r.require(name + "/two_task_CE", tasks.shape == (2,) and np.all(tasks >= -2e-5))
        r.close(name + "/replay_average", metrics["replay_ce"], tasks.sum() / 2, 2e-5)
        r.close(name + "/weighted_replay", metrics["regularization_loss"], tasks.sum() / 3, 2e-5)
        r.close(name + "/weighted_total", metrics["loss"], (metrics["train_ce"] + tasks.sum()) / 3, 2e-5)
        r.close(name + "/no_L2", metrics["squared_l2"], 0., 0.)
    else:
        square = sum(float(np.sum(np.asarray(leaves[index], np.float64)**2)) for index in PARAMETERS)
        r.close(name + "/squared_norm_from_physical_parameters", metrics["squared_l2"], square, 1e-4, 2e-6)
        r.close(name + "/L2_loss", metrics["regularization_loss"], 1e-4 * square, 2e-5, 2e-6)
        r.close(name + "/total_loss", metrics["loss"], metrics["train_ce"] + metrics["regularization_loss"], 2e-5)
        r.close(name + "/no_replay", metrics["replay_ce"], 0., 0.)
        r.close(name + "/no_replay_tasks", metrics["replay_task_ce"], np.zeros(2), 0.)


def actual_gradient_adam(r: Reader, raw: Any, records: list[dict[str, Any]],
                         before: list[np.ndarray], after: list[np.ndarray], name: str) -> float:
    """Forward Adam reconstruction from the gradient actually used, never mu inversion."""
    keys = {key for key in raw.files if key.startswith("g_used/")}
    expected_keys = {f"g_used/{slot:04d}" for slot in range(16)}
    r.require(name + "/exact_16_actual_gradient_leaves", keys == expected_keys and len(records) == 16)
    count = int(before[44]) + 1
    r.close(name + "/Adam_count", after[44], count, 0.)
    square_sum = 0.
    for slot, index in enumerate(PARAMETERS):
        key = f"g_used/{slot:04d}"
        value = np.asarray(raw[key])
        r.require(name + "/actual_gradient_layout/" + str(slot), value.shape == before[index].shape and value.dtype == np.float32 and np.isfinite(value).all())
        record = records[slot]
        r.require(name + "/gradient_manifest/" + str(slot), record["npz_key"] == key and record["shape"] == list(value.shape) and record["dtype"] == "float32" and isinstance(record["path"], str))
        g = value.astype(np.float64)
        square_sum += float(np.sum(g * g))
        # Optax computes (1-decay) as a Python scalar before its FP32 cast.
        # Computing 1-np.float32(decay) would change the beta2 coefficient.
        expected_mu = (np.float32(1.0 - .9) * value
                       + np.float32(.9) * before[45 + slot])
        expected_nu = (np.float32(1.0 - .999) * np.square(value)
                       + np.float32(.999) * before[61 + slot])
        r.close(name + "/forward_Adam_mu/" + str(slot), after[45 + slot], expected_mu, 1e-4)
        r.close(name + "/forward_Adam_nu/" + str(slot), after[61 + slot], expected_nu, 1e-4)
        corrected_mu = expected_mu.astype(np.float64) / (1 - float(np.float32(.9))**count)
        corrected_nu = expected_nu.astype(np.float64) / (1 - float(np.float32(.999))**count)
        expected_theta = (before[index].astype(np.float64)
                          - float(np.float32(1e-5)) * corrected_mu
                          / (np.sqrt(corrected_nu) + float(np.float32(1e-8))))
        r.close(name + "/forward_original_Adam_theta/" + str(slot), after[index], expected_theta, 1e-4)
    norm = math.sqrt(square_sum)
    r.close(name + "/actual_gradient_norm", raw["metrics/grad_norm"], norm, 2e-5, 2e-5)
    r.close(name + "/one_microbatch_zero_stddev", raw["metrics/grad_batch_stddev"], 0., 0.)
    r.counts["actual_used_gradient_Adam_reconstructions"] += 1
    return norm


def actual_step_outputs(r: Reader, raw: Any, saved: dict[str, Any], arm: str,
                        before: list[np.ndarray], name: str) -> None:
    for key in raw.files:
        value = np.asarray(raw[key])
        r.require(name + "/finite/" + key, np.isfinite(value).all())
    x, y = np.asarray(raw["actual/train_examples"]), np.asarray(raw["actual/train_labels"])
    targets = np.asarray(raw["actual/train_targets"])
    logits = np.asarray(raw["actual/train_logits"])
    rows = np.asarray(raw["actual/train_rows"])
    r.require(name + "/normal_input_layout", x.shape == (32, 3, 512) and x.dtype == np.float32 and y.shape == (32, 3) and y.dtype == np.int32)
    r.require(name + "/normal_logit_layout", logits.shape == (32, 12800) and logits.dtype == np.float32 and rows.shape == (32,) and rows.dtype == np.float32)
    r.close(name + "/normal_target_identity", targets, y[:, -1], 0.)
    r.require(name + "/normal_batch_digest", hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest() == saved["normal_batch_sha256"])
    independent_rows = logsumexp(logits) - logits.astype(np.float64)[np.arange(32), targets]
    r.close(name + "/normal_full_CE_rows_from_actual_logits", rows, independent_rows, .001)
    r.close(name + "/normal_mean_CE_from_actual_logits", raw["actual/train_ce"], float(independent_rows.mean()), .001)
    for field, value in saved["metrics_before"].items():
        r.close(name + "/recorded_metric/" + field, raw["metrics/" + field], value, 0.)
    r.close(name + "/actual_train_CE_is_used_component", raw["actual/train_ce"], raw["metrics/train_ce"], 0.)
    px, py = np.asarray(raw["actual/protection_examples"]), np.asarray(raw["actual/protection_labels"])
    pt = np.asarray(raw["actual/protection_targets"])
    r.require(name + "/protection_input_layout", px.shape == (2, 32, 3, 512) and px.dtype == np.float32 and py.shape == (2, 32, 3) and py.dtype == np.int32)
    r.close(name + "/protection_target_identity", pt, py[:, :, -1], 0.)
    for task_index, task in enumerate(("icl", "flip_icl")):
        h = hashlib.sha256(px[task_index].tobytes() + py[task_index].tobytes()).hexdigest()
        r.require(name + "/protection_batch_digest/" + task, saved["protection_batch_sha256"][task] == h)
    if arm == "REPLAY":
        pl, pr = np.asarray(raw["actual/protection_logits"]), np.asarray(raw["actual/protection_rows"])
        r.require(name + "/protection_logit_layout", pl.shape == (2, 32, 12800) and pl.dtype == np.float32 and pr.shape == (2, 32) and pr.dtype == np.float32)
        independent = np.stack([logsumexp(pl[i]) - pl[i].astype(np.float64)[np.arange(32), pt[i]] for i in range(2)])
        r.close(name + "/protection_full_CE_rows_from_actual_logits", pr, independent, .001)
        r.close(name + "/protection_mean_CE_from_actual_logits", raw["actual/protection_ce"], independent.mean(axis=1), .001)
        r.close(name + "/protection_CE_is_used_component", raw["actual/protection_ce"], raw["metrics/replay_task_ce"], 0.)
        expected_loss = (independent_rows.mean() + independent.sum(axis=1).sum() / 32) / 3
        r.close(name + "/actual_weighted_96_sequence_CE", raw["metrics/loss"], expected_loss, .001)
    else:
        norm = sum(float(np.sum(before[index].astype(np.float64)**2)) for index in PARAMETERS)
        r.close(name + "/actual_norm_from_parameters", raw["actual/squared_l2"], norm, 1e-4, 2e-6)
        r.close(name + "/actual_norm_is_used_component", raw["actual/squared_l2"], raw["metrics/squared_l2"], 0.)
        r.close(name + "/actual_CE_plus_squared_L2", raw["metrics/loss"], float(independent_rows.mean()) + 1e-4 * norm, .001)
    audit_objective(r, saved["metrics_before"], arm, before, name + "/weighted_component_algebra")
    r.counts["actual_used_logit_rows"] += 96 if arm == "REPLAY" else 32


def update_audits(r: Reader, s: dict[str, Any], arm: str, receipt: dict[str, Any], code: str,
                  base: list[np.ndarray], timeline: dict[int, Any], eval_timeline: dict[int, Any]) -> dict[int, Any]:
    sd = r.out / "s5" / arm
    wanted = s["diagnostic_update_offsets"]
    r.require("audit/" + arm + "/five_offsets", wanted == [0, 3125, 31250, 143750, 174999])
    files = {p.name for p in (sd / "update_audit").glob("*.json")}
    expected = {f"{arm}_{4400000 + 32 * offset:011d}.json" for offset in wanted}
    r.require("audit/" + arm + "/exact_file_set", files == expected)
    identities = {}
    with h5py.File(r.out / "data/protection.h5", "r") as pool:
        for offset in wanted:
            stage = 4400000 + 32 * offset
            name = "audit/" + arm + "/" + str(stage)
            path = sd / "update_audit" / f"{arm}_{stage:011d}.json"
            saved = read_json(path)
            r.require(name + "/code_settings", saved.get("code_manifest_sha256") == code and saved.get("settings_sha256") == digest(r.work / "settings.json"))
            r.require(name + "/arm_stage", saved.get("arm") == arm and saved.get("stage") == stage and saved.get("update_offset") == offset and saved.get("seed") == {"id": "s5", "init": 5, "train": 2})
            r.require(name + "/coefficient", saved.get("coefficient") == (2. if arm == "REPLAY" else 1e-4))
            r.require(name + "/normal32", saved.get("normal_batch_size") == 32)
            r.require(name + "/protection_task_order", saved.get("protection_tasks") == ["icl", "flip_icl"])
            r.require(name + "/protection_use", saved.get("protection_used_for_gradient") is (arm == "REPLAY"))
            indices = np.arange(32, dtype=np.int64) + offset * 32
            indices %= 4096
            r.close(name + "/cyclic_protection_indices", saved["protection_indices"], indices, 0.)
            r.require(name + "/pool_SHA", saved.get("protection_data_sha256") == s["preserved_data_sha256"]["protection.h5"])
            for task in ("icl", "flip_icl"):
                x = np.asarray(pool[task + "/examples"][indices.tolist()], np.float32)
                y = np.asarray(pool[task + "/labels"][indices.tolist()], np.int32)
                h = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
                r.require(name + "/protection_batch_SHA/" + task, saved["protection_batch_sha256"].get(task) == h)
            r.require(name + "/normal_batch_digest_present", is_sha(saved.get("normal_batch_sha256")))
            _, used_data = split_key(timeline[offset]["train_data_seed"])
            _, used_model = split_key(timeline[offset]["train_model_seed"])
            r.close(name + "/normal_data_key_used", saved["train_data_key_used"], used_data, 0.)
            r.close(name + "/normal_model_key_used", saved["train_model_key_used"], used_model, 0.)
            calls = (stage - 4400000) // 100000 + 1
            states = []
            for label, at_offset, at_stage in (("before", offset, stage), ("after", offset + 1, stage + 32)):
                record = saved["checkpoints"][label]
                cp = at(r.work, record["path"])
                r.file(name + "/checkpoint/" + label, cp, record["sha256"])
                r.require(name + "/checkpoint_registered/" + label, receipt["output_sha256"].get(record["path"]) == record["sha256"])
                state = checkpoint(r, cp, at_stage, base, timeline[at_offset], eval_timeline[calls], name + "/" + label)
                for field, index in zip(SEEDS, (77, 78, 79)):
                    r.close(name + "/receipt_key/" + label + "/" + field, saved["training_prng_" + label][field], state[index], 0.)
                states.append(state)
            if offset == 0:
                for index in range(77):
                    r.close(name + "/exact_branch_model_Adam/" + str(index), states[0][index], base[index], 0.)
            before, after = states
            r.close(name + "/eval_key_not_consumed_by_update", before[77], after[77], 0.)
            r.close(name + "/Adam_count_increment", after[44], int(before[44]) + 1, 0.)
            r.require(name + "/fixed_production_capture_interface", saved.get("production_capture_interface_fixed") is True and saved.get("gradient_source") == "actual gradient used by this formal Adam update")
            actual_path = at(r.work, saved["actual_step_npz_path"])
            r.file(name + "/actual_step_NPZ", actual_path, saved["actual_step_npz_sha256"])
            r.require(name + "/actual_step_registered", receipt["output_sha256"].get(saved["actual_step_npz_path"]) == saved["actual_step_npz_sha256"])
            with np.load(actual_path, allow_pickle=False) as raw:
                actual_gradient_adam(r, raw, saved["gradient_leaves"], before, after, name + "/actual_gradient")
                actual_step_outputs(r, raw, saved, arm, before, name + "/actual_outputs")
            identities[offset] = {key: saved[key] for key in ("normal_batch_sha256", "protection_batch_sha256", "protection_indices", "train_data_key_used", "train_model_key_used")}
            r.counts["update_audits"] += 1
    return identities


def new_outputs(r: Reader, s: dict[str, Any], code: str, provenance: dict[str, Any], assets: Path,
                base: list[np.ndarray], labels: dict[str, Any], ids: dict[str, Any], audit_x: dict[str, Any]) -> dict[str, Any]:
    r.require("PRNG/known_legacy_split_vector", split_key((0, 0)) == ((4146024105, 967050713), (2718843009, 1272950319)))
    start = {field: base[index].tolist() for field, index in zip(SEEDS, (77, 78, 79))}
    wanted = {0, 175000}
    wanted.update((stage - 4400000) // 32 for stage in STAGES)
    wanted.update(s["diagnostic_update_offsets"])
    wanted.update(offset + 1 for offset in s["diagnostic_update_offsets"])
    timeline = key_timeline(start, wanted)
    eval_timeline = {0: start["eval_model_seed"]}
    for calls in range(1, 58):
        eval_timeline[calls] = list(split_key(eval_timeline[calls - 1])[0])
    data, diagnostics = {}, {}
    for arm in ARMS:
        sd, name = r.out / "s5" / arm, "output/" + arm
        cp = sd / "training_complete.json"
        r.file(name + "/completion", cp)
        completion = read_json(cp)
        for key, value in {"passed": True, "arm": arm, "seed": {"id": "s5", "init": 5, "train": 2}, "coefficient": 2. if arm == "REPLAY" else 1e-4,
                           "start_stage": 4400000, "end_stage": 10000000, "steps": 175000,
                           "model_and_adam_finite": True, "adam_count": 312500,
                           "training_prng_exact": True, "mechanism_diagnostics": False, "projection": False,
                           "code_manifest_sha256": code, "settings_sha256": digest(r.work / "settings.json")}.items():
            r.require(name + "/completion/" + key, completion.get(key) == value)
        for key in ("inputs", "source", "new_inputs"):
            r.require(name + "/completion/" + key, completion.get(key) == provenance.get(key))
        r.require(name + "/base_same", completion.get("base_checkpoint_sha256") == digest(assets / "checkpoints/00004400000.eqx"))
        registered = completion["output_sha256"]
        present = {}
        for group, pattern in (("original", "*.npz"), ("monitor", "*.npz"), ("confirm", "*.npz"),
                               ("checkpoints", "*.eqx"), ("update_audit", "*"), ("evaluated", "*.json")):
            present.update({str(path.relative_to(r.work)): path for path in sorted((sd / group).glob(pattern)) if path.is_file()})
        present[str((sd / "training_prng.json").relative_to(r.work))] = sd / "training_prng.json"
        r.require(name + "/exact_registered_output_set", set(registered) == set(present))
        for filename, path in present.items():
            r.file(name + "/artifact/" + filename, path, registered.get(filename))
        r.require(name + "/all_artifact_SHAs_present", all(is_sha(value) for value in registered.values()))
        confirm_set = {path.name for path in (sd / "confirm").glob("*.npz")}
        r.require(name + "/exact_11_confirm_files", confirm_set == {f"{arm}_{stage:011d}.npz" for stage in STAGES})
        original_set = {path.name for path in (sd / "original").glob("*.npz")}
        r.require(name + "/all_57_original_stages", original_set == {f"{arm}_{stage:011d}.npz" for stage in range(4400000, 10000001, 100000)})
        monitor_set = {path.name for path in (sd / "monitor").glob("*.npz")}
        monitor_stages = [4400000] + list(range(4500000, 10000001, 500000))
        r.require(name + "/all_fixed_monitor_stages", monitor_set == {f"{arm}_{stage:011d}.npz" for stage in monitor_stages})
        sums, means = completion["statistics_sum"], completion["statistics_mean"]
        r.require(name + "/statistics_steps", sums.get("steps") == 175000)
        for key, value in means.items():
            r.close(name + "/statistics_mean/" + key, value, sums[key] / 175000, 1e-12)
        prng = read_json(sd / "training_prng.json")
        r.require(name + "/PRNG_SHA", completion["training_prng_sha256"] == registered[str((sd / "training_prng.json").relative_to(r.work))])
        r.require(name + "/PRNG_receipt", prng.get("passed") is True and prng.get("steps") == 175000 and prng.get("train_data_and_model_streams_exact") is True and prng.get("protection_consumes_training_stream") is False and prng.get("code_manifest_sha256") == code)
        for field in SEEDS:
            r.close(name + "/PRNG_start/" + field, prng["start"][field], start[field], 0.)
            expected = eval_timeline[57] if field == "eval_model_seed" else timeline[175000][field]
            r.close(name + "/PRNG_end/" + field, prng["end"][field], expected, 0.)
        r.close(name + "/expected_train_data_receipt", prng["expected_train_data_seed"], timeline[175000]["train_data_seed"], 0.)
        r.close(name + "/expected_train_model_receipt", prng["expected_train_model_seed"], timeline[175000]["train_model_seed"], 0.)
        rows = {task: {metric: [] for metric in SCALARS} for task in TASKS}
        for stage in STAGES:
            path = sd / "confirm" / f"{arm}_{stage:011d}.npz"
            key = name + "/" + str(stage)
            expected_path = sd / "checkpoints" / f"{arm}_{stage:011d}.eqx"
            with np.load(path, allow_pickle=False) as raw:
                expected_meta = {"arm": arm, "stage": stage, "seed_id": "s5", "init_seed": 5, "train_seed": 2,
                                 "code_manifest_sha256": code, "settings_sha256": digest(r.work / "settings.json"),
                                 "data_sha256": s["preserved_data_sha256"]["confirm.h5"],
                                 "checkpoint_path": str(expected_path.relative_to(r.work)),
                                 "checkpoint_sha256": registered.get(str(expected_path.relative_to(r.work)))}
                for field, value in expected_meta.items():
                    saved = np.asarray(raw["_meta/" + field])
                    r.require(key + "/meta/" + field, saved.shape == () and saved.item() == value)
                    if isinstance(value, int):
                        r.require(key + "/meta_int64/" + field, saved.dtype == np.int64)
                for task in TASKS:
                    values = metric_group(r, raw, task, labels[task], ids[task], audit_x[task], key + "/" + task, True)
                    for metric, value in values.items():
                        rows[task][metric].append(value)
            r.file(key + "/checkpoint", expected_path, registered.get(str(expected_path.relative_to(r.work))))
            checkpoint(r, expected_path, stage, base, timeline[(stage - 4400000) // 32], eval_timeline[(stage - 4400000) // 100000 + 1], key + "/checkpoint_stream")
        r.require(name + "/final_checkpoint_is_10M_confirm", completion["final_checkpoint_path"] == str((sd / "checkpoints" / f"{arm}_00010000000.eqx").relative_to(r.work)) and completion["final_checkpoint_sha256"] == registered.get(completion["final_checkpoint_path"]))
        diagnostics[arm] = update_audits(r, s, arm, completion, code, base, timeline, eval_timeline)
        data[arm] = {task: {metric: np.sum(np.asarray(values), axis=0, dtype=np.float64) / 11 for metric, values in item.items()} for task, item in rows.items()}
    for offset in s["diagnostic_update_offsets"]:
        r.require("matched_arms/normal_and_protection_batches/" + str(offset), diagnostics["REPLAY"][offset] == diagnostics["L2"][offset])
    return data


def verify(work: Path = WORK) -> bool:
    r, independent = Reader(work), {}
    try:
        s = read_json(work / "settings.json")
        code, provenance, assets, base = identity(r, s)
        labels, ids, audit_x, query = fixed_data(r, s)
        preflight(r, s, code, provenance, base)
        data = references(r, s, labels, ids, audit_x)
        data.update(new_outputs(r, s, code, provenance, assets, base, labels, ids, audit_x))
        analysis_path = r.out / "analysis.json"
        r.file("analysis/file", analysis_path)
        saved = read_json(analysis_path)
        for field, path in (("code_manifest_sha256", work / "code_manifest.json"), ("settings_sha256", work / "settings.json"),
                            ("protocol_sha256", work / "protocol.md"), ("analysis_plan_sha256", work / "analysis_plan.md"),
                            ("references_manifest_sha256", work / "references_manifest.json"), ("confirm_data_sha256", r.out / "data/confirm.h5")):
            r.require("analysis/identity/" + field, saved.get(field) == digest(path))
        if (work / "postprocess_manifest.json").exists():
            r.require("analysis/identity/postprocessing", saved.get("postprocess_manifest_sha256") == digest(work / "postprocess_manifest.json"))
        r.require("analysis/fixed_window", saved.get("window") == [9000000, 10000000] and saved.get("stages") == STAGES and saved.get("n_per_task") == 5000)
        r.require("analysis/no_cross_seed_claim", saved.get("cross_seed_confirmation_performed") is False and saved.get("analysis_scope") == "single_seed_matched_baseline_pilot")
        independent = recalculate(data, query)
        for field, value in independent.items():
            r.subset("analysis/independent/" + field, saved.get(field), value)
    except Exception as error:
        r.require("reader/no_exception", False, {"type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
    passed = not r.failures
    support = {}
    for arm, methods in independent.get("methods", {}).items():
        for name in ("joint_vs_A", "P25_extra_value", "baseline_extra_value"):
            support[arm + "/" + name] = bool(passed and methods[name]["statistical_supported"])
    result = {"passed": passed, "status": "verified" if passed else "failed", "checks": r.checks,
              "failures": r.failures, "counts": dict(r.counts), "maximum_errors": r.maximum_errors,
              "verified_file_sha256": r.hashes, "independent_statistics": independent,
              "preflight_scope_validation": r.preflight_scopes,
              "scientific_support": support,
              "scope": "Independent NumPy identity, saved confirmatory rows, fixed eight-row full logits, checkpoint stream, actual used gradients with forward Adam reconstruction, actual training logits and batch hashes at five saved updates, Threefry PRNG and statistical readback. Distinct default production and float32 mathematical preflight receipts checked; no independent training or full model-forward rerun.",
              "limitations": ["One training seed; uncertainty is conditional on this trajectory.", "All confirmatory row summaries are read; full evaluation logits are checked only for eight prespecified audit rows per task and checkpoint.", "Actual training logits and actual gradients are read at five prespecified updates; no independent full sampler/model execution is claimed.", "The mathematical preflight uses a separate float32-matmul process. Production remains default matmul. Earlier default cross-graph failures remain failed and are not claimed to have passed.", "Failed noninferiority does not establish inferiority, and noninferiority is not strict equivalence."],
              "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    r.out.mkdir(parents=True, exist_ok=True)
    (r.out / "verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if passed:
        report_path = r.out / "report.md"
        if report_path.exists():
            report_path.write_text(report_path.read_text(encoding="utf-8").replace("统计结果待独立读取器核验。", "统计结果已通过独立产物读取与统计重算核验。"), encoding="utf-8")
    print(json.dumps({"passed": passed, "checks": r.checks, "failures": r.failures[:5], "scientific_support": support}, ensure_ascii=False))
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=WORK)
    raise SystemExit(0 if verify(parser.parse_args().work.resolve()) else 1)

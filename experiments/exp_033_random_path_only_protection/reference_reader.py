"""Independent NumPy reader for exp030 raw artifacts and fixed analysis.

No analyze/common/evaluation/JAX/Equinox import, model execution or sampling.
All saved row summaries are checked algebraically; full 12800-way logits and
log probabilities are recomputed for the eight fixed audit rows only. This is
an output-integrity check, not a claim that every unrecorded model output was
independently reproduced. Missing evidence fails visibly.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np

TASKS = ("train_eval", "iwl_copy_avail", "flip_icl", "icl")
SCALARS = ("acc", "loss", "in_context_acc", "pair_loss", "pair_margin",
           "context_mass", "log_context_mass")
PRIMARY = (("C25", "A", "icl", "min_icl_gain"),
           ("C25", "A", "flip_icl", "min_icl_gain"),
           ("C25", "B", "iwl_copy_avail", "min_icl_gain"),
           ("D25", "C25", "icl", "incremental_icl_gain"),
           ("D25", "C25", "flip_icl", "incremental_icl_gain"),
           ("D25", "R25", "icl", "incremental_icl_gain"),
           ("D25", "R25", "flip_icl", "incremental_icl_gain"))
NI_PAIRS = (("C25", "A", ("iwl_copy_avail", "train_eval")),
            ("C25", "B", ("icl", "flip_icl")),
            ("D25", "C25", ("iwl_copy_avail", "train_eval")),
            ("D25", "R25", ("iwl_copy_avail", "train_eval")))
GRID = {group: tuple(group + suffix for suffix in ("00", "25", "75")) for group in ("C", "D", "R")}
AUDIT_ROWS = np.asarray([0, 1, 127, 1024, 2499, 4095, 4998, 4999], np.int32)
STAGES = list(range(9000000, 10000001, 100000))
ONE_NORMAL = 1.6448536269514722
TWO_NORMAL = 1.959963984540054
ONE_T2 = 2.919985580353725
TWO_T2 = 4.302652729749461
RAW_ATOL = 2e-5
PILOT_ACTIVE_DATA_SETS = ["mechanism_confirm", "protection", "monitor", "confirm"]
INACTIVE_DATA_POLICY = "preserved and audited but not used for selection, protection or evaluation"


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for data in iter(lambda: source.read(1 << 20), b""):
            hasher.update(data)
    return hasher.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(work: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else work / path


def is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def file_records(work: Path, section: Any) -> dict[str, str]:
    """Accept the two explicit manifest record forms, never infer missing SHA."""
    if not isinstance(section, dict) or not section:
        raise ValueError("An identity section requires nonempty file records")
    if set(section) == {"sha256"} and isinstance(section["sha256"], dict):
        section = section["sha256"]
    records = {}
    for name, record in section.items():
        if isinstance(record, str):
            path, sha = name, record
        elif isinstance(record, dict) and "sha256" in record:
            path, sha = record.get("path", name), record["sha256"]
        else:
            raise ValueError(f"Malformed identity record {name}")
        if not isinstance(path, str) or not is_sha(sha):
            raise ValueError(f"Malformed path/SHA record {name}")
        key = str(resolve(work, path).resolve())
        if key in records and records[key] != sha:
            raise ValueError("Conflicting identities for the same file")
        records[key] = sha
    return records


class Reader:
    def __init__(self, work: Path):
        self.work, self.out = work, work / "outputs"
        self.assets_root = work / "assets"
        self.new_inputs: dict[str, str] = {}
        self.late_functional: dict[str, Any] = {}
        self.candidate_heads: dict[str, int] = {}
        self.dataset_overlaps: dict[str, Any] = {}
        self.data_generator_manifest_sha256: str | None = None
        self.failures: list[dict[str, Any]] = []
        self.checks = 0
        self.counts = Counter()
        self.hashes: dict[str, str] = {}
        self.maximum_errors: dict[str, float] = {}

    def require(self, name: str, passed: Any, detail: Any = None) -> bool:
        self.checks += 1
        okay = bool(passed)
        if not okay:
            self.failures.append({"check": name, "detail": detail})
        return okay

    def close(self, name: str, observed: Any, expected: Any, tolerance: float = RAW_ATOL) -> bool:
        a, b = np.asarray(observed), np.asarray(expected)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            return self.require(name, False, {"reason": "shape_or_nonfinite", "observed": list(a.shape), "expected": list(b.shape)})
        error = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.
        category = name.rsplit("/", 1)[-1]
        self.maximum_errors[category] = max(error, self.maximum_errors.get(category, 0.))
        return self.require(name, error <= tolerance, {"max_abs": error, "absolute_tolerance": tolerance})

    def file(self, name: str, path: Path, expected_sha: str | None = None) -> str | None:
        if not self.require(name + "/exists", path.is_file(), str(path)):
            return None
        key = str(path.resolve())
        digest = self.hashes.get(key)
        if digest is None:
            digest = digest_file(path)
            self.hashes[key] = digest
            self.counts["hashed_files"] += 1
        if expected_sha is not None:
            self.require(name + "/sha256", is_sha(expected_sha) and digest == expected_sha,
                         {"path": str(path), "actual": digest, "expected": expected_sha})
        return digest

    def subset(self, name: str, observed: Any, expected: Any) -> None:
        """Compare every independently recomputed field; diagnostic extras allowed."""
        if isinstance(expected, dict):
            if not self.require(name + "/mapping", isinstance(observed, dict)):
                return
            for key, value in expected.items():
                if self.require(name + "/key/" + key, key in observed):
                    self.subset(name + "/" + key, observed[key], value)
        elif isinstance(expected, list):
            if self.require(name + "/length", isinstance(observed, list) and len(observed) == len(expected)):
                for i, (left, right) in enumerate(zip(observed, expected)):
                    self.subset(name + f"/{i}", left, right)
        elif expected is None or isinstance(expected, (str, bool)):
            self.require(name, observed == expected, {"observed": observed, "expected": expected})
        elif isinstance(expected, (int, float)):
            self.close(name, observed, expected, 1e-10)
        else:
            self.require(name, observed == expected)


def clustered(values: Any, classes: Any) -> dict[str, Any]:
    delta, classes = np.asarray(values, np.float64), np.asarray(classes)
    if delta.ndim != 1 or delta.size == 0 or delta.shape != classes.shape or not np.isfinite(delta).all():
        raise ValueError("Malformed independently paired rows")
    names, membership, counts = np.unique(classes, return_inverse=True, return_counts=True)
    totals = np.zeros(names.size, np.float64)
    np.add.at(totals, membership, delta)
    estimate = float(np.sum(delta) / delta.size)
    score = totals - counts * estimate
    variance = (float(np.sum(score * score) * names.size / (names.size - 1) / delta.size**2)
                if names.size > 1 else None)
    standard_error = math.sqrt(max(variance, 0.)) if variance is not None else None
    valid = standard_error is not None and standard_error > 1e-15
    return {"mean": estimate, "se": standard_error, "n": int(delta.size), "clusters": int(names.size),
            "valid": bool(valid),
            "lower_one_sided_95": estimate - ONE_NORMAL * standard_error if valid else None,
            "upper_one_sided_95": estimate + ONE_NORMAL * standard_error if valid else None,
            "ci95_lower": estimate - TWO_NORMAL * standard_error if valid else None,
            "ci95_upper": estimate + TWO_NORMAL * standard_error if valid else None}


def repeated_run_stat(values: Any) -> dict[str, Any]:
    estimates = np.asarray(values, np.float64)
    if estimates.shape != (3,) or not np.isfinite(estimates).all():
        return {"mean": None, "se": None, "valid": False, "n_training_seeds": int(estimates.size),
                "df": 2, "lower_one_sided_95": None, "upper_one_sided_95": None,
                "ci95_lower": None, "ci95_upper": None}
    mean = float(np.sum(estimates) / 3)
    se = math.sqrt(float(np.sum((estimates - mean)**2) / 2 / 3))
    valid = se > 1e-15
    return {"mean": mean, "se": se, "valid": bool(valid), "n_training_seeds": 3, "df": 2,
            "seed_means": estimates.tolist(), "lower_one_sided_95": mean - ONE_T2 * se if valid else None,
            "upper_one_sided_95": mean + ONE_T2 * se if valid else None,
            "ci95_lower": mean - TWO_T2 * se if valid else None,
            "ci95_upper": mean + TWO_T2 * se if valid else None}


def upper_tail_t2(value: float) -> float:
    if value < 0:
        return 1. - upper_tail_t2(-value)
    hypotenuse = math.hypot(value, math.sqrt(2))
    return 1. / (hypotenuse * (hypotenuse + value))


def gain_test(stat: dict[str, Any], name: str, threshold: float, seed_level: bool = False) -> dict[str, Any]:
    z = (stat["mean"] - threshold) / stat["se"] if stat["valid"] else None
    p = (upper_tail_t2(z) if seed_level else .5 * math.erfc(z / math.sqrt(2))) if z is not None else 1.
    return dict(stat, name=name, threshold=float(threshold), z_or_t=z, p_raw=float(p))


def absent_test(name: str, threshold: float) -> dict[str, Any]:
    return {"name": name, "mean": None, "se": None, "valid": False, "threshold": float(threshold),
            "p_raw": 1., "z_or_t": None}


def adjust_holm(items: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    tests = [dict(item) for item in items]
    ordering = np.argsort(np.asarray([item["p_raw"] for item in tests]), kind="stable")
    adjusted = np.maximum.accumulate(np.minimum(1., np.asarray([tests[int(slot)]["p_raw"] * (len(tests) - rank)
                                      for rank, slot in enumerate(ordering)])))
    for slot, value in zip(ordering, adjusted):
        test = tests[int(slot)]
        test.update(p_holm=float(value), family_size=len(tests),
                    passed=bool(test["valid"] and value < alpha and test["mean"] > test["threshold"]))
    return {"tests": tests, "alpha": float(alpha), "family_size": len(tests),
            "all_passed": all(item["passed"] for item in tests), "correction": "Holm"}


def independent_ni(stat: dict[str, Any], metric: str, settings: dict[str, Any]) -> dict[str, Any]:
    threshold = -float(settings["ni_acc_tolerance"]) if metric == "acc" else float(settings["ni_ce_tolerance"])
    bound = stat["lower_one_sided_95"] if metric == "acc" else stat["upper_one_sided_95"]
    passed = bool(bound is not None and (bound > threshold if metric == "acc" else bound < threshold))
    return dict(stat, threshold=threshold, metric=metric, passed=passed)


def strength_matching(completions: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    d, r = (completions.get(arm, {}).get("update_statistics", {}) for arm in ("D25", "R25"))
    tests = {}
    for metric, tolerance in (("mean_update_norm_ratio", settings["primary_d_r_update_ratio_tolerance"]),
                              ("mean_correction_norm_ratio", settings["primary_d_r_update_ratio_tolerance"]),
                              ("conflict_rate", settings["primary_d_r_conflict_rate_tolerance"])):
        a, b = d.get(metric), r.get(metric)
        valid = isinstance(a, (int, float)) and isinstance(b, (int, float))
        valid = valid and math.isfinite(float(a)) and math.isfinite(float(b)) and a >= 0 and b >= 0
        if metric == "conflict_rate":
            valid = valid and a <= 1 and b <= 1
        difference = abs(a - b) if valid else None
        tests[metric] = {"D25": a, "R25": b, "absolute_difference": difference,
                         "tolerance": float(tolerance), "passed": bool(valid and difference <= tolerance)}
    return {"tests": tests, "passed": all(item["passed"] for item in tests.values())}


def recalculate_seed(data: dict[str, Any], query: dict[str, Any], settings: dict[str, Any],
                     mechanism_passed: bool, completions: dict[str, Any]) -> dict[str, Any]:
    tests = []
    for arm, reference, task, threshold_key in PRIMARY:
        name, threshold = f"{arm}-{reference}/{task}/acc", settings[threshold_key]
        tests.append(gain_test(clustered(data[arm][task]["acc"] - data[reference][task]["acc"], query[task]), name, threshold)
                     if arm in data and reference in data else absent_test(name, threshold))
    primary = adjust_holm(tests, settings["alpha"])
    noninferiority = {}
    for arm, reference, tasks in NI_PAIRS:
        gates = {}
        for task in tasks:
            for metric in ("acc", "loss"):
                gates[f"{task}/{metric}"] = (independent_ni(clustered(data[arm][task][metric] - data[reference][task][metric], query[task]), metric, settings)
                                              if arm in data and reference in data else
                                              {"passed": False, "status": "missing_prespecified_arm_no_support"})
        noninferiority[f"{arm}-{reference}"] = {"tests": gates, "all_passed": all(test["passed"] for test in gates.values())}
        if (arm, reference) == ("C25", "B"):
            noninferiority["C25-B"].update(required_for_primary_confirmation=False,
                role="descriptive_retention_comparison_with_freeze")
    matching = strength_matching(completions, settings)
    c_gains, d_gains = (all(primary["tests"][i]["passed"] for i in indices) for indices in (range(3), range(3, 7)))
    c_ni = noninferiority["C25-A"]["all_passed"]
    d_ni = noninferiority["D25-C25"]["all_passed"] and noninferiority["D25-R25"]["all_passed"]
    methods = {"ordinary_task_protection": {"gains_passed": c_gains, "noninferiority_passed": c_ni,
                                             "confirmation_target": "relative_A_tradeoff_improvement_plus_CIWL_learning_gain_over_B",
                                             "B_retention_noninferiority_required": False,
                                             "B_retention_noninferiority_descriptive_passed": noninferiority["C25-B"]["all_passed"],
                                             "statistical_supported": bool(c_gains and c_ni)},
               "causal_path_increment": {"gains_passed": d_gains, "noninferiority_passed": d_ni,
                                         "mechanism_passed": bool(mechanism_passed),
                                         "d_r_strength_matching_passed": matching["passed"],
                                         "statistical_supported": bool(d_gains and d_ni and mechanism_passed and matching["passed"])}}
    return {"primary_Holm7": primary, "noninferiority": noninferiority, "d_r_matching": matching, "methods": methods}


def recalculate_aggregate(data: dict[str, Any], reports: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    seeds = [entry["id"] for entry in settings["training_seeds"]]
    if settings.get("analysis_scope") == "single_seed_pilot":
        tag = seeds[0]
        available = set(data) == {tag} and set(reports) == {tag}
        methods = {}
        for label in ("ordinary_task_protection", "causal_path_increment"):
            supported = bool(available and reports[tag]["methods"][label]["statistical_supported"])
            methods[label] = {"pilot_statistical_supported": supported,
                              "confirmation_statistical_supported": False,
                              "cross_seed_statistical_supported": False,
                              "independent_reader_required": True}
        return {"status": "not_performed_single_seed_pilot", "primary_Holm7": None,
                "methods": methods, "require_seed_aggregate_holm": False,
                "n_training_seeds": 1, "pilot_seed": tag, "unit": "one_fixed_training_trajectory",
                "note": "Pilot support is conditional on s5 class-cluster Holm7 and joint gates; no between-seed confirmation"}
    tests = []
    for arm, reference, task, threshold_key in PRIMARY:
        name, threshold = f"{arm}-{reference}/{task}/acc", settings[threshold_key]
        if all(seed in data and arm in data[seed] and reference in data[seed] for seed in seeds):
            values = [float(np.sum(data[seed][arm][task]["acc"] - data[seed][reference][task]["acc"]) / settings["n"]) for seed in seeds]
            tests.append(gain_test(repeated_run_stat(values), name, threshold, True))
        else:
            tests.append(absent_test(name, threshold))
    family = adjust_holm(tests, settings["alpha"])
    methods = {}
    for label, indices in (("ordinary_task_protection", range(3)), ("causal_path_increment", range(3, 7))):
        tests_for_method = [family["tests"][i] for i in indices]
        per_seed = all(seed in reports and reports[seed]["methods"][label]["statistical_supported"] for seed in seeds)
        lower_passed = all(item.get("lower_one_sided_95") is not None and item["lower_one_sided_95"] > item["threshold"] for item in tests_for_method)
        holm_passed = all(item["passed"] for item in tests_for_method)
        methods[label] = {"all_three_training_seed_joint_gates_passed": per_seed,
                          "seed_mean_one_sided_lower_bounds_passed": lower_passed,
                          "seed_mean_Holm7_passed": holm_passed,
                          "confirmation_statistical_supported": bool(per_seed and lower_passed and holm_passed),
                          "independent_reader_required": True}
        if label == "ordinary_task_protection":
            methods[label].update(
                confirmation_target="relative_A_tradeoff_improvement_plus_CIWL_learning_gain_over_B",
                B_retention_noninferiority_required=False)
    return {"primary_Holm7": family, "methods": methods, "require_seed_aggregate_holm": True}


def independently_select_frontier(means: dict[str, Any], difference: Any, settings: dict[str, Any]) -> dict[str, Any]:
    if settings.get("analysis_scope") == "single_seed_pilot":
        return {"status": "not_planned_single_rho_pilot", "confirmed": False, "budgets": {},
                "reason": "Only rho=.25 is authorized; no strength-grid frontier or seed/class bootstrap is estimated"}
    budgets = {}
    for budget in settings["frontier_icl_tolerances"]:
        families = {}
        for family, arms in GRID.items():
            candidates = []
            for arm in arms:
                if arm not in means or "A" not in means or "B" not in means:
                    candidates.append({"arm": arm, "feasible": False, "status": "missing_arm"})
                    continue
                gates = {}
                for task in ("icl", "flip_icl"):
                    lower = difference(arm, "B", task, "acc")["lower_one_sided_95"]
                    gates[task + "/budget_acc"] = {"lower_one_sided_95": lower, "threshold": -float(budget),
                                                   "passed": bool(lower is not None and lower > -budget)}
                for metric in ("acc", "loss"):
                    gates["train_eval/" + metric] = independent_ni(difference(arm, "A", "train_eval", metric), metric, settings)
                candidates.append({"arm": arm, "feasible": all(value["passed"] for value in gates.values()), "gates": gates,
                                   "ciwl_acc": float(means[arm]["iwl_copy_avail"]["acc"]),
                                   "ciwl_ce": float(means[arm]["iwl_copy_avail"]["loss"])})
            possible = [item for item in candidates if item["feasible"]]
            selected = sorted(possible, key=lambda item: -item["ciwl_acc"])[0] if possible else None
            families[family] = {"candidates": candidates, "selected_arm": selected["arm"] if selected else None,
                                "ciwl_acc": selected["ciwl_acc"] if selected else None,
                                "ciwl_ce": selected["ciwl_ce"] if selected else None,
                                "status": "finite_three_point_frontier" if selected else "no_feasible_prespecified_point"}
        budgets[str(budget)] = families
    return {"budgets": budgets}


def logsumexp_numpy(values: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(values, np.float64)
    maximum = np.max(z, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(z - maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def verify_metric_group(reader: Reader, raw: Any, task: str, labels: np.ndarray, n: int, name: str) -> dict[str, np.ndarray]:
    values = {}
    for metric in (*SCALARS, "full_pred", "pair_pred", "target"):
        key = f"{task}/{metric}"
        if not reader.require(name + "/key/" + metric, key in raw):
            raise ValueError(f"Missing raw row field {key}")
        value = np.asarray(raw[key])
        if not reader.require(name + "/shape/" + metric, value.shape == (n,), list(value.shape)):
            raise ValueError(f"Malformed raw row field {key}")
        reader.require(name + "/finite/" + metric, np.isfinite(value).all())
        values[metric] = value
    y = np.asarray(labels)
    target, other = y[:, -1], y[:, 0] + y[:, 1] - y[:, -1]
    reader.require(name + "/two_unique_candidate_labels", np.all(y[:, 0] != y[:, 1]))
    reader.require(name + "/target_exactly_once_in_context", np.all(np.sum(y[:, :2] == target[:, None], axis=1) == 1))
    for metric in ("full_pred", "pair_pred", "target"):
        reader.require(name + "/integer/" + metric, np.issubdtype(values[metric].dtype, np.integer))
        reader.require(name + "/range/" + metric, np.all((values[metric] >= 0) & (values[metric] < 12800)))
    reader.close(name + "/target_input_identity", values["target"], target, 0.)
    reader.require(name + "/acc_binary", np.isin(values["acc"], (0., 1.)).all())
    reader.require(name + "/pair_acc_binary", np.isin(values["in_context_acc"], (0., 1.)).all())
    reader.close(name + "/full_acc_prediction", values["acc"], (values["full_pred"] == target).astype(np.float64), 0.)
    margin = values["pair_margin"].astype(np.float64)
    pair_prediction = np.where((margin > 0) | ((margin == 0) & (target < other)), target, other)
    reader.close(name + "/pair_pred_margin", values["pair_pred"], pair_prediction, 0.)
    reader.close(name + "/pair_acc_prediction", values["in_context_acc"], (pair_prediction == target).astype(np.float64), 0.)
    reader.close(name + "/pair_CE_margin", values["pair_loss"], np.logaddexp(0., -margin))
    reader.close(name + "/fullCE_pairCE_logmass", values["loss"], values["pair_loss"].astype(np.float64) - values["log_context_mass"])
    reader.close(name + "/context_mass_exp", values["context_mass"], np.exp(values["log_context_mass"].astype(np.float64)))
    reader.require(name + "/fullCE_nonnegative", np.all(values["loss"] >= -RAW_ATOL))
    reader.require(name + "/context_mass_range", np.all((values["context_mass"] >= 0) & (values["context_mass"] <= 1 + RAW_ATOL))
                   and np.all(values["log_context_mass"] <= RAW_ATOL))
    reader.require(name + "/full_correct_implies_pair_correct", np.all(values["acc"] <= values["in_context_acc"]))
    for metric in ("audit_rows", "audit_logits", "audit_log_probs"):
        if not reader.require(name + "/key/" + metric, f"{task}/{metric}" in raw):
            raise ValueError("Missing fixed full-output audit")
    rows = np.asarray(raw[f"{task}/audit_rows"])
    logits, logp = (np.asarray(raw[f"{task}/{metric}"]) for metric in ("audit_logits", "audit_log_probs"))
    reader.require(name + "/fixed_audit_rows", rows.dtype == np.int32 and np.array_equal(rows, AUDIT_ROWS))
    if not reader.require(name + "/audit_shapes", logits.shape == logp.shape == (8, 12800)):
        raise ValueError("Malformed full-output audit")
    reader.require(name + "/audit_FP32", logits.dtype == logp.dtype == np.float32)
    reader.require(name + "/audit_finite", np.isfinite(logits).all() and np.isfinite(logp).all())
    z, lp = logits.astype(np.float64), logp.astype(np.float64)
    denominator = logsumexp_numpy(z, axis=1)
    reader.close(name + "/log_probs_from_logits", lp, z - denominator[:, None])
    reader.close(name + "/log_probs_normalized", logsumexp_numpy(lp, axis=1), np.zeros(8))
    row_target, row_other, ix = target[AUDIT_ROWS], other[AUDIT_ROWS], np.arange(8)
    reader.close(name + "/fullCE_from_logits", values["loss"][AUDIT_ROWS], denominator - z[ix, row_target])
    reader.close(name + "/fullCE_from_log_probs", values["loss"][AUDIT_ROWS], -lp[ix, row_target])
    reader.close(name + "/full_pred_from_logits", values["full_pred"][AUDIT_ROWS], np.argmax(z, axis=1), 0.)
    reader.close(name + "/full_pred_from_log_probs", values["full_pred"][AUDIT_ROWS], np.argmax(lp, axis=1), 0.)
    true_z, other_z = z[ix, row_target], z[ix, row_other]
    audit_margin = true_z - other_z
    predicted_pair = np.where((true_z > other_z) | ((true_z == other_z) & (row_target < row_other)), row_target, row_other)
    logmass = np.logaddexp(true_z, other_z) - denominator
    reader.close(name + "/pair_margin_from_logits", margin[AUDIT_ROWS], audit_margin)
    reader.close(name + "/pairCE_from_logits", values["pair_loss"][AUDIT_ROWS], np.logaddexp(0., -audit_margin))
    reader.close(name + "/pair_pred_from_logits", values["pair_pred"][AUDIT_ROWS], predicted_pair, 0.)
    reader.close(name + "/log_context_mass_from_logits", values["log_context_mass"][AUDIT_ROWS], logmass)
    reader.close(name + "/context_mass_from_logits", values["context_mass"][AUDIT_ROWS], np.exp(logmass))
    reader.counts["raw_summary_rows"] += n
    reader.counts["fixed_full_logit_rows"] += 8
    reader.counts["metric_groups"] += 1
    return {metric: values[metric].astype(np.float64) for metric in SCALARS}


def verify_identity(reader: Reader) -> tuple[dict[str, Any], dict[str, str], dict[str, str], str | None]:
    manifest_path = reader.work / "code_manifest.json"
    manifest_digest = reader.file("identity/code_manifest", manifest_path)
    if manifest_digest is None:
        return {}, {}, {}, None
    manifest = load_json(manifest_path)
    required = ("settings.json", "protocol.md", "input_spec.json", "analyze.py", "verify_results.py")
    reader.require("identity/manifest_required_files", all(name in manifest for name in required))
    for name, expected in manifest.items():
        reader.require("identity/manifest_record/" + name, is_sha(expected))
        reader.file("identity/code/" + name, resolve(reader.work, name), expected)
    settings = load_json(reader.work / "settings.json")
    if settings.get("analysis_scope") == "single_seed_pilot":
        generator_digest = settings.get("data_generator_manifest_sha256")
        reader.require("identity/data_generator_anchor", is_sha(generator_digest))
        creation = reader.work / "data_creation_manifest.json"
        reader.require("identity/creation_manifest_sealed_by_execution_manifest", "data_creation_manifest.json" in manifest)
        archived = reader.file("identity/data_generator_manifest", creation, generator_digest)
        reader.data_generator_manifest_sha256 = generator_digest if archived == generator_digest else None
        if archived is not None:
            generator = load_json(creation)
            original_source = reader.work / "attempts/attempt_001_unused_discovery_overlap/source"
            for name in ("settings.json", "datasets.py", "common.py", "input_spec.json"):
                expected = generator.get(name)
                reader.require("identity/data_generator/record/" + name, is_sha(expected))
                reader.file("identity/data_generator/original_source/" + name, original_source / name, expected)
            original_settings = original_source / "settings.json"
            if original_settings.is_file():
                reader.require("identity/data_generator/purpose_seeds_and_counts_preserved",
                               load_json(original_settings).get("data_sets") == settings["data_sets"])
        preserved = settings.get("preserved_data_sha256")
        reader.require("identity/five_preserved_data_hashes", isinstance(preserved, dict)
                       and set(preserved) == {name + ".h5" for name in settings["data_sets"]}
                       and all(is_sha(value) for value in preserved.values()))
    else:
        reader.data_generator_manifest_sha256 = manifest_digest
    provenance_path = reader.out / "provenance.json"
    if reader.file("identity/provenance", provenance_path) is None:
        return {}, {}, {}, manifest_digest
    provenance = load_json(provenance_path)
    reader.require("identity/provenance_code_manifest", provenance.get("code_manifest_sha256") == manifest_digest)
    reader.assets_root = resolve(reader.work, provenance.get("assets_root", "assets")).resolve()
    try:
        inputs = file_records(reader.assets_root, provenance.get("inputs"))
        source = file_records(reader.assets_root / "source", provenance.get("source"))
        reader.new_inputs = file_records(reader.work, provenance.get("new_inputs"))
    except ValueError as error:
        reader.require("identity/provenance_records", False, str(error))
        return provenance, {}, {}, manifest_digest
    specification = load_json(reader.work / "input_spec.json")
    expected_inputs = {str((reader.assets_root / name).resolve()): item["sha256"]
                       for name, item in specification["files"].items()}
    reader.require("identity/inputs_equal_sealed_scientific_specification", inputs == expected_inputs)
    expected_source = {str((reader.assets_root / "source" / name).resolve()):
                       specification["files"]["source/" + name]["sha256"]
                       for name in specification["source_files"]}
    reader.require("identity/source_equal_sealed_specification", source == expected_source)
    asset_manifest_path = reader.assets_root / "assets_manifest.json"
    asset_digest = reader.file("identity/assets_manifest", asset_manifest_path)
    reader.require("identity/assets_manifest_equal_registered_preparation",asset_digest==specification.get("registered_asset_manifest_sha256"))
    reader.require("identity/assets_manifest_bound_to_provenance",
                   is_sha(provenance.get("assets_manifest_sha256")) and provenance.get("assets_manifest_sha256") == asset_digest)
    if asset_digest is not None:
        assets = file_records(reader.assets_root, load_json(asset_manifest_path))
        for path, expected in assets.items():
            reader.file("identity/exported_asset/" + path, Path(path), expected)
        replay_path = str((reader.assets_root / specification["replay_checkpoint"]["to"]).resolve())
        reader.require("identity/replay_checkpoint_registered_in_asset_manifest", replay_path in assets)
        reader.require("identity/replay_checkpoint_equal_registered_preparation",assets.get(replay_path)==specification['replay_checkpoint']['sha256'])
        reader.require("identity/specification_present_in_asset_manifest", all(assets.get(name) == digest for name, digest in inputs.items()))
    for section, records in (("inputs", inputs), ("source", source), ("new_inputs", reader.new_inputs)):
        for path, expected in records.items():
            reader.file(f"identity/{section}/{path}", Path(path), expected)
    for name in ("mechanism_discovery", "mechanism_confirm", "protection", "monitor", "confirm"):
        path = str((reader.out / "data" / f"{name}.h5").resolve())
        reader.require("identity/fixed_data_registered/" + name, path in reader.new_inputs)
    return provenance, inputs, source, manifest_digest


def feature_lookup(reader: Reader, inputs: dict[str, str]) -> dict[bytes, tuple[int, int]] | None:
    import h5py

    path = reader.assets_root / "data/omniglot_features_norotate.h5"
    if not reader.require("data/registered_evaluation_feature_file", str(path.resolve()) in inputs and path.is_file()):
        return None
    with h5py.File(path, "r") as source:
        if not reader.require("data/evaluation_feature_dataset", "resnet18/224/feat" in source):
            return None
        shape = source["resnet18/224/feat"].shape
        if not reader.require("data/evaluation_feature_shape", len(shape) == 3 and shape[0] >= 1600 and shape[1:] == (20, 512), list(shape)):
            return None
        features = np.asarray(source["resnet18/224/feat"][:1600, :5], np.float32)
        reader.require("data/evaluation_feature_used_shape", features.shape == (1600, 5, 512), list(features.shape))
        reader.require("data/feature_finite", np.isfinite(features).all())
        if "original_class" in source:
            reader.close("data/feature_class_order", source["original_class"][:1600], np.arange(1600), 0.)
        if "augmentation" in source:
            reader.require("data/feature_no_rotation", np.all(source["augmentation"][:1600] == 0))
    lookup = {}
    for class_id in range(1600):
        for exemplar in range(5):
            key = hashlib.sha256(np.ascontiguousarray(features[class_id, exemplar]).tobytes()).digest()
            if key in lookup:
                reader.require("data/feature_hash_unique", False, {"class": class_id, "exemplar": exemplar})
            lookup[key] = (class_id, exemplar)
    return lookup


def verify_dataset_overlaps(reader: Reader, settings: dict[str, Any], exact: dict[str, set[bytes]],
                            visible: dict[str, set[bytes]], saved: dict[str, Any]) -> None:
    """Audit all purposes, gate only the frozen purposes actually used by s5."""
    names = list(settings["data_sets"])
    pilot = settings.get("analysis_scope") == "single_seed_pilot"
    active = settings.get("active_data_sets") if pilot else names
    reader.require("data/active_purpose_scope", active == PILOT_ACTIVE_DATA_SETS if pilot else active == names)
    if not isinstance(active, list) or not all(name in names for name in active):
        active = []
    inactive = [name for name in names if name not in active]
    pairs = [left + "/" + right for left, right in combinations(active, 2)]
    if pilot:
        reader.require("data/inactive_discovery_only", inactive == ["mechanism_discovery"])
        reader.subset("data/audit/active_data_sets", saved.get("active_data_sets"), active)
        reader.subset("data/audit/active_independence_pairs", saved.get("active_independence_pairs"), pairs)
        reader.subset("data/audit/inactive_data_sets", saved.get("inactive_data_sets"), inactive)
        reader.require("data/audit/inactive_policy", saved.get("inactive_policy") == INACTIVE_DATA_POLICY)
    all_pairs = [left + "/" + right for left, right in combinations(names, 2)]
    for field in ("exact_sequence_overlap", "visible_model_input_overlap"):
        reader.require("data/audit/complete_pair_set/" + field,
                       isinstance(saved.get(field), dict) and set(saved[field]) == set(all_pairs))
    active_gate_passed = True
    for left, right in combinations(names, 2):
        pair = left + "/" + right
        complete = all(name in exact and name in visible for name in (left, right))
        reader.require("data/overlap_inputs_present/" + pair, complete)
        if not complete:
            active_gate_passed = False
            continue
        counts = {"exact_sequence_overlap": len(exact[left] & exact[right]),
                  "visible_model_input_overlap": len(visible[left] & visible[right])}
        is_active = left in active and right in active
        reader.dataset_overlaps[pair] = dict(counts, active_independence_gate=is_active,
            status="zero_required" if is_active else "unused_not_selection")
        for field, count in counts.items():
            reader.require("data/audit/recomputed/" + field + "/" + pair,
                           saved.get(field, {}).get(pair) == count, {"overlap": count})
            if is_active:
                active_gate_passed = active_gate_passed and count == 0
                reader.require("data/no_" + field + "/" + pair, count == 0, {"overlap": count})
    reader.require("data/audit/active_gate_recomputed", saved.get("passed") is active_gate_passed)


def verify_generation_metadata(reader: Reader, name: str, metadata: dict[str, Any]) -> bool:
    digest = reader.data_generator_manifest_sha256
    return reader.require("data/" + name + "/metadata_generation_code_identity",
                          is_sha(digest) and metadata.get("code_manifest_sha256") == digest,
                          {"data_generator_manifest_sha256": digest,
                           "execution_code_manifest_sha256": reader.hashes.get(str((reader.work / "code_manifest.json").resolve()))})


def verify_datasets(reader: Reader, settings: dict[str, Any], lookup: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    import h5py

    labels, query = {}, {}
    sequences: dict[str, set[bytes]] = {}
    exact_sequences: dict[str, set[bytes]] = {}
    for name, spec in settings["data_sets"].items():
        path = reader.out / "data" / f"{name}.h5"
        if not reader.require("data/" + name + "/exists", path.is_file(), str(path)):
            continue
        if settings.get("analysis_scope") == "single_seed_pilot":
            original_digest = settings.get("preserved_data_sha256", {}).get(name + ".h5")
            reader.require("data/" + name + "/preserved_hash_registered", is_sha(original_digest))
            reader.file("data/" + name + "/preserved_generated_bytes", path, original_digest)
        required_tasks = TASKS
        hashes, exact_hashes = set(), set()
        with h5py.File(path, "r") as source:
            try:
                metadata = json.loads(source.attrs["metadata"])
            except (KeyError, TypeError, ValueError) as error:
                reader.require("data/" + name + "/metadata", False, str(error))
                metadata = {}
            reader.require("data/" + name + "/metadata_purpose", metadata.get("name") == name)
            reader.require("data/" + name + "/metadata_seed", metadata.get("seed") == spec["seed"])
            reader.require("data/" + name + "/metadata_n", metadata.get("n") == spec["n"])
            verify_generation_metadata(reader, name, metadata)
            reader.require("data/" + name + "/metadata_author_task_order",
                           metadata.get("task_order") == [*TASKS, "pure_iwl"])
            sampler_digest = reader.hashes.get(str((reader.assets_root / "source/samplers.py").resolve()))
            feature_digest = reader.hashes.get(str((reader.assets_root / "data/omniglot_features_norotate.h5").resolve()))
            reader.require("data/" + name + "/metadata_sampler", metadata.get("sampler_sha256") == sampler_digest)
            reader.require("data/" + name + "/metadata_feature", metadata.get("feature_sha256") == feature_digest)
            for task in required_tasks:
                tag = f"data/{name}/{task}"
                if not reader.require(tag + "/group", task in source):
                    continue
                group = source[task]
                if not all(reader.require(tag + "/key/" + key, key in group) for key in ("examples", "labels", "identities")):
                    continue
                x, y, identities = (np.asarray(group[key]) for key in ("examples", "labels", "identities"))
                n = int(spec["n"])
                shape = x.shape == (n, 3, 512) and y.shape == (n, 3) and identities.shape == (n, 3, 2)
                if not reader.require(tag + "/shapes", shape, {"examples": list(x.shape), "labels": list(y.shape), "identities": list(identities.shape)}):
                    continue
                reader.require(tag + "/FP32_examples", x.dtype == np.float32)
                reader.require(tag + "/finite_examples", np.isfinite(x).all())
                reader.require(tag + "/integer_labels_identities", np.issubdtype(y.dtype, np.integer) and np.issubdtype(identities.dtype, np.integer))
                reader.require(tag + "/label_range", np.all((y >= 0) & (y < 12800)))
                reader.require(tag + "/class_range", np.all((identities[..., 0] >= 0) & (identities[..., 0] < 1600)))
                reader.require(tag + "/exemplar_range", np.all((identities[..., 1] >= 0) & (identities[..., 1] < 5)))
                reader.require(tag + "/unique_context_labels", np.all(y[:, 0] != y[:, 1]))
                reader.require(tag + "/unique_target_support", np.all(np.sum(y[:, :2] == y[:, -1, None], axis=1) == 1))
                if task in ("icl", "flip_icl"):
                    matches = identities[:, :2, 0] == identities[:, -1, 0, None]
                    reader.require(tag + "/unique_query_class_support", np.all(np.sum(matches, axis=1) == 1))
                    reader.close(tag + "/class_label_support_alignment", np.argmax(matches, axis=1), np.argmax(y[:, :2] == y[:, -1, None], axis=1), 0.)
                actual_classes = identities[..., 0]
                support = actual_classes[:, :2] == actual_classes[:, -1, None]
                target_support = y[:, :2] == y[:, -1, None]
                reader.require(tag + "/distinct_context_classes", np.all(actual_classes[:, 0] != actual_classes[:, 1]))
                if task == "iwl_copy_avail":
                    reader.require(tag + "/query_class_absent", np.all(support.sum(axis=1) == 0))
                    reader.close(tag + "/query_original_label", y[:, -1], actual_classes[:, -1], 0.)
                    reader.close(tag + "/noncopied_context_label", y[:, :2][~target_support], actual_classes[:, :2][~target_support], 0.)
                elif task == "train_eval":
                    reader.require(tag + "/unique_query_class_support", np.all(support.sum(axis=1) == 1))
                    reader.close(tag + "/original_class_labels", y, actual_classes, 0.)
                elif task == "flip_icl":
                    reader.close(tag + "/flipped_context_labels", y[:, :2], actual_classes[:, 1::-1], 0.)
                    reader.require(tag + "/query_label_flipped", np.all(y[:, -1] != actual_classes[:, -1]))
                else:
                    reader.close(tag + "/relabelled_01_candidates", np.sort(y[:, :2], axis=1), np.tile([0, 1], (n, 1)), 0.)
                if lookup is not None:
                    okay = True
                    first_mismatch = None
                    for row in range(n):
                        for slot in range(3):
                            key = hashlib.sha256(np.ascontiguousarray(x[row, slot]).tobytes()).digest()
                            expected = lookup.get(key)
                            actual = tuple(int(v) for v in identities[row, slot])
                            if expected != actual:
                                okay = False
                                if first_mismatch is None:
                                    first_mismatch = {"row": row, "slot": slot, "expected": expected, "actual": actual}
                    reader.require(tag + "/identities_from_actual_features", okay, first_mismatch)
                for row in range(n):
                    # Include ordered examples and context labels. Invisible query
                    # target is intentionally excluded when identifying inputs.
                    hasher = hashlib.sha256()
                    hasher.update(np.ascontiguousarray(x[row]).tobytes())
                    hasher.update(np.ascontiguousarray(y[row, :2].astype(np.int32)).tobytes())
                    hashes.add(hasher.digest())
                    full_hasher = hashlib.sha256()
                    full_hasher.update(np.ascontiguousarray(x[row]).tobytes())
                    full_hasher.update(np.ascontiguousarray(y[row].astype(np.int32)).tobytes())
                    exact_hashes.add(full_hasher.digest())
                if name == "confirm":
                    labels[task], query[task] = y, identities[:, -1, 0]
                reader.counts["data_rows"] += n
        sequences[name], exact_sequences[name] = hashes, exact_hashes
    audit_path = reader.out / "data/identity_audit.json"
    audit = load_json(audit_path) if audit_path.is_file() else {}
    verify_dataset_overlaps(reader, settings, exact_sequences, sequences, audit)
    return labels, query


def verify_prerequisite_gates(reader: Reader, settings: dict[str, Any], inputs: dict[str, str],
                              source: dict[str, str], code_digest: str | None) -> None:
    path = reader.out / "preflight.json"
    if reader.file("prerequisite/GPU_preflight", path) is not None:
        preflight = load_json(path)
        reader.require("prerequisite/GPU_preflight/passed", preflight.get("passed") is True)
        reader.require("prerequisite/GPU_preflight/device_verified", preflight.get("model_forward_and_gradient_gpu_verified") is True)
        reader.require("prerequisite/GPU_preflight/code_identity", preflight.get("code_manifest_sha256") == code_digest)
        reader.require("prerequisite/GPU_preflight/sealed_limits", preflight.get("limits") == settings)
        for section, expected, base in (("inputs", inputs, reader.assets_root), ("source", source, reader.assets_root / "source")):
            try:
                actual = file_records(base, preflight.get(section))
                reader.require("prerequisite/GPU_preflight/" + section, actual == expected)
            except ValueError as error:
                reader.require("prerequisite/GPU_preflight/" + section, False, str(error))
        checks = preflight.get("checks", {})
        obligations = {}
        for kind in ("P",):
            for field in ("adam_max_abs", "model_max_abs"):
                obligations[kind + "_" + field] = settings["numeric_model_adam_max_abs"]
            if kind != "B":
                obligations[kind + "_numpy_projection_max_abs"] = settings["numeric_projection_reference_max_abs"]
                obligations[kind + "_numpy_projection_relative"] = settings["numeric_projection_reference_max_relative"]
                obligations[kind + "_projection_relative_violation"] = settings["numeric_projection_relative_violation_max"]
                obligations[kind + "_jit_projection_relative_violation"] = settings["numeric_projection_relative_violation_max"]
        for task in (*TASKS, "pure_iwl"):
            obligations[task + "_eval_ce_max_mean"] = settings["numeric_eval_mean_ce_max_abs"]
            obligations[task + "_eval_acc_rows"] = settings["numeric_eval_acc_max_rows"]
        for name, limit in obligations.items():
            value = checks.get(name)
            valid = isinstance(value, (float, int)) and math.isfinite(value) and 0 <= value <= limit
            reader.require("prerequisite/GPU_preflight/check/" + name, valid, {"value": value, "limit": limit})
        derivative = checks.get("functional_derivative", {})
        norm, finite_difference, saved_error = (derivative.get(key) for key in ("autodiff", "finite_difference", "abs_error"))
        finite = all(isinstance(value, (int, float)) and math.isfinite(value) for value in (norm, finite_difference, saved_error))
        reader.require("prerequisite/GPU_preflight/functional_derivative_fields", finite)
        if finite:
            reader.require("prerequisite/GPU_preflight/functional_derivative_nonzero", norm > 0)
            error = abs(finite_difference - norm)
            reader.close("prerequisite/GPU_preflight/functional_derivative_error", saved_error, error, 1e-10)
            limit = settings["numeric_functional_grad_absolute_tolerance"] + settings["numeric_functional_grad_relative_tolerance"] * norm
            reader.require("prerequisite/GPU_preflight/functional_derivative_gate", error <= limit,
                           {"error": error, "limit": limit})
    audit_path = reader.out / "data/identity_audit.json"
    if reader.file("prerequisite/fixed_data_gate", audit_path) is not None:
        audit = load_json(audit_path)
        reader.require("prerequisite/fixed_data_gate/passed", audit.get("passed") is True)
        reader.require("prerequisite/fixed_data_gate/no_resampling", audit.get("resampling") is False)
        for name in settings["data_sets"]:
            expected = reader.new_inputs.get(str((reader.out / "data" / f"{name}.h5").resolve()))
            reader.require("prerequisite/fixed_data_gate/hash/" + name,
                           audit.get("data", {}).get(name, {}).get("sha256") == expected)
    replay_path = reader.work / "references/replay.json"
    if reader.file("prerequisite/s5_author_replay", replay_path) is not None:
        replay = load_json(replay_path)
        reader.require("prerequisite/s5_author_replay/passed", replay.get("passed") is True)
        reader.require("prerequisite/s5_author_replay/training_prng_exact", replay.get("training_prng_exact") is True)
        for field in ("model_error", "adam_error"):
            error = replay.get(field)
            reader.require("prerequisite/s5_author_replay/" + field,
                           isinstance(error, (int, float)) and math.isfinite(error) and 0 <= error <= settings["numeric_model_adam_max_abs"])


def check_completion(reader: Reader, seed: str, arm: str, settings: dict[str, Any],
                     input_records: dict[str, str], source_records: dict[str, str], manifest_digest: str | None) -> dict[str, Any] | None:
    path = reader.out / seed / "training_complete" / f"{arm}.json"
    prefix = f"completion/{seed}/{arm}"
    if reader.file(prefix, path) is None:
        return None
    record = load_json(path)
    reader.require(prefix + "/passed", record.get("passed") is True)
    reader.require(prefix + "/code_manifest", record.get("code_manifest_sha256") == manifest_digest)
    reader.require(prefix + "/end_stage", record.get("end_stage") == settings["end_stage"])
    for section, expected in (("inputs", input_records), ("source", source_records)):
        try:
            base = reader.assets_root if section == "inputs" else reader.assets_root / "source"
            registered = file_records(base, record.get(section))
        except ValueError as error:
            reader.require(prefix + "/" + section, False, str(error))
            continue
        reader.require(prefix + "/" + section + "_identity", registered == expected)
    try:
        registered_new = file_records(reader.work, record.get("new_inputs"))
        reader.require(prefix + "/new_inputs_identity", registered_new == reader.new_inputs)
    except ValueError as error:
        reader.require(prefix + "/new_inputs_identity", False, str(error))
    compute = record.get("compute_seconds")
    reader.require(prefix + "/compute_seconds_recorded", isinstance(compute, (int, float)) and math.isfinite(compute) and compute >= 0)
    checkpoint_sha = record.get("final_checkpoint_sha256")
    reader.require(prefix + "/final_checkpoint_sha_registered", is_sha(checkpoint_sha))
    checkpoint_path = record.get("final_checkpoint_path")
    if not reader.require(prefix + "/final_checkpoint_path_registered", isinstance(checkpoint_path, str)):
        return record
    reader.file(prefix + "/final_checkpoint", resolve(reader.work, checkpoint_path), checkpoint_sha)
    return record


def inspect_late(reader: Reader, seed: str, arm: str, stage: int, settings: dict[str, Any]) -> None:
    path = reader.out / seed / "mechanism_late" / f"{arm}_{stage:011d}.npz"
    prefix = f"late_mechanism/{seed}/{arm}/{stage}"
    if reader.file(prefix, path) is None:
        return
    values, classes, structural = read_mechanism_rows(reader, path, reader.out / "data/confirm.h5", settings,
                                                    prefix, reader.candidate_heads.get(seed))
    summaries = {}
    for task, data in values.items():
        summaries[task] = {"donor_ce_gain": clustered(data["native"]["donor_ce"] - data["donor"]["donor_ce"], classes[task]),
                           "candidate_K_ce_gain": clustered(data["native"]["donor_ce"] - data["path"]["donor_ce"], classes[task]),
                           "candidate_K_minus_V_ce_gain": clustered(data["v_path"]["donor_ce"] - data["path"]["donor_ce"], classes[task]),
                           "candidate_donor_margin_gain": clustered(data["path"]["donor_margin"] - data["native"]["donor_margin"], classes[task]),
                           "reverse_native_ce_gain": clustered(data["donor"]["native_ce"] - data["reverse_path"]["native_ce"], classes[task])}
    reader.require(prefix + "/structural_gate", structural)
    reader.late_functional[seed + "/" + arm] = {"stage": stage, "structure_passed": structural, "tasks": summaries,
        "interpretation": "Fixed-endpoint causal diagnostics; do not select a new head or substitute these for full behavior"}


MECHANISM_STEMS = ("native", "donor", "path", "v_path", "complete_k", "complete_v",
                   "self", "query_label_control", "reverse_path", "reverse_v_path")
MECHANISM_FIELDS = ("donor_ce", "native_ce", "donor_margin", "donor_pair_acc",
                    "native_pair_acc", "native_full_acc", "donor_full_acc", "full_pred", "pair_pred")
EXACT_PATH_FIELDS = ("embedding_max_abs", "l1_upstream_max_abs", "l1_values_pre_proj_max_abs",
                     "l1_values_post_proj_max_abs", "l2_norm_input_max_abs", "l2_raw_qkv_max_abs",
                     "q_max_abs", "other_channel_max_abs", "nonlabel_channel_max_abs",
                     "target_channel_max_abs", "hybrid_nonlabel_residual_max_abs")
TOLERANT_PATH_FIELDS = ("qk_recompute_max_abs", "attention_row_sum_max_abs", "future_attention_max")
EXTRA_EXACT = ("query_input_max_abs",)
EXTRA_TOLERANT = ("self_logits_max_abs", "invisible_query_label_logits_max_abs",
                  "l1_native_head_reconstruction_max_abs", "l1_donor_head_reconstruction_max_abs")
RESIDUAL_CACHE_FIELDS = ("recipient_embedding", "patched_embedding", "recipient_l1_out", "donor_l1_out",
    "patched_l1_out", "native_l1_residual", "hybrid_l1_residual", "recipient_l1_residual_post_attn",
    "donor_l1_residual_post_attn", "patched_l1_residual_post_attn", "recipient_l1_values_pre_proj",
    "donor_l1_values_pre_proj", "patched_l1_values_pre_proj", "recipient_l1_values_post_proj",
    "donor_l1_values_post_proj", "patched_l1_values_post_proj", "native_head_values_pre_proj",
    "donor_head_values_pre_proj", "native_head_projected", "donor_head_projected",
    "hybrid_l2_norm_input", "recipient_l2_norm_input", "patched_l2_norm_input")
CHANNEL_CACHE_FIELDS = ("recipient_l2_q", "recipient_l2_k", "recipient_l2_v", "patched_l2_q",
                       "patched_l2_k", "patched_l2_v", "hybrid_l2_k", "hybrid_l2_v")
ATTENTION_CACHE_FIELDS = ("recipient_l2_attn_scores", "patched_l2_attn_scores",
                         "recipient_l2_attn_pre_softmax", "patched_l2_attn_pre_softmax")


def byte_equal(reader: Reader, name: str, observed: Any, expected: Any) -> bool:
    left, right = np.asarray(observed), np.asarray(expected)
    same = left.shape == right.shape and left.dtype == right.dtype
    same = same and np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(right).tobytes()
    return reader.require(name, same, {"observed_shape": list(left.shape), "expected_shape": list(right.shape),
                                      "observed_dtype": str(left.dtype), "expected_dtype": str(right.dtype)})


def verify_cache_rows(reader: Reader, raw: Any, task: str, examples: np.ndarray, labels: np.ndarray,
                       settings: dict[str, Any], name: str, expected_head: int | None = None) -> None:
    """First128 actual cached tensors, independently checking preserved sites."""
    rows = min(128, len(labels))
    row_key = f"{task}/cache_rows"
    if not reader.require(name + "/cache_rows_key", row_key in raw):
        raise ValueError("Missing fixed first128 cache-row identities")
    reader.close(name + "/fixed_first128_rows", raw[row_key], np.arange(rows), 0.)
    reader.require(name + "/cache_rows_integer", np.issubdtype(np.asarray(raw[row_key]).dtype, np.integer))
    shapes = {"source_x": (rows, 3, 512), "source_y": (rows, 3), "swapped_x": (rows, 3, 512),
              "forward_key": (rows, 2), "native_target": (rows,), "donor_target": (rows,),
              "selected_head": (rows,), "patched_channel_is_k": (rows,), "label_token_mask": (rows, 5),
              "head_dim_mask": (rows, 64)}
    shapes.update({field: (rows, 5, 64) for field in RESIDUAL_CACHE_FIELDS})
    shapes.update({field: (rows, 1, 8, 5, 8) for field in CHANNEL_CACHE_FIELDS})
    shapes.update({field: (rows, 1, 8, 5, 5) for field in ATTENTION_CACHE_FIELDS})
    shapes.update({field: (rows, 3, 8, 5, 8) for field in ("recipient_l2_qkv", "patched_l2_qkv")})
    cache = {}
    for field, shape in shapes.items():
        key = f"{task}/cache/{field}"
        if not reader.require(name + "/key/" + field, key in raw):
            raise ValueError("Missing first128 functional tensor cache")
        value = np.asarray(raw[key])
        if not reader.require(name + "/shape/" + field, value.shape == shape, list(value.shape)):
            raise ValueError("Malformed first128 functional tensor cache")
        reader.require(name + "/finite/" + field, np.isfinite(value).all())
        if field in (*RESIDUAL_CACHE_FIELDS, *CHANNEL_CACHE_FIELDS, *ATTENTION_CACHE_FIELDS,
                     "recipient_l2_qkv", "patched_l2_qkv", "source_x", "swapped_x"):
            reader.require(name + "/FP32/" + field, value.dtype == np.float32)
        cache[field] = value
    byte_equal(reader, name + "/source_examples", cache["source_x"], examples[:rows])
    byte_equal(reader, name + "/source_labels", cache["source_y"], labels[:rows])
    byte_equal(reader, name + "/swapped_context_examples", cache["swapped_x"], examples[:rows, [1, 0, 2]])
    byte_equal(reader, name + "/query_unchanged", cache["source_x"][:, -1], cache["swapped_x"][:, -1])
    reader.close(name + "/native_target", cache["native_target"], labels[:rows, -1], 0.)
    reader.close(name + "/donor_target", cache["donor_target"], labels[:rows, 0] + labels[:rows, 1] - labels[:rows, -1], 0.)
    heads = cache["selected_head"]
    reader.require(name + "/head_integer_range", np.issubdtype(heads.dtype, np.integer) and np.all((heads >= 0) & (heads < 8)))
    reader.require(name + "/one_frozen_head", np.unique(heads).size == 1)
    if expected_head is not None:
        reader.require(name + "/expected_head", np.all(heads == expected_head))
    reader.require(name + "/K_channel", cache["patched_channel_is_k"].dtype == bool and cache["patched_channel_is_k"].all())
    reader.require(name + "/forward_key_uint32", cache["forward_key"].dtype == np.uint32)
    label_mask = np.tile(np.asarray([False, True, False, True, False]), (rows, 1))
    byte_equal(reader, name + "/label_site_mask", cache["label_token_mask"], label_mask)
    head_mask = (np.arange(64)[None, :] // 8) == heads[:, None]
    byte_equal(reader, name + "/head_site_mask", cache["head_dim_mask"], head_mask)
    for native, patched in (("recipient_embedding", "patched_embedding"),
                             ("recipient_l1_out", "patched_l1_out"),
                             ("recipient_l1_residual_post_attn", "patched_l1_residual_post_attn"),
                             ("recipient_l1_values_pre_proj", "patched_l1_values_pre_proj"),
                             ("recipient_l1_values_post_proj", "patched_l1_values_post_proj"),
                             ("recipient_l2_norm_input", "patched_l2_norm_input"),
                             ("recipient_l2_qkv", "patched_l2_qkv"),
                             ("recipient_l2_q", "patched_l2_q"), ("recipient_l2_v", "patched_l2_v"),
                             ("recipient_l1_out", "native_l1_residual")):
        byte_equal(reader, name + "/preserved/" + patched, cache[patched], cache[native])
    nonlabels, labels_at = [0, 2, 4], [1, 3]
    byte_equal(reader, name + "/nonlabel_K_preserved", cache["patched_l2_k"][:, :, :, nonlabels, :], cache["recipient_l2_k"][:, :, :, nonlabels, :])
    byte_equal(reader, name + "/label_K_from_hybrid", cache["patched_l2_k"][:, :, :, labels_at, :], cache["hybrid_l2_k"][:, :, :, labels_at, :])
    byte_equal(reader, name + "/hybrid_nonlabel_residual", cache["hybrid_l1_residual"][:, nonlabels], cache["native_l1_residual"][:, nonlabels])
    for prefix in ("recipient", "donor"):
        source_field = prefix + "_l1_values_pre_proj"
        masked_field = ("native" if prefix == "recipient" else "donor") + "_head_values_pre_proj"
        reader.close(name + "/masked_head_values/" + prefix, cache[masked_field], cache[source_field] * head_mask[:, None, :], 0.)
    expected_hybrid = cache["native_l1_residual"] + np.where(label_mask[:, :, None],
        cache["donor_head_projected"] - cache["native_head_projected"], np.float32(0))
    reader.close(name + "/hybrid_residual_algebra", cache["hybrid_l1_residual"], expected_hybrid, settings["numeric_sham_logit_max_abs"])
    for channel, position in (("q", 0), ("k", 1), ("v", 2)):
        byte_equal(reader, name + "/recipient_qkv_split/" + channel,
                   cache["recipient_l2_" + channel], cache["recipient_l2_qkv"][:, position:position + 1])
    causal = np.tril(np.ones((5, 5), bool))
    for prefix in ("recipient", "patched"):
        q, k = cache[prefix + "_l2_q"], cache[prefix + "_l2_k"]
        computed = np.matmul(q, np.swapaxes(k, -1, -2)) * np.float32(8**-.5)
        before = cache[prefix + "_l2_attn_pre_softmax"]
        reader.close(name + "/QK_scores/" + prefix, before[..., causal], computed[..., causal], settings["numeric_path_qk_max_abs"])
        reader.close(name + "/causal_future_mask/" + prefix, before[..., ~causal], np.full_like(before[..., ~causal], -1e20), 0.)
        score = cache[prefix + "_l2_attn_scores"]
        normalized = np.exp(before.astype(np.float64) - logsumexp_numpy(before, -1)[..., None])
        reader.close(name + "/softmax/" + prefix, score, normalized, settings["numeric_path_rowsum_max_abs"])
        reader.close(name + "/attention_rowsum/" + prefix, score.sum(-1, dtype=np.float64), np.ones(score.shape[:-1]), settings["numeric_path_rowsum_max_abs"])
        reader.require(name + "/future_attention/" + prefix, np.all(score[..., ~causal] <= settings["numeric_path_future_attention_max"]))
    reader.counts["functional_full_cache_rows"] += rows


def read_mechanism_rows(reader: Reader, path: Path, data_path: Path, settings: dict[str, Any],
                        name: str, expected_head: int | None = None) -> tuple[dict[str, Any], dict[str, Any], bool]:
    import h5py

    values, classes, structure_passed = {}, {}, True
    if reader.file(name, path) is None:
        return values, classes, False
    with np.load(path, allow_pickle=False) as raw, h5py.File(data_path, "r") as source:
        for task in settings["protection_tasks"]:
            prefix = name + "/" + task
            labels = np.asarray(source[f"{task}/labels"])
            examples = np.asarray(source[f"{task}/examples"])
            identities = np.asarray(source[f"{task}/identities"])
            n = len(labels)
            native_target = labels[:, -1]
            donor_target = labels[:, 0] + labels[:, 1] - native_target
            classes[task] = identities[:, -1, 0]
            values[task] = {}
            verify_cache_rows(reader, raw, task, examples, labels, settings, prefix + "/cache128", expected_head)
            for field, expected in (("native_target", native_target), ("donor_target", donor_target),
                                    ("pair_valid", np.ones(n, bool)), ("identities", identities)):
                if not reader.require(prefix + "/key/" + field, f"{task}/{field}" in raw):
                    raise ValueError("Missing mechanism row identities")
                reader.close(prefix + "/" + field, raw[f"{task}/{field}"], expected, 0.)
            audit_rows = AUDIT_ROWS[AUDIT_ROWS < n]
            reader.close(prefix + "/audit_rows", raw[f"{task}/audit_rows"], audit_rows, 0.)
            ix = np.arange(audit_rows.size)
            nt, dt = native_target[audit_rows], donor_target[audit_rows]
            for stem in MECHANISM_STEMS:
                group = {}
                for field in MECHANISM_FIELDS:
                    key = f"{task}/{stem}/{field}"
                    if not reader.require(prefix + "/key/" + stem + "/" + field, key in raw):
                        raise ValueError("Missing saved functional metric")
                    row = np.asarray(raw[key])
                    if not reader.require(prefix + "/shape/" + stem + "/" + field, row.shape == (n,)):
                        raise ValueError("Malformed functional rows")
                    reader.require(prefix + "/finite/" + stem + "/" + field, np.isfinite(row).all())
                    group[field] = row.astype(np.float64)
                values[task][stem] = group
                tag = prefix + "/" + stem
                reader.close(tag + "/two_CEs_imply_margin", group["native_ce"] - group["donor_ce"], group["donor_margin"])
                reader.require(tag + "/CE_nonnegative", np.all(group["donor_ce"] >= -RAW_ATOL) and np.all(group["native_ce"] >= -RAW_ATOL))
                for field in ("donor_pair_acc", "native_pair_acc", "native_full_acc", "donor_full_acc"):
                    reader.require(tag + "/binary/" + field, np.isin(group[field], (0., 1.)).all())
                donor_choice = (group["donor_margin"] > 0) | ((group["donor_margin"] == 0) & (donor_target < native_target))
                reader.close(tag + "/donor_pair_choice", group["donor_pair_acc"], donor_choice, 0.)
                reader.close(tag + "/native_pair_choice", group["native_pair_acc"], ~donor_choice, 0.)
                reader.close(tag + "/pair_prediction", group["pair_pred"], np.where(donor_choice, donor_target, native_target), 0.)
                reader.close(tag + "/native_full_prediction", group["native_full_acc"], group["full_pred"] == native_target, 0.)
                reader.close(tag + "/donor_full_prediction", group["donor_full_acc"], group["full_pred"] == donor_target, 0.)
                for field in ("full_pred", "pair_pred"):
                    saved_prediction = np.asarray(raw[f"{task}/{stem}/{field}"])
                    reader.require(tag + "/integer/" + field, np.issubdtype(saved_prediction.dtype, np.integer))
                    reader.require(tag + "/range/" + field, np.all((saved_prediction >= 0) & (saved_prediction < 12800)))
                reader.require(tag + "/full_implies_pair", np.all(group["native_full_acc"] <= group["native_pair_acc"]))
                reader.require(tag + "/donor_full_implies_pair", np.all(group["donor_full_acc"] <= group["donor_pair_acc"]))
                logit_key = f"{task}/audit_{stem}_logits"
                if not reader.require(tag + "/audit_logits_key", logit_key in raw):
                    raise ValueError("Missing functional full-output audit")
                logits = np.asarray(raw[logit_key])
                if not reader.require(tag + "/audit_logits_shape", logits.shape == (len(audit_rows), 12800)):
                    raise ValueError("Malformed functional full-output audit")
                reader.require(tag + "/audit_logits_FP32", logits.dtype == np.float32)
                reader.require(tag + "/audit_logits_finite", np.isfinite(logits).all())
                z = logits.astype(np.float64)
                denominator = logsumexp_numpy(z, 1)
                reader.close(tag + "/audit_donorCE", group["donor_ce"][audit_rows], denominator - z[ix, dt])
                reader.close(tag + "/audit_nativeCE", group["native_ce"][audit_rows], denominator - z[ix, nt])
                reader.close(tag + "/audit_margin", group["donor_margin"][audit_rows], z[ix, dt] - z[ix, nt])
                reader.close(tag + "/audit_full_native_acc", group["native_full_acc"][audit_rows], np.argmax(z, 1) == nt, 0.)
                reader.close(tag + "/audit_full_donor_acc", group["donor_full_acc"][audit_rows], np.argmax(z, 1) == dt, 0.)
                reader.close(tag + "/audit_full_pred", group["full_pred"][audit_rows], np.argmax(z, 1), 0.)
                reader.counts["functional_summary_rows"] += n
                reader.counts["functional_fixed_full_logit_rows"] += len(audit_rows)
            for branch in ("audit", "reverse_audit"):
                exact = EXACT_PATH_FIELDS + (EXTRA_EXACT if branch == "audit" else ())
                tolerant = TOLERANT_PATH_FIELDS + (EXTRA_TOLERANT if branch == "audit" else ())
                for field in (*exact, *tolerant):
                    key = f"{task}/{branch}/{field}"
                    if not reader.require(prefix + "/key/" + branch + "/" + field, key in raw):
                        raise ValueError("Missing functional structural audit")
                    row = np.asarray(raw[key], np.float64)
                    if not reader.require(prefix + "/audit_shape_finite/" + branch + "/" + field,
                                          row.shape == (n,) and np.isfinite(row).all() and np.all(row >= 0)):
                        raise ValueError("Malformed structural audit rows")
                    special = {"qk_recompute_max_abs": settings["numeric_path_qk_max_abs"],
                               "future_attention_max": settings["numeric_path_future_attention_max"],
                               "attention_row_sum_max_abs": settings["numeric_path_rowsum_max_abs"]}
                    limit = 0. if field in exact else special.get(field, settings["numeric_sham_logit_max_abs"])
                    structure_passed = structure_passed and bool(np.all(row <= limit))
    return values, classes, structure_passed


def verify_mechanism_gate(reader: Reader, seed: dict[str, Any], settings: dict[str, Any], record: dict[str, Any],
                          inputs: dict[str, str], source: dict[str, str], code_digest: str | None) -> bool:
    tag = seed["id"]
    prefix = "mechanism/" + tag
    reader.require(prefix + "/code_manifest", record.get("code_manifest_sha256") == code_digest)
    for section, expected, base in (("inputs", inputs, reader.assets_root),
                                    ("source", source, reader.assets_root / "source"),
                                    ("new_inputs", reader.new_inputs, reader.work)):
        try:
            actual = file_records(base, record.get(section))
            reader.require(prefix + "/" + section, actual == expected)
        except ValueError as error:
            reader.require(prefix + "/" + section, False, str(error))
    checkpoint = (reader.assets_root / "checkpoints/00004400000.eqx" if tag == "s5" else
                  reader.out / tag / "prefix/00004400000.eqx")
    reader.file(prefix + "/base_checkpoint", checkpoint, record.get("checkpoint_sha256"))
    raw_path = reader.out / tag / "mechanism_confirm.npz"
    reader.file(prefix + "/confirm_raw_identity", raw_path, record.get("confirm_raw_sha256"))
    values, classes, structural = read_mechanism_rows(reader, raw_path, reader.out / "data/mechanism_confirm.h5", settings,
                                                     prefix + "/raw", record.get("candidate_head"))
    if not values:
        return False
    tests, floors = [], {}
    for task in settings["protection_tasks"]:
        raw = values[task]
        donor = raw["native"]["donor_ce"] - raw["donor"]["donor_ce"]
        k = raw["native"]["donor_ce"] - raw["path"]["donor_ce"]
        v = raw["native"]["donor_ce"] - raw["v_path"]["donor_ce"]
        contrasts = (("donor_ce", donor, settings["mechanism_donor_ce_min"]),
                     ("path_ce", k, settings["mechanism_path_ce_min"]),
                     ("retention", k - settings["mechanism_retention_min"] * donor, 0.),
                     ("k_vs_v", k - v, settings["mechanism_k_vs_v_min"]),
                     ("pair_margin", raw["path"]["donor_margin"] - raw["native"]["donor_margin"], settings["mechanism_pair_margin_min"]))
        for name, rows, threshold in contrasts:
            tests.append(gain_test(clustered(rows, classes[task]), task + "/" + name, threshold))
        for stem, field, threshold in (("native", "native_pair_acc", settings["mechanism_pair_acc_floor"]),
                                       ("donor", "donor_pair_acc", settings["mechanism_pair_acc_floor"]),
                                       ("native", "native_full_acc", settings["mechanism_full_acc_floor"])):
            stat = clustered(raw[stem][field], classes[task])
            floors[task + "/" + stem + "/" + field] = dict(stat, threshold=float(threshold),
                passed=bool(stat["valid"] and stat["lower_one_sided_95"] > threshold))
    family = adjust_holm(tests, settings["alpha"])
    reader.require(prefix + "/Holm10_size", family["family_size"] == 10)
    for test in family["tests"]:
        expected = {"mean": test["mean"], "se": test["se"], "p": test["p_raw"],
                    "threshold": test["threshold"], "p_holm": test["p_holm"], "passed": test["passed"],
                    "classes": test["clusters"], "rows": test["n"]}
        if test["valid"]:
            expected.update(lower95=test["lower_one_sided_95"], upper95=test["upper_one_sided_95"])
        reader.subset(prefix + "/contrast/" + test["name"], record.get("checks", {}).get(test["name"]), expected)
    for name, floor in floors.items():
        expected = {"mean": floor["mean"], "se": floor["se"], "threshold": floor["threshold"], "passed": floor["passed"]}
        if floor["valid"]:
            expected.update(lower95=floor["lower_one_sided_95"], upper95=floor["upper_one_sided_95"])
        reader.subset(prefix + "/floor/" + name, record.get("floors", {}).get(name), expected)
    if tag == "s5":
        expected_head = 0
        if settings.get("analysis_scope") == "single_seed_pilot":
            reader.require(prefix + "/unused_discovery_not_selection", record.get("discovery_scores") == {})
    else:
        scores = {}
        for head in range(8):
            path = reader.out / tag / "mechanism_discovery" / f"head{head}.npz"
            discovered, _, _ = read_mechanism_rows(reader, path, reader.out / "data/mechanism_discovery.h5", settings,
                                                  prefix + f"/discovery/head{head}", head)
            if not discovered:
                return False
            score = float(np.mean([np.mean(discovered[task]["native"]["donor_ce"] - discovered[task]["path"]["donor_ce"])
                                   for task in settings["protection_tasks"]]))
            scores[head] = score
            saved_score = record.get("discovery_scores", {}).get(str(head))
            reader.close(prefix + f"/discovery_score/head{head}", saved_score, score, RAW_ATOL)
        expected_head = min(range(8), key=lambda head: (-scores[head], head))
    reader.require(prefix + "/prespecified_candidate_head", record.get("candidate_head") == expected_head)
    reader.candidate_heads[tag] = expected_head
    material = (f"exp030/random-head/v1|init={seed['init']}|train={seed['train']}"
                f"|random_path_seed={settings['random_path_seed']}").encode("ascii")
    random_seed = int.from_bytes(hashlib.sha256(material).digest(), "big")
    expected_random = random.Random(random_seed).choice([head for head in range(8) if head != expected_head])
    reader.require(prefix + "/outcome_blind_random_head", record.get("random_head") == expected_random)
    passed = bool(structural and family["all_passed"] and all(item["passed"] for item in floors.values()))
    reader.require(prefix + "/structure_gate_recomputed", record.get("structure_passed") is structural)
    reader.require(prefix + "/overall_gate_recomputed", record.get("passed") is passed)
    return passed


def verify_outputs(reader: Reader, settings: dict[str, Any], labels: dict[str, Any],
                   input_records: dict[str, str], source_records: dict[str, str], manifest_digest: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    all_data, receipt = {}, {}
    for seed in settings["training_seeds"]:
        tag = seed["id"]
        mechanism_path = reader.out / tag / "mechanism_confirmation.json"
        if reader.file(f"mechanism/{tag}", mechanism_path) is None:
            mechanism = {"passed": False}
        else:
            mechanism = load_json(mechanism_path)
            reader.require(f"mechanism/{tag}/passed_boolean", isinstance(mechanism.get("passed"), bool))
            reader.require(f"mechanism/{tag}/code_manifest", mechanism.get("code_manifest_sha256") == manifest_digest)
            for field in ("candidate_head", "random_head"):
                reader.require(f"mechanism/{tag}/{field}", isinstance(mechanism.get(field), int) and 0 <= mechanism[field] < 8)
            if all(isinstance(mechanism.get(field), int) for field in ("candidate_head", "random_head")):
                reader.require(f"mechanism/{tag}/distinct_heads", mechanism["candidate_head"] != mechanism["random_head"])
            reader.require(f"mechanism/{tag}/checks_present", isinstance(mechanism.get("checks"), (dict, list)))
        eligible = verify_mechanism_gate(reader, seed, settings, mechanism, input_records, source_records, manifest_digest) if mechanism_path.is_file() else False
        arms = [arm for arm in settings["arms"] if eligible or not arm.startswith(("D", "R"))]
        if not eligible and mechanism_path.is_file():
            for arm in settings["arms"]:
                if not arm.startswith(("D", "R")):
                    continue
                stopped = reader.out / tag / "not_run" / (arm + ".json")
                if reader.file(f"mechanism/{tag}/prespecified_stop/{arm}", stopped) is not None:
                    record = load_json(stopped)
                    reader.require(f"mechanism/{tag}/prespecified_stop/{arm}/reason", record.get("reason") == "prespecified mechanism gate failed")
                    reader.require(f"mechanism/{tag}/prespecified_stop/{arm}/code", record.get("code_manifest_sha256") == manifest_digest)
        data, completions = {}, {}
        expected_names = {f"{arm}_{stage:011d}.npz" for arm in arms for stage in STAGES}
        existing = {path.name for path in (reader.out / tag / "confirm").glob("*.npz")}
        reader.require(f"window/{tag}/exact_stage_arm_file_set", existing == expected_names,
                       {"missing": sorted(expected_names - existing), "unexpected": sorted(existing - expected_names)})
        for arm in arms:
            completion = check_completion(reader, tag, arm, settings, input_records, source_records, manifest_digest)
            if completion is not None:
                completions[arm] = completion
            metrics = {task: {metric: [] for metric in SCALARS} for task in TASKS}
            complete = True
            for stage in STAGES:
                path = reader.out / tag / "confirm" / f"{arm}_{stage:011d}.npz"
                if reader.file(f"window/{tag}/{arm}/{stage}", path) is None:
                    complete = False
                    continue
                with np.load(path, allow_pickle=False) as raw:
                    for task in TASKS:
                        if task not in labels:
                            complete = False
                            continue
                        rows = verify_metric_group(reader, raw, task, labels[task], settings["n"], f"raw/{tag}/{arm}/{stage}/{task}")
                        for metric, value in rows.items():
                            metrics[task][metric].append(value)
            inspect_late(reader, tag, arm, settings["end_stage"], settings)
            if complete and completion is not None:
                # Separate summation implementation, equal weight for the exact
                # 11 fixed checkpoints within each immutable evaluation row.
                data[arm] = {task: {metric: np.sum(np.asarray(rows), axis=0, dtype=np.float64) / len(STAGES)
                                   for metric, rows in item.items()} for task, item in metrics.items()}
        all_data[tag], receipt[tag] = data, {"mechanism_passed": eligible, "completions": completions}
    return all_data, receipt


def independently_bootstrap(data: dict[str, Any], query: dict[str, Any], settings: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    """Independent reconstruction of the sealed finite-grid bootstrap receipt."""
    if settings.get("analysis_scope") == "single_seed_pilot":
        return {"status": "not_planned_single_rho_pilot", "confirmed": False, "budgets": {},
                "reason": "Only rho=.25 is authorized; no strength-grid frontier or seed/class bootstrap is estimated"}
    draws, random_seed = settings.get("frontier_bootstrap_draws"), settings.get("frontier_bootstrap_seed")
    if not isinstance(draws, int) or draws < 999 or not isinstance(random_seed, int):
        return {"status": "not_sealed_or_not_enabled", "confirmed": False}
    seeds = [entry["id"] for entry in settings["training_seeds"]]
    if not all(seed in data for seed in seeds):
        return {"status": "incomplete_training_seeds", "confirmed": False}
    arms = [arm for arm in settings["arms"] if all(arm in data[seed] for seed in seeds)]
    if any(arm not in arms for group in GRID.values() for arm in group):
        return {"status": "missing_or_mechanism_ineligible_frontier_arms", "confirmed": False}
    classes = np.unique(np.concatenate([query[task] for task in TASKS]))
    totals, counts = {}, {}
    for task in TASKS:
        membership = np.searchsorted(classes, query[task])
        counts[task] = np.bincount(membership, minlength=classes.size)
        totals[task] = np.zeros((3, len(arms), 2, classes.size), np.float64)
        for si, seed in enumerate(seeds):
            for ai, arm in enumerate(arms):
                for mi, metric in enumerate(("acc", "loss")):
                    np.add.at(totals[task][si, ai, mi], membership, data[seed][arm][task][metric])
    rng = np.random.Generator(np.random.PCG64(random_seed))
    results = {str(budget): {ref: [] for ref in ("C", "R")} for budget in settings["frontier_icl_tolerances"]}
    choices = {str(budget): {family: {arm: 0 for arm in GRID[family]} | {"none": 0} for family in GRID}
               for budget in settings["frontier_icl_tolerances"]}
    for _ in range(draws):
        chosen_seeds = rng.integers(0, 3, size=3)
        replicate = {task: np.empty((3, len(arms), 2), np.float64) for task in TASKS}
        for draw_slot, si in enumerate(chosen_seeds):
            class_count = np.bincount(rng.integers(0, classes.size, size=classes.size), minlength=classes.size)
            for task in TASKS:
                denominator = float(np.sum(counts[task] * class_count))
                if denominator == 0:
                    raise ValueError("No sampled class rows; no silent bootstrap redraw permitted")
                replicate[task][draw_slot] = np.sum(totals[task][si] * class_count[None, None, :], axis=-1) / denominator
        means = {arm: {task: {metric: float(np.sum(replicate[task][:, ai, mi]) / 3) for mi, metric in enumerate(("acc", "loss"))}
                       for task in TASKS} for ai, arm in enumerate(arms)}
        arm_index = {arm: ai for ai, arm in enumerate(arms)}
        def difference(a: str, b: str, task: str, metric: str) -> dict[str, Any]:
            mi = 0 if metric == "acc" else 1
            return repeated_run_stat(replicate[task][:, arm_index[a], mi] - replicate[task][:, arm_index[b], mi])
        front = independently_select_frontier(means, difference, settings)
        for budget, groups in front["budgets"].items():
            for family, point in groups.items():
                choices[budget][family][point["selected_arm"] or "none"] += 1
            for reference in ("C", "R"):
                a, b = groups["D"]["ciwl_acc"], groups[reference]["ciwl_acc"]
                results[budget][reference].append(a - b if a is not None and b is not None else None)
    tests, comparisons = [], {}
    for budget, pairs in results.items():
        details = {}
        for reference, values in pairs.items():
            point = observed["budgets"][budget]
            a, b = point["D"]["ciwl_acc"], point[reference]["ciwl_acc"]
            effect = a - b if a is not None and b is not None else None
            distribution = sorted(-math.inf if value is None else value for value in values)
            low_slot = min(draws - 1, max(0, math.floor(settings["alpha"] * (draws - 1))))
            high_slot = min(draws - 1, max(0, math.ceil((1 - settings["alpha"]) * (draws - 1))))
            low, high = distribution[low_slot], distribution[high_slot]
            details[f"D-{reference}"] = {"mean": effect, "lower_one_sided_95": low if math.isfinite(low) else None,
                                        "upper_percentile_95": high if math.isfinite(high) else None,
                                        "p_raw": (1 + sum(value <= 0 for value in distribution)) / (draws + 1),
                                        "undefined_draws": sum(value is None for value in values),
                                        "defined_draws": sum(value is not None for value in values),
                                        "valid": effect is not None and math.isfinite(low)}
        comparisons[budget] = details
        effects = [value["mean"] for value in details.values()]
        tests.append({"name": f"frontier_D_exceeds_C_and_R/budget_{budget}", "threshold": 0.,
                      "mean": min(effects) if all(value is not None for value in effects) else None,
                      "p_raw": max(value["p_raw"] for value in details.values()),
                      "valid": all(value["valid"] and value["lower_one_sided_95"] > 0 for value in details.values())})
    return {"status": "executed", "draws": draws, "seed": random_seed, "comparisons": comparisons,
            "selection_counts": choices, "increment_family_Holm3": adjust_holm(tests, settings["alpha"])}

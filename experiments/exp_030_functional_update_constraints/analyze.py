"""Frozen-window analysis for exp030; never runs a model or samples science data.

The 11 checkpoints are averaged within each evaluation row. Class-cluster
uncertainty is conditional on a training seed. The single-seed pilot retains
the seven-test family but does not perform between-run inference, a strength
grid, or frontier bootstrap. Missing or failed mechanism comparisons stay in
that family. The independent reader must validate the resulting report before
a scientific support claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

TASKS = ("train_eval", "iwl_copy_avail", "flip_icl", "icl")
METRICS = ("acc", "loss", "in_context_acc", "pair_loss", "pair_margin",
           "context_mass", "log_context_mass")
PRIMARY = (("C25", "A", "icl", "min_icl_gain"),
           ("C25", "A", "flip_icl", "min_icl_gain"),
           ("C25", "B", "iwl_copy_avail", "min_icl_gain"),
           ("D25", "C25", "icl", "incremental_icl_gain"),
           ("D25", "C25", "flip_icl", "incremental_icl_gain"),
           ("D25", "R25", "icl", "incremental_icl_gain"),
           ("D25", "R25", "flip_icl", "incremental_icl_gain"))
NI_COMPARISONS = (("C25", "A", ("iwl_copy_avail", "train_eval")),
                  ("C25", "B", ("icl", "flip_icl")),
                  ("D25", "C25", ("iwl_copy_avail", "train_eval")),
                  ("D25", "R25", ("iwl_copy_avail", "train_eval")))
FAMILIES = {name: tuple(f"{name}{rho}" for rho in ("00", "25", "75"))
            for name in ("C", "D", "R")}
SQRT_TWO = math.sqrt(2.0)


def single_seed_pilot(settings: dict[str, Any]) -> bool:
    return settings.get("analysis_scope") == "single_seed_pilot"


def validate_design(settings: dict[str, Any]) -> None:
    if single_seed_pilot(settings):
        if (settings["training_seeds"] != [{"id": "s5", "init": 5, "train": 2}]
                or settings["arms"] != ["A", "B", "C25", "D25", "R25"]
                or settings["relaxation"] != {"C25": .25, "D25": .25, "R25": .25}
                or settings["primary_relaxation"] != .25
                or settings.get("require_seed_aggregate_holm") is not False):
            raise ValueError("Expected the authorized single s5, five-arm, rho=.25 pilot")
    elif settings.get("require_seed_aggregate_holm") is not True or len(settings["training_seeds"]) != 3:
        raise ValueError("The three-run design must require seed aggregate Holm7")
    if settings["fresh_tasks"] != list(TASKS):
        raise ValueError("Expected all four frozen evaluation tasks")
    if (settings["start_stage"] != 4400000 or settings["end_stage"] != 10000000
            or settings["confirm_stages"] != list(range(9000000, 10000001, 100000))
            or settings["late_window"] != [9000000, 10000000] or settings["n"] != 5000):
        raise ValueError("Training endpoints or confirmatory count/window changed from the frozen protocol")


def unplanned_pilot_frontier() -> dict[str, Any]:
    return {"status": "not_planned_single_rho_pilot", "confirmed": False, "budgets": {},
            "reason": "Only rho=.25 is authorized; no strength-grid frontier or seed/class bootstrap is estimated"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normal_ppf(p: float) -> float:
    """Deterministic normal inverse CDF; no optional library changes the test."""
    if not 0 < p < 1:
        raise ValueError("normal quantile requires 0 < p < 1")
    low, high = -12.0, 12.0
    for _ in range(80):
        mid = (low + high) / 2
        if .5 * math.erfc(-mid / SQRT_TWO) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def t2_ppf(p: float) -> float:
    """Exact inverse of Student t with df=2, also usable without SciPy."""
    if not 0 < p < 1:
        raise ValueError("t quantile requires 0 < p < 1")
    v = 2 * p - 1
    return math.sqrt(2.0) * v / math.sqrt(1 - v * v)


def t2_sf(t: float) -> float:
    # Use the rationalized positive-tail form to avoid cancellation.
    if t >= 0:
        root = math.sqrt(t * t + 2)
        return 1 / (root * (root + t))
    return 1 - t2_sf(-t)


def class_stats(values: Any, query_classes: Any, alpha: float = .05) -> dict[str, Any]:
    values, query_classes = np.asarray(values, np.float64), np.asarray(query_classes)
    if values.ndim != 1 or values.size == 0 or query_classes.shape != values.shape:
        raise ValueError("Expected paired nonempty rows and true query classes")
    if not np.isfinite(values).all() or not np.issubdtype(query_classes.dtype, np.integer):
        raise ValueError("Nonfinite values or noninteger query-class identities")
    classes, inverse, counts = np.unique(query_classes, return_inverse=True, return_counts=True)
    mean = float(values.mean(dtype=np.float64))
    scores = np.bincount(inverse, weights=values - mean, minlength=classes.size)
    variance = (float(classes.size / (classes.size - 1) * np.dot(scores, scores) / values.size**2)
                if classes.size > 1 else None)
    se = math.sqrt(max(0, variance)) if variance is not None else None
    valid = bool(se is not None and se > 1e-15)
    q_one, q_two = normal_ppf(1 - alpha), normal_ppf(1 - alpha / 2)
    return {"mean": mean, "se": se, "n": int(values.size), "clusters": int(classes.size),
            "valid": valid, "lower_one_sided_95": mean - q_one * se if valid else None,
            "upper_one_sided_95": mean + q_one * se if valid else None,
            "ci95_lower": mean - q_two * se if valid else None,
            "ci95_upper": mean + q_two * se if valid else None,
            "status": "normal_query_class_cluster" if valid else "degenerate_no_confirmatory_support",
            "estimand": "evaluation_row_weighted_mean", "cluster_unit": "true_query_class"}


def seed_stats(values: Any, alpha: float = .05) -> dict[str, Any]:
    values = np.asarray(values, np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        return {"mean": None, "se": None, "valid": False, "n_training_seeds": int(values.size),
                "status": "requires_exactly_three_complete_training_seeds", "df": 2,
                "lower_one_sided_95": None, "upper_one_sided_95": None,
                "ci95_lower": None, "ci95_upper": None}
    mean, se = float(values.mean()), float(values.std(ddof=1) / math.sqrt(3))
    valid = se > 1e-15
    one, two = t2_ppf(1 - alpha), t2_ppf(1 - alpha / 2)
    return {"mean": mean, "se": se, "valid": valid, "n_training_seeds": 3, "df": 2,
            "seed_means": values.tolist(), "lower_one_sided_95": mean - one * se if valid else None,
            "upper_one_sided_95": mean + one * se if valid else None,
            "ci95_lower": mean - two * se if valid else None,
            "ci95_upper": mean + two * se if valid else None,
            "status": "student_t_exact_df2" if valid else "degenerate_seed_se_no_confirmatory_support",
            "estimand": "equally_weighted_mean_over_three_independent_training_seeds",
            "limitation": "Three runs give limited precision and do not establish broad seed robustness"}


def greater(stat: dict[str, Any], threshold: float, name: str, kind: str = "normal") -> dict[str, Any]:
    valid = stat["valid"]
    z = (stat["mean"] - threshold) / stat["se"] if valid else None
    p = (t2_sf(z) if kind == "t2" else .5 * math.erfc(z / SQRT_TWO)) if valid else 1.0
    return dict(stat, name=name, threshold=float(threshold), alternative="mean > threshold",
                z_or_t=z, p_raw=float(p), p_underflow=bool(valid and p == 0))


def missing_test(name: str, threshold: float) -> dict[str, Any]:
    return {"name": name, "threshold": float(threshold), "alternative": "mean > threshold",
            "mean": None, "se": None, "valid": False, "p_raw": 1.0, "z_or_t": None,
            "p_underflow": False, "status": "missing_prespecified_arm_no_support"}


def holm(tests: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    tests = [dict(item) for item in tests]
    if not tests:
        raise ValueError("An empty family cannot establish multiplicity control")
    running = 0.0
    for rank, slot in enumerate(sorted(range(len(tests)), key=lambda i: (tests[i]["p_raw"], i))):
        running = max(running, min(1., (len(tests) - rank) * tests[slot]["p_raw"]))
        tests[slot]["p_holm"] = float(running)
    for test in tests:
        test["passed"] = bool(test["valid"] and test["p_holm"] < alpha
                              and test["mean"] > test["threshold"])
        test["family_size"] = len(tests)
    return {"tests": tests, "alpha": float(alpha), "family_size": len(tests),
            "all_passed": all(item["passed"] for item in tests), "correction": "Holm",
            "interval_note": "Reported 95% intervals are ordinary, not Holm simultaneous intervals"}


def ni(stat: dict[str, Any], metric: str, settings: dict[str, Any]) -> dict[str, Any]:
    accuracy = metric == "acc"
    threshold = -float(settings["ni_acc_tolerance"]) if accuracy else float(settings["ni_ce_tolerance"])
    bound = stat["lower_one_sided_95"] if accuracy else stat["upper_one_sided_95"]
    passed = bool(bound is not None and (bound > threshold if accuracy else bound < threshold))
    return dict(stat, threshold=threshold, metric=metric, passed=passed,
                alternative="mean > -acc_tolerance" if accuracy else "mean < ce_tolerance",
                multiplicity="intersection_union_all_required_gates")


def primary_family(data: dict[str, Any], classes: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    tests = []
    for arm, reference, task, key in PRIMARY:
        name = f"{arm}-{reference}/{task}/acc"
        if arm not in data or reference not in data:
            tests.append(missing_test(name, settings[key]))
        else:
            stat = class_stats(data[arm][task]["acc"] - data[reference][task]["acc"],
                               classes[task], settings["alpha"])
            tests.append(greater(stat, settings[key], name))
    return holm(tests, settings["alpha"])


def noninferiority(data: dict[str, Any], classes: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    results = {}
    for arm, reference, tasks in NI_COMPARISONS:
        tests = {}
        for task in tasks:
            for metric in ("acc", "loss"):
                if arm in data and reference in data:
                    stat = class_stats(data[arm][task][metric] - data[reference][task][metric],
                                       classes[task], settings["alpha"])
                    tests[f"{task}/{metric}"] = ni(stat, metric, settings)
                else:
                    tests[f"{task}/{metric}"] = {"passed": False, "status": "missing_prespecified_arm_no_support"}
        results[f"{arm}-{reference}"] = {"tests": tests, "all_passed": all(t["passed"] for t in tests.values())}
        if (arm, reference) == ("C25", "B"):
            results["C25-B"].update(required_for_primary_confirmation=False,
                role="descriptive_retention_comparison_with_freeze")
    return results


def d_r_matching(completions: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    left = completions.get("D25", {}).get("update_statistics", {})
    right = completions.get("R25", {}).get("update_statistics", {})
    tests = {}
    for name, tolerance in (("mean_update_norm_ratio", settings["primary_d_r_update_ratio_tolerance"]),
                            ("mean_correction_norm_ratio", settings["primary_d_r_update_ratio_tolerance"]),
                            ("conflict_rate", settings["primary_d_r_conflict_rate_tolerance"])):
        a, b = left.get(name), right.get(name)
        valid = (isinstance(a, (float, int)) and isinstance(b, (float, int))
                 and math.isfinite(float(a)) and math.isfinite(float(b)) and a >= 0 and b >= 0)
        if name == "conflict_rate":
            valid = valid and a <= 1 and b <= 1
        difference = abs(float(a) - float(b)) if valid else None
        tests[name] = {"D25": a, "R25": b, "absolute_difference": difference,
                       "tolerance": float(tolerance), "passed": bool(valid and difference <= tolerance)}
    return {"tests": tests, "passed": all(test["passed"] for test in tests.values()),
            "note": "Norm/conflict matching is necessary but cannot exclude every mechanical-strength difference"}


def joint_methods(primary: dict[str, Any], gates: dict[str, Any], mechanism_passed: bool,
                  matching: dict[str, Any]) -> dict[str, Any]:
    names = {test["name"]: test["passed"] for test in primary["tests"]}
    c_gain = all(names.get(f"C25-{ref}/{task}/acc", False)
                 for ref, task in (("A", "icl"), ("A", "flip_icl"), ("B", "iwl_copy_avail")))
    c_ni = gates["C25-A"]["all_passed"]
    d_gain = all(names.get(f"D25-{ref}/{task}/acc", False)
                 for ref in ("C25", "R25") for task in ("icl", "flip_icl"))
    d_ni = gates["D25-C25"]["all_passed"] and gates["D25-R25"]["all_passed"]
    return {"ordinary_task_protection": {"gains_passed": c_gain, "noninferiority_passed": c_ni,
                                          "confirmation_target": "relative_A_tradeoff_improvement_plus_CIWL_learning_gain_over_B",
                                          "B_retention_noninferiority_required": False,
                                          "B_retention_noninferiority_descriptive_passed": gates["C25-B"]["all_passed"],
                                          "statistical_supported": bool(c_gain and c_ni)},
            "causal_path_increment": {"gains_passed": d_gain, "noninferiority_passed": d_ni,
                                      "mechanism_passed": bool(mechanism_passed),
                                      "d_r_strength_matching_passed": matching["passed"],
                                      "statistical_supported": bool(d_gain and d_ni and mechanism_passed and matching["passed"])}}


def descriptive(data: dict[str, Any], classes: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    return {arm: {task: {metric: class_stats(values, classes[task], settings["alpha"])
                         for metric, values in metrics.items()} for task, metrics in tasks.items()}
            for arm, tasks in data.items()}


def descriptive_comparisons(data: dict[str, Any], classes: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    # Prespecified contextual comparisons; no extra discoveries are promoted to main claims.
    comparisons = {}
    for arm, reference in (("C25", "A"), ("C25", "B"), ("D25", "A"), ("D25", "B"),
                           ("D25", "C25"), ("D25", "R25")):
        if arm in data and reference in data:
            comparisons[f"{arm}-{reference}"] = {
                task: {metric: class_stats(data[arm][task][metric] - data[reference][task][metric],
                                            classes[task], settings["alpha"])
                       for metric in METRICS} for task in TASKS}
    return comparisons


def frontier_from_arm_stats(arm_means: dict[str, Any], difference_stats: Any,
                            settings: dict[str, Any]) -> dict[str, Any]:
    """Same finite-grid selector for observed and hierarchical bootstrap data.

    difference_stats(arm, reference, task, metric) returns the appropriate
    class-cluster or seed-t statistic. All eligibility gates use one-sided
    bounds; ties are broken in the frozen order 00,25,75, never by later data.
    """
    if single_seed_pilot(settings):
        return unplanned_pilot_frontier()
    budgets = {}
    for budget in settings["frontier_icl_tolerances"]:
        families = {}
        for family, arms in FAMILIES.items():
            candidates = []
            for arm in arms:
                if arm not in arm_means or "A" not in arm_means or "B" not in arm_means:
                    candidates.append({"arm": arm, "feasible": False, "status": "missing_arm"})
                    continue
                gates = {}
                for task in ("icl", "flip_icl"):
                    stat = difference_stats(arm, "B", task, "acc")
                    bound = stat["lower_one_sided_95"]
                    gates[task + "/budget_acc"] = {"lower_one_sided_95": bound, "threshold": -float(budget),
                                                   "passed": bool(bound is not None and bound > -budget)}
                for metric in ("acc", "loss"):
                    gates["train_eval/" + metric] = ni(difference_stats(arm, "A", "train_eval", metric), metric, settings)
                feasible = all(gate["passed"] for gate in gates.values())
                candidates.append({"arm": arm, "feasible": feasible, "gates": gates,
                                   "ciwl_acc": float(arm_means[arm]["iwl_copy_avail"]["acc"]),
                                   "ciwl_ce": float(arm_means[arm]["iwl_copy_avail"]["loss"])})
            available = [item for item in candidates if item["feasible"]]
            best = max(available, key=lambda item: item["ciwl_acc"]) if available else None
            families[family] = {"candidates": candidates, "selected_arm": best["arm"] if best else None,
                                "ciwl_acc": best["ciwl_acc"] if best else None,
                                "ciwl_ce": best["ciwl_ce"] if best else None,
                                "status": "finite_three_point_frontier" if best else "no_feasible_prespecified_point"}
        budgets[str(budget)] = families
    return {"budgets": budgets, "selection": "highest_full_CIWL_acc_among_three_prespecified_rho_points",
            "tie_break": "00_then_25_then_75", "eligibility": "ICL_and_Flip_acc_vs_B_one_sided_budget_and_train_acc_CE_NI_vs_A",
            "scope": "finite_grid_only_not_a_global_or_continuous_optimum"}


def per_seed_frontier(data: dict[str, Any], classes: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    means = {arm: {task: {metric: float(value.mean()) for metric, value in item.items()}
                   for task, item in tasks.items()} for arm, tasks in data.items()}
    return frontier_from_arm_stats(means,
        lambda a, b, task, metric: class_stats(data[a][task][metric] - data[b][task][metric], classes[task], settings["alpha"]), settings)


def aggregate_primary(seed_data: dict[str, Any], seed_reports: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    seeds = [item["id"] for item in settings["training_seeds"]]
    if single_seed_pilot(settings):
        tag = seeds[0]
        available = set(seed_data) == {tag} and set(seed_reports) == {tag}
        methods = {name: {"pilot_statistical_supported": bool(available and seed_reports[tag]["methods"][name]["statistical_supported"]),
                          "confirmation_statistical_supported": False,
                          "cross_seed_statistical_supported": False,
                          "independent_reader_required": True}
                   for name in ("ordinary_task_protection", "causal_path_increment")}
        return {"status": "not_performed_single_seed_pilot", "primary_Holm7": None,
                "methods": methods, "require_seed_aggregate_holm": False,
                "n_training_seeds": 1, "pilot_seed": tag,
                "unit": "one_fixed_training_trajectory",
                "note": "Pilot support is conditional on s5 class-cluster Holm7 and joint gates; no between-seed confirmation"}
    tests = []
    for arm, reference, task, key in PRIMARY:
        name = f"{arm}-{reference}/{task}/acc"
        if all(seed in seed_data and arm in seed_data[seed] and reference in seed_data[seed] for seed in seeds):
            values = [float(np.mean(seed_data[seed][arm][task]["acc"] - seed_data[seed][reference][task]["acc"])) for seed in seeds]
            tests.append(greater(seed_stats(values, settings["alpha"]), settings[key], name, "t2"))
        else:
            tests.append(missing_test(name, settings[key]))
    family = holm(tests, settings["alpha"])
    names = {test["name"]: test for test in family["tests"]}
    methods = {}
    for method, relevant in (("ordinary_task_protection", range(3)), ("causal_path_increment", range(3, 7))):
        all_seed_pass = all(seed in seed_reports and seed_reports[seed]["methods"][method]["statistical_supported"] for seed in seeds)
        selected = [family["tests"][i] for i in relevant]
        lower_pass = all(item.get("lower_one_sided_95") is not None and item["lower_one_sided_95"] > item["threshold"] for item in selected)
        holm_pass = all(item["passed"] for item in selected)
        methods[method] = {"all_three_training_seed_joint_gates_passed": all_seed_pass,
                           "seed_mean_one_sided_lower_bounds_passed": lower_pass,
                           "seed_mean_Holm7_passed": holm_pass,
                           "confirmation_statistical_supported": bool(all_seed_pass and lower_pass and holm_pass),
                           "independent_reader_required": True}
        if method == "ordinary_task_protection":
            methods[method].update(
                confirmation_target="relative_A_tradeoff_improvement_plus_CIWL_learning_gain_over_B",
                B_retention_noninferiority_required=False)
    return {"primary_Holm7": family, "methods": methods,
            "require_seed_aggregate_holm": True, "unit": "independent_training_run",
            "note": "Evaluation rows, query classes and window checkpoints do not increase the training-run count"}


def aggregate_frontier(seed_data: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    if single_seed_pilot(settings):
        return unplanned_pilot_frontier()
    seeds = [item["id"] for item in settings["training_seeds"]]
    arms = [arm for arm in settings["arms"] if all(seed in seed_data and arm in seed_data[seed] for seed in seeds)]
    means = {arm: {task: {metric: float(np.mean([seed_data[seed][arm][task][metric].mean() for seed in seeds]))
                         for metric in ("acc", "loss")} for task in TASKS} for arm in arms}
    def differences(arm: str, reference: str, task: str, metric: str) -> dict[str, Any]:
        return seed_stats([float(np.mean(seed_data[seed][arm][task][metric] - seed_data[seed][reference][task][metric]))
                           for seed in seeds], settings["alpha"])
    return frontier_from_arm_stats(means, differences, settings)


def bootstrap_frontier(seed_data: dict[str, Any], classes: dict[str, Any], settings: dict[str, Any],
                       observed: dict[str, Any]) -> dict[str, Any]:
    """Prespecified paired seed/class hierarchical bootstrap, with reselection.

    A class draw is shared by every arm and task within one selected training
    run. Sampling class blocks keeps the row-weighted ratio estimand. Undefined
    frontiers remain undefined and count against support rather than being
    discarded. This routine belongs in the authorized full cloud analysis.
    """
    if single_seed_pilot(settings):
        return unplanned_pilot_frontier()
    draws = settings.get("frontier_bootstrap_draws")
    random_seed = settings.get("frontier_bootstrap_seed")
    if not isinstance(draws, int) or draws < 999 or not isinstance(random_seed, int):
        return {"status": "not_sealed_or_not_enabled", "confirmed": False,
                "reason": "Fixed bootstrap draws/seed must be sealed before first scientific result"}
    seeds = [item["id"] for item in settings["training_seeds"]]
    if not all(seed in seed_data for seed in seeds):
        return {"status": "incomplete_training_seeds", "confirmed": False}
    arms = [arm for arm in settings["arms"] if all(arm in seed_data[seed] for seed in seeds)]
    if not all(arm in arms for group in FAMILIES.values() for arm in group):
        return {"status": "missing_or_mechanism_ineligible_frontier_arms", "confirmed": False}
    union_classes = np.unique(np.concatenate([classes[task] for task in TASKS]))
    block_counts, block_totals = {}, {}
    for task in TASKS:
        block = np.searchsorted(union_classes, classes[task])
        block_counts[task] = np.bincount(block, minlength=union_classes.size).astype(np.float64)
        block_totals[task] = np.stack([
            np.stack([np.bincount(block, weights=seed_data[seed][arm][task][metric], minlength=union_classes.size)
                      for metric in ("acc", "loss")]) for seed in seeds for arm in arms
        ]).reshape(3, len(arms), 2, union_classes.size)
    rng = np.random.Generator(np.random.PCG64(random_seed))
    distributions = {str(budget): {ref: [] for ref in ("C", "R")} for budget in settings["frontier_icl_tolerances"]}
    selected_counts = {str(budget): {family: {arm: 0 for arm in FAMILIES[family]} | {"none": 0}
                                    for family in FAMILIES} for budget in settings["frontier_icl_tolerances"]}
    for _ in range(draws):
        seed_sample = rng.integers(0, 3, size=3)
        sampled = {arm: {task: {} for task in TASKS} for arm in arms}
        task_values = {task: np.empty((3, len(arms), 2), np.float64) for task in TASKS}
        for slot, seed_index in enumerate(seed_sample):
            counts = np.bincount(rng.integers(0, union_classes.size, size=union_classes.size), minlength=union_classes.size)
            for task in TASKS:
                denominator = float(np.dot(block_counts[task], counts))
                if denominator == 0:
                    raise ValueError("Bootstrap sampled no rows; preserve failure instead of silently redrawing")
                task_values[task][slot] = np.einsum("amk,k->am", block_totals[task][seed_index], counts) / denominator
        for i, arm in enumerate(arms):
            for task in TASKS:
                for metric_slot, metric in enumerate(("acc", "loss")):
                    sampled[arm][task][metric] = task_values[task][:, i, metric_slot]
        means = {arm: {task: {metric: float(values.mean()) for metric, values in metrics.items()}
                       for task, metrics in tasks.items()} for arm, tasks in sampled.items()}
        front = frontier_from_arm_stats(means,
            lambda a, b, task, metric: seed_stats(sampled[a][task][metric] - sampled[b][task][metric], settings["alpha"]), settings)
        for budget, families in front["budgets"].items():
            for family, item in families.items():
                selected_counts[budget][family][item["selected_arm"] or "none"] += 1
            for reference in ("C", "R"):
                a, b = families["D"]["ciwl_acc"], families[reference]["ciwl_acc"]
                distributions[budget][reference].append(float(a - b) if a is not None and b is not None else None)
    tests, detailed = [], {}
    alpha = float(settings["alpha"])
    for budget, comparisons in distributions.items():
        details = {}
        for reference, values in comparisons.items():
            observed_families = observed["budgets"][budget]
            a, b = observed_families["D"]["ciwl_acc"], observed_families[reference]["ciwl_acc"]
            effect = float(a - b) if a is not None and b is not None else None
            # Undefined draws are a lower-tail sentinel and are never dropped.
            conservative = np.asarray([-np.inf if value is None else value for value in values])
            order = np.sort(conservative)
            lower_slot = min(draws - 1, max(0, int(math.floor(alpha * (draws - 1)))))
            upper_slot = min(draws - 1, max(0, int(math.ceil((1 - alpha) * (draws - 1)))))
            lower = float(order[lower_slot]) if np.isfinite(order[lower_slot]) else None
            upper = float(order[upper_slot]) if np.isfinite(order[upper_slot]) else None
            p = float((1 + np.sum(conservative <= 0)) / (draws + 1))
            details[f"D-{reference}"] = {"mean": effect, "lower_one_sided_95": lower,
                                        "upper_percentile_95": upper, "p_raw": p,
                                        "undefined_draws": sum(value is None for value in values),
                                        "defined_draws": sum(value is not None for value in values),
                                        "valid": effect is not None and lower is not None,
                                        "note": "Percentile bootstrap tail diagnostic; three-seed uncertainty remains limited"}
        detailed[budget] = details
        effects = [item["mean"] for item in details.values()]
        tests.append({"name": f"frontier_D_exceeds_C_and_R/budget_{budget}",
                      "mean": min(effects) if all(v is not None for v in effects) else None,
                      "threshold": 0.0, "p_raw": max(item["p_raw"] for item in details.values()),
                      "valid": all(item["valid"] and item["lower_one_sided_95"] > 0 for item in details.values()),
                      "alternative": "D CIWL frontier gain > 0 against both C and R"})
    return {"status": "executed", "draws": draws, "seed": random_seed,
            "method": "paired_hierarchical_training_seed_then_true_query_class_blocks",
            "class_pairing": "same_class_multiplicity_for_all_arms_and_tasks_in_a_selected_run",
            "reselect_every_draw": True, "undefined_draws": "conservatively_counted_against_support_not_omitted",
            "comparisons": detailed, "selection_counts": selected_counts,
            "increment_family_Holm3": holm(tests, alpha),
            "scope": "finite_grid_seed_class_bootstrap_not_proof_over_all_training_seeds"}


def plot_tradeoffs(work: Path, seed_data: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Standalone scientific figures after real data are complete, never at import."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "pending_missing_matplotlib", "paths": [],
                "note": "Numeric finite-grid comparisons remain available; no package is installed automatically"}
    means = {seed: {arm: {task: {metric: float(values.mean()) for metric, values in item.items()}
                          for task, item in tasks.items()} for arm, tasks in data.items()}
             for seed, data in seed_data.items()}
    seeds = [item["id"] for item in settings["training_seeds"]]
    common = [arm for arm in settings["arms"] if all(seed in means and arm in means[seed] for seed in seeds)]
    pilot = single_seed_pilot(settings)
    if not pilot:
        means["three_training_seeds"] = {arm: {task: {metric: float(np.mean([means[seed][arm][task][metric] for seed in seeds]))
                                                  for metric in METRICS} for task in TASKS} for arm in common}
    directory = work / "outputs" / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    colors = {"A": "#222222", "B": "#855b13", "C": "#2171b5", "D": "#b2182b", "R": "#666666"}
    for seed, arms in means.items():
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.4), constrained_layout=True)
        for ax, (task, metric) in zip(axes.ravel(), (("icl", "acc"), ("flip_icl", "acc"),
                                                 ("icl", "loss"), ("flip_icl", "loss"))):
            for family in ("A", "B", "C", "D", "R"):
                selected = (family,) if family in ("A", "B") else FAMILIES[family]
                points = [arm for arm in selected if arm in arms]
                if not points:
                    continue
                x = [arms[arm][task][metric] for arm in points]
                y = [arms[arm]["iwl_copy_avail"][metric] for arm in points]
                ax.plot(x, y, marker="s" if family == "B" else "o", color=colors[family],
                        linestyle="-" if len(points) > 1 else "none", label=family, linewidth=1.4)
                for arm, px, py in zip(points, x, y):
                    ax.annotate(arm, (px, py), xytext=(4, 4), textcoords="offset points", fontsize=8)
            xlabel = "ICL" if task == "icl" else "Flip ICL"
            suffix = "full 12,800-label accuracy" if metric == "acc" else "full 12,800-label CE (nats)"
            ax.set_xlabel(xlabel + " " + suffix)
            ax.set_ylabel("CIWL " + suffix)
            if metric == "acc":
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, loc="best")
        title_note = ("Single s5 pilot, rho=.25; five observed arm points, no frontier estimate" if pilot else
                      "Three fixed strengths; connecting lines guide the eye and do not claim interpolation")
        fig.suptitle(f"exp030 | {seed} | fixed 9.0-10.0M window\n" + title_note, fontsize=12)
        for extension in ("png", "svg"):
            path = directory / f"tradeoffs_{seed}.{extension}"
            fig.savefig(path, dpi=160)
            paths.append(str(path.relative_to(work)))
        plt.close(fig)
    return {"status": "written", "paths": paths,
            "reduction": ("same-row eleven-checkpoint means within the single s5 trajectory" if pilot else
                          "same-row eleven-checkpoint means; aggregate uses equally weighted three training-seed means"),
            "scope": ("single_rho_single_seed_descriptive_points_no_frontier_estimate" if pilot else
                      "finite_grid_descriptive_plots_not_a_global_optimal_frontier")}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_window(work: Path, settings: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    import h5py
    out, issues = work / "outputs", []
    classes = {}
    data_path = out / "data" / "confirm.h5"
    if not data_path.is_file():
        return {}, {}, {}, [{"status": "missing_confirm_data", "path": str(data_path)}]
    with h5py.File(data_path, "r") as source:
        for task in TASKS:
            identities = np.asarray(source[f"{task}/identities"])
            if identities.shape != (settings["n"], 3, 2) or not np.issubdtype(identities.dtype, np.integer):
                raise ValueError(f"Malformed true class/exemplar identity: {task}")
            classes[task] = identities[:, -1, 0]
    all_data, receipts = {}, {}
    for seed in settings["training_seeds"]:
        tag = seed["id"]
        gate_path = out / tag / "mechanism_confirmation.json"
        mechanism = read_json(gate_path) if gate_path.is_file() else {"passed": False, "status": "missing_confirmation"}
        if not isinstance(mechanism.get("passed"), bool):
            raise ValueError("Mechanism eligibility must be an explicit Boolean")
        # Failed functional gates do not remove the training run from main inference.
        wanted = [arm for arm in settings["arms"] if not arm.startswith(("D", "R")) or mechanism["passed"]]
        data, completions = {}, {}
        for arm in wanted:
            files = [out / tag / "confirm" / f"{arm}_{stage:011d}.npz" for stage in settings["confirm_stages"]]
            completion_path = out / tag / "training_complete" / f"{arm}.json"
            missing = [str(path) for path in files if not path.is_file()]
            if not completion_path.is_file():
                missing.append(str(completion_path))
            if missing:
                issues.append({"seed": tag, "arm": arm, "status": "incomplete_window_or_training", "missing": missing})
                continue
            completion = read_json(completion_path)
            if completion.get("passed") is not True or completion.get("end_stage") != settings["end_stage"]:
                issues.append({"seed": tag, "arm": arm, "status": "training_completion_not_passed_at_fixed_endpoint"})
                continue
            completions[arm] = completion
            arrays = {task: {metric: [] for metric in METRICS} for task in TASKS}
            for path in files:
                with np.load(path, allow_pickle=False) as raw:
                    for task in TASKS:
                        for metric in METRICS:
                            value = np.asarray(raw[f"{task}/{metric}"], np.float64)
                            if value.shape != (settings["n"],) or not np.isfinite(value).all():
                                raise ValueError(f"Bad frozen-window array {path}: {task}/{metric}")
                            arrays[task][metric].append(value)
            data[arm] = {task: {metric: np.mean(np.stack(rows), axis=0, dtype=np.float64)
                               for metric, rows in item.items()} for task, item in arrays.items()}
            late = out / tag / "mechanism_late" / f"{arm}_{settings['end_stage']:011d}.npz"
            if not late.is_file():
                issues.append({"seed": tag, "arm": arm, "status": "missing_late_functional_diagnostics", "missing": [str(late)]})
        all_data[tag], receipts[tag] = data, {"mechanism": mechanism, "completions": completions}
        if not gate_path.is_file():
            issues.append({"seed": tag, "status": "missing_mechanism_confirmation", "path": str(gate_path)})
    return all_data, classes, receipts, issues


def analyze(work: Path) -> dict[str, Any]:
    settings_path = work / "settings.json"
    settings = read_json(settings_path)
    validate_design(settings)
    pilot = single_seed_pilot(settings)
    seed_data, classes, receipts, issues = load_window(work, settings)
    reports = {}
    for seed, data in seed_data.items():
        primary = primary_family(data, classes, settings)
        gates = noninferiority(data, classes, settings)
        matching = d_r_matching(receipts[seed]["completions"], settings)
        reports[seed] = {"primary_Holm7": primary, "noninferiority": gates,
                         "d_r_matching": matching,
                         "mechanism_confirmation_passed": receipts[seed]["mechanism"]["passed"],
                         "methods": joint_methods(primary, gates, receipts[seed]["mechanism"]["passed"], matching),
                         "descriptive": descriptive(data, classes, settings),
                         "paired_comparisons": descriptive_comparisons(data, classes, settings),
                         "finite_grid_frontier": per_seed_frontier(data, classes, settings),
                         "update_statistics": {arm: record.get("update_statistics", {})
                                               for arm, record in receipts[seed]["completions"].items()}}
    complete = not issues
    aggregate = aggregate_primary(seed_data, reports, settings)
    frontier = aggregate_frontier(seed_data, settings) if pilot or seed_data else {"status": "incomplete"}
    bootstrap = bootstrap_frontier(seed_data, classes, settings, frontier) if pilot or complete else {
        "status": "incomplete_required_artifacts", "confirmed": False}
    plots = plot_tradeoffs(work, seed_data, settings) if complete else {"status": "pending_incomplete_artifacts", "paths": []}
    for method in aggregate["methods"].values():
        method["all_required_artifacts_present"] = complete
        support_key = "pilot_statistical_supported" if pilot else "confirmation_statistical_supported"
        method["supported_pending_independent_readback"] = bool(complete and method[support_key])
        method["scientific_supported"] = False
    return {"experiment": "exp030", "settings_sha256": sha256(settings_path),
            "protocol_sha256": sha256(work / "protocol.md"),
            "code_manifest_sha256": sha256(work / "code_manifest.json") if (work / "code_manifest.json").is_file() else None,
            "confirm_data_sha256": sha256(work / "outputs/data/confirm.h5") if (work / "outputs/data/confirm.h5").is_file() else None,
            "status": "complete_analysis_pending_independent_reader" if complete else "incomplete",
            "window": settings["late_window"], "stages": settings["confirm_stages"], "n_per_task": settings["n"],
            "checkpoint_reduction": "average_same_row_across_all_11_fixed_stages_before_inference",
            "analysis_scope": "single_seed_pilot" if pilot else "three_seed_confirmation",
            "cross_seed_confirmation_performed": not pilot,
            "issues": issues, "seeds": reports, "aggregate": aggregate,
            "aggregate_finite_grid_frontier": frontier, "frontier_bootstrap": bootstrap,
            "tradeoff_plots": plots,
            "claim_boundary": ("Single s5 pilot only; class-cluster uncertainty cannot establish between-seed robustness. " if pilot else "")
                              + "No mechanism or method support claim is final until raw-output and source/input identities pass the independent verifier"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, help="Default: <work>/outputs/analysis.json")
    args = parser.parse_args()
    work = args.work.resolve()
    output = args.output or work / "outputs" / "analysis.json"
    result = analyze(work)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"status": result["status"], "report": str(output), "issues": len(result["issues"])}))
    return 0 if result["status"] == "complete_analysis_pending_independent_reader" else 2


if __name__ == "__main__":
    raise SystemExit(main())

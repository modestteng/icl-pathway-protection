"""Independent raw-result analysis for the frozen exp023/024 inference runs.

This script imports no model, JAX, training or SSH module. It reads completed
per-row HDF5 groups and shared input identities, then writes both experiments'
scientific results. All confirmatory choices come from frozen settings and
time_selection.json. It never selects a new window from intervention effects.
"""

import argparse
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from analysis_stats import adjusted_family, cluster_stats, equivalence, one_sided


CIWL = "iwl_copy_avail"
SCALARS = (
    "full_loss", "pair_loss", "output_margin", "full_acc", "pair_acc",
    "outside_pair", "full_prob", "pair_prob", "context_prob", "target_logit",
    "other_logit", "full_logsumexp",
)
PREDICTIONS = ("full_pred", "pair_pred")
AUXILIARIES = {
    "l2_qk_margin": (8,), "l2_attention_margin": (8,),
    "l1_previous_attention": (8, 2), "l1_self_attention": (8, 2),
    "l1_previous_score": (8, 2), "head_label_delta_norm": (2, 2),
}
METRICS = SCALARS + PREDICTIONS + tuple(AUXILIARIES)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(serializable(value), ensure_ascii=False, indent=2,
                              allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def input_identity(path, tasks, n):
    output = {}
    with h5py.File(path, "r") as f:
        for task in tasks:
            labels, ids = f[f"{task}/labels"][:], f[f"{task}/identities"][:]
            require(labels.shape == (n, 3) and ids.shape == (n, 3, 2),
                    f"Bad input identity shapes for {task}")
            require(np.issubdtype(labels.dtype, np.integer) and np.issubdtype(ids.dtype, np.integer),
                    f"Input identities must be integral: {task}")
            target = labels[:, :2] == labels[:, -1, None]
            require(np.all(target.sum(axis=1) == 1), f"Nonunique correct candidate: {task}")
            if task == CIWL:
                require(np.array_equal(labels[:, -1], ids[:, -1, 0]), "CIWL target/class mismatch")
                require(np.all(ids[:, :2, 0] != ids[:, -1, 0, None]), "CIWL query has support")
            output[task] = {
                "labels": labels, "classes": ids[:, -1, 0],
                "target_position": np.argmax(target, axis=1),
                "classes_count": int(len(np.unique(ids[:, -1, 0]))),
            }
    return output


class RawBook:
    """Read-on-demand arrays; only completed, finite, correctly sized groups."""

    def __init__(self, path, n, identities):
        self.path, self.n, self.identities = Path(path), int(n), identities
        self.file = h5py.File(self.path, "r")
        self.summaries = []

    def close(self):
        self.array.cache_clear()
        self.file.close()

    @staticmethod
    def key(category, stage, code, evaluator):
        return f"{category}/{int(stage)}/{code}/{evaluator}"

    @lru_cache(maxsize=96)
    def array(self, category, stage, code, evaluator, metric):
        key = self.key(category, stage, code, evaluator)
        require(key in self.file, f"Missing frozen group: {self.path}: {key}")
        group = self.file[key]
        require(bool(group.attrs.get("completed", False)), f"Incomplete group: {key}")
        require(metric in group, f"Missing metric {metric}: {key}")
        data = np.asarray(group[metric][:])
        require(data.shape == (self.n,) + AUXILIARIES.get(metric, ()),
                f"Bad shape {data.shape} for {key}/{metric}")
        require(np.isfinite(data).all(), f"Nonfinite result: {key}/{metric}")
        return data.astype(np.float64)

    def average(self, stages, code, metric, evaluator=CIWL, category="confirmation"):
        stages = list(stages)
        require(bool(stages), "An analysis window is empty")
        total = np.zeros_like(self.array(category, stages[0], code, evaluator, metric))
        for stage in stages:
            total += self.array(category, stage, code, evaluator, metric)
        return total / len(stages)

    def expect(self, stages, codes, tasks, category="confirmation"):
        for stage in stages:
            for code in codes:
                for task in tasks:
                    self.array(category, stage, code, task, "full_loss")

    def summarize_all(self):
        """Independent means and metric algebra; no model forward or resampling."""
        self.summaries = []
        for category in ("discovery", "confirmation", "original"):
            if category not in self.file:
                continue
            require(category in self.identities, f"Missing row identities for {category}")
            for stage_text in sorted(self.file[category], key=int):
                stage = int(stage_text)
                for code in self.file[f"{category}/{stage_text}"]:
                    prefix = f"{category}/{stage_text}/{code}"
                    for task in self.file[prefix]:
                        require(task in self.identities[category], f"Unknown evaluator: {task}")
                        group = self.file[f"{prefix}/{task}"]
                        require(bool(group.attrs.get("completed", False)), f"Incomplete group: {group.name}")
                        require("summary" in group.attrs, f"Missing saved summary: {group.name}")
                        summary = json.loads(serializable(group.attrs["summary"]))
                        arrays = {metric: self.array(category, stage, code, task, metric)
                                  for metric in METRICS}
                        labels = self.identities[category][task]["labels"]
                        require(np.array_equal(arrays["full_acc"], arrays["full_pred"] == labels[:, -1]),
                                f"full_acc/pred mismatch: {group.name}")
                        require(np.array_equal(arrays["pair_acc"], arrays["pair_pred"] == labels[:, -1]),
                                f"pair_acc/pred mismatch: {group.name}")
                        require(np.all((arrays["pair_pred"] == labels[:, 0]) |
                                       (arrays["pair_pred"] == labels[:, 1])), f"Pair prediction out of pair: {group.name}")
                        outside = (arrays["full_pred"] != labels[:, 0]) & (arrays["full_pred"] != labels[:, 1])
                        require(np.array_equal(arrays["outside_pair"], outside), f"outside_pair mismatch: {group.name}")
                        algebra = {
                            "full_loss": np.max(np.abs(arrays["full_loss"] -
                                (arrays["full_logsumexp"] - arrays["target_logit"]))),
                            "margin": np.max(np.abs(arrays["output_margin"] -
                                (arrays["target_logit"] - arrays["other_logit"]))),
                            "pair_loss": np.max(np.abs(arrays["pair_loss"] -
                                np.logaddexp(0., -arrays["output_margin"]))),
                            "full_prob": np.max(np.abs(arrays["full_prob"] - np.exp(-arrays["full_loss"]))),
                            "pair_prob": np.max(np.abs(arrays["pair_prob"] -
                                np.exp(-np.logaddexp(0., -arrays["output_margin"])))),
                            "candidate_mass": np.max(np.abs(arrays["full_prob"] -
                                arrays["context_prob"] * arrays["pair_prob"])),
                        }
                        require(all(value <= 1e-4 for value in algebra.values()),
                                f"Saved metric algebra inconsistent: {group.name}: {algebra}")
                        row = {"category": category, "stage": stage, "condition": code,
                               "evaluator": task, "n": self.n,
                               "clusters": self.identities[category][task]["classes_count"],
                               "completed": True, "saved_summary": summary,
                               "metric_algebra_max_errors": serializable(algebra)}
                        row.update({metric: float(arrays[metric].mean()) for metric in SCALARS})
                        for metric in AUXILIARIES:
                            row[metric] = arrays[metric].mean(axis=0).tolist()
                        self.summaries.append(row)
        return self.summaries


def describe_difference(values, identity, structural=False):
    result = cluster_stats(values, identity["classes"], structural_identity=structural)
    result["by_correct_label_position"] = {
        str(position): cluster_stats(values[identity["target_position"] == position],
                                    identity["classes"][identity["target_position"] == position],
                                    structural_identity=structural)
        for position in (0, 1)
    }
    return result


def window_metric_descriptions(book, stages, codes, identity):
    output = {}
    for code in codes:
        output[code] = {}
        for metric in SCALARS:
            native = book.average(stages, "N", metric)
            changed = book.average(stages, code, metric)
            # For loss/outside-pair, positive means a reduction. For other
            # metrics, positive means changed minus native, explicitly labeled.
            reduced = metric in ("full_loss", "pair_loss", "outside_pair")
            delta = native - changed if reduced else changed - native
            output[code][metric] = {
                "native_mean": float(native.mean()), "condition_mean": float(changed.mean()),
                "difference_direction": "N - condition" if reduced else "condition - N",
                "paired_difference": describe_difference(delta, identity, structural=code == "N"),
            }
    return output


def rebound_comparisons(book, selection, identity):
    definitions = {
        "window_vs_pre": (selection["windows"]["divot"], selection["windows"]["pre"]),
        "window_vs_post": (selection["windows"]["divot"], selection["windows"]["post"]),
        "peak_vs_left": ([selection["peak_stage"]], [selection["left_stage"]]),
        "peak_vs_right": ([selection["peak_stage"]], [selection["right_stage"]]),
    }
    output = {}
    for name, (peaks, flanks) in definitions.items():
        dn = book.average(peaks, "N", "full_loss") - book.average(flanks, "N", "full_loss")
        ds = book.average(peaks, "P1", "full_loss") - book.average(flanks, "P1", "full_loss")
        output[name] = {
            "peak_stages": peaks, "flank_stages": flanks,
            "native_rebound": describe_difference(dn, identity),
            "suppressed_rebound": describe_difference(ds, identity),
            "rebound_reduction": describe_difference(dn - ds, identity),
            "note": "Ordinary descriptive 95% intervals; nonsignificance does not establish disappearance",
        }
    return output


def curves_and_heads(book, stages, codes, identity):
    curves, heads = [], []
    for stage in stages:
        current = {"stage": int(stage), "conditions": {}, "effects": {}}
        head_row = {"stage": int(stage), "conditions": {}}
        native_loss = book.array("confirmation", stage, "N", CIWL, "full_loss")
        for code in codes:
            current["conditions"][code] = {metric: float(book.array(
                "confirmation", stage, code, CIWL, metric).mean()) for metric in SCALARS}
            current["effects"][code] = {}
            for metric in ("full_loss", "pair_loss", "output_margin", "context_prob"):
                native = book.array("confirmation", stage, "N", CIWL, metric)
                changed = book.array("confirmation", stage, code, CIWL, metric)
                delta = native - changed if metric.endswith("loss") else changed - native
                current["effects"][code][metric] = cluster_stats(
                    delta, identity["classes"], structural_identity=code == "N")
            head_row["conditions"][code] = {}
            for metric in ("l2_qk_margin", "l2_attention_margin"):
                actual = book.array("confirmation", stage, code, CIWL, metric)
                native = book.array("confirmation", stage, "N", CIWL, metric)
                head_row["conditions"][code][metric] = {
                    "mean_by_head": actual.mean(axis=0).tolist(),
                    "delta_by_head": [cluster_stats(actual[:, h] - native[:, h], identity["classes"],
                        structural_identity=code == "N") for h in range(8)],
                    "inference": "descriptive ordinary intervals; no post-hoc confirmatory head selection",
                }
            for metric in ("l1_previous_attention", "l1_self_attention", "l1_previous_score", "head_label_delta_norm"):
                head_row["conditions"][code][metric] = book.array(
                    "confirmation", stage, code, CIWL, metric).mean(axis=0).tolist()
            if code != "N":
                current["effects"][code]["full_loss"]["native_loss_floor_fraction"] = (
                    float((native_loss < 0.02).mean()))
        curves.append(current)
        heads.append(head_row)
    return curves, heads


def fallacy_coverage():
    items = [
        ("Simpson's paradox", "partially_assessed", "Correct-label-position strata and class effect distributions saved; no exhaustive covariate stratification"),
        ("Ecological fallacy", "interpretation_bounded", "Paired row effects retained; group averages do not identify each row's strategy"),
        ("Berkson's paradox", "design_review_only", "Frozen author sampling, no effect-based row exclusion; no general population selection guarantee"),
        ("Collider bias", "design_review_only", "No adjustment for post-intervention covariates; historical native locator independent of intervention"),
        ("Base rate neglect", "partially_assessed", "Full and two-candidate outcomes plus candidate probability mass reported; no diagnostic prevalence claim"),
        ("Regression to the mean", "design_mitigated", "Native divot confirmed on independent seed20261012 with frozen historical points; selection uncertainty remains"),
        ("Survivorship bias", "completion_checked", "All prescribed groups must be completed; failed executions belong in run records, not removed scientific conditions"),
        ("Look-elsewhere effect", "design_mitigated", "All primary contrasts reported with family correction; curve maxima and heads remain descriptive"),
        ("Garden of forking paths", "design_review_only", "Frozen settings/window files referenced; protocol changes and execution history require separate provenance audit"),
        ("Correlation versus causation", "interpretation_bounded", "Within-checkpoint interventions establish conditional forward effects; temporal alignment is not training causation"),
        ("Reverse causality", "not_resolved_for_training", "Inference interventions do not identify the direction or unique cause of natural training dynamics"),
    ]
    return {"reviewed_items": 11, "empirical_exclusion_of_all_fallacies": False,
            "items": [{"name": name, "status": status, "basis": basis} for name, status, basis in items]}


def analyze_window(book, selection, settings, identities):
    identity, discovery = identities["confirmation"][CIWL], identities["discovery"][CIWL]
    windows, stats = selection["windows"], settings["statistics"]
    peak, left, right = (int(selection[key]) for key in ("peak_stage", "left_stage", "right_stage"))
    dp = book.array("discovery", peak, "N", CIWL, "full_loss")
    divot_tests = []
    for name, flank in (("peak_minus_left", left), ("peak_minus_right", right)):
        diff = dp - book.array("discovery", flank, "N", CIWL, "full_loss")
        divot_tests.append(one_sided(diff, discovery["classes"], 0.02, name))
    divot_family = adjusted_family(divot_tests, method="bonferroni")
    divot_confirmed = bool(selection["located_divot"] and divot_family["all_pass"])
    effects = {name: book.average(stages, "N", "full_loss") - book.average(stages, "P1", "full_loss")
               for name, stages in windows.items()}
    control = book.average(windows["divot"], "N", "full_loss") - book.average(windows["divot"], "H6R", "full_loss")
    primary = [
        one_sided(effects["divot"], identity["classes"], stats["minimum_window_benefit_nats"], "divot_P1"),
        one_sided(effects["divot"] - effects["pre"], identity["classes"], stats["minimum_concentration_nats"], "divot_vs_pre_P1"),
        one_sided(effects["divot"] - effects["post"], identity["classes"], stats["minimum_concentration_nats"], "divot_vs_post_P1"),
        one_sided(effects["divot"] - control, identity["classes"], stats["minimum_concentration_nats"], "divot_P1_vs_H6R"),
    ]
    family = adjusted_family(primary)
    for test, vector in zip(family["tests"], (effects["divot"], effects["divot"] - effects["pre"],
                                             effects["divot"] - effects["post"], effects["divot"] - control)):
        test["by_correct_label_position"] = describe_difference(vector, identity)["by_correct_label_position"]
    late_n = book.array("confirmation", 20000000, "N", CIWL, "full_loss")
    late_p = book.array("confirmation", 20000000, "P1", CIWL, "full_loss")
    late = equivalence(late_n - late_p, identity["classes"], stats["late_equivalence_nats"], "I_P1_20M")
    late.update(native_loss=float(late_n.mean()), suppressed_loss=float(late_p.mean()),
                native_rows_below_0_02=float((late_n < 0.02).mean()),
                note="Small late benefit may reflect a loss floor; equivalence does not establish missing information or mechanism")
    stages = sorted(int(stage) for stage in selection["window_stages"])
    curves, heads = curves_and_heads(book, stages, settings["window_codes"], identity)
    maximum = max(curves, key=lambda row: (row["effects"]["P1"]["full_loss"]["mean"], -row["stage"]))
    peak_i = {"stage": maximum["stage"], "mean": maximum["effects"]["P1"]["full_loss"]["mean"],
              "distance_from_historical_peak": maximum["stage"] - peak,
              "selection": "descriptive maximum of the whole frozen grid, not a new confirmatory window"}
    metric_windows = {name: window_metric_descriptions(book, stages_, settings["window_codes"], identity)
                      for name, stages_ in windows.items()}
    return {
        "experiment": "exp023_ciwl_divot_window", "primary": {test["name"]: test for test in family["tests"]},
        "families": {"divot_confirmation": divot_family, "window": family},
        "historical_located_divot": bool(selection["located_divot"]),
        "divot_confirmed": divot_confirmed,
        "fixed_native_contrasts_pass": divot_family["all_pass"],
        "window_family_pass": family["all_pass"],
        "supports_time_concentration": bool(divot_confirmed and family["all_pass"]),
        "rebound_comparisons": rebound_comparisons(book, selection, identity),
        "late_equivalence": late, "window_metrics": metric_windows,
        "whole_grid_curves": curves, "whole_grid_interference_maximum": peak_i,
        "head_diagnostics": heads,
        "control_scope": "H6R matches relative previous-attention decline, not absolute mass or output disturbance; SELF is a direction diagnostic",
        "uncertainty": "Single training seed; query-class sampling uncertainty only; checkpoint points are not independent replicates",
    }


def equivalence_audit(book, stages, identities):
    output = {}
    for first, second in (("KB", "V"), ("VB", "K"), ("KV", "S")):
        errors, exact_predictions, groups = {}, True, 0
        for stage in stages:
            for task in identities:
                key = book.key("confirmation", stage, first, task)
                if key not in book.file:
                    continue
                groups += 1
                for metric in SCALARS + ("l2_qk_margin", "l2_attention_margin"):
                    a, b = (book.array("confirmation", stage, code, task, metric) for code in (first, second))
                    errors[metric] = max(errors.get(metric, 0.0), float(np.max(np.abs(a - b))))
                for metric in PREDICTIONS:
                    a, b = (book.array("confirmation", stage, code, task, metric) for code in (first, second))
                    exact_predictions = bool(exact_predictions and np.array_equal(a, b))
        output[f"{first}={second}"] = {
            "groups_compared": groups, "metric_max_abs_errors": errors,
            "predictions_exact": exact_predictions,
            "passes_saved_metric_tolerance": bool(exact_predictions and all(v <= 1e-4 for v in errors.values())),
            "role": "computational equivalence audit, not independent causal evidence; full-logit/structural audit belongs to runner/verification",
        }
    return output


def analyze_path(book, window_book, selection, settings, identity, window_result):
    stages, stats = selection["windows"]["divot"], settings["statistics"]
    n, s, k, v, kb = [book.average(stages, code, "full_loss") for code in ("N", "S", "K", "V", "KB")]
    total, transfer, block = n - s, n - k, kb - s
    margin = book.average(stages, "K", "output_margin") - book.average(stages, "N", "output_margin")
    primary = [
        one_sided(transfer, identity["classes"], 0.01, "K_transfer"),
        one_sided(block, identity["classes"], 0.01, "K_block"),
        one_sided(v - k, identity["classes"], 0.01, "K_vs_V"),
        one_sided(margin, identity["classes"], 0.0, "output_margin_K"),
    ]
    family = adjusted_family(primary)
    for test, vector in zip(family["tests"], (transfer, block, v - k, margin)):
        test["by_correct_label_position"] = describe_difference(vector, identity)["by_correct_label_position"]
    indexed = {test["name"]: test for test in family["tests"]}
    total_test = one_sided(total, identity["classes"], stats["minimum_window_benefit_nats"], "total_effect")
    fraction = stats["path_retention_fraction"]
    retention = adjusted_family([
        one_sided(transfer - fraction * total, identity["classes"], 0.0, "K_transfer_retains_80pct"),
        one_sided(block - fraction * total, identity["classes"], 0.0, "K_block_removes_80pct"),
    ])
    point_eligible = bool(total_test["mean"] > stats["minimum_window_benefit_nats"])
    retention.update(total_effect=total_test, eligible_point_estimate=point_eligible,
                     eligible_total_effect_test=total_test["pass_unadjusted"],
                     supports_retention=bool(total_test["pass_unadjusted"] and retention["all_pass"]),
                     transfer_ratio=transfer.mean() / total.mean() if point_eligible else None,
                     block_ratio=block.mean() / total.mean() if point_eligible else None,
                     ratio_note="Descriptive same-effect ratios only when total mean >0.02; no independent additive mediation proportions")
    veq = equivalence(n - v, identity["classes"], stats["v_equivalence_nats"], "V_only_gain")
    pair_gain = book.average(stages, "N", "pair_loss") - book.average(stages, "K", "pair_loss")
    pair_diagnostic = describe_difference(pair_gain, identity)
    path_supported = bool(indexed["K_transfer"]["pass_adjusted"] and indexed["K_block"]["pass_adjusted"]
                          and indexed["output_margin_K"]["pass_adjusted"])
    full_improves_pair_worsens = bool(transfer.mean() > 0 and pair_gain.mean() < 0)
    path_stages = sorted(int(stage) for stage in selection["path_stages"])
    curves, heads = curves_and_heads(book, path_stages, settings["path_codes"], identity)
    head_window = {}
    for metric in ("l2_qk_margin", "l2_attention_margin"):
        native, patched = (book.average(stages, code, metric) for code in ("N", "K"))
        head_window[metric] = {
            "native_means": native.mean(axis=0).tolist(), "K_means": patched.mean(axis=0).tolist(),
            "K_minus_N": [cluster_stats(patched[:, head] - native[:, head], identity["classes"]) for head in range(8)],
            "inference": "All eight heads; ordinary descriptive intervals, no head confirmation",
        }
    agreements = {}
    for path_code, window_code in (("N", "N"), ("S", "P1")):
        max_errors, exact_predictions = {}, True
        for stage in path_stages:
            for metric in SCALARS + tuple(AUXILIARIES):
                a = book.array("confirmation", stage, path_code, CIWL, metric)
                b = window_book.array("confirmation", stage, window_code, CIWL, metric)
                max_errors[metric] = max(max_errors.get(metric, 0.0), float(np.max(np.abs(a - b))))
            for metric in PREDICTIONS:
                exact_predictions = bool(exact_predictions and np.array_equal(
                    book.array("confirmation", stage, path_code, CIWL, metric),
                    window_book.array("confirmation", stage, window_code, CIWL, metric)))
        agreements[f"path_{path_code}=window_{window_code}"] = {
            "max_abs_errors": max_errors, "predictions_exact": exact_predictions,
            "passes_saved_metric_tolerance": bool(exact_predictions and all(v <= 1e-4 for v in max_errors.values())),
        }
    return {
        "experiment": "exp024_l2_label_k_path", "primary": indexed,
        "families": {"path": family, "retention_diagnostic": retention},
        "supports_K_forward_path": path_supported,
        "supports_K_over_V": indexed["K_vs_V"]["pass_adjusted"],
        "all_path_tests_pass": family["all_pass"],
        "divot_confirmed_in_exp023": window_result["divot_confirmed"],
        "window_concentration_confirmed_in_exp023": window_result["supports_time_concentration"],
        "full_improves_but_pair_loss_worsens": full_improves_pair_worsens,
        "static_matching_interpretation": (
            "not_supported_full_gain_pair_worsening" if full_improves_pair_worsens else
            "output_margin_support_with_all_head_diagnostics" if path_supported and pair_gain.mean() > 0 else
            "not_confirmed_by_primary_margin_or_pair_loss"),
        "total_effect": total_test, "V_equivalence": veq,
        "factorial_interaction_gain": describe_difference(k + v - n - s, identity),
        "interaction_definition": "(N-S)-(N-K)-(N-V); gain interaction, not independent mediated fractions",
        "pair_loss_K_gain": pair_diagnostic,
        "window_metrics": window_metric_descriptions(book, stages, settings["path_codes"], identity),
        "window_head_diagnostics": head_window, "whole_grid_curves": curves,
        "head_diagnostics": heads, "computational_equivalences": equivalence_audit(book, path_stages, book.identities["confirmation"]),
        "cross_experiment_agreement": agreements,
        "uncertainty": "Single training seed; forward activation interventions do not identify incorrect-class content or unique training cause",
    }


def write_summary(path, summaries):
    columns = ["category", "stage", "condition", "evaluator", "n", "clusters"] + list(SCALARS)
    for metric, shape in AUXILIARIES.items():
        columns.extend(f"{metric}_{'_'.join(map(str, index))}" for index in np.ndindex(shape))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, columns, delimiter="\t")
        writer.writeheader()
        for summary in summaries:
            row = {key: summary[key] for key in columns if key in summary}
            for metric, shape in AUXILIARIES.items():
                array = np.asarray(summary[metric])
                for index in np.ndindex(shape):
                    row[f"{metric}_{'_'.join(map(str, index))}"] = float(array[index])
            writer.writerow(row)


def fmt(value, digits=6):
    return "NA" if value is None else f"{value:.{digits}g}"


def interval(stat, prefix="ci95"):
    return f"[{fmt(stat.get(prefix + '_lower'))}, {fmt(stat.get(prefix + '_upper'))}]"


def family_table(family):
    lines = ["|对比|均值|SE|门槛|普通95%区间|校正p|Bonf单侧下界|通过|",
             "|---|---:|---:|---:|---|---:|---:|---|"]
    for test in family["tests"]:
        lines.append(f"|{test['name']}|{fmt(test['mean'])}|{fmt(test['se'])}|{fmt(test['threshold'])}|"
                     f"{interval(test)}|{fmt(test['p_adjusted'])}|{fmt(test['lower_bonferroni'])}|{test['pass_adjusted']}|")
    return "\n".join(lines)


def write_test_report(path, result):
    lines = ["# 预设统计检验", "", "效应按逐条配对、窗口内等权平均后，以真实查询类别聚类；单位为nat，输出margin项为logit。",
             "普通95%区间与Bonferroni同时单侧下界分列。Holm家族的p值检验相对预设门槛；门槛未获支持不等于效应为零。", ""]
    for name, family in result["families"].items():
        lines.extend([f"## {name}", "", f"校正：{family['method']}；alpha={family['alpha']}；"
                      f"全部通过={family['all_pass']}；全部Bonf下界通过={family['all_bonferroni_bounds_pass']}。", "", family_table(family), ""])
    eq = result.get("late_equivalence", result.get("V_equivalence"))
    lines.extend(["## 预设等价诊断", "", f"{eq['name']}：均值={fmt(eq['mean'])}，90%区间={interval(eq, 'ci90')}，"
                  f"界限±{eq['equivalence_margin']}，TOST p={fmt(eq['p_tost'])}，通过={eq['equivalent']}。", "",
                  "p的数值下溢为0不代表真实概率为零。退化簇标准误使用保守不通过，未伪造无限精度。", ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def diagnostic_table(summaries, codes):
    rows = [row for row in summaries if row["category"] == "confirmation" and row["evaluator"] != CIWL
            and row["condition"] in codes]
    lines = ["|阶段|任务|条件|完整loss|两候选loss|完整acc|两候选acc|候选外预测|候选概率质量|",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(rows, key=lambda value: (value["stage"], value["evaluator"], value["condition"])):
        lines.append(f"|{row['stage']}|{row['evaluator']}|{row['condition']}|{fmt(row['full_loss'])}|"
                     f"{fmt(row['pair_loss'])}|{row['full_acc']:.4%}|{row['pair_acc']:.4%}|"
                     f"{row['outside_pair']:.4%}|{fmt(row['context_prob'])}|")
    return "\n".join(lines)


def write_scientific_report(path, result, selection, summaries):
    is_window = result["experiment"].startswith("exp023")
    lines = [f"# {result['experiment']}：实际推理结果", "", "## Material Passport", "",
             "Origin Skill: academic-research-suite / experiment-agent；Mode: run → validate；"
             "Origin Date: 2026-10-10（America/Los_Angeles）；Verification Status: ANALYZED（实际逐条结果读回；"
             "实现/来源独立验收另见审计）；Version: exp023_exp024_ciwl_divot_v1。", "",
             "## 实际结论", ""]
    if is_window:
        lines.extend([f"历史曲线定位到预定义divot={result['historical_located_divot']}；独立新样本确认="
                      f"{result['divot_confirmed']}；主时间窗口四项全部通过={result['window_family_pass']}。",
                      f"本协议的“干扰集中在divot附近”得到支持={result['supports_time_concentration']}。"
                      "结果仅涉及该自然训练轨迹的固定模型前向计算。", "",
                      f"固定峰/左谷/右恢复：{selection['peak_stage']}/{selection['left_stage']}/{selection['right_stage']}序列；"
                      f"历史prominence={fmt(selection.get('selected_prominence'))} nat。"
                      "fallback点不用于支持divot时间吻合。", "",
                      "## 独立原生divot确认", "", family_table(result["families"]["divot_confirmation"]), "",
                      "## 主时间家族", "", family_table(result["families"]["window"]), "",
                      "## 原生与抑制后的实际损失回升", "",
                      "|比较|N回升及普通95%区间|P1回升及普通95%区间|回升减少及普通95%区间|",
                      "|---|---|---|---|"])
        for name, comparison in result["rebound_comparisons"].items():
            values = [comparison[key] for key in ("native_rebound", "suppressed_rebound", "rebound_reduction")]
            lines.append(f"|{name}|" + "|".join(f"{fmt(value['mean'])} {interval(value)}" for value in values) + "|")
        maximum, late = result["whole_grid_interference_maximum"], result["late_equivalence"]
        lines.extend(["", "抑制后仍存在的正回升直接保留；未显著不等于回升消失。", "",
                      f"冻结全网格的描述性I最大点={maximum['stage']}，均值={fmt(maximum['mean'])} nat，"
                      f"相对历史峰距离={maximum['distance_from_historical_peak']}序列，不用于重新选窗口。", "",
                      f"20M I={fmt(late['mean'])}，90%区间={interval(late, 'ci90')}，±0.02 nat等价通过={late['equivalent']}；"
                      f"原生loss={fmt(late['native_loss'])}，抑制loss={fmt(late['suppressed_loss'])}，"
                      f"原生逐条loss<0.02比例={late['native_rows_below_0_02']:.3%}。小净收益也可能受到损失地板约束。", ""])
        codes = ("N", "P1", "H6R")
    else:
        lines.extend([f"K两种V背景中的前向路径效应及正确标签margin方向得到支持={result['supports_K_forward_path']}；"
                      f"K优于V={result['supports_K_over_V']}；四个主路径检验全部通过={result['all_path_tests_pass']}。",
                      f"exp023独立原生divot确认={result['divot_confirmed_in_exp023']}；时间集中确认="
                      f"{result['window_concentration_confirmed_in_exp023']}。一般路径效应不能替代这两项确认。", "",
                      "## 主路径家族", "", family_table(result["families"]["path"]), "",
                      "## 两因素交互与更强诊断", ""])
        interaction, retain, veq = (result["factorial_interaction_gain"], result["families"]["retention_diagnostic"], result["V_equivalence"])
        lines.extend([f"gain交互=(N−S)−(N−K)−(N−V)={fmt(interaction['mean'])} nat，普通95%区间={interval(interaction)}。"
                      "K/V条件是相同四角的不同背景效应，不解释成可相加的独立中介比例。", "",
                      f"总收益={fmt(result['total_effect']['mean'])} nat；总收益超过0.02的单侧检验通过="
                      f"{retain['eligible_total_effect_test']}；80%线性诊断得到支持={retain['supports_retention']}。"
                      f"仅在总收益点估计>0.02时保存描述性传递/撤销比：{fmt(retain['transfer_ratio'])}/{fmt(retain['block_ratio'])}。", "",
                      family_table(retain), "",
                      f"V-only收益={fmt(veq['mean'])}，90%区间={interval(veq, 'ci90')}，±0.02 nat等价通过={veq['equivalent']}。"
                      "未通过等价不说明V无用。", "",
                      "## 匹配解释与实现等价", "",
                      f"K-only的两候选loss收益={fmt(result['pair_loss_K_gain']['mean'])} nat，"
                      f"普通95%区间={interval(result['pair_loss_K_gain'])}；full改善而pair loss恶化="
                      f"{result['full_improves_but_pair_loss_worsens']}；匹配解释状态={result['static_matching_interpretation']}。",
                      "所有8头的QK/attention方向保存在analysis.json和图中，未按结果选择有利头。"
                      "若只有full loss改善而pair判断恶化，不支持静态正确标签匹配受到干扰的说法。", ""])
        for name, audit in result["computational_equivalences"].items():
            lines.append(f"- {name}：已保存指标误差门槛通过={audit['passes_saved_metric_tolerance']}，"
                         f"预测精确={audit['predictions_exact']}；这是计算审计，不是独立重复证据。")
        codes = ("N", "S", "K", "V")
    lines.extend(["", "## 完整输出与策略权衡诊断", "", diagnostic_table(summaries, codes), "",
                  "其他固定条件、全部逐头均值和所有时间点见summary.tsv/analysis.json；上述诊断表没有新增确认检验。", "",
                  "## 统计与结论边界", "",
                  "各窗口内先按检查点等权取得每条输入的平均，再按真实query class聚类。"
                  "单训练seed init5/train2；区间只覆盖评价类别抽样不确定性，不包含跨训练seed变异。"
                  "主家族单侧Holm校正与Bonf同时单侧下界分列；普通95%区间不称同时区间。"
                  "0.02/0.01 nat是本轮预设实用门槛，未获支持不等于效应不存在；退化SE保守处理。", "",
                  "前向干预和K注入可以定位条件性路径，不能确定写入内容必为错误类别，不能证明唯一自然训练原因、"
                  "首创或跨seed普遍性。H6R仅匹配相对读取下降；SELF是方向诊断。"
                  "本轮没有继续训练或相似度剂量实验。", "",
                  "## 11类统计误读核查", "",
                  "完成11项设计/解释审阅；不宣称实证排除全部11类风险。", "",
                  "|事项|核查层级|依据与残余边界|", "|---|---|---|"])
    for item in result["fallacy_coverage"]["items"]:
        lines.append(f"|{item['name']}|{item['status']}|{item['basis']}|")
    lines.extend(["", "idea → change → result → decision：检验CIWL损失回升的时间集中与L2标签K路径 → "
                  "固定自然检查点与不看答案的注意力抑制/激活传递 → 上述实际检验结果 → "
                  "保留实测方向与不确定性，不用失败门槛证明零效应，不自动追加计算。", ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def figures(window, path, selection, outputs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none"})
    stages = np.array([row["stage"] for row in window["whole_grid_curves"]]) / 1e6
    rows = window["whole_grid_curves"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for code, color in (("N", "#262626"), ("P1", "#276fb5"), ("P2", "#769fcd"), ("PB", "#9aa1a7")):
        axes[0].plot(stages, [row["conditions"][code]["full_loss"] for row in rows], label=code, color=color)
    hist = selection.get("historical_curve", [])
    if hist:
        axes[0].plot([row["stage"] / 1e6 for row in hist], [row["full_loss"] for row in hist],
                     color="#b8b8b8", linestyle="--", label="Historical N")
    peak = selection["peak_stage"] / 1e6
    axes[0].set(xlim=(max(0, peak-1.2), peak+2.2), xlabel="Training sequences (M)", ylabel="CIWL full CE (nats)")
    for code, color in (("P1", "#276fb5"), ("H6R", "#c56a25"), ("SELF", "#777777")):
        values = [row["effects"][code]["full_loss"]["mean"] for row in rows]
        axes[1].plot(stages, values, marker=".", label=code, color=color)
    for ax in axes:
        for name, color in (("pre", "#ececec"), ("divot", "#dbebf7"), ("post", "#ececec")):
            ax.axvspan(min(selection["windows"][name])/1e6, max(selection["windows"][name])/1e6, color=color, alpha=.5, zorder=0)
        ax.axvline(peak, color="#999999", linestyle=":", linewidth=.8)
        ax.legend(frameon=False)
    axes[1].axhline(0, color="#999999", linewidth=.8)
    axes[1].set(xlabel="Training sequences (M)", ylabel="I = native CE - intervention CE")
    fig.suptitle("Frozen checkpoints: native loss and forward interference")
    for extension in ("png", "svg"):
        fig.savefig(outputs[0] / f"loss_and_interference.{extension}", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    tests = window["families"]["window"]["tests"]
    for y, test in enumerate(tests):
        mean = test["mean"] - test["threshold"]
        lower, upper = test["ci95_lower"], test["ci95_upper"]
        error = [[test["mean"]-lower], [upper-test["mean"]]] if lower is not None else None
        ax.errorbar(mean, y, xerr=error, fmt="o", color="#276fb5")
    ax.set(yticks=range(len(tests)), yticklabels=[test["name"] for test in tests],
           xlabel="Effect minus prespecified threshold (nats)", title="Primary time contrasts; ordinary 95% intervals")
    ax.axvline(0, color="#888888", linestyle="--", linewidth=.8)
    ax.invert_yaxis()
    for extension in ("png", "svg"):
        fig.savefig(outputs[0] / f"window_contrasts.{extension}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    prows = path["whole_grid_curves"]
    pstages = np.array([row["stage"] for row in prows]) / 1e6
    for code, color in (("S", "#777777"), ("K", "#276fb5"), ("V", "#c56a25")):
        axes[0].plot(pstages, [row["effects"][code]["full_loss"]["mean"] for row in prows],
                     marker="o", markersize=3, label=code, color=color)
    axes[0].axhline(0, color="#999999", linewidth=.8)
    axes[0].set(xlabel="Training sequences (M)", ylabel="Native CE - condition CE", title="All frozen path checkpoints")
    axes[0].legend(frameon=False)
    tests = path["families"]["path"]["tests"][:3]
    for y, test in enumerate(tests):
        error = [[test["mean"]-test["ci95_lower"]], [test["ci95_upper"]-test["mean"]]] if test["ci95_lower"] is not None else None
        axes[1].errorbar(test["mean"]-test["threshold"], y, xerr=error, fmt="o", color="#276fb5")
    axes[1].set(yticks=range(3), yticklabels=[test["name"] for test in tests],
                xlabel="Effect minus threshold (nats)", title="W path contrasts; ordinary 95% intervals")
    axes[1].axvline(0, color="#888888", linestyle="--", linewidth=.8)
    axes[1].invert_yaxis()
    for extension in ("png", "svg"):
        fig.savefig(outputs[1] / f"path_effects.{extension}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for ax, metric, label in zip(axes, ("l2_qk_margin", "l2_attention_margin"),
                                 ("Correct - other QK score", "Correct - other attention")):
        values = path["window_head_diagnostics"][metric]["K_minus_N"]
        means = [value["mean"] for value in values]
        errors = [1.959963984540054 * value["se"] if value["valid_normal_approximation"] else 0. for value in values]
        ax.errorbar(range(8), means, yerr=errors, fmt="o", capsize=2, color="#276fb5")
        ax.axhline(0, color="#888888", linestyle="--", linewidth=.8)
        ax.set(xticks=range(8), xlabel="L2 head", ylabel=f"K - N: {label}")
    fig.suptitle("All L2 heads in W; descriptive ordinary 95% intervals")
    for extension in ("png", "svg"):
        fig.savefig(outputs[1] / f"label_matching_heads.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--path-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true", help="Numerical analysis/report only")
    args = parser.parse_args()
    window_dir = args.window_dir.resolve()
    path_dir = args.path_dir.resolve() if args.path_dir else window_dir.parent / "exp_024_l2_label_k_path"
    settings = json.loads((window_dir / "settings.json").read_text())
    output23, output24 = window_dir / "outputs", path_dir / "outputs"
    selection = json.loads((output23 / "time_selection.json").read_text())
    n = settings["n"]
    require(n == 5000 and settings["execution"]["no_training"], "Unrecognized frozen sample/compute design")
    require(set(selection["windows"]) == {"pre", "divot", "post"}, "Unrecognized windows")
    require(all(len(stages) == 11 and len(set(stages)) == 11 for stages in selection["windows"].values()),
            "Each frozen window must contain 11 distinct checkpoints")
    identities = {
        "discovery": input_identity(output23 / "discovery_data.h5", settings["eval_sets"], n),
        "confirmation": input_identity(output23 / "confirmation_data.h5", settings["eval_sets"], n),
    }
    original_path = output23 / "original_data_identity.h5"
    if original_path.exists():
        identities["original"] = input_identity(original_path, settings["eval_sets"], n)
    window_book = RawBook(output23 / "per_sequence.h5", n, identities)
    path_book = RawBook(output24 / "per_sequence.h5", n, {"confirmation": identities["confirmation"]})
    try:
        window_book.expect(selection["window_stages"], settings["window_codes"], [CIWL])
        window_book.expect(selection["diagnostic_stages"], settings["window_codes"], settings["eval_sets"])
        window_book.expect([selection[key] for key in ("left_stage", "peak_stage", "right_stage")], ["N"], [CIWL], "discovery")
        path_book.expect(selection["path_stages"], settings["path_codes"], [CIWL])
        path_book.expect(selection["diagnostic_stages"], settings["path_codes"], settings["eval_sets"])
        summary23, summary24 = window_book.summarize_all(), path_book.summarize_all()
        window_result = analyze_window(window_book, selection, settings, identities)
        path_result = analyze_path(path_book, window_book, selection, settings, identities["confirmation"][CIWL], window_result)
        metadata = {
            "analysis_created_utc": datetime.now(timezone.utc).isoformat(),
            "settings_sha256": sha(window_dir / "settings.json"),
            "time_selection_sha256": sha(output23 / "time_selection.json"),
            "analyzer_sha256": sha(Path(__file__)), "statistics_sha256": sha(Path(__file__).with_name("analysis_stats.py")),
            "confirmation_input_sha256": sha(output23 / "confirmation_data.h5"),
            "discovery_input_sha256": sha(output23 / "discovery_data.h5"),
            "settings": settings, "time_selection": selection,
            "material_verification_status": "ANALYZED; raw metrics read; separate source/model/logit audit required",
        }
        for result, output, summaries, source in ((window_result, output23, summary23, window_book.path),
                                                 (path_result, output24, summary24, path_book.path)):
            output.mkdir(parents=True, exist_ok=True)
            result.update(metadata=metadata, raw_file=str(source), completed_groups=len(summaries),
                          raw_metric_summaries=summaries, fallacy_coverage=fallacy_coverage())
            write_json(output / "analysis.json", result)
            write_summary(output / "summary.tsv", summaries)
            write_test_report(output / "prespecified_tests.md", result)
            write_scientific_report(output.parent / "report.md", result, selection, summaries)
        if not args.no_plots:
            figures(window_result, path_result, selection, (output23, output24))
        print(json.dumps({"status": "ANALYZED", "window_support": window_result["supports_time_concentration"],
                          "K_path_support": path_result["supports_K_forward_path"],
                          "groups": {"exp023": len(summary23), "exp024": len(summary24)}}, ensure_ascii=False))
    finally:
        window_book.close()
        path_book.close()


if __name__ == "__main__":
    main()

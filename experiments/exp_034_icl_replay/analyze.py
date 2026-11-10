"""Fixed, paired 11-checkpoint analysis; no model code or science sampling.

This producer uses transparent formulas. verify_results.py independently reads
the raw artifacts and reconstructs the statistics without importing this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np

WORK = Path(__file__).resolve().parent
TASKS = ("train_eval", "iwl_copy_avail", "flip_icl", "icl")
METRICS = ("acc", "loss", "in_context_acc", "pair_loss", "pair_margin",
           "context_mass", "log_context_mass")
ARMS = ("REPLAY", "L2")
REFERENCES = ("A", "C25", "P25")
STAGES = tuple(range(9000000, 10000001, 100000))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(work: Path, path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else work / value


def validate_settings(s: dict[str, Any]) -> None:
    expected = {"arms": list(ARMS), "reference_arms": list(REFERENCES),
                "training_seeds": [{"id": "s5", "init": 5, "train": 2}],
                "start_stage": 4400000, "end_stage": 10000000,
                "confirm_stages": list(STAGES), "late_window": [9000000, 10000000],
                "fresh_tasks": list(TASKS), "n": 5000, "train_bs": 32,
                "protection_tasks": ["icl", "flip_icl"], "protection_bs_per_task": 32,
                "alpha": .05, "min_icl_gain": .05,
                "ni_acc_tolerance": .02, "ni_ce_tolerance": .1,
                "replay_coefficient": 2., "replay_loss_normalizer": 3.,
                "l2_coefficient": 1e-4,
                "l2_form": "sum_of_squared_all_trainable_array_entries_without_half",
                "l2_optimizer": "coupled_to_original_adam"}
    for name, value in expected.items():
        if s.get(name) != value:
            raise ValueError("Changed fixed design: " + name)
    if s["data_sets"]["protection"]["n"] != 4096:
        raise ValueError("The full protection pool must be 4096 per task")


def clustered(values: Any, classes: Any) -> dict[str, Any]:
    x, group = np.asarray(values, np.float64), np.asarray(classes)
    if x.shape != group.shape or x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("Malformed paired rows")
    if not np.issubdtype(group.dtype, np.integer):
        raise ValueError("Query classes must be true integer identities")
    unique, membership = np.unique(group, return_inverse=True)
    average = float(x.mean(dtype=np.float64))
    scores = np.bincount(membership, weights=x - average, minlength=len(unique))
    variance = float(len(unique) / (len(unique) - 1) * np.dot(scores, scores) / x.size**2) if len(unique) > 1 else None
    se = math.sqrt(max(variance, 0.)) if variance is not None else None
    valid = bool(se is not None and se > 1e-15)
    return {"mean": average, "se": se, "n": int(x.size), "clusters": len(unique),
            "valid": valid,
            "lower_one_sided_95": average - 1.6448536269514722 * se if valid else None,
            "upper_one_sided_95": average + 1.6448536269514722 * se if valid else None,
            "ci95_lower": average - 1.959963984540054 * se if valid else None,
            "ci95_upper": average + 1.959963984540054 * se if valid else None,
            "estimand": "evaluation_row_weighted_mean", "cluster_unit": "true_query_class",
            "status": "normal_query_class_cluster" if valid else "degenerate_no_confirmatory_support"}


def test_gain(stat: dict[str, Any], threshold: float, name: str) -> dict[str, Any]:
    z = (stat["mean"] - threshold) / stat["se"] if stat["valid"] else None
    p = .5 * math.erfc(z / math.sqrt(2)) if z is not None else 1.
    return dict(stat, name=name, threshold=float(threshold), alternative="mean > threshold",
                z_or_t=z, p_raw=p, p_underflow=bool(z is not None and p == 0.))


def holm(items: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    tests = [dict(item) for item in items]
    running = 0.
    for rank, slot in enumerate(sorted(range(len(tests)), key=lambda i: (tests[i]["p_raw"], i))):
        running = max(running, min(1., tests[slot]["p_raw"] * (len(tests) - rank)))
        tests[slot]["p_holm"] = float(running)
    for item in tests:
        item["family_size"] = len(tests)
        item["passed"] = bool(item["valid"] and item["p_holm"] < alpha and item["mean"] > item["threshold"])
    return {"tests": tests, "alpha": alpha, "family_size": len(tests),
            "all_passed": all(item["passed"] for item in tests), "correction": "Holm",
            "interval_note": "Reported 95% intervals are ordinary, not Holm simultaneous intervals"}


def noninferiority(stat: dict[str, Any], metric: str, s: dict[str, Any]) -> dict[str, Any]:
    accuracy = metric == "acc"
    margin = -s["ni_acc_tolerance"] if accuracy else s["ni_ce_tolerance"]
    bound = stat["lower_one_sided_95"] if accuracy else stat["upper_one_sided_95"]
    passed = bound is not None and (bound > margin if accuracy else bound < margin)
    return dict(stat, threshold=float(margin), metric=metric, passed=bool(passed),
                alternative="mean > -acc_tolerance" if accuracy else "mean < ce_tolerance",
                multiplicity="intersection_union_all_required_gates")


def read_window(paths: dict[int, Path], s: dict[str, Any]) -> dict[str, Any]:
    rows = {task: {metric: [] for metric in METRICS} for task in TASKS}
    if set(paths) != set(STAGES):
        raise ValueError("Missing or extra fixed checkpoint")
    for stage in STAGES:
        with np.load(paths[stage], allow_pickle=False) as raw:
            for task in TASKS:
                for metric in METRICS:
                    value = np.asarray(raw[task + "/" + metric], np.float64)
                    if value.shape != (s["n"],) or not np.isfinite(value).all():
                        raise ValueError("Incomplete/nonfinite row: " + str(paths[stage]))
                    rows[task][metric].append(value)
    return {task: {metric: np.stack(values).mean(axis=0, dtype=np.float64)
                   for metric, values in item.items()} for task, item in rows.items()}


def input_data(work: Path, s: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load(work / "references_manifest.json")
    for name, digest in manifest["files"].items():
        if sha256(resolve(work, name)) != digest:
            raise ValueError("Changed sealed reference: " + name)
    registration = {(item["arm"], item["stage"]): item for item in manifest["raw_records"]}
    if set(registration) != {(arm, stage) for arm in REFERENCES for stage in STAGES} or len(manifest["raw_records"]) != 33:
        raise ValueError("Expected exactly 33 sealed matched references")
    data = {}
    for arm in REFERENCES:
        paths = {}
        for stage in STAGES:
            record = registration[arm, stage]
            path, vp = resolve(work, record["path"]), resolve(work, record["verifier_path"])
            verifier = load(vp)
            if (sha256(path) != record["sha256"] or sha256(vp) != record["verifier_sha256"]
                    or verifier.get("passed") is not True or verifier.get("failures")
                    or verifier["verified_file_sha256"].get(str(path.resolve())) != record["sha256"]):
                raise ValueError("Reference raw not bound to its successful reader: " + str(path))
            paths[stage] = path
        data[arm] = read_window(paths, s)
    confirm = work / "outputs/data/confirm.h5"
    if sha256(confirm) != s["preserved_data_sha256"]["confirm.h5"]:
        raise ValueError("Changed fixed confirmation data")
    with h5py.File(confirm, "r") as stream:
        query = {task: np.asarray(stream[task + "/identities"][:, -1, 0]) for task in TASKS}
    for arm in ARMS:
        paths = {stage: work / f"outputs/s5/{arm}/confirm/{arm}_{stage:011d}.npz" for stage in STAGES}
        data[arm] = read_window(paths, s)
    return data, query


def calculate(data: dict[str, Any], query: dict[str, Any], s: dict[str, Any]) -> dict[str, Any]:
    descriptions = {arm: {task: {metric: clustered(value, query[task]) for metric, value in group.items()}
                          for task, group in tasks.items()} for arm, tasks in data.items()}
    directions = [(arm, ref) for arm in ARMS for ref in REFERENCES]
    directions += [(ref, arm) for arm in ARMS for ref in ("C25", "P25")]
    directions += [("REPLAY", "L2"), ("L2", "REPLAY")]
    paired = {left + "-" + right: {task: {metric: clustered(data[left][task][metric] - data[right][task][metric], query[task])
                                         for metric in METRICS} for task in TASKS} for left, right in directions}
    primary_items = [test_gain(paired[arm + "-A"][task]["acc"], s["min_icl_gain"], arm + "-A/" + task + "/acc")
                     for arm in ARMS for task in ("icl", "flip_icl")]
    primary = holm(primary_items, s["alpha"])
    primary_per_arm = {arm: holm(primary_items[index * 2:index * 2 + 2], s["alpha"]) for index, arm in enumerate(ARMS)}
    gates = {}
    for left, right in directions:
        name = left + "-" + right
        tasks = ("iwl_copy_avail", "train_eval") if right == "A" else TASKS
        tests = {task + "/" + metric: noninferiority(paired[name][task][metric], metric, s)
                 for task in tasks for metric in ("acc", "loss")}
        gates[name] = {"tests": tests, "all_passed": all(item["passed"] for item in tests.values())}
    value_items = {"P25_advantage": [], "baseline_advantage": []}
    directional = {}
    for arm in ARMS:
        for left, right, family in (("P25", arm, "P25_advantage"), (arm, "P25", "baseline_advantage")):
            name = left + "-" + right
            tests = [test_gain(paired[name][task]["acc"], 0., name + "/" + task + "/acc") for task in ("icl", "flip_icl")]
            directional[name] = holm(tests, s["alpha"])
            value_items[family].extend(tests)
    value_families = {name: holm(items, s["alpha"]) for name, items in value_items.items()}
    def family_pass(family: dict[str, Any], prefix: str) -> bool:
        tests = [item for item in family["tests"] if item["name"].startswith(prefix + "/")]
        return len(tests) == 2 and all(item["passed"] for item in tests)
    methods = {}
    for arm in ARMS:
        p0, p1 = "P25-" + arm, arm + "-P25"
        main_gain = family_pass(primary, arm + "-A")
        v0, v1 = family_pass(value_families["P25_advantage"], p0), family_pass(value_families["baseline_advantage"], p1)
        methods[arm] = {"joint_vs_A": {"gain_Holm4_passed": main_gain,
                           "noninferiority_four_passed": gates[arm + "-A"]["all_passed"],
                           "statistical_supported": main_gain and gates[arm + "-A"]["all_passed"],
                           "auxiliary_Holm2_gain_passed": primary_per_arm[arm]["all_passed"]},
                       "noninferior_to_P25": gates[p1]["all_passed"],
                       "noninferior_to_C25": gates[arm + "-C25"]["all_passed"],
                       "P25_extra_value": {"gain_Holm4_passed": v0,
                           "noninferiority_eight_passed": gates[p0]["all_passed"],
                           "statistical_supported": v0 and gates[p0]["all_passed"]},
                       "baseline_extra_value": {"gain_Holm4_passed": v1,
                           "noninferiority_eight_passed": gates[p1]["all_passed"],
                           "statistical_supported": v1 and gates[p1]["all_passed"]}}
    return {"descriptive": descriptions, "paired_comparisons": paired, "primary_Holm4": primary,
            "primary_Holm2_auxiliary": primary_per_arm, "noninferiority": gates,
            "value_Holm4": value_families, "directed_Holm2_zero_auxiliary": directional, "methods": methods}


def report(result: dict[str, Any]) -> str:
    display = ("icl", "flip_icl", "iwl_copy_avail", "train_eval")
    lines = ["# ICL 回放与平方 L2：匹配基线结果", "", "## Material Passport", "",
             "- 实验：exp034 / exp035，用户授权的两个固定系数基线。",
             "- 数据：init5/train2；同 SHA 的保护池、确认集及已核验 A/C25/P25 逐题参考。",
             "- 验证边界：保存产物和统计重算；没有独立重训或全量模型前向复现。", "",
             "统计结果待独立读取器核验。",
             "", "唯一 init5/train2，4.4M→10M；固定 9–10M 共 11 点先按行平均，再按真实 query 类聚类。",
             "本轮两新臂的收益主门统一 Holm4，逐臂 Holm2 只作原标准的辅助比较。", "",
             "| 组 | ICL准确率 | Flip准确率 | CIWL准确率 | 原任务准确率 |", "|---|---:|---:|---:|---:|"]
    for arm in (*REFERENCES, *ARMS):
        lines.append("| " + arm + " | " + " | ".join(f"{100*result['descriptive'][arm][task]['acc']['mean']:.3f}%" for task in display) + " |")
    lines += ["", "| 组 | ICL CE | Flip CE | CIWL CE | 原任务 CE |", "|---|---:|---:|---:|---:|"]
    for arm in (*REFERENCES, *ARMS):
        lines.append("| " + arm + " | " + " | ".join(f"{result['descriptive'][arm][task]['loss']['mean']:.6f}" for task in display) + " |")
    lines += ["", "所有 CE 均为完整 12800 标签输出的 nat。区间为普通 95% 区间，非 Holm 同时区间。", "",
              "| 对比 | ICL差（pp，95%CI） | Flip差（pp，95%CI） | CIWL差（pp） | 原任务差（pp） |", "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        for ref in REFERENCES:
            values = result["paired_comparisons"][arm + "-" + ref]
            cells = []
            for task in display:
                st = values[task]["acc"]
                cell = f"{100*st['mean']:+.3f}"
                if task in ("icl", "flip_icl") and st["valid"]:
                    cell += f" [{100*st['ci95_lower']:+.3f}, {100*st['ci95_upper']:+.3f}]"
                cells.append(cell)
            lines.append("| " + arm + "−" + ref + " | " + " | ".join(cells) + " |")
    lines += ["", "| 臂 | 相对A收益Holm4 | 相对A非劣4项 | 联合保留 | 相对P25非劣8项 | 相对C25非劣8项 | P25额外价值 | 基线反向额外价值 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    yes = lambda value: "通过" if value else "未通过"
    for arm in ARMS:
        m = result["methods"][arm]
        values = [m["joint_vs_A"]["gain_Holm4_passed"], m["joint_vs_A"]["noninferiority_four_passed"], m["joint_vs_A"]["statistical_supported"],
                  m["noninferior_to_P25"], m["noninferior_to_C25"], m["P25_extra_value"]["statistical_supported"], m["baseline_extra_value"]["statistical_supported"]]
        lines.append("| " + arm + " | " + " | ".join(yes(v) for v in values) + " |")
    lines += ["", "额外价值须 ICL/Flip 同时严格大于零、通过所属方向的全局 Holm4，且四任务八项非劣门全通过。未通过非劣门不等于证明劣势；双向联合门均失败时不能称任一方向整体优越。非劣不等于严格等价。",
              "", "REPLAY 每步为正常 32 条、ICL 32 条、Flip 32 条逐序列等权平均；固定池为每任务 4096 条，总 8192 条。它增加直接监督。L2 为原 Adam 下固定 1e-4 的全参数平方和，不含 1/2、无参数数目归一化，无 BatchNorm 参数。两臂不扫系数。",
              "", "数值核验范围：生产训练使用作者 default matmul；每步固定返回真正送入 Adam 的梯度、实际 logits 和分项。读取器从五次审计保存的真实梯度正向重建 Adam，并重算实际 logits 的完整标签 CE。另一个独立进程在 float32 matmul 下核验联合目标与独立数学分解；它不证明 default 生产梯度与另一编译图逐位一致。首次 default 跨编译图预检失败继续保留，不能称该旧门已通过。",
              "", "完整配对差、CE区间、各项 p 值、Holm校正、双向非劣和辅助指标见 analysis.json。所有结论只适用于该训练种子、数据与固定窗口；产物读取核验不是独立重训或全量模型前向复现。"]
    return "\n".join(lines) + "\n"


def analyze(work: Path = WORK) -> dict[str, Any]:
    s = load(work / "settings.json")
    validate_settings(s)
    data, query = input_data(work, s)
    result = {"experiment": "exp034_exp035", "analysis_scope": "single_seed_matched_baseline_pilot",
              "status": "analysis_complete_pending_independent_readback", "window": s["late_window"],
              "stages": list(STAGES), "n_per_task": s["n"], "cross_seed_confirmation_performed": False,
              "code_manifest_sha256": sha256(work / "code_manifest.json"),
              "settings_sha256": sha256(work / "settings.json"),
              "protocol_sha256": sha256(work / "protocol.md"),
              "analysis_plan_sha256": sha256(work / "analysis_plan.md"),
              "references_manifest_sha256": sha256(work / "references_manifest.json"),
              "confirm_data_sha256": sha256(work / "outputs/data/confirm.h5"),
              "fixed_hyperparameters": {key: s[key] for key in ("replay_coefficient", "replay_loss_normalizer", "l2_coefficient", "l2_form", "l2_optimizer")},
              "claim_boundary": "One fixed training seed, two fixed coefficients and a prespecified 11-point window. Conditional evaluation-row uncertainty, not cross-seed inference; noninferiority is not equivalence; no superiority is inferred from failed noninferiority."}
    if (work / "postprocess_manifest.json").exists():
        result["postprocess_manifest_sha256"] = sha256(work / "postprocess_manifest.json")
    result.update(calculate(data, query, s))
    (work / "outputs/analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (work / "outputs/report.md").write_text(report(result), encoding="utf-8")
    print(json.dumps({"analysis_complete": True, "methods": result["methods"]}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=WORK)
    analyze(parser.parse_args().work.resolve())

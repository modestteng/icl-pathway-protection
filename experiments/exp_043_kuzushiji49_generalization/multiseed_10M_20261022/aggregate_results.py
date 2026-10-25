#!/usr/bin/env python3
"""Aggregate two frozen evaluations using five paired training seeds.

Only standard-library modules and scipy.stats are imported. Per-query-class
intervals and ``primary_standing`` from older single-trajectory analyses are
deliberately not used as multi-seed evidence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from scipy.stats import t


SEEDS = ("s18", "s19", "s20", "s21", "s22")
ARMS = ("A", "P25")
METRICS = ("acc", "ce", "pair_acc")
STAGE = 10_000_000
MODEL_IDENTITIES = {
    "s18": {
        "A": "15bd62fee78388da39803f1c771086ef04dd600d7b9fece30a065910264fc1e2",
        "P25": "a8eccea9f6a372a3725907b64083d4d0e4eec56eb58e51ff78a25a3b171afd9e",
    },
    "s19": {
        "A": "7399d650d47082223b4be6aca52635d32ed6d489f9f881017023474eaeea2bb0",
        "P25": "a923db73a9baa33b9e2535ec2a5e45cb0986db247ee1a6e1c6026c19f84228be",
    },
    "s20": {
        "A": "c5b8c7deaff2d6c9658f0d095876cb26ddce5a0ea435c4bf9b59d6622d0bd8b9",
        "P25": "e28e581576a999ac9afc102c387a18d90cc3356b99c5b37cbd981f72f8a00728",
    },
    "s21": {
        "A": "a57674449714e8fcc06931cdf84228dd2612ccd9d7da64dc52090a7ff311ae5d",
        "P25": "61bf66dab78a102ca77f6cf6465624ddad7a6d55b2a4a757e3341cf1d42450e3",
    },
    "s22": {
        "A": "04e1c392cdb605fb283b736b01a12566ec25aa169b64b4178d0adf2d673f1e98",
        "P25": "dee7843cff34c0fc6e8f2fa39f1f1b43014f0c3657d20a87801f697fbc01f05c",
    },
}
DATASETS = {
    "k49": {"name": "Kuzushiji-49", "classes": 49, "per_class": 50},
    "cifar100": {"name": "CIFAR-100", "classes": 100, "per_class": 24},
}
REQUIRED_DOCUMENTS = (
    "settings.json", "outputs/analysis.json", "outputs/verification.json",
    "outputs/runtime_complete.json", "outputs/data_manifest.json",
    "outputs/encoding_receipt.json", "outputs/A_evaluation_receipt.json",
    "outputs/P25_evaluation_receipt.json",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def valid_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def read_documents(root):
    documents, evidence = {}, {}
    for name in REQUIRED_DOCUMENTS:
        path = root / name
        require(path.is_file(), "Missing required evidence: " + str(path))
        with path.open(encoding="utf-8") as stream:
            documents[name] = json.load(stream)
        evidence[name] = {"path": str(path.resolve()), "sha256": digest(path)}
    return documents, evidence


def finite_number(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), "Nonfinite or nonnumeric metric: " + label)
    return float(value)


def seed_summary(values, intervals=False):
    require(len(values) == len(SEEDS), "All five seed values are required")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    se = sd / math.sqrt(len(SEEDS))
    result = {"mean": mean, "sample_sd": sd, "se": se, "n_training_seeds": 5}
    if intervals:
        for key, probability, confidence in (
                ("ci95", .975, .95),
                ("bonferroni_two_datasets_ci97_5", .9875, .975)):
            critical = float(t.ppf(probability, 4))
            radius = critical * se
            result[key] = {
                "lower": mean - radius, "upper": mean + radius,
                "individual_confidence": confidence, "df": 4,
                "critical_t": critical,
            }
        result["bonferroni_two_datasets_ci97_5"].update(
            family_confidence=.95, family_size=2,
            family_scope="the two datasets, separately within this metric")
    return result


def validate_seed(dataset_key, dataset_root, seed):
    spec = DATASETS[dataset_key]
    root = dataset_root / seed
    docs, evidence = read_documents(root)
    cfg = docs["settings.json"]
    analysis = docs["outputs/analysis.json"]
    verified = docs["outputs/verification.json"]
    complete = docs["outputs/runtime_complete.json"]
    data = docs["outputs/data_manifest.json"]
    encoder = docs["outputs/encoding_receipt.json"]
    context = dataset_key + "/" + seed

    require(cfg.get("dataset_kind") == dataset_key, context + ": wrong dataset")
    require(cfg.get("stage") == STAGE, context + ": settings stage is not 10M")
    require(cfg.get("classes") == spec["classes"], context + ": class count")
    require(cfg.get("support_per_class") == cfg.get("query_per_class")
            == spec["per_class"], context + ": support/query counts")
    require(cfg.get("forward_seed") == 0, context + ": forward seed changed")
    require(set(cfg.get("models", {})) == set(ARMS), context + ": A/P25 required")
    for arm in ARMS:
        require(cfg["models"][arm].get("sha256") == MODEL_IDENTITIES[seed][arm],
                context + ": unexpected " + arm + " checkpoint identity")

    require(complete.get("status") == "completed" and complete.get("exit_code") == 0,
            context + ": no successful runtime completion")
    phases = complete.get("phases")
    require(isinstance(phases, list) and phases, context + ": phase receipts missing")
    phase_names = [phase.get("phase") for phase in phases]
    require(len(phase_names) == len(set(phase_names)), context + ": duplicate phases")
    require({"evaluate_A", "evaluate_P25", "verify"}.issubset(phase_names),
            context + ": evaluation/verification phase missing")
    for phase in phases:
        require(phase.get("exit_code") == 0 and not phase.get("stop_reason")
                and phase.get("remaining_group_members") == [],
                context + ": unsuccessful or unclosed phase " + str(phase.get("phase")))
    require(verified.get("passed") is True and verified.get("failures") == []
            and verified.get("independent_reader") is True,
            context + ": independent verification did not pass")
    require(verified.get("source_manifest_sha256") == complete.get("code_manifest_sha256")
            and valid_sha(verified.get("source_manifest_sha256")),
            context + ": verification/runtime code identity mismatch")

    require(encoder.get("status") == "complete", context + ": encoder incomplete")
    require(encoder.get("parameters_before") == encoder.get("parameters_after")
            and bool(encoder.get("parameters_before")), context + ": encoder changed")
    require(encoder.get("weights_sha256") == encoder.get("original_weights_sha256")
            and valid_sha(encoder.get("weights_sha256")), context + ": encoder identity")
    features_sha = encoder.get("features_sha256")
    episodes_sha = data.get("episodes_sha256")
    require(valid_sha(features_sha) and valid_sha(episodes_sha),
            context + ": shared input hashes missing")
    n_queries = spec["classes"] * spec["per_class"]
    require(encoder.get("shape") == [n_queries * 2, 512],
            context + ": encoded feature shape")
    require(analysis.get("stage") == STAGE and analysis.get("n_queries") == n_queries
            and analysis.get("n_classes") == spec["classes"]
            and analysis.get("training_trajectories") == 1,
            context + ": analysis stage, sample count or trajectory scope")

    metrics = {arm: {} for arm in ARMS}
    for arm in ARMS:
        evaluation = docs["outputs/" + arm + "_evaluation_receipt.json"]
        require(evaluation.get("status") == "complete" and evaluation.get("arm") == arm,
                context + ": evaluation receipt status/arm")
        require(evaluation.get("checkpoint_sha256") == MODEL_IDENTITIES[seed][arm],
                context + ": evaluated checkpoint identity mismatch")
        require(evaluation.get("stage") == STAGE and evaluation.get("rows") == n_queries
                and evaluation.get("full_label_count") == 12800,
                context + ": evaluation stage/rows/output space")
        require(evaluation.get("features_sha256") == features_sha
                and evaluation.get("episodes_sha256") == episodes_sha,
                context + ": A/P25 input mismatch")
        for metric in METRICS:
            value = finite_number(analysis["models"][arm][metric]["mean"],
                                  context + "/" + arm + "/" + metric)
            require(value >= 0 and (metric == "ce" or value <= 1),
                    context + ": metric out of range")
            metrics[arm][metric] = value
    differences = {}
    for metric in METRICS:
        delta = metrics["P25"][metric] - metrics["A"][metric]
        saved = finite_number(analysis["P25_minus_A"][metric]["mean"],
                              context + "/P25_minus_A/" + metric)
        require(math.isclose(delta, saved, rel_tol=0, abs_tol=1e-10),
                context + ": saved paired difference disagrees with arm means")
        differences[metric] = delta
    counts = analysis.get("paired_full_accuracy_counts", {})
    count_keys = ("A_wrong_P25_correct", "A_correct_P25_wrong", "both_correct", "both_wrong")
    require(all(isinstance(counts.get(key), int) and not isinstance(counts[key], bool)
                and counts[key] >= 0 for key in count_keys), context + ": paired counts")
    require(sum(counts[key] for key in count_keys) == n_queries, context + ": paired total")
    require(math.isclose((counts[count_keys[0]] - counts[count_keys[1]]) / n_queries,
                         differences["acc"], rel_tol=0, abs_tol=1e-10),
            context + ": paired accuracy count identity")
    # Any locally present compact input files are bound again. These can be
    # omitted from a receipt-only transfer; the independent full reader above
    # remains the evidence for large logits and remote checkpoint contents.
    for filename, expected in (("features.h5", features_sha), ("episodes.npz", episodes_sha)):
        candidate = root / "outputs" / filename
        if candidate.exists():
            require(digest(candidate) == expected, context + ": local " + filename + " changed")
    return {
        "seed": seed, "seed_identity": {"init": int(seed[1:]), "train": int(seed[1:]) - 3},
        "stage": STAGE, "checkpoint_sha256": MODEL_IDENTITIES[seed],
        "models": metrics, "P25_minus_A": differences,
        "n_queries": n_queries, "features_sha256": features_sha,
        "episodes_sha256": episodes_sha, "encoder_weights_sha256": encoder["weights_sha256"],
        "evidence": evidence,
    }


def aggregate_dataset(key, root):
    rows = [validate_seed(key, root, seed) for seed in SEEDS]
    for identity in ("features_sha256", "episodes_sha256", "encoder_weights_sha256"):
        require(len({row[identity] for row in rows}) == 1,
                key + ": " + identity + " differs across training seeds")
    return {
        "dataset_name": DATASETS[key]["name"], "n_training_seeds": len(SEEDS),
        "seed_ids": list(SEEDS), "n_queries_per_seed": rows[0]["n_queries"],
        "features_sha256": rows[0]["features_sha256"],
        "episodes_sha256": rows[0]["episodes_sha256"],
        "encoder_weights_sha256": rows[0]["encoder_weights_sha256"],
        "models": {arm: {metric: seed_summary([row["models"][arm][metric] for row in rows])
                         for metric in METRICS} for arm in ARMS},
        "P25_minus_A": {metric: seed_summary([row["P25_minus_A"][metric] for row in rows],
                                             intervals=True) for metric in METRICS},
        "per_seed": rows,
    }


def report_text(result):
    lines = ["# 两个外部数据集：五组10M模型冻结评测", "",
             "统计单位为五条独立训练轨迹 s18–s22（每条轨迹内 A/P25 配对）。"
             "A/P25 均值和样本标准差先在每个 seed 内汇总固定题目，再跨训练 seed 计算。", "",
             "主指标在完整 12,800 标签空间计分；二候选准确率仅为辅助。"
             "普通差值区间为95%双侧 t 区间（df=4）。两个数据集的主准确率差值另给"
             "Bonferroni 家族95%同时区间，每个区间置信度97.5%，临界值 t.ppf(0.9875,4)。"
             "CE与二候选差值也列同样的逐指标区间，但不声称所有指标合并构成一个95%家族。", ""]
    for key in DATASETS:
        dataset = result["datasets"][key]
        lines += ["## " + dataset["dataset_name"], "",
                  f"每个 seed 共用同一组 {dataset['n_queries_per_seed']} 个查询、相同特征和上下文。", "",
                  "| 指标 | A：均值 ± seed样本SD | P25：均值 ± seed样本SD | P25−A：均值 ± seed样本SD | 差值普通95%区间 | 差值两数据集家族95%区间 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for metric, title, scale in (("acc", "完整准确率（% / pp）", 100),
                                     ("ce", "完整交叉熵（nat）", 1),
                                     ("pair_acc", "二候选准确率：辅助（% / pp）", 100)):
            a, p = (dataset["models"][arm][metric] for arm in ARMS)
            d = dataset["P25_minus_A"][metric]
            ci, simultaneous = d["ci95"], d["bonferroni_two_datasets_ci97_5"]
            lines.append(
                f"| {title} | {a['mean']*scale:.4f} ± {a['sample_sd']*scale:.4f} | "
                f"{p['mean']*scale:.4f} ± {p['sample_sd']*scale:.4f} | "
                f"{d['mean']*scale:+.4f} ± {d['sample_sd']*scale:.4f} | "
                f"[{ci['lower']*scale:+.4f}, {ci['upper']*scale:+.4f}] | "
                f"[{simultaneous['lower']*scale:+.4f}, {simultaneous['upper']*scale:+.4f}] |")
        lines += ["", "完整标签计分的逐 seed 结果：", "",
                  "| seed（init / train） | A准确率 % | P25准确率 % | 差值 pp | A交叉熵 | P25交叉熵 | 差值 nat |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in dataset["per_seed"]:
            a, p, d = row["models"]["A"], row["models"]["P25"], row["P25_minus_A"]
            identity = row["seed_identity"]
            lines.append(f"| {row['seed']}（{identity['init']} / {identity['train']}） | "
                         f"{a['acc']*100:.4f} | {p['acc']*100:.4f} | {d['acc']*100:+.4f} | "
                         f"{a['ce']:.6f} | {p['ce']:.6f} | {d['ce']:+.6f} |")
        lines += ["", "二候选辅助计分的逐 seed 结果：", "",
                  "| seed | A % | P25 % | 差值 pp |", "|---|---:|---:|---:|"]
        for row in dataset["per_seed"]:
            lines.append(f"| {row['seed']} | {row['models']['A']['pair_acc']*100:.4f} | "
                         f"{row['models']['P25']['pair_acc']*100:.4f} | "
                         f"{row['P25_minus_A']['pair_acc']*100:+.4f} |")
        lines += ["", "共享特征 SHA-256：`" + dataset["features_sha256"] + "`。",
                  "共享题目 SHA-256：`" + dataset["episodes_sha256"] + "`。", ""]
    lines += ["## 核验范围与结论边界", ""]
    lines += ["- " + item for item in result["limitations"]]
    lines += ["", "逐 seed 模型哈希、输入文件路径和读取时SHA-256见 combined_analysis.json。"
              "本汇总不沿用单 seed 的按查询类别区间或 primary_standing 作为整体判据，也不新增p值。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k49-root", type=Path, required=True)
    parser.add_argument("--cifar-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    datasets = {
        "k49": aggregate_dataset("k49", args.k49_root.resolve()),
        "cifar100": aggregate_dataset("cifar100", args.cifar_root.resolve()),
    }
    require(datasets["k49"]["encoder_weights_sha256"]
            == datasets["cifar100"]["encoder_weights_sha256"],
            "Encoder weight identity differs between datasets")
    limitations = [
        "原 ResNet18 编码器和 A/P25 Transformer 全部冻结，没有在这两个目标数据集上更新参数；"
        "编码器参数前后身份及 Transformer 检查点身份由逐 seed 完成/核验收据绑定。",
        "原编码器使用 ImageNet 预训练权重，未排除与外部图像数据的预训练内容或语义重叠；"
        "结果只检验这些冻结 Transformer 在指定新数据来源上的上下文使用能力。",
        "五条训练轨迹是主统计单位；查询类别、题目和两个数据集均不充当额外训练 seed。"
        "固定支持池、查询和上下文的抽样变异未由跨训练seed区间全面覆盖。",
        "t区间采用五个配对训练seed差值，df=4；小样本正态近似与固定评价集条件需要保留。"
        "同时区间仅针对每个指标内的两个数据集，不扩展为全部指标或所有分析的联合覆盖。",
        "五组模型为 exp042 s18–s22 的10M端点。旧 EMNIST 单轨迹结果使用不同模型，"
        "不能并入本次五seed汇总，也不能作为本模型组的第三个配对数据集。",
        "本脚本复核全部必要收据、指标恒等式和存在于本地的紧凑输入文件；"
        "完整原始图像/logits读取和模型文件哈希检查依赖已通过的独立CPU验证收据，"
        "本汇总本身不是第二次模型前向或编码器重跑。",
    ]
    result = {
        "schema": "external_dataset_multiseed_10M_v1", "passed": True,
        "utc": datetime.now(timezone.utc).isoformat(), "stage": STAGE,
        "training_seed_ids": list(SEEDS), "n_training_seeds": 5,
        "statistical_unit": "paired training seed", "full_label_count": 12800,
        "primary_metric": "full-label accuracy, P25 minus A",
        "secondary_metrics": ["full-label cross entropy", "two-candidate accuracy (auxiliary)"],
        "family_size": 2, "family_scope": "two datasets, separately within each metric",
        "p_values_computed": False, "datasets": datasets, "limitations": limitations,
    }
    # Validation of every seed and both complete datasets precedes output writes.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = args.output_dir / "combined_analysis.json"
    report_path = args.output_dir / "combined_report.md"
    for path, payload in ((analysis_path, json.dumps(result, ensure_ascii=False,
                                                   indent=2, allow_nan=False) + "\n"),
                          (report_path, report_text(result))):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
    print(json.dumps({"passed": True, "n_training_seeds": 5,
                      "analysis": str(analysis_path.resolve()),
                      "report": str(report_path.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()

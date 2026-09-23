"""Summarize the frozen matrix from private evaluation JSONs, without inference.

API: summarize(out, protocol) -> JSON-serializable dict. Writes summary.json and
SUMMARY_ZH.md only after every declared fit and its four evaluations are valid.
No raw industrial arrays, checkpoints, training, or scientific significance
tests are needed. Physical metrics are never pooled across tasks or targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TERMINAL = {"EARLY_STOPPED", "BUDGET_EXHAUSTED"}
PHYSICAL_METRICS = ("MAE", "RMSE", "persistence_MAE", "increment_MAE")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_text(path, content):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _number(value, label, *, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"Expected finite metric {label}: {value!r}")
    if value < 0:
        raise ValueError(f"Expected nonnegative error metric {label}")
    return float(value)


def _count(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Expected nonnegative integer {label}")
    return value


def _stats(values, seeds):
    """Preserve missing values and expose n; do not turn undefined into zero."""
    usable = [float(values[s]) for s in seeds if values[s] is not None]
    return {"mean": statistics.mean(usable) if usable else None,
            "sample_std": statistics.stdev(usable) if len(usable) > 1 else None,
            "n": len(usable), "expected_n": len(seeds),
            "complete": len(usable) == len(seeds),
            "per_seed": {str(s): values[s] for s in seeds}}


def _validate_report(report, label):
    _number(report["target_macro_normalized_mse"], label + "/normalized_mse")
    if _count(report["origins"], label + "/origins") == 0:
        raise ValueError("Empty evaluation: " + label)
    if _count(report["horizon_steps"], label + "/horizon_steps") == 0:
        raise ValueError("Empty horizon: " + label)
    if not report["targets"]:
        raise ValueError("Missing targets: " + label)
    for name, target in report["targets"].items():
        if _count(target["count"], name + "/count") == 0:
            raise ValueError("Target has no eligible labels: " + name)
        for metric in PHYSICAL_METRICS:
            _number(target.get(metric), name + "/" + metric, optional=metric == "increment_MAE")
        increment_count = _count(target.get("increment_count", 0), name + "/increment_count")
        if (increment_count == 0) != (target.get("increment_MAE") is None):
            raise ValueError("Increment metric/count disagree: " + name)
        for lead, endpoint in target.get("endpoints", {}).items():
            count = _count(endpoint["count"], name + "/endpoint/" + lead)
            for metric in ("MAE", "RMSE"):
                value = _number(endpoint[metric], name + "/endpoint/" + metric, optional=count == 0)
                if count == 0 and value is not None:
                    raise ValueError("Zero-label endpoint must be undefined: " + name)


def _same_support(reference, report, label):
    """Counts/roles are necessary support checks; array identities stay in runner."""
    for key in ("origins", "horizon_steps"):
        if reference[key] != report[key]:
            raise ValueError(f"Evaluation support mismatch ({key}): {label}")
    if set(reference["targets"]) != set(report["targets"]):
        raise ValueError("Evaluation target mismatch: " + label)
    for name, ref in reference["targets"].items():
        target = report["targets"][name]
        if (ref["count"], ref.get("increment_count", 0)) != (target["count"], target.get("increment_count", 0)):
            raise ValueError("Evaluation mask counts mismatch: " + label)
        if not math.isclose(ref["persistence_MAE"], target["persistence_MAE"], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Persistence baseline mismatch: " + label)
        if set(ref.get("endpoints", {})) != set(target.get("endpoints", {})):
            raise ValueError("Endpoint support mismatch: " + label)
        for lead, endpoint in ref.get("endpoints", {}).items():
            if endpoint["count"] != target["endpoints"][lead]["count"]:
                raise ValueError("Endpoint mask count mismatch: " + label)


def _fit_summary(path, task, arm, seed, state):
    row = {"task": task, "arm": arm, "seed": seed, "status": state["status"],
           "epoch": state.get("epoch"), "epoch_cap_reached": state["status"] == "BUDGET_EXHAUSTED",
           "best_epoch": state.get("best_epoch"), "checkpoint_sha256": state.get("checkpoint_sha256"),
           "parameter_count": state.get("identity", {}).get("model_parameters")}
    ledger = path / "ledger.jsonl"
    if ledger.is_file():
        records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
        improved = [entry for entry in records if entry.get("improved")]
        if row["best_epoch"] is None and improved:
            row["best_epoch"] = improved[-1]["epoch"]
        row["logged_epochs"] = len(records)
    return row


def _format(stat):
    if stat["mean"] is None:
        return "未定义"
    result = f'{stat["mean"]:.6g}'
    if stat["sample_std"] is not None:
        result += f' ± {stat["sample_std"]:.4g}'
    if not stat["complete"]:
        result += f'（有效种子 {stat["n"]}/{stat["expected_n"]}）'
    return result


def _markdown(summary):
    lines = ["# 三任务统一模型：私有结果汇总", "", f'实验：`{summary["experiment_id"]}`。', "",
             "表中为预先声明种子的均值 ± 样本标准差（ddof=1）。物理误差逐目标保留，未跨任务或目标混合单位。",
             "同种子差值定义为 mechanism − mechanism_no_slow；误差差值为负表示完整模型误差较低。未进行显著性推断。",
             "H180 是 H60 训练后的长时域外推；直接模型沿用同一个连续时距查询头。历史测试仅用于冻结矩阵的最终评价。", "",
             f'完成拟合 {summary["fits_completed"]}；达到 epoch 上限 {summary["epoch_cap_count"]}。'
             " BUDGET_EXHAUSTED 表示预算用尽，不能据此写成已收敛；EARLY_STOPPED 表示满足注册的早停规则。", ""]
    for section in summary["sections"]:
        lines += [f'## {section["task"]} / {section["split"]} / {section["profile"]}', "",
                  f'起点数：{section["origins"]}；预测步数：{section["horizon_steps"]}。', "",
                  "| 模型 | 目标宏平均 normalized MSE | epoch上限种子数 |", "|---|---:|---:|"]
        for arm, group in section["arms"].items():
            lines.append(f'| {arm} | {_format(group["target_macro_normalized_mse"])} | {group["epoch_cap_count"]} |')
        unit = "°C（配置约定）" if section["task"] in {"main_steam", "reheat_steam"} else "原记录单位"
        lines += ["", f'物理误差单位：{unit}。increment_MAE 对应相邻 10 秒预测步的变化误差。', "",
                  "| 目标 | 模型 | MAE | RMSE | 持续值MAE | 增量MAE |", "|---|---|---:|---:|---:|---:|"]
        for arm, group in section["arms"].items():
            for name, target in group["targets"].items():
                columns = [_format(target[metric]) for metric in PHYSICAL_METRICS]
                lines.append(f'| {name} | {arm} | ' + " | ".join(columns) + " |")
        paired = section.get("paired_mechanism_minus_no_slow")
        if paired:
            lines += ["", f'配对 normalized MSE 差值：{_format(paired["target_macro_normalized_mse"])}。', "",
                      "| 目标 | 配对MAE差 | 配对RMSE差 | 配对增量MAE差 |", "|---|---:|---:|---:|"]
            for name, target in paired["targets"].items():
                lines.append(f'| {name} | ' + " | ".join(_format(target[m]) for m in ("MAE", "RMSE", "increment_MAE")) + " |")
        lines.append("")
    lines += ["## 训练终态", "", "| 任务 | 模型 | 种子 | 终态 | 已完成epoch | 最佳epoch |",
              "|---|---|---:|---|---:|---:|"]
    for row in summary["fits"]:
        lines.append(f'| {row["task"]} | {row["arm"]} | {row["seed"]} | {row["status"]} | '
                     f'{row["epoch"] if row["epoch"] is not None else "未记录"} | '
                     f'{row["best_epoch"] if row["best_epoch"] is not None else "未记录"} |')
    lines += ["", "逐种子原值、端点指标、标签数量及输入文件哈希保存在同目录 summary.json。"
              "这些结果用于预测和模块比较；本轮没有据此报告现场因果响应、闭环效果或跨任务权重迁移。", ""]
    return "\n".join(lines)


def summarize(out, protocol):
    """Validate complete matrix, aggregate same-seed results, write private files."""
    out = Path(out)
    seeds, tasks, arms = list(protocol["seeds"]), list(protocol["tasks"]), list(protocol["arms"])
    for label, values in (("seeds", seeds), ("tasks", tasks), ("arms", arms)):
        if not values or len(values) != len(set(values)):
            raise ValueError("Empty or duplicate " + label)
    if len(seeds) < 2:
        raise ValueError("At least two seeds are needed for sample standard deviation")
    expected_fits = len(seeds) * len(tasks) * len(arms)
    if protocol.get("fit_count", expected_fits) != expected_fits:
        raise ValueError("fit_count does not match declared matrix")
    profiles = list(dict.fromkeys((protocol["primary_profile"], protocol["long_profile"])))
    reports, fits, source_files = {}, [], []
    for task in tasks:
        for arm in arms:
            for seed in seeds:
                path = out / "runs" / f"{task}__{arm}__s{seed}"
                status = _read(path / "status.json")
                if status["status"] not in TERMINAL:
                    raise ValueError("Incomplete fit: " + path.name)
                identity = status.get("identity", {})
                for key, expected in (("task", task), ("arm", arm), ("seed", seed)):
                    if key in identity and identity[key] != expected:
                        raise ValueError("Fit identity mismatch: " + path.name)
                fits.append(_fit_summary(path, task, arm, seed, status))
                for split in ("validation", "historical_test"):
                    for profile in profiles:
                        file = path / f"evaluation_{split}_{profile}.json"
                        report = _read(file)
                        _validate_report(report, str(file))
                        expected_horizon = protocol.get("training_horizon" if profile == protocol["primary_profile"] else "long_horizon")
                        if expected_horizon is not None and report["horizon_steps"] != expected_horizon:
                            raise ValueError("Evaluation horizon disagrees with profile: " + str(file))
                        reports[task, split, profile, arm, seed] = report
                        source_files.append({"path": file.relative_to(out).as_posix(),
                                             "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
    fit_index = {(f["task"], f["arm"], f["seed"]): f for f in fits}
    sections = []
    for task in tasks:
        for split in ("validation", "historical_test"):
            for profile in profiles:
                reference = reports[task, split, profile, arms[0], seeds[0]]
                groups = {}
                for arm in arms:
                    entries = {seed: reports[task, split, profile, arm, seed] for seed in seeds}
                    for seed, entry in entries.items():
                        _same_support(reference, entry, f"{task}/{split}/{profile}/{arm}/{seed}")
                    targets = {}
                    for name, ref_target in reference["targets"].items():
                        target = {"count": ref_target["count"], "increment_count": ref_target.get("increment_count", 0)}
                        for metric in PHYSICAL_METRICS:
                            target[metric] = _stats({s: entries[s]["targets"][name].get(metric) for s in seeds}, seeds)
                        target["endpoints"] = {
                            lead: {"count": endpoint["count"], **{
                                metric: _stats({s: entries[s]["targets"][name]["endpoints"][lead][metric] for s in seeds}, seeds)
                                for metric in ("MAE", "RMSE")}}
                            for lead, endpoint in ref_target.get("endpoints", {}).items()}
                        targets[name] = target
                    groups[arm] = {"target_macro_normalized_mse": _stats({s: entries[s]["target_macro_normalized_mse"] for s in seeds}, seeds),
                                   "epoch_cap_count": sum(fit_index[task, arm, s]["epoch_cap_reached"] for s in seeds), "targets": targets}
                section = {"task": task, "split": split, "profile": profile,
                           "origins": reference["origins"], "horizon_steps": reference["horizon_steps"], "arms": groups}
                if "mechanism" in groups and "mechanism_no_slow" in groups:
                    def difference(metric, target=None):
                        values = {}
                        for seed in seeds:
                            full = reports[task, split, profile, "mechanism", seed]
                            ablated = reports[task, split, profile, "mechanism_no_slow", seed]
                            if target is not None:
                                full, ablated = full["targets"][target], ablated["targets"][target]
                            left, right = full.get(metric), ablated.get(metric)
                            values[seed] = None if left is None or right is None else float(left) - float(right)
                        return _stats(values, seeds)
                    section["paired_mechanism_minus_no_slow"] = {
                        "direction": "mechanism minus mechanism_no_slow; negative error difference favors mechanism",
                        "target_macro_normalized_mse": difference("target_macro_normalized_mse"),
                        "targets": {name: {metric: difference(metric, name) for metric in PHYSICAL_METRICS}
                                    for name in reference["targets"]}}
                sections.append(section)
    summary = {"schema_version": 1, "experiment_id": protocol["experiment_id"],
               "generated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "COMPLETE_FROZEN_MATRIX_SUMMARY",
               "private_results": True, "seeds": seeds, "fits_completed": len(fits),
               "epoch_cap_count": sum(f["epoch_cap_reached"] for f in fits), "fit_status_counts": dict(Counter(f["status"] for f in fits)),
               "standard_deviation": "sample; ddof=1; seeds are the replicate unit",
               "physical_aggregation": "per target within task only; no cross-task physical metric pooling",
               "significance_inference": "not performed", "fits": fits, "sections": sections, "source_files": source_files}
    _atomic_text(out / "SUMMARY_ZH.md", _markdown(summary))
    _atomic_text(out / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    result = summarize(args.out, _read(args.protocol or args.out / "protocol.json"))
    print(json.dumps({"status": result["status"], "fits_completed": result["fits_completed"],
                      "epoch_cap_count": result["epoch_cap_count"]}))


if __name__ == "__main__":
    main()

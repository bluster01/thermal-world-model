# Thermal world model：GitHub 任务交接

2026-09-09 用户确认：本地 Windows 与远程 Linux 没有 SSH/组网，通过本仓库共享任务状态；Linux 可持续工作。研究主目标为实现 thermal world model，先使用原训练区间 1/10 数据做原型，Applied Energy 是后续发表目标。

当前可领取任务为 **thermal_world_model_tenth_preflight_v1**。这是实际可执行的环境/资产预检；训练代码与数据合同齐备后，另发有独立身份的冻结训练任务。预检成功不等于训练已运行。

## 领取与回传

1. 在对应 GitHub issue 评论 `CLAIMED: thermal_world_model_tenth_preflight_v1`。写入本次实际代码 commit、Python 版本和匿名 worker 标识即可；先检查是否已有未结束的领取记录。
2. 在独立 checkout/worktree 取 issue 指定的 commit，不覆盖 Linux 上已有实验或输出。不要重启旧 synthetic pilot 或旧 v0.7 test。
3. 在已有 Python 环境运行下列命令。三个路径参数对应 **Linux 机本地**的数据和物性资产；不需要传送原始数据。

```bash
python3 -m experiments.world_model_plant.preflight \
  --canonical "$TWM_CANONICAL" \
  --raw "$TWM_RAW_MERGED" \
  --properties "$TWM_PROPERTIES" \
  --output "$TWM_PREFLIGHT_OUTPUT"
```

`TWM_CANONICAL` 为冻结 canonical_sideA_v2.npz；`TWM_RAW_MERGED` 为原 all_merged_10s.csv；`TWM_PROPERTIES` 为 iapws_surrogate.npz。`TWM_PREFLIGHT_OUTPUT` 是**不存在的新目录**，其父目录必须存在。不确定路径时先省略相应参数，程序会如实回报缺失；不要用不明来源的同名文件替代。

4. `local_details.json` 留在 Linux。把 `public_receipt.json` 的内容作为 issue 评论回传：成功标 `PREFLIGHT_COMPLETE`；退出码2时标 `PREREQUISITE_MISSING` 并说明缺失项。程序不会安装包、读入电厂数组或启动训练。
5. 当前任务到此结束。主线程核查回执后，发布确切数据/代码哈希、预算与命令的训练任务。没有领取回执及真实运行日志时，不把任务写成 running。

## 已知资产身份

| 资产 | SHA-256 |
|---|---|
| canonical_sideA_v2.npz | `24da77960e05e3636cc7b97a60a75e9b4ba470a3abb4f8ba920ddf11c6dad1d0` |
| iapws_surrogate.npz | `9fd7a1dba96a5b968661f644fa185ac755266c85853d689d065432a2d41f6e92` |

预检只对上述文件核对字节身份，raw CSV 此阶段只核存在性。下一数据准备任务按时间有界读取旧训练侧：训练分母530,779，取前53,077个连续10s点，并在这段内重新划分原型训练/验证。它不是对全量数据每十行采样，不访问旧 reserved test，也不把旧验证集当成新独立测试。

## 状态语义

`QUEUED → CLAIMED → PREFLIGHT_COMPLETE / PREREQUISITE_MISSING` 属于本预检任务。未来训练任务使用单独的 `QUEUED → CLAIMED → RUNNING → RETURNED / FAILED → AUDITED`，返回内容须绑定代码 commit、数据身份、配置和 checkpoint。

超时或没有新评论不是失败证据，不应自动重跑。任务重复领取、源码/配置不符、已有输出冲突时保留原记录并回报；不能覆盖已完成结果。断点恢复须使用协议明定的同一次训练状态，不能悄悄从头换一次随机训练。

## 共享范围

本仓库公开。共享范围为实现代码、执行配置、任务状态及明确允许的汇总回执。原始 CSV、选取后的真实数据 NPZ、逐时刻测量/预测、含本地路径的详细报告及真实数据训练权重保持在各机器本地。不要使用 `git add .` 提交实验输出。

本地已完成的12模型合成 pilot 实际运行于 Windows 上的 Ubuntu WSL2，不能计作这台远程 Linux 的任务。该 pilot 的结果没有支持默认采用内容attention；目前正在检验固定总缓存128内的64观测/历史＋64预测配额，软件测试通过不等于已学出更好的模型。

实际阀位条件回放仍须与最终阀门指令控制区分。未确认的串级主调输出不能直接替代阀门开度命令。原型结果不自动证明现场干预效果、风险控制或跨机组通用性。

# Phase 3.5 MS3-R Gate C RM2 Linux 并行验证设计

## 1. 目的与证据边界

RM1-A 的真实 1/100 结果只完成架构归因 smoke：未观察到 residual capacity 增大导致 A1 response 单调坍缩，并证明 terminal-only 会牺牲中间 `Tin-Tout` 语义。RM2 不再由本地承担训练，而由 Hermes 在冻结完整 train/validation 支持域上验证跨 seed、跨时间 fold 的稳定性。本批仍是闭环观测 validation：可以支持 observed-policy prediction、扰动条件响应和结构稳健性，不能支持任意 `do(valve)`、喷水流量真值、独立 test 或闭环投运。

本地职责是冻结假设、代码、配置、产物合同与 Supervisor replay；Hermes 只运行最终授权提交，允许多 GPU 并行，不修改候选、预算、阈值、seed、fold 或文档。普通提交在 `linux_authorized_gate=null` 时必须 no-op；只有最终单一提交把 `ms3_r` 切到 `ready_for_linux`。

## 2. 54-run 封闭矩阵

每个候选运行 seeds `0/1/2` × expanding folds `F0/F1`，共 6 runs。

| 组 | 候选 | 作用 | runs |
|---|---|---|---:|
| A attribution | paired-free、additive-base、A1 scheduled-base、A1 scheduled-large | 响应、调度和 residual capacity 稳定性 | 24 |
| B operators | LPV-Koopman、PI-Neural-ODE、causal DeepONet；复用 A1-base | 表示路线稳定性，不按单 seed MAE设冠军 | 18 新 runs |
| C topology | common-only response、no-downstream-latent | differential 与 downstream latent 必要性 | 12 |

矩阵中 A1-base 只出现一次，总计 9 个唯一候选、54 runs。B 组比较共享 A1-base 作为参考，不重复生成 6 个 A1 runs。

## 3. 时间 fold 与信息流

原始 cache 按 60/20/20 划分。RM2 test 区间固定为最后 20%，任何代码不得生成其 anchors。

- `F0`: train `[0, 0.60N)`；validation `[0.60N, 0.70N)`；
- `F1`: train `[0, 0.70N)`；validation `[0.70N, 0.80N)`。

两个 fold 都采用 expanding history，validation window 不借用前一 split 的历史。训练 batch 从该 fold 全部合格 anchors 采样；train-only normalization/robust scales 使用冻结统计样本。selector 与最终 episode anchors 对所有候选和 seed 固定，避免候选间样本漂移。

forecast-boundary 不读取未来 Tin、terminal 或 logged valve；logged future valve 只进入 response auxiliary 和 validation diagnosis。oracle Tin 只作 ceiling。`common_only` 只限制显式 response 的 differential mode，不删除观测输出；`no_downstream_latent` 用无递归直接下游映射替代 latent mixer，不能偷读未来末温。

## 4. 训练与选择

统一 batch 128、最多 4000 optimizer updates、每 100 updates 在固定 selector anchors 上评估；至少 1000 updates 后才允许 patience=8 的 early stop。所有 run 使用 AdamW、gradient clipping 和相同 shared prediction score。不同监督目标的训练 loss 不跨候选比较。

checkpoint 先满足 finite rollout、SP prefix causality、constant-action identity、future-truth isolation；再最小化 validation shared score。paired-free 的 response non-collapse 标记为不适用。完整 checkpoint、train curve、selector history、最终聚合指标和压缩 episode trajectories 必须回传。

RM2 不设自动科学 PASS。任何单 run 或整组失败都原样记录，其他并行 run 可继续；失败 run 不自动重试。`--skip-complete` 只用于 webhook/进程恢复时跳过 ledger 已闭合的 run，不能覆盖或补训不完整 run。

## 5. 诊断与 Supervisor 判决

每个 final episode 保存 anchor/timestamp、future SP/valve、local/terminal truth、forecast prediction、logged/shuffled response prediction、predicted/logged effect。由本地统一计算：

- UTC 日与连续时间块的 paired uncertainty；
- 60/180 s 正确路径、错侧和 lead/placebo；
- opening/closing、负荷/压力工况不变性；
- common/differential action support；
- logged 相对 held/shuffled 的局部优势；
- terminal 改善是否以 Tin/local/valve 退化为代价；
- seed 与 fold 方向、时延和量级一致性。

operator 最多保留两条，但只有在跨 seed/fold 响应方向、时延、量级与 Gate B 一致且 prediction 不退化时才有资格。否则只保留预注册 A1-base 作为工程参考，不声称最优或完全物理辨识。

## 6. Hermes 并行边界

Runner 接受设备池并把 run specs 静态分片到长期 worker；每个 worker 只绑定一个 CUDA device，顺序执行自己的 runs。Hermes 可以调整并行设备列表、CPU 线程数和输出挂载路径，但不能调整科学矩阵。必须回传根 manifest、逐 run 目录、failure records、summary、checkpoint archive、stdout/stderr、环境、资源使用和 artifact ledger。Hermes 不运行 Supervisor audit、不访问 test、不更新 TODO/registry，不因结果不理想自行新增实验。

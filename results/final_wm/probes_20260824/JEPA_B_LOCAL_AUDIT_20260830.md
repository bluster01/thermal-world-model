# JEPA-B 系列本地实现审计（2026-08-30）

## 审计结论

`C0/B1/B2/B3/B3-SHUFFLE/B4` 已按预注册 v1 实现并达到 **READY FOR LINUX**。
本结论只覆盖代码、注册合同和合成数据身份门，不是实验效果结论。Linux 尚未回传真实
canonical v2.2 结果，test 与正式论文 verdict 继续锁定。

冻结 matrix SHA-256：
`b664c06272318775ad5aa89cc93c337c09a72806e5b16340552d536c66224751`。

## 原文核对与采用边界

- LeJEPA（arXiv:2511.08544v3）：采用无 EMA/stop-grad 的 SIGReg 思路；仓内实现明确是固定切片特征函数 Gaussian-CF adaptation，不声称逐行复现官方实现。
- LeWorldModel（arXiv:2603.19312v3）：采用 action-conditioned latent predictor 的职责划分，不替换现有 Fan2020-UDE 物理 transition。
- JEPA-x（arXiv:2608.24044v2）：采用双分支四路自/交叉预测、共享 action predictor 和部署丢弃特权分支；名称已从误写的 XP-JEPA 更正为 JEPA-x。B3 仅使用 canonical v2.2 已注册的 32 维富通道，并设置固定错配负控制。
- Phys-JEPA（arXiv:2606.16076v1）：采用物理/残差分解及 static/dynamic consistency；原文单种子、弱 descriptor 的证据限制已保留，不作项目效果背书。

## 关键合同审计

1. 数据：只允许侧 A canonical v2.2；完整窗口执行 A5 水煤比/负荷/燃料质量门；任何 test split 读取直接报错。
2. 防泄漏：特权归一化只拟合 train；B1/B3 representation target 与 B4 residual target encoder 不读未来动作。B4 确定性 physical anchor 仅使用同刻已记录动作完成物理反演。
3. 对照：六臂预算、seed0 与验证锚冻结；B3-SHUFFLE 使用 corpus-wide 固定无不动点错配，结构和预算与 B3 相同。
4. 身份门：B1/B2/B3/B3-SHUFFLE/B4 在机制关闭时均与 C0 温度 rollout 逐位相同。
5. 判定：只用 validation 的 H18 主门、H18 负荷稳健门和两阀 H18/H60 v0.3 方向门；H36/H60 漂移与 UTC-day bias 只报告，不事后增门。
6. 恢复：完整臂复用必须由 report、final ledger、checkpoint 同时匹配 arm、git commit 和 matrix SHA-256；半臂、空 ledger 或任一指纹不符均拒绝自动重跑。
7. 权限：registry 只授权 `jepa_b_series_v1` 单 GPU 顺序执行；seed 仅 0；自动重试、test、seeds 1/2 和论文 verdict 均未授权。

## 本地验证记录

- `python -m pytest tests/final_wm/test_jepa.py tests/phase35/test_experiment_status.py -q`：20 passed。
- `python -m pytest tests/final_wm -q`：181 passed。
- `python -m pytest tests/phase35 -q --import-mode=importlib`：472 passed。
- `python experiments/phase3_5/experiment_status.py --check --json`：valid，唯一 active/Linux gate 均为 `jepa_b_series`。
- `py_compile` 与 Linux runner `--help`：通过。

仓库根目录的裸 `pytest -q` 仍会在收集阶段遇到既有环境问题：历史探针硬编码 Linux
绝对路径、未随仓库提供的原始 CSV、不同目录同名测试模块，以及本地未跟踪 `tmp` 测试副本。
本次未修改旧测试发现规则；改用受影响测试域的完整回归。Windows 未保存真实
`canonical_sideA_v2.npz`/IAPWS 产物，因此真实 `--sanity` 和训练必须由 Linux 按运行单执行。

## Linux 接手条件

Linux 必须先核对 clean tracked tree、registry、matrix 指纹和 canonical v2.2/IAPWS 产物，
再执行 `--sanity`；只有五个机制身份均为 `exact=true` 才能运行 `--queue`。结果无论正负均
原样回传，不改阈值、不搜索、不补种子、不升级论文结论。

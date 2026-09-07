# FMTS 图 2：norew 配对证据预注册草案

状态更新：用户明确授权三臂（physics_only、closure_cons、closure_cons_norew）各三种子 H18 预测和双阀 H18/H60 响应重评估，并澄清“补充跑”不重新训练。随后用户提出 push 后交由 Linux 执行，现准备独立分支与冻结执行单；本任务不远程启动 Linux。未授权新增训练或 leakage 辅助训练。reserved test 保持锁定。下文最初的执行前确认事项由本条授权及独立 protocol.json 更新。

## 科学问题与主线

norew 的选择依据是此前其他配置出现方向问题；不能把这些配置重新预设为可信基线。现有 norew 的方向证据符合冷却预期，但完整 R1 因 counterfactual_support_violation 为 INCOMPLETE。

1. closure_cons 对 closure_cons_norew：量化关闭再润湿的预测差及同口径方向变化。已有三种子五温度 H18 MAE 差 +0.283368°C（17.681%），尚不能称为可信响应的代价。
2. 无 closure 的 norew 对 closure_cons_norew：量化相同 norew 结构内增加 closure 的收益，并检查方向是否受损。两臂均保留 hybrid 初态，结论仅涉及 closure。独立训练的对比估计配置及其拟合结果的整体效果，不等于在固定参数上只开关网络的即时效果。
3. 原 physics_only 为方向问题诊断对照。closure_steam 可作已观察到的 T1 最佳预测参照，明确其为验证结果已知后的描述性选择，不称独立选出的预测上界。

本地已核对的 v0.7 Side A manifest、checkpoint 目录与 T1 注册表均无无-closure/norew 臂。training.py 可解析 none_norew 的结构语义，但该臂尚未注册；不能据此声称已有可直接执行的训练协议。Linux 可先仅列出已有候选产物及训练 spec/指纹，不搜索参数、不启动实验。

## 输入回传清单

只传原执行包中的文件，不重建 canonical，不以同名新文件替代：

- /home/bluster/final_wm_v07_full_reissue_v1/inputs/canonical_sideA_v2.npz
  - manifest SHA-256: 24da77960e05e3636cc7b97a60a75e9b4ba470a3abb4f8ba920ddf11c6dad1d0
- /home/bluster/final_wm_v07_full_reissue_v1/inputs/iapws_surrogate.npz
  - manifest SHA-256: 9fd7a1dba96a5b968661f644fa185ac755266c85853d689d065432a2d41f6e92
- 同目录 canonical_sideA_v2_meta.json；本地已有回传副本，可用于一致性核对。
- 原执行代码 commit 身份与环境记录（manifest 指向 105c61a4cbd3d8eefe7c4a166e6594472f5b2b94）；能提供则附原命令日志及依赖版本。

本地已有 Side A 全部 T1 checkpoint、metrics、manifest、summary、R1 报告，无需默认重传。按原目录结构将 inputs 放在 sideA 的同级。先按文件字节核验身份，不加载 test 数值；传输完整封存文件不代表解锁 test。后续模型重放前必须确认加载器及支持域计算不会读取或使用 reserved test，必要时另行设计保留原索引与来源身份的隔离读取路径，不静默裁剪或重新编号。

## 预定验证口径

- canonical v2.2、Side A、原 75/15/10 划分；训练只用 train，评估只用 validation；不混合 Side B 或旧 canonical。
- 预测复用原三种子 256 个 H18 验证锚点及日标识，五温度 MAE；保留 hybrid 初态、oracle 未来边界和完整模型链。
- 响应沿用冻结双阀各自 +0.05、上限截断，H18/H60，首个未来边界固定，末十步末过温度差及日块统计。各臂使用同一组原协议响应索引，不声称其与预测锚点相同。
- 原有 R1 同时含 blindness、残差功率、leakage、支持域。leakage_probe 包含辅助训练，必须与纯模型推理分开列入执行范围；建议训练工作均由 Linux 承担。
- 支持域基线和干预掩码完整保留；不筛掉失败样本、不改阈值、不缩小干预求通过。若将来另提局部干预协议，须先阐明科学目标并独立预注册，经用户批准后执行，不能追溯修改原 R1。
- 已知验证结果须明确披露。沿用现有按日统计，报告各 seed 和配对差，具体新增 MAE 区间的估计对象及聚合顺序在执行前冻结；不将窗口视作独立重复、不选择最优 seed。

## 分阶段与停止条件

1. 输入身份核验：仅文件与元数据核对。哈希或来源不一致则停止，不重新生成文件补齐身份。
2. 执行方案冻结后，本地独立验证既有 checkpoint 的预测重放与响应；辅助训练另列 Linux 工作。任何数值差异先解释，不静默放宽容差。重放容差须在运行前结合精度与平台差异冻结。
3. 若 Linux 也没有匹配的无-closure/norew 产物，再提新增三种子训练，继承 120/20、相同数据与初态，不做搜索；新臂 spec、选择规则、指纹和验证锚点先冻结。
4. 科学结果不利或 INCOMPLETE 即如实报告，不触发补跑；运行异常保留输出，停止并讨论。运行成本未核实，不估算。

## 结论边界与可选增强

方向正确和幅值非零最多支持方向一致性，不能证明现场响应保真度。已知真值 MS5 可支持限定系统中的机制证据，不能代替现场验证，也不能强行制造正的精度代价。现场幅值和时序保真需独立干预数据或有不确定性范围的工程校准证据；取得何种证据后再单列预注册。不默认增加现场实验。

执行前待确认：本地模型推理/重评估的明确范围；Linux 辅助探针训练与缺失匹配臂训练的明确授权；统计与重放容差的最终冻结。当前未运行任何上述步骤。

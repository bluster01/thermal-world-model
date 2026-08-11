# Phase 3.5-MS5 Full Free+Response Coupling Design

## Material Passport

- Material Type: preregistered synthetic experiment design
- Evidence Scope: `synthetic_full_free_response_coupling_validation_not_field_causality`
- Upstream: MS2-D3 `CLOSED / VALIDATION_STRESS_PASS / NO_TEST_BY_BUDGET_DECISION`
- Status: FROZEN / READY FOR LINUX VALIDATION
- Boundary: 不读取 A/B，不访问 synthetic test，不做路线冠军，不启动 MS3/MS4

## 1. 单一研究问题

在动作与工况相关、自由温度轨迹与动作响应幅度相近的 known-truth 中，仅以总温度误差训练完整

\[
\widehat y_{1:H}=\widehat f_{free}(c) + \widehat g_R(c,a_{1:H},r_{1:H})
\]

时，`g_R` 是否仍恢复真实多步动作响应；若 joint-from-scratch 发生 component absorption，短阶段 `free pretrain → response with frozen free → low-LR joint` 是否能够恢复且不牺牲总预测？

MS5 的核心不是比较 D2/D3 路线，而是判定完整加法分解的训练可辨识性。free head 接口仍不读取未来动作；这只能阻止直接 leakage，不能阻止它利用 context 与 policy 的相关性吸收条件平均动作响应。

## 2. 备选方案与冻结取舍

| 方案 | 训练量 | 能回答 | 决定 |
|---|---:|---|---|
| A. 4 modes×3 seeds | 12 runs | free-only 预测陷阱、joint、staged、component-supervised oracle | **采用** |
| B. 额外扫描 free capacity/loss weight | 18–24+ runs | 超参数鲁棒性 | 不采用；会产生事后挑配置空间 |
| C. 直接上 A/B | 计算较少 | 真实预测 | 不采用；没有 component truth，无法判定 absorption |

## 3. 冻结生成式

response truth 继承 D2 的 R50、context scheduling、三极点 `[40,70,210] s`、无 pure delay；D3 colored nuisance 不带入 MS5，以免把 decomposition 与 nuisance 两个问题混在一起。新增 clean free trajectory：

\[
f_h(c)=0.8\tanh c_0 +0.5\tanh c_1 u_h
+0.4\tanh c_2(1-e^{-h\Delta t/180})
+0.2\tanh c_3\sin(\pi u_h),
\quad u_h=h/H.
\]

非 hold action 在原 profile 的 active support 内额外加入 `4·tanh(c0)` percentage points 的 policy offset，使 action 与 free context 相关；hold 保持 `a=r`，为 free pretraining 提供可观测零响应样本。target 为

\[
y_h=f_h(c)+g_{clean,h}(c,a,r)+\epsilon_h,
\quad \epsilon_h\sim\mathcal N(0,0.02^2).
\]

生成器必须保存 `clean_free`、`clean_effect`、`clean_total`，且 legacy regimes 在新字段关闭时保持原 action/response 数值不变。

## 4. 冻结 12-run 矩阵

所有模式使用相同数据、三极点 scheduled monotone graybox response、free MLP hidden=32、train/validation=1024/256、seeds `[0,1,2]`、300-epoch optimizer-update cap。

1. `ms5_free_only`：只训练 free head；response 恒零，作为 prediction-only negative control。
2. `ms5_joint_total`：free 与 response 从头联合，只用 total loss。
3. `ms5_staged_total`：Stage A 只在 hold episodes 训练 free；Stage B 冻结 free、全样本训练 response；Stage C 全部以 0.2×学习率联合微调，只用 total loss。
4. `ms5_component_oracle`：joint total loss 加 synthetic clean component loss；只作 decomposition 可解性正对照，不传播到真实数据训练。

不扫描阶段长度：A/B/C caps 固定为 80/140/80，总 cap 300，patience 20。joint 与 oracle 同为最多 300 epochs。

## 5. 指标与动作吸收定义

validation checkpoint 只按 noisy total MAE 选择。每个 run 保存 episode-level：total clean MAE/scale、free clean MAE/scale、response clean MAE/scale、response amplitude ratio 和 profile。

\[
NMAE_R=\frac{E|\hat g_R-g_R|}{E|g_R|},\qquad
q_R=\frac{E|\hat g_R|}{E|g_R|}.
\]

当 `NMAE_R>0.15` 或 `q_R<0.80` 时，记为 response absorption；`q_R>1.20` 记为 over-attribution。free-only 必须精确 `q_R=0`，用于证明 total prediction 不能替代 component recovery。

## 6. 主门与决策规则

结构/产物门逐 run 全部通过。正对照 `ms5_component_oracle` 每 seed必须同时满足：total clean NMAE `<0.10`、free clean NMAE `<0.10`、response clean NMAE `<0.10`、`0.80<=q_R<=1.20`。

训练策略资格门对 `joint_total` 与 `staged_total` 分别逐 seed检查：total clean NMAE `<0.10`、free clean NMAE `<0.10`、response clean NMAE `<0.15`、`0.80<=q_R<=1.20`。

决策顺序冻结：

1. oracle 失败：生成式/容量正对照失败，MS5 fail closed；
2. joint 资格门全过：选择更简单的 joint，不因 staged 数字更好而升级复杂流程；
3. joint 失败而 staged 全过，且 staged/joint total clean MAE ratio 每 seed `<=1.10`：选择 staged；
4. oracle 通过但 joint/staged 都失败：判 total-only decomposition 不可辨识，阻断 MS3 的强 component claim；
5. free-only 预测误差只作陷阱诊断，不参与选择。

同 seed、同 validation truth 下 total clean scale 完全相同，因此实现中记录的 staged/joint total clean NMAE ratio 与上述 MAE ratio 数值严格相同。

按项目负责人决策，MS5 validation 通过即可关闭，不再开 synthetic test；因此最终标签必须包含 `VALIDATION_ONLY`，不能写成 confirmatory test。

## 7. Claim boundary

MS5 PASS 最多证明一套 synthetic full-model 训练流程在冻结 policy confounding 下能保留 known-truth response component。它不证明现场 free/response 分解、观测因果识别、完整状态 simulator、反事实真实性或闭环可用性。MS5 关闭前不启动 MS3。

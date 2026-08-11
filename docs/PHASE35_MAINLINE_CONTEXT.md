# Phase 3.5-MS 主线实验上下文

## Material Passport

- Material Type: active experiment-purpose specification
- Scope: Phase 3.5-MS0–MS5、真实数据适配与闭环响应验证
- Verification Status: ANALYZED；MS5 validation 已完成权重级独立审计
- Deprecated Track: 原 E1–E5，仅作历史失败证据
- Active Decision: 完成完整 MS 系列；MS5 选择 joint 并关闭，当前仅授权 MS3 A/B observational validation

## 1. 一句话主线

项目当前不是写论文，也不是证明已经得到完全物理世界模型；当前任务是逐级证明一个结构化多步动作响应架构能在已知真值下可解、在结构失配下稳健、接入自由预测头后不丢失动作链，并最终迁移到现场串级闭环数据。

统一模型为

\[
\widehat T_{1:H}=f_{free}(\text{history, context})+
g_{response}(\text{context},a_{1:H},r_{1:H}),
\qquad g_{response}(c,r,r)=0.
\]

`f_free` 不读取未来动作；`g_response` 负责实际阀位相对参考阀位的增量响应。实际阀位是有效喷水作用代理，不是喷水质量流量。

## 2. 完整验证链

| 层级 | Gate | 回答的问题 |
|---|---|---|
| 结构合同 | MS0 | 不同路线是否共享同一 estimand、零参考、时间因果和递推接口？ |
| 最小可解性 | MS1 | 同型二阶多步系统能否恢复？ |
| 模块辨识 | MS2-V/C/J | 非线性开度、工况调度及其联合是否可解？ |
| 结构压力 | MS2-D1/D2/D3 | 纯迟延、额外阶次、未建模扰动下结论是否仍成立？ |
| 完整耦合 | MS5 | `free+response` 联合训练时动作响应会不会被 free head 吸收？ |
| 真实适配 | MS3 | 在 A/B 观测数据上能否保持预测、结构和动作敏感性？ |
| 物理闭环 | MS4 | 是否复现 SP→控制器→阀位→温度的现场闭环响应？ |

顺序冻结为：

```text
MS0 → MS1 → MS2-V/C/J → MS2-D1/D2/D3 → MS5 → MS3 → MS4 → 模型选择/论文
```

## 3. 已完成证据

- MS1：六条表示在同型二阶 truth 上达到噪声区；只作正对照。
- MS2-V：learned monotone 相对 identity 的 validation/test 改善稳定超过门槛；没有单独恢复真实 `phi`。
- MS2-C：scheduled K/τ 相对 global 双层通过。
- MS2-J：joint 相对两个单模块双层通过；response-internal staged 在 1.10 非劣界双层失败，但相对 Stage A 有效。因此当前 response 主训练采用 joint，不能据此决定完整 MS5 staging。
- MS2-D1：learned-delay 的改善方向跨 split 稳定，但 test CI 下界未达到冻结的 20%，按阴性结果关闭；不传播 delay 结构。
- MS2-D2 test：oracle 与三阶绝对 NMAE 逐 seed过门；三阶相对二阶点改善 23.74%–25.36%，冻结 bootstrap CI 下界 19.90%–21.22%，逐 seed高于 10%。只确认 frozen known-truth 下的响应优势，不确认现场阶次唯一性。
- MS2-D3 validation：21/21 产物门通过；oracle 0.0357–0.0446、三阶 0.0558–0.0633，相对二阶 CI 下界 10.8%–14.3%。按预算不追加 test，只保留为 colored-nuisance validation 压力证据。
- MS5 validation：12/12 产物闭合并完成 checkpoint 重算。joint response NMAE 0.047–0.050、amplitude ratio 0.988–0.994，逐 seed过门；冻结 staged 协议相对 joint 误差比 11.14–14.11，拒绝；free-only 证明“预测准不等于组件恢复”。只作 synthetic validation，不追加 test。

Koopman、PI-ODE 与 DeepONet 的历史 synthetic 结果不构成最终路线冠军，但不在 MS3 当前 Gate 重新赛马。MS3 先验证 MS5 选中主干能否迁移到真实 A/B；表示路线比较在模型选择阶段另行冻结。

## 4. 原 E 系列的定位

原 E1–E5 事件匹配路线已废弃。其价值只在于记录为什么不能把闭环阀位事件直接当外生干预：反馈内生性、开关 support 不平衡、balance/pre-trend 不足和低基率 no-execution。

现场物理验证改由 MS4 承担。当前最强外部锚点是 SP held-step 的闭环响应：A/B 两侧具有双向事件和 110+ 日块，`SP↑→T2↑→Tm↑` 方向率约 70–86%。它验证的是包含串级 PID 的闭环系统，不是开环 `do(valve)`；双侧 SP 联动仍限制单侧归因。

## 5. 当前 Gate：MS3

MS2-D 采用顺序压力测试，不一次铺开大矩阵：

1. D1：在 R50 + context-scheduled truth 上加入 20 s pure delay；
2. D2：加入第三个惯性环节；
3. D3：加入 action-independent colored disturbance。

D1 已关闭：learned causal delay 的改善方向跨 validation/test 稳定，但 test bootstrap CI 下界 17.2–18.8% 未达到预注册 20%，且 delay kernel 未唯一恢复。该结果不能把 learned-delay 传播为主架构的已证实部件。

D2 采用正交的阶次压力设计：真值取消 pure delay，只加入第三个惯性极点 `[40,70,210] s`。one-shot test 的 oracle、三阶绝对误差和三阶相对二阶 CI 主门逐 seed通过，故以 `CONFIRMED_SYNTHETIC_ORDER_RESPONSE` 关闭。但二极点+learned-delay 与 DeepONet 在有限 horizon 仍接近三阶；D2 没有建立现场唯一阶次、参数唯一性或迟延机制。

D3 保持 D2 clean truth，只加入 response operator 不可观察的 stationary AR(1) output nuisance。validation 的冻结主门逐 seed通过，本地 episode 重算与独立 50k bootstrap 一致；由于 validation 参与 checkpoint 选择，按预算决定以 `VALIDATION_STRESS_PASS / NO_TEST_BY_BUDGET_DECISION` 关闭，不称独立确认。

MS5 已证明在冻结 full-coupled synthetic truth 下，total-only joint 能恢复 free/response 组件；当前冻结 staged 协议失败。该结论已经权重级重算，以 `JOINT_SELECTED` 关闭，但不等于真实数据中的组件真值可观察。

MS3 使用 SHA 冻结的 `all_merged_10s.csv` 构造两个控制回路 cache，把已确认现场映射写死为 A阀→右(B)温、B阀→左(A)温。冻结矩阵为 joint/free-only×A/B×3 seeds=12。主门检验 joint 相对 free-only 预测非劣、动态动作分支非坍缩，以及 logged future valve 相对保持阀位/置乱 delta-path 的 UTC-day block-bootstrap 预测增益。由于未来阀位处于串级 PID 闭环，这仍是 observational conditional-prediction Gate，不是 `do(valve)`；闭环物理响应继续由 MS4 负责。

## 6. 当前不能声称

当前不能声称真实阀门曲线或喷水流量已恢复、A1phys 参数唯一、现场 `do(valve)` 已识别、完整状态闭合 simulator 已成立、路线冠军已确定、模型已可嵌入闭环，或论文已到收口阶段。

## 7. 状态恢复

新会话先读 [`PHASE35_CONTEXT_SNAPSHOT.md`](PHASE35_CONTEXT_SNAPSHOT.md)，再运行：

```bash
python experiments/phase3_5/experiment_status.py --check --json
```

机器注册表是 `configs/phase3_5/experiment_registry.json`。任何 Linux 运行必须同时满足 `active_gate` 与 `linux_authorized_gate`，不能只依据聊天或旧 handoff。

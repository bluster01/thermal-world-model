# Phase 3.5-MS 主线实验上下文

## Material Passport

- Material Type: active experiment-purpose specification
- Scope: Phase 3.5-MS0–MS5、真实数据适配与闭环响应验证
- Verification Status: ANALYZED；MS2-D2 test 已独立审计
- Deprecated Track: 原 E1–E5，仅作历史失败证据
- Active Decision: 完成完整 MS 系列；MS2-D2 已确认关闭，当前仅授权 MS2-D3 validation

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

Koopman、PI-ODE 与 DeepONet 继续作为表示路线参与压力和真实适配验证；MS2-J 的 secondary 数值不构成最终路线冠军。

## 4. 原 E 系列的定位

原 E1–E5 事件匹配路线已废弃。其价值只在于记录为什么不能把闭环阀位事件直接当外生干预：反馈内生性、开关 support 不平衡、balance/pre-trend 不足和低基率 no-execution。

现场物理验证改由 MS4 承担。当前最强外部锚点是 SP held-step 的闭环响应：A/B 两侧具有双向事件和 110+ 日块，`SP↑→T2↑→Tm↑` 方向率约 70–86%。它验证的是包含串级 PID 的闭环系统，不是开环 `do(valve)`；双侧 SP 联动仍限制单侧归因。

## 5. 当前 Gate：MS2-D3

MS2-D 采用顺序压力测试，不一次铺开大矩阵：

1. D1：在 R50 + context-scheduled truth 上加入 20 s pure delay；
2. D2：加入第三个惯性环节；
3. D3：加入 action-independent colored disturbance。

D1 已关闭：learned causal delay 的改善方向跨 validation/test 稳定，但 test bootstrap CI 下界 17.2–18.8% 未达到预注册 20%，且 delay kernel 未唯一恢复。该结果不能把 learned-delay 传播为主架构的已证实部件。

D2 采用正交的阶次压力设计：真值取消 pure delay，只加入第三个惯性极点 `[40,70,210] s`。one-shot test 的 oracle、三阶绝对误差和三阶相对二阶 CI 主门逐 seed通过，故以 `CONFIRMED_SYNTHETIC_ORDER_RESPONSE` 关闭。但二极点+learned-delay 与 DeepONet 在有限 horizon 仍接近三阶；D2 没有建立现场唯一阶次、参数唯一性或迟延机制。

D3 保持 D2 的 R50、context scheduling、三阶、无 pure delay clean truth，只加入 response operator 不可观察的 action-independent stationary AR(1) output nuisance：`sigma_d=0.03 °C`、`tau_d=120 s`。21-run validation 主门仍为 oracle `<0.05`、三阶 `<0.10`、三阶相对二阶的配对/profile 分层 bootstrap CI 下界逐 seed `>=10%`，且全部基于已知 clean response。扰动 realization、tau、伪迟延、profile/horizon、D2→D3 漂移及 Koopman/PI-ODE/DeepONet 排名只作诊断。当前不访问 test，不启动 MS5。

## 6. 当前不能声称

当前不能声称真实阀门曲线或喷水流量已恢复、A1phys 参数唯一、现场 `do(valve)` 已识别、完整状态闭合 simulator 已成立、路线冠军已确定、模型已可嵌入闭环，或论文已到收口阶段。

## 7. 状态恢复

新会话先读 [`PHASE35_CONTEXT_SNAPSHOT.md`](PHASE35_CONTEXT_SNAPSHOT.md)，再运行：

```bash
python experiments/phase3_5/experiment_status.py --check --json
```

机器注册表是 `configs/phase3_5/experiment_registry.json`。任何 Linux 运行必须同时满足 `active_gate` 与 `linux_authorized_gate`，不能只依据聊天或旧 handoff。

# 物理修复批 ②③④ 设计稿（矩阵修正案 v0.3 子项，2026-08-20）

> 状态：设计冻结，随代码实施入仓。靶向：动作保真（R1）根因——H1 瞬时过冲与
> 60 步符号反转。实施对象：`src/final_wm/{contracts,transition}.py`，接口形状不变，
> 状态维度 9+L → 11+L（fingerprint 变化自动触发矩阵重训，旧 checkpoint 失效为预期行为）。

## ② 喷水→混合链路时滞

**现状缺陷**：混合焓 `hm_i = (D·h + Dsw_i·h_spray)/(D + Dsw_i)` 用瞬时喷水率
`Dsw_i = th_i(pm)·varphi(v_i)`，同时进入下一级入口焓 `h_in` 与输出方程——阀位阶跃
在**同一 dt** 改变出口温度测量。真实对象事件研究显示 H1（10 s）响应 ≈0、符号在
1–3 min 内建立（v2 down: H1 −0.04 / H6 −0.15 / H18 +0.56 °C，auditpack_A.json）。

**改动**：新增两个状态 `dsw_lag1/2` [kg/s]（注册于 `PHYSICAL_STATE_ELEMENTS` 位置 9–10，
latent 块起点顺延至 11），一阶滞后：

```
d(dsw_lag_i)/dt = (Dsw_i^target − dsw_lag_i) / tau_mix_i      i ∈ {1, 2}
```

- 混合焓与输出方程一律使用 `dsw_lag_i`（状态驱动），**输出方程不再瞬时依赖当前
  动作**——H1 构造性过冲在结构层消除，而非参数层压小；
- `m_liq` 沉积仍用瞬时 `Dsw_i`（液滴沉积是本地过程，τ_evap=15 s 已自带短滞后），不变；
- 输出方程的干湿混合（dry-out blend）改用 **lag 状态等效湿润度** `dsw_lag·τ_evap` 而非壁面
  液滴库存 `m_liq`：传感器位于传输路径下游，否则沉积通道（τ_evap=15 s）会绕过混合时滞
  把瞬时动作响应泄漏进测量（实施中实测确认：仅用 m 时首步响应占稳态 43%，改后为
  一阶滞后量级 ~15%）。稳态下 lag=Dsw 且 m=Dsw·τ_evap，两口径一致，锚定初始化不受影响；
- 初值（`initial_steady_state`）：稳态恒等 `dsw_lag_i = Dsw_i`；
- 新参数 `tau_mix1/tau_mix2`：**可学习参数**（`raw` ParameterDict 内 softplus 正参数化，
  随矩阵训练被 Adam 更新，先验仅定初始化中心，数据可推翻）；先验 **80 s**，锚定 adhoc2
  学习证据（`fix3_learnlag_summary.json`：tau_sw=73–86 s 由数据学出）；
- **备选结构（登记在案，本轮不实施）**：adhoc2 同实验还学出 tau_sens=19–24 s，与 tau_sw
  明确分离——传感湿润通道比混合传输快约 4×。当前 blend 挂在混合 lag 上（80 s）；若重跑后
  R1/事件研究显示 sh1_out/sh2_out 快分量仍失配，再拆出独立湿润滞后状态（+2 状态，
  先验 20 s）。拆分时需重新走本设计稿的状态注册流程；
- 守恒性：滞后是速率状态的低通滤波，稳态 `dsw_lag = Dsw`，boundary 模式 κW 总量
  守恒在稳态保持；瞬态不守恒是物理（管道/混合腔储液），且守恒审计只查 `aux` 中的
  瞬时 target，不受影响。

## ③ 再湿项符号-量级硬契约

**现状缺陷**：`q_w_i = aW_i·(Tm_i − Tsat_i)·(1 − dry_i)` 无上界——喷水增加 → m 上升 →
(1−dry) 上升 → q_w 正向加热蒸汽，构成正反馈（aW=0 消融 0.27→1.00 实锤其为 60 步
反转根因）。

**契约（质量平衡闭锁）**：再湿加热的载体只能是正在蒸发的沉积液滴（质量通量
`m_i/τ_evap`），每 kg 蒸发液滴至多向蒸汽相净转移 `(h_pre_i − h_spray)` 焓差：

```
q_w_i^raw = aW_i·(Tm_i − Tsat_i)·(1 − dry_i)                    # 现有形式
q_w_i^cap = (m_i / tau_evap) · max(h_pre_i − h_spray, 0)       # 质量平衡上界 [kW]
q_w_i     = min(q_w_i^raw, q_w_i^cap)
```

其中 `h_pre_1 = h_1`、`h_pre_2 = h_2`（该级喷水前蒸汽焓）。

- m=0（干态）或蒸汽焓低于喷水焓时 q_w=0——与 dry sigmoid 冗余但构成双保险；
- `q_w^raw < 0`（壁温低于饱和，冷凝冷却）时 `min` 自然放行负值，方向契约不被破坏；
- step() 与 output_temperatures() 共用同一 helper（`_rewetting_powers`），消除两处
  公式漂移风险；
- aW1/aW2 参数与消融探针（`analysis.rewetting_ablation` 置零 raw）语义不变。

## ④ 喷水灵敏度先验锚定

数据回归（auditpack `spray_sensitivity`，OLS，闭环混杂警告在案）：
dW/dv1 = 27.76、dW/dv2 = 70.01 t/h/满开度。锚定（÷3.6 换 kg/s）：

| 参数 | 旧先验 | 新先验 | 依据 |
|---|---|---|---|
| th1 | 10.0 | 7.71 | 27.76/3.6 |
| th2 | 20.0 | 19.45 | 70.01/3.6 |
| th1d/th2d | 10.0/20.0 | 7.71/19.45 | 干态无分辨口径，同值锚定 |

γ 先验保持 1.0。先验是 softplus 中心而非硬约束，参数仍可学习；OLS 闭环混杂风险
已通过"锚先验不锁参数"消化。

## 不变量核对（实施必须保持）

- varphi(0)=0、valve 单调、参数正值化：不动；
- 零动作恒等：v=0 → Dsw=0 → dsw_lag 衰减至 0 → 混合趋无喷水、再湿闭锁至 0；
- forecast 模式不读 W；closure action-blind；残余注入位置不变；
- `aux` 继续暴露瞬时 target（dsw1/dsw2）供守恒审计，另增 `dsw_lag1/2` 供诊断。

## 验收门禁（本地，先于 Hermes 重跑）——结果已闭合

1. ~~既有 110 项测试适配后全过~~ **116/116 通过**（110 + 6 项新契约测试）；
2. 新契约测试全过：阶跃响应渐进性（首步/稳态 ≈ 一阶滞后量级）、输出方程状态驱动
   （同状态任意动作输出逐位相同）、再湿上界与干态闭锁（m=0 → q_w=0、冷凝方向保留）、
   先验数值（th1=7.71/th2=19.45/τ_mix=60）、零动作下 lag 衰减；
3. D-SYN 同型可解性：quick 档（2 epoch 冒烟）per-seed 标志失败——诊断为**纯冒烟档假象**：
   修复后 skeleton（未训练同型模型）从 ~150 降到 11.6（名义物理本身显著变准），
   2 epoch 无法再压 30%；中等规模探针（8k 步记录、8 epoch、CPU）skeleton=11.58、
   student best=0.88（ep4），门禁以大幅裕度通过。quick 档 verdict 本就强制 PASS，
   per-seed 标志在冒烟档仅供参考；正式 D-SYN（全尺寸 3 seeds）随 Hermes 重跑复核。

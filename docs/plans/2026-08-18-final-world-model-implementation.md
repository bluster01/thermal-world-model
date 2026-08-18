# 最终世界模型接口包实现记录（src/final_wm）

> 状态：本地接口 + micro-smoke 完成。本文记录 `src/final_wm/` 的模块划分、关键设计决策和与
> [pipeline 设计稿](2026-08-18-final-world-model-pipeline-design.md) 的对应关系。
> 不授权任何长训；O1/B1/T1/R1/J1/K1 判别实验需独立冻结提交。

## 1. 模块划分

| 文件 | 职责 | 设计稿对应 |
|---|---|---|
| `contracts.py` | 状态/边界/动作/观测注册表、权限标签、fail-closed 配置校验、动作支持域 | §3 模块职责与硬边界 |
| `properties.py` | 可注入可微热物性接口：`GridThermoProperties`（生产，legacy IAPWS 网格）+ `AnalyticThermoProperties`（定性 fallback，仅测试） | §2 数学合同中的 T(p,h)/h(p,T) |
| `transition.py` | Fan2020-UDE 状态转移：三段焓、金属蓄热、燃料滞后、喷水混合、蒸发/干燥 latent；半隐式 Euler 5×2s 子步 | §2 `x_{t+1}=Φ(F_Fan20 + S r_θ)` |
| `closure.py` | action-blind 残差 `r_θ(x,b,ε)`，固定注入位置，幅值 tanh 饱和 | §3 Closure 行 |
| `observer.py` | 概率初态后验 q(x0\|H)，GRU 编码器 + tanh 有界均值 + 相邻窗口 state-continuity 度量 | §3 Observer 行 |
| `boundary.py` | forecast/oracle 双模式未来边界分布，GRU encoder-decoder | §3 Boundary 行 |
| `observation.py` | 观测均值 = transition 输出方程 g(x,b,u) + 有界异方差 σ | §3 Observation 行 |
| `controller.py` | SP→PI（含饱和/死区/抗积分饱和）→执行器一阶滞后+速率限制→阀位 | §3 Controller/actuator 行 |
| `model.py` | 装配层：forecast / counterfactual / closed_loop 共用同一 transition | §2 共享 transition 硬约束 |
| `synthetic.py` | 合成工况数据与同型 teacher rollout（仅测试用） | §7 本地 micro-smoke |

## 2. 关键设计决策

### 2.1 权限合同的代码化

- **forecast 模式在签名层面没有真实未来边界的入口**；`BoundaryModel.forecast` 不接受 true future，
  `FinalWorldModel.forecast(boundary_mode="forecast", true_future_boundary=...)` 直接 raise。
- **实测喷水总流量 W 是 oracle-only 通道**：`spray_total_mode="boundary"`（硬质量守恒读 W）与
  `boundary_mode="forecast"` 的组合在 `validate_world_model_config` 中被拒绝。forecast 下喷水只能经
  `varphi(u)` 进入。
- **closure action-blind**：特征只含当前物理/latent 状态与白名单边界通道（不含 W）；
  `reads_actions` 字段恒为 False，任何改动都会 fail-closed。动作不可表示 ⇒ 运行时不变性测试兜底。
- **被拒绝的 double-injection 不可表示**：注入模式枚举只有 `none/steam_only/conservative`。

### 2.2 动作通道

`varphi(v) = v^γ`，`γ>0`（softplus 参数化）：`varphi(0)=0` 给出 constant-action 零喷水恒等式，
`[0,1]` 上单调满足"阀位映射单调"合同。喷水增益 `th_i(pm)` 跨临界 sigmoid 平滑混合（沿袭 legacy
干湿分模态修复）。

### 2.3 状态与初态

打包状态 9+L 维：`h[3], Tm[3], rB, m_liq[2]` + 可选 latent block（漂移 `rho=tanh(raw)`，|rho|<1
有界稳定合同）。`initial_steady_state` 复刻 legacy 的观测锚定稳态初始化（h 由观测温度反演、
金属稳态热平衡偏置、`m=Dsw·τ_evap`），作为 O1 实验的 steady 臂；learned posterior 臂由 observer 提供。

### 2.4 热物性技术债的处理

legacy `iapws_surrogate.npz` 未导入主仓，因此热物性被抽象为注入接口：生产路径由执行侧注入真实
IAPWS 网格（`load_grid_properties`，与 legacy npz 格式一致）；本地测试用解析 fallback
（h 对 T 仿射、闭式反演、单调有界），**定性而非 IAPWS 精确，禁止用于科学数字**。

### 2.5 closure 零初始化

closure 输出层零初始化：启用 closure 不会跳变已验证骨架的行为（残差从精确的 0 开始生长）。

## 3. 测试覆盖（tests/final_wm/，82 项，本地全过）

| 文件 | 覆盖 |
|---|---|
| `test_contracts.py` | 注册表完整性、配置 fail-closed（forecast×boundary-spray 拒绝、double-injection 不可表示、closure 禁读动作）、动作支持域 |
| `test_properties.py` | 单调性、p-h 往返一致、临界 clamp、网格加载器、越界有限性、STE clamp 梯度 |
| `test_transition.py` | 前向有限性、零动作→零喷水且液滴干燥、输出方程零动作恒等、观测锚定初态复现、定常有界且收敛、开阀长期降温方向、boundary-spray 守恒、残差注入方向、latent 衰减 |
| `test_closure.py` | 运行时动作不变性、三种注入模式符号、零初始化、幅值饱和、随机性合同 |
| `test_observer.py` | 后验形状/正值性/物理有界、历史长度合同、continuity 度量 |
| `test_boundary.py` | forecast 形状与模式标签、oracle 透传、模式 fail-closed、scenario 合同 |
| `test_observation.py` | σ 正值有界、均值恒等于 transition 输出方程 |
| `test_controller.py` | 方向、饱和、速率限制、死区、执行器滞后、抗积分饱和 |
| `test_model.py` | 共享 transition、forecast/oracle 权限、counterfactual 支持域门、闭环 smoke、NLL、state continuity |
| `test_micro_smoke.py` | 同型 teacher 上 observer NLL 训练下降、closure 经共享 transition 收梯度、latent+closure 全栈、定种子确定性、纯 forecast 全链路 |

复现命令：

```bash
python -m pytest tests/final_wm/ -q
```

## 4. 明确未做

- 不在本地训练真实参数（micro-smoke 只证明同型可训练性）；
- 不实现 Koopman student（P5 需母模型过门禁后另行授权）；
- 真实数据适配、真实 IAPWS 网格注入需 D0a 发现报告回传后冻结通道映射再授权执行；
- Direct-WM 高容量 backbone 尚未接入为 observer 替代实现（当前 observer 为 GRU 基线）。

## 4.1 执行层（2026-08-18 增补）

判别矩阵的执行代码已就位，Linux 只做执行与产物回传，不改代码/阈值：

- `src/final_wm/data.py` — D0 两阶段：`discover_dataset`（有界 schema/质量发现报告）→
  本地冻结通道映射（`configs/final_wm/channel_mapping.json`）→ `build_canonical`
  （10 s 重采样 + fail-closed 质量门 + train/val/test 锁定时序切分 + 唯一 canonical npz）；
- `src/final_wm/training.py` — 单臂训练循环（Adam + clip + val-NLL 早停 + checkpoint + JSONL ledger，
  含 commit/properties/device 审计字段）；
- `src/final_wm/evaluation.py` — 窗口 rollout 指标（NLL/MAE/CRPS@H1/H6/H18）、UTC-day block
  bootstrap 的相对改善 CI、阶跃方向审计、残差功率分位数、B1 persistence 基线；
- `src/final_wm/diagnostics.py` — R1 负对照残差泄漏探针（blind vs action-aware，诊断件不进入生产装配）；
- `experiments/final_wm/matrix_spec.py` — 冻结臂/种子/阈值（矩阵文档的可执行镜像）；
- `experiments/final_wm/run_matrix.py` — 四阶段统一入口：`discover` / `build` / `dsyn` / `matrix`
  （O1→T1→B1→J1→R1 顺序执行，自动产出 `matrix_summary.json` 判决与逐项 ledger）。

新增 13 项测试（`test_data/test_evaluation/test_training/test_matrix_smoke`），全套件 95 项本地通过；
`--quick` dry-run 已验证（D-SYN quick 门禁 skeleton NLL 150.1 → student 40.3，PASS）。

## 5. 已知环境备注

`tests/phase35/multistep/test_rm3b_execution.py` 在本机 Windows+numpy/MKL 下于 `np.cov`
（`rm3av_diagnostics.valve_innovation_rank`）处进程级 abort，属先行存在的环境问题，与本包无关；
本包全部为新增文件，不触碰任何既有代码路径。

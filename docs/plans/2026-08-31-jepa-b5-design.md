# JEPA-B5 动作盲慢态设计与实现

## 目的

B2（慢态）是 B 系列唯一精度提升臂（H18 −5.3%），但破坏 valve2 方向证书
（H18 +0.010 / H60 +0.079°C，c0 同口径 −0.105）。B5 在保持 B2 机制的前提下
做因果隔离，回答："慢态收益是否依赖读取动作效果（因果盗窃），还是
独立于动作因果的状态记忆？"

## 机制差异（唯一）

`B2SlowState.update()` 输入：
- B2：`[slow, normalized_physical, normalized_boundary]`
- B5：`[slow, normalized_boundary]`（`use_physical=False`）

物理状态是已记录动作的函数（喷水→温度→h/Tm），B2 慢态从 state 读到动作
效果→能"代偿"降温→稀释 direction 响应。boundary（7 通道：负荷/压力/给水/
燃料等工况量）不含喷水阀动作，慢态只从工况演化——严格动作盲。

其余全部保持：slow_dim=4、stride=6、power_net→3 维守恒注入
（steam +extra / metal −extra，scale 3e4·tanh）、Gaussian-CF（SLICED_CF
16 slices/17 knots/seed 260830）、初始投影（observer_hidden→slow）。

## 实现清单

1. `src/final_wm/jepa.py`：
   - `B2SlowState.__init__` 加 `use_physical: bool = True`；update_in 按需拼；
     `update()` 条件拼 physical。
   - `build_jepa_model` 接受 `b5`：B2 同款 slow + `use_physical=False`；
     臂集合白名单加 b5。
   - `JepaBModel` dispatch（forecast/counterfactual/auxiliary_terms）：
     `arm != "b2"` → `arm not in ("b2","b5")`。
2. `experiments/final_wm/jepa_b5_spec.py`：冻结合同（SHA
   `28dcb4b6…`、ORDERED_ARMS=(c0,b5)、损失表、方向门语义
   `original_trajectory_base` 校验、`require_linux_authorization` 查
   active_gate == `jepa_b5`）。
3. `experiments/final_wm/run_jepa_b.py`：`_spec_for()` 按 protocol_version
   分派 spec；`_SPEC` module-level；sanity/queue 用 `_SPEC.ORDERED_ARMS`；
   sanity 的 slow 机制关闭分支覆盖 b5。
4. `configs/final_wm/jepa_b5_series_v1.json`：冻结矩阵（c0+b5，
   result_root `results/final_wm/jepa_b5_series_v1`）。
5. `configs/phase3_5/experiment_registry.json`：注册 `jepa_b5` 条目，
   active_gate/linux_authorized_gate → `jepa_b5`（B 系列已 6/6 完成）。
6. `tests/final_wm/test_jepa.py`：b5 用例（机制关闭身份、use_physical 选项
   shape、b5 更新输入不含 state）。

## 执行

```bash
python experiments/final_wm/run_jepa_b.py --sanity
python experiments/final_wm/run_jepa_b.py --queue   # c0 → b5，顺序单 GPU
```

c0 复用 B 系列 v1 结果（同 commit 链 + 同记录 SHA 时"完整臂复用"跳过重训；
若指纹不同则重训 c0——预算不变，两臂 ~6h）。

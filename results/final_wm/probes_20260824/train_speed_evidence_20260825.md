# 训练速度优化证据包（execution-side, 2026-08-25）

## 结论摘要

**前向是 100% CPU 分发绑定，真正的瓶颈不是 IAPWS 插值，而是锚定反演 + 标量同步。**

## 证据

### 1. 前向解剖（torch.profiler，anchored seed0，batch32）
- CPU 1.24s / forecast vs CUDA 36.5ms —— GPU 97% 空闲
- `aten::item`/`_local_scalar_dense`: **5,598 次标量同步 = 722ms CPU (58%)**
- 其余是数千个 ~1µs 的小 elementwise kernel（mul 8100 / add 6676 / sub 4754）

### 2. 同步来源定位
- `properties.py _polyval`: `float(coef[i])` × 5 / 每次 saturation_temperature 调用
  （tsat_coef 是 GPU tensor；5 个标量却在 GPU 上，每次多项式求值 5 次 DtoH）
- 放大者: `transition.py _invert_spray_anchor` —— **24 次二分迭代**，每次迭代跑
  完整三级 transition 步（含 saturation_temperature → 5 次同步）
- `boundary.py:116` oracle: 每次调用创建 tensor + `float(torch.log(torch.tensor(sigma)))`

### 3. 迭代数扫描（monkeypatch，探针侧）
| iters | forecast | 加速 | temps_mu 漂移 vs 24 |
|---|---|---|---|
| 24（协议值） | 1316ms | 1.00x | — |
| 16 | 1011ms | 1.30x | **0.05°C** |
| 12 | 1245ms | 1.06x | 1.16°C |
（计时含 v1fix 争用噪声；16→12 漂移跳变说明 24 对 float32 精度是过度迭代）

### 4. torch.compile 全前向（compile_bench_probe.py）
- 前向 1.50x / 前向+反向 1.45x；数值漂移 0.027°C
- 只编译 transition: 0.89x（图断裂开销），无效
- dynamo 对 interp1d/interp2d 触发 recompile_limit（33 次重编译）——缓存脆弱

### 5. IAPWS 等距索引插值（iapws_fast_probe.py）
- 全部网格等距（spacing 比 1.000000），算术索引版数值验证通过（1.2e-4°C）
- 但插值占前向 <2%（每 query 7–20ns）——**非瓶颈，优先级下调**

## 优化提案（按收益排序，均需对侧 src 权限）

1. **P1（零数值风险）**: `_polyval` 系数上移到 CPU float 一次性 hoist；
   `boundary.oracle` 的 sigma tensor 改 float。消灭 58% 的标量同步。
   预期前向 1.5–2x，数值逐位不变。
2. **P2（需数值门）**: 锚定二分 24→16 迭代，实测漂移 0.05°C。
   协议影响：初始状态值变化 → 全部产物需重跑重裁定。
3. **P3（需数值门）**: 全 forecast torch.compile（探针实测 1.45x，漂移 0.027°C）；
   P1 后再开，预期叠加乘性 2.5–4x 前向加速。需解决 dynamo 对插值函数的重编译
   风暴（等距索引版插值天然 dynamo-friendly，可与 P2 同批落地）。
4. **P4（低优先级）**: IAPWS 等距索引插值——正确但收益 <2%，顺手做。

## 组合预期

P1+P2+P3 落地后：前向 340ms → 估 60–90ms/batch，epoch 101s → 估 35–50s，
**整体训练提速 2–3x**。每个 2–3h 的探针臂 → 1h 左右。

## 探针侧现状

- 编译（P3）可在脚本层立即启用，标注 flag + 0.027°C 漂移，与冻结产物不可直接比；
- iters=16（P2）同理，标注 0.05°C 漂移。

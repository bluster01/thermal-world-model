# A5 真实水煤比机制臂：执行前审计与解决方案（冻结稿）

日期：2026-08-28
范围：只解决 A5 的真实测点进入、机制注入、可复现实验与判定；不改正式 v0.7 判决。

## 1. 审计结论

1. 当前 7 通道世界模型没有发现需要借 A5 顺手修复的重大逻辑漏洞；历史 v1 契约和既有证据链不动。
2. A5 的实际阻塞是数据契约：canonical v2.1 没有 DCS `水煤比`，基础 transition 又明确只接收 7 列。把代理量塞进旧通道会破坏语义，不能采用。
3. A5 应保持独立机制臂。A4 已因负荷均匀性恶化而未晋级，不能替代真实水煤比，也不能先验决定 A5 的符号。
4. 论文可信度的关键不是增加网络容量，而是：真实测点、训练集内基线、嵌套零初始化、物理量级上界、验证集冻结判据和完整来源指纹。

## 2. 最小解决方案

### 2.1 数据层（canonical v2.2）

- 在 `boundary_ext` 末尾追加两个 A5 必需上下文：
  - `water_coal_ratio` ← `水煤比`；
  - `unit_load` ← `机组负荷_GENERATOR_POWER`，只用于预注册的运行工况门和 `wc_ref(L)`。
- 保留原 `boundary/actions/obs/valid/timestamps/split` 契约；不把扩展列加入全局 `BOUNDARY_ELEMENTS`。
- 构建仍走既有时间网格、覆盖率、量程、stuck、指纹和双侧对齐门。
- A5 有效样本按预注册固定为：原记录 valid、`unit_load > 160 MW`、`1 < water_coal_ratio < 8`、`fuel_corrected > 50 t/h`。窗口必须完整落在连续有效段内。

### 2.2 机制层（A5 独立探针）

- A5 运行视图只给 transition 增加 `water_coal_ratio` 和 `unit_load`，即 7+2；observer、closure 和输出方程继续只读原 7 列。
- `wc_ref(L)` 用二次多项式在 **train split 的有效样本**上一次拟合并冻结；不得看 validation/test。拟合系数、负荷中心/尺度、残差尺度写入报告与 checkpoint。
- 注入：

  `delta_Q = residual_scale_kw * tanh(tanh(w_raw) * z_wc)`，

  其中 `z_wc = (wc - wc_ref(L)) / train_residual_std`。`delta_Q` 平均分到三个 metal-power stage；符号由数据学习，幅值受既有 closure 的 30 MW 标度硬界约束。
- `w_raw=0` 时严格复现基线，形成嵌套身份门；不增加网络、不修改原闭包、不启用 A5-lite 代理。

### 2.3 运行与判定

- 训练协议与主基线一致：side A、oracle、history=96、H18、seed0、120 epochs / patience20 / batch32 / 200 batches、GridThermoProperties、`conservative_norew`。
- 因 A5 有效工况门会改变约 5% 的训练/验证窗口，先在**同一筛选记录**上跑一个无 A5
  机制的匹配对照；A5 与对照同 seed、同初始化规则、同预算、同采窗。否则无法区分机制收益
  与筛选收益。旧 0.484/1.13x 只保留作外部锚，不作为 A5 的因果差值分母。
- 主指标：validation H18 final-outlet-temperature MAE，256 windows，seed=50000；并报 Q1-Q5。
- 晋级规则沿用已冻结三条件，但比较对象改为上述匹配对照：H18 至少改善 5%；分箱极差
  恶化不超过 10%；双阀 H18/H60 方向按 v0.3（均值、day-block CI、正确方向比例和支持域）通过。
  旧 0.4840/1.13x 同报，用于检查记录迁移幅度，不替代匹配对照。
- seed0 只产生探索性结论；即使通过也只进入 seed1/2 复核，不升级正式路线冠军。

## 3. 明确不做

- 不修改正式 v0.7 matrix、paper verdict 或历史结果。
- 不把水煤比伪装成现有 7 通道之一，不使用 `D/uB` 代理替代真实测点。
- 不联合 A4、不做超参扫描、不增加额外结构臂。
- Linux 只执行冻结的 build → sanity → train/eval 命令；失败即留痕停止，不自动改判据重跑。

## 4. 验收门

1. v2.2 构建测试证明 v1 核心键不变、两列来源和顺序正确、质量门 fail-closed。
2. A5 记录测试证明无效工况被排除且窗口不跨无效段。
3. `w_raw=0` 逐位身份测试通过；非零权重能激活且功率有界。
4. 训练集拟合隔离测试通过，validation 值变化不影响 `wc_ref`。
5. Windows 本地契约测试通过后才提交；Linux 用提交哈希运行并输出 ledger、checkpoint、report。

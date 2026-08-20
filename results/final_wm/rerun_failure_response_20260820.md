# T1+R1 重跑失败回执（本地侧修复闭合，2026-08-20）

> 对应：`results/final_wm/rerun_failure_report_20260820.md`（执行侧 23:35 回传）。
> 三项问题全部已修复，118/118 测试通过，可重跑。

## §3 指纹失效（根因，本地侧全责）

- **修复**：`src/final_wm/training.py` 新增 `model_structure_fingerprint()`（状态/边界/
  动作/观测注册表 + transition 先验表），`config_fingerprint` = hash(spec, structure)。
  任何结构级修复（布局、参数、先验值）自动作废旧产物，不再静默 RESUMED。
- **回归测试**：`tests/final_wm/test_training.py::test_config_fingerprint_covers_model_structure`
  ——mock 改动先验表即要求指纹变化。
- 本次部署后所有旧指纹自然不匹配，T1+R1 将全新训练（无需手动清产物）。

## §5 红项（裁定 A 落实）

- `test_spray_sensitivity_recovers_slopes` 容差 1e-6 → **1e-5**，注释注明 aarch64/BLAS
  相对误差 ~4e-8 来源。非矩阵路径，不影响判决。

## §6 --quick 覆写

- quick 档改写独立文件名：`matrix_summary*_quick.json` / `dsyn_verdict_quick.json` /
  `auditpack*_quick.json`，已审计产物不可再被冒烟覆写；
- 冒烟测试已改为同时断言 quick 文件存在 **且** 正式文件名不产生（防回归）。

## 执行侧重跑命令（不变）

```bash
python experiments/final_wm/run_matrix.py --phase matrix \
  --record artifacts/final_wm/canonical_sideA.npz --side A \
  --units t1,r1 --out artifacts/final_wm \
  --properties-npz artifacts/final_wm/iapws_surrogate.npz
```

预期：所有 arm 打印 fresh 训练（旧 v0.2 产物指纹不匹配自动跳过复用），不再出现
state_dict 崩溃。quick 冒烟现在写 `_quick` 文件名，安全。

---

## 追加（2026-08-21 00:35，用户反馈两项）

### A. 训练加速（15h 不可接受）

- 瓶颈定位：rollout 每 batch = horizon 18 × n_substeps 5 = **90 次顺序小算子调用**，
  GPU 上 kernel-launch bound，与算力无关；
- 修复：子步物理体抽取为 `transition._substep`（逐字搬运，数值等价），runner 新增
  **`--compile`** 开关 → `torch.compile(_substep, dynamic=False)` 融合。fp32 不变、
  预算口径（60/10、评估窗口、阈值）一律不动——速度只从实现层出；
- 本机无 cl 编译器，inductor 无法用；已用 `aot_eager` 验证 dynamo 图捕获
  **逐位一致**（max|Δstate|=0.0）。Hermes（gcc 齐全）首次执行侧先做 5 分钟冒烟：

```bash
python experiments/final_wm/run_matrix.py --phase dsyn --quick --compile \
  --out artifacts/final_wm/_compile_smoke
# 通过（不崩、student_val 有限）即可删目录，正式命令加 --compile
```

- 若 --compile 冒烟失败：去掉该旗标按原 eager 路径跑（行为与修复前完全一致）。
- bf16 明确不用：焓量级 ~3000 kJ/kg，bf16 相对精度 0.4% → ±12 kJ/kg 量化误差，
  会污染物理数值与审计口径。

### B. 改动归因纪律（"改动应一个一个加"）

- **指纹再升级**：`config_fingerprint` 现纳入 `src/final_wm` 与 `experiments/final_wm`
  的 git tree hash——任何代码提交（含只改动力学的 revert/二分）都自动作废旧产物，
  单改单跑的归因实验从此安全；
- **本批 ②③④ 归因方案**（不额外花算力）：矩阵自带逐机制诊断——事件研究 H1 渐进性
  对应②、rewetting_ablation 探针对应③、checkpoint 先验读出对应④，一次训练分别读取；
  仅当 T1/R1 相对 v0.2 出现回归时才 git revert 二分；
- **今后纪律**：修复批严格一次一项一次重跑（修正案 ①→⑤ 顺序不变）。

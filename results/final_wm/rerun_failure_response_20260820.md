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

# T1+R1 重跑失败回传（Hermes 执行侧，2026-08-20 23:35）

> 授权来源：用户指示"先把实验继续跑吧"+ 用户裁定 A（红项如实回传不阻塞）；
> registry 状态行"待授权 Hermes 重跑 T1+R1（①全锚定随后）"。

## 1. 执行命令（冻结 README §3 矩阵命令 + --units 子集）

```bash
python experiments/final_wm/run_matrix.py --phase matrix \
  --record artifacts/final_wm/canonical_sideA.npz --side A \
  --units t1,r1 --out artifacts/final_wm \
  --properties-npz artifacts/final_wm/iapws_surrogate.npz
```

## 2. 失败现象（REAL_EXIT=1，40 秒内崩溃，无训练发生）

1. T1 四臂 × 3 seeds 全部打印 `RESUMED (artifacts match spec)` —— spec 指纹比对
   **未识别**修复批②③④引入的模型结构变更，判定旧 v0.2 产物可复用；
2. 进入后续步骤加载 t1 checkpoint 到新模型时崩溃：

```
RuntimeError: Error(s) in loading state_dict for FinalWorldModel:
  Missing key(s) in state_dict: "transition.raw.tau_mix1", "transition.raw.tau_mix2".
  size mismatch for closure.state_loc: shape [9] vs current [11]
  size mismatch for closure.net.0.weight: shape [64, 15] vs current [64, 17]
  size mismatch for observer.state_loc / mu_head / observation.head: 9 → 11
```

## 3. 根因判断（供本地侧修改，执行侧未动代码）

- 修复批给 transition 增加了 `tau_mix1/tau_mix2`（一阶混合时滞状态），state 维 9→11；
- runner 的 spec 指纹（checkpoint/metrics 复用判定）不覆盖模型架构/代码版本，
  因此"spec 变更自动触发重训"的 v0.2 语义对本次代码级变更失效；
- 建议：spec 指纹纳入模型架构指纹或代码 commit（如 matrix_spec 版本号 v0.3
  进指纹），或对修复批明确加版本戳强制重训。

## 4. 环境快照

- git HEAD: ef92ef78e718c063c1c375f570b24b58c49a3eac
- torch 2.11.0+cu130 / CUDA 13.0 / 驱动 580.95.05 / NVIDIA GB10
- 测试：116 passed, 1 failed（见 §5）

## 5. 预检红项（用户裁定 A：如实回传不阻塞）

`tests/final_wm/test_analysis.py::test_spray_sensitivity_recovers_slopes`：
`abs(30.0000011403 - 30.0) < 1e-6` → 1.14e-6 超差 14%。
本机 aarch64/BLAS 浮点差异导致的容差边缘超差（相对误差 3.8e-8），
与矩阵训练路径无关；是否放宽至 1e-5 由本地侧裁定。

## 6. 执行侧观察（--quick 陷阱）

`--quick` dry-run 会**覆写** `--out` 下的 `matrix_summary_sideA.json`（写 quick:True 版本）。
本侧以 --quick 冒烟时冲掉了已审计 summary，已从 6305b50 恢复；
README 的 "--quick dry-run sizes; no verdicts" 描述与实际行为不符，建议：
quick 模式不写 summary，或默认写独立文件名。

## 7. 产物状态

- artifacts/ 已从 6305b50 恢复，git 干净（0 改动）；
- 无训练发生、无新 ledger 条目、无 checkpoint 覆盖。

# final_wm 判别矩阵 — Linux 执行提交（冻结）

> 授权范围：本文件列出的命令与参数。执行侧不改代码/配置/阈值；失败原样回传；
> test 锁定；K1 不解冻。产物目录 `artifacts/final_wm/` 整体回传（含 ledger.jsonl、
> checkpoints/、metrics/、各 summary/report JSON）。
>
> 上游合同：`docs/plans/2026-08-18-final-wm-discrimination-matrix.md`（矩阵 v0.2）、
> `docs/plans/2026-08-18-final-world-model-implementation.md` §4.1。

## 重跑语义（v0.2 起）

runner 具备 run 级断点续跑：`checkpoints/<run_id>.pt` + `metrics/<run_id>.pt` 存在且
spec 指纹匹配（旧产物回退比对 ledger 末次 final 块的 spec）时**跳过重训**，直接复算判决；
spec 变更（如 v0.2 的 T1 预算修正）自动触发对应臂重训，其余臂复用。matrix_summary 每单元
增量落盘，中途崩溃不丢已完成判决。**首轮侧 A 重跑**：O1/B1/J1 臂 spec 未变将自动复用，
T1 四臂按新预算重训，R1 用新 T1 权重复跑；ledger 中首轮重复块按既有约定以末次出现为准。

## 0. 环境准备

```bash
cd <repo>
git checkout main && git pull origin main
python -m pytest tests/final_wm/ -q        # 必须全过（101 项）；任何失败立即停止并回传输出
```

## 1. D-SYN 同型可解性门禁（先于真实数据，必过）

```bash
python experiments/final_wm/run_matrix.py --phase dsyn --out artifacts/final_wm
```

- 产出 `artifacts/final_wm/dsyn_verdict.json`；`verdict=PASS` 才允许进入第 2 步，FAIL 则回传并停止。
- 预算 ≤1 GPU 小时（3 seeds）。

## 2. 双侧桥接（D0 → 每侧 canonical 记录）

```bash
python experiments/final_wm/run_matrix.py --phase split-sides \
  --record <D0 双侧记录路径>/canonical_record.npz \
  --out artifacts/final_wm
```

- 产出 `canonical_sideA.npz` / `canonical_sideB.npz` + meta + `split_sides_report.json`。
- 桥接复跑全部质量门（fail-closed）；任何一侧不过门即回传停止。
- 校验 `canonical_side*_meta.json` 的 `provenance.dual_record_sha256` 与
  `results/final_wm/d0/canonical_manifest.json` 的 sha256 一致；不一致立即停止。

## 3. 判别矩阵（逐侧执行）

```bash
# 侧 A
python experiments/final_wm/run_matrix.py --phase matrix \
  --record artifacts/final_wm/canonical_sideA.npz --side A \
  --out artifacts/final_wm [--properties-npz <真实 IAPWS 网格路径>]
# 侧 B
python experiments/final_wm/run_matrix.py --phase matrix \
  --record artifacts/final_wm/canonical_sideB.npz --side B \
  --out artifacts/final_wm [--properties-npz <真实 IAPWS 网格路径>]
```

- 顺序执行 O1 → T1 → B1 → J1 → R1（runner 内置），产出
  `matrix_summary_sideA.json` / `matrix_summary_sideB.json`、`r1_report.json`、
  `ledger.jsonl`、`checkpoints/`、`metrics/`。
- **热物性**：若可提供真实 IAPWS 网格 npz（`load_grid_properties` 兼容格式，同 legacy
  `iapws_surrogate.npz`），必须经 `--properties-npz` 注入；否则 runner 用解析 fallback 且
  ledger/summary 中 `properties=AnalyticThermoProperties` —— 该运行的全部判决标记为
  **provisional（定性骨架）**，本地审计据此降级处理，不回填为正式判决。
- 预算：每侧 ≤36 GPU 小时（矩阵 §4），两侧合计 ≤72。

## 3.5 证据包（auditpack，判决审计后执行）

证据链全部分析已协议化入仓（`src/final_wm/analysis.py`）：真实对象阀位阶跃事件研究、
persistence 增量基线、喷水灵敏度回归 + 混合冷却参考带、误差地板三锚点、残差负荷分箱、
再湿消融探针。记录级分析只需 canonical 记录；模型探针需训练权重：

```bash
python experiments/final_wm/run_matrix.py --phase auditpack \
  --record artifacts/final_wm/canonical_sideA.npz --side A \
  --checkpoint artifacts/final_wm/checkpoints/t1_closure_cons_seed0.pt --arm closure_cons --seed 0
```

产出 `auditpack_A.json`；论文与证据链文档的数值只准引用该产物口径。

## 4. 回传清单

- `artifacts/final_wm/` 整目录（ledger.jsonl、matrix_summary_side{A,B}.json、dsyn_verdict.json、
  r1_report.json、split_sides_report.json、checkpoints/、metrics/）；
- 执行机的 `git rev-parse HEAD`、`git status --porcelain` 输出（必须干净）、GPU/驱动/PyTorch 版本；
- 任何非零退出或异常栈原样附在回传说明中。

## 禁止事项（执行侧）

- 不修改 `src/final_wm/`、`experiments/final_wm/`、任何配置/阈值/种子；
- 不访问 split id=2（test）；不对失败运行做补跑/调参（预算内失败原样回传）；
- 不执行 K1、MS4 或任何 `experiments/phase3_5/` 历史命令；
- 不把 `results/final_wm/d0/` 下执行方自写脚本当作生产管线（桥接后唯一入口是本 runner）。

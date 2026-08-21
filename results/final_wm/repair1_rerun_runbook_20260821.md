# 修复批① 重跑 runbook（2026-08-21，执行侧）

前置：`git pull` 至 ≥ 本批提交（①设计冻结
`docs/plans/2026-08-21-repair-batch-1-design.md`；本地 128/128 + D-SYN quick
门禁已过）。指纹变化为预期：所有旧 checkpoint 失效，全量重训。

**顺序执行（单命令链，串行）：**

```bash
# 1) D-SYN 全量门禁（新栈必须先自证可辨识，~1h）
python -m experiments.final_wm.run_matrix --phase dsyn \
  --out artifacts/final_wm --device cuda --compile --tf32

# 2) T1 减臂 closure_cons x3 + R1（~5h；判决纪律不变）
python -m experiments.final_wm.run_matrix --phase matrix \
  --record data/canonical_sideA.npz --side A --units t1,r1 \
  --arm-filter closure_cons --properties-npz data/iapws_surrogate.npz \
  --out artifacts/final_wm --device cuda --compile --tf32

# 3) O1 三臂 x3 seed（新观测器语义下重判，~5h）
python -m experiments.final_wm.run_matrix --phase matrix \
  --record data/canonical_sideA.npz --side A --units o1 \
  --properties-npz data/iapws_surrogate.npz \
  --out artifacts/final_wm --device cuda --compile --tf32

# 4) auditpack 带 CF/D1 新探针，对 closure_cons 三 seed 各一次（各 ~20min）
for S in 0 1 2; do
  python -m experiments.final_wm.run_matrix --phase auditpack \
    --record data/canonical_sideA.npz --side A \
    --checkpoint artifacts/final_wm/checkpoints/t1_closure_cons_seed$S.pt \
    --arm closure_cons --seed $S --properties-npz data/iapws_surrogate.npz \
    --out artifacts/final_wm --device cuda
done

git add artifacts/final_wm && git commit -m "artifacts: repair-batch-1 rerun (dsyn + T1 closure_cons x3 + R1 + O1 + auditpack w/ CF-D1 probes)" && git push
```

注意：

- 任一阶段失败原样回传，不本地改码；
- auditpack 会**覆写** `auditpack_A.json`——每 seed 跑完先
  `cp auditpack_A.json auditpack_A_seed$S.json` 再跑下一个（runner 现行为按 side
  命名，未按 seed 分文件；本批维持现状，本地审计时用分 seed 副本）；
- 旧指纹产物（v0.2/修复批234）已在仓内留档，不删；
- 预算/seed/判决纪律全部不变（T1 60 epoch/early-stop-10，R1 现行冻结判据）。

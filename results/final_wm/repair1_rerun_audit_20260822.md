# 修复批①重跑审计（2026-08-22，本地设计侧）

对象：Hermes 回传 922d599/774ec9f/0e4cf37（dsyn×3、t1 closure_cons×3、
R1、auditpack×3、 ledger/metrics）。审计口径：逐位复核 + 指纹/时间戳交叉验证。

## 判决表（新鲜度已核实）

| 单元 | 判决 | 新鲜度 | 关键数字 |
|---|---|---|---|
| D-SYN 全量 | **PASS ×3** | ✓ 新栈（890bd15） | student NLL −0.03/−0.10/−0.30 vs skeleton ~313-317（≈100% 削减 ≫ 30% 判据）；CF-1 探针随门禁运行：终值符号一致 1.0 |
| T1 closure_cons×3 | 重训完成（减臂设计下单臂无比对判决） | ✓ 新栈 | best_val 1.214/1.260/1.228（旧栈 1.324–1.65）；H18 MAE 2.65/2.79/2.62°C |
| R1 | **REJECTED** | ✓ 新栈（读 t1 ckpt 8/22） | seed0 frac_negative 28/32 < 1.0 冻结门；seed1/2 全项通过；**leakage 三 seed 全清白（1.09/0.87/0.55pp ≪ 5pp）** |
| O1 | **无效——陈旧产物** | ✗ 逐位等于 v0.2 | 见 F1；当前 matrix_summary 的 o1 块**不得引用** |

## ①成效对照（目标签名 → 实测）

- sh1_in H1 箱均值 **5.3–13.3°C → 0.35–1.15°C**（三 seed auditpack 一致）；
- 出口 t=0 锚 −18.1/−6.3 → **−3.6/−0.38°C**（残余 sh1_out −3.6°C 为结构性，已登记 AE）；
- R1 泄漏签名（旧栈 seed1 5.15pp）在新栈**完全消失**——支持"欠拟合/初态失配"
  根因说；方向探针 closure 臂均值全负（物理方向正确）；
- auditpack 三 seed 齐全（seed 分副本已按 runbook 留存）；两笔执行侧热修
  （fc549bb numpy JSON hook、4282c3d detach）复审通过：最小、加性。

## F1（完整性，已修复+回归测试）

**症状**：matrix_summary 的 o1 块与 v0.2 逐位相同；ledger 无 890bd15 之后的
o1 训练条目；o1 checkpoints/metrics 时间戳仍为 8/20。
**根因**：`_try_resume` 的 legacy 平铺格式（无指纹）经 `_spec_matches_ledger`
仅按 spec 字段放行——①只改代码不改 spec 字段 → 旧 O1 判决被当新判决重发。
同类洞：J1 staged-boundary 的 ledger 匹配续跑。
**次生**：`dump_summary` 整文件覆写 → o1 调用抹掉了 t1/r1 块（runbook 只预警了
auditpack，漏了 summary）。
**修复**：legacy 格式永不续跑（强制重训）；J1 同级修复；summary 按 unit 键合并。
回归测试：`test_legacy_metrics_blob_never_resumes`、`test_summary_merges_across_invocations`。
全套 130/130。

## F3（物理，新栈首次可测）：再湿项幅值使 v1 下游响应反号

新探针暴露：v1 +5% 阶跃下 closure_cons seed0 的下游响应 **sh2_in +2.7°C/满开度、
final +0.5°C（60s）/ +1.2°C（600s）——方向错误**（sh1_out 本身 −5.4°C 正确）。
证伪实验（aW1/aW2→−30，q_w≈0）：sh1_out −11.9、**sh2_in −0.3、final −0.05，全部转负**。
解释：③把"壁膜再蒸发"建成壁→蒸汽的导热导管（金属 −q_w、蒸汽 +q_w，能量自闭合），
该机制本身成立；但 v1 在训练数据中冻结（σ=0.005，D1 数据侧 n=1-2/箱不可约束），
**两个反向机制（喷水直接混合冷却 vs 再湿导热加热）的拆分在侧A上不可辨识**，
训练把 aW 学到 ~1.26 后加热项在下游压过冷却项。R1 seed0 的边界失败（28/32）
与此幅值主导一致（再湿消融下 v2 响应从 −0.061 增强到 −0.333）。

**建议（修正案 v0.4 提案，待批准）**：增设第一类消融臂 `closure_cons_norew`
（aW 冻结 −30，不可学习），quick 档与 closure_cons 同预算对跑，判决性判据：
(i) v1/v2 方向探针；(ii) quick val NLL（ parsimony：消融臂不差于 intact 则取消融臂）；
(iii) H18 MAE。若消融臂胜 → 论文口径变为"最小闭包 SUPPORTED；再湿机制在侧A
不可辨识"（强负结果，FMTS 欢迎）；若 intact 胜 → 保留③并在论文披露 v1 下游
反号为已知结构局限。**在判决前不改 R1 门、不改③主路径。**

## 后续 runbook（执行侧）

```bash
git pull   # 须 ≥ 本审计提交（含 resume 洞修复），否则 O1 仍会复陈
# 1) O1 三臂 x3 全量重跑（~13h；①修正案所令，且论文 O1 声明依赖）
python -m experiments.final_wm.run_matrix --phase matrix \
  --record data/canonical_sideA.npz --side A --units o1 \
  --properties-npz data/iapws_surrogate.npz \
  --out artifacts/final_wm --device cuda --compile --tf32
# （v0.4 若批准：先 quick 对跑 closure_cons vs closure_cons_norew，
#   再按判决决定是否以 v0.4 栈重跑 T1+R1，避免双重全量周期）
```

## 日程影响

- 8/23 决策点输入已齐：①达标（精度层），R1 REJECTED（边际+机制已定位），
  O1 待重跑；论文 v2 草稿（执行侧已编译 PDF）中的 O1 数字须标 stale-pending。
- CWM 整合提案（3046741，执行侧文档）排入 8/23 评审，不阻塞当前链。

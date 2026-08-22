# 矩阵修正案 v0.4：再湿项第一类消融臂（audit F3 响应）

日期：2026-08-22。状态：**已批准（用户 09:06）**。前置证据：
`results/final_wm/repair1_rerun_audit_20260822.md` F3。

## 发现回顾（F3）

①栈（五点锚定后）首次可测：closure_cons seed0 的 v1 +5% 阶跃下游响应
**反号**（sh2_in +2.7°C/满开度、final +0.5°C @60s；sh1_out −5.4°C 本身正确）。
aW→0 消融（③探针同款）后全部转负（sh2_in −0.3、final −0.05）。
③的"壁膜再蒸发=壁→蒸汽导热导管"机制本身能量自闭合（金属 −q_w、蒸汽 +q_w），
但 v1 在训练数据冻结（σ=0.005）→ **直接混合冷却与再湿加热的拆分在侧A不可辨识**，
训练后 aW≈1.26 的幅值使加热项在下游压过冷却项。R1 seed0 的 28/32<1.0
边界失败与该幅值主导一致（消融下 v2 响应 −0.061→−0.333）。

## 修正案内容

1. **新增第一类消融臂 `closure_cons_norew`**：closure_mode 后缀 `_norew` →
   `TransitionConfig.rewet_ablate=True`：aW1/aW2 raw 冻结于 −30
   （softplus≈0，与审计证伪探针同设置），requires_grad=False。
   - 仅 TrainSpec.closure_mode 字符串不同 → 既有臂指纹**不变**；
     TransitionConfig 不入结构指纹（已核）；
   - 该臂为**证据臂**：不进 T1 冻结 verdict `nested` 对；判决走本条预注册规则；
   - closure 注入语义、其余参数、预算（60/10）与 closure_cons 完全一致。
2. **判决规则（预注册，两级）**：
   - **本地判决档**（epochs=10, patience=4, 其余同全档；quick 档 2 epoch
     已实证无区分度：两臂 best_val≈10800）：采纳 norew 当且仅当
     (i) intact 的 v1 下游反号复现（sh2_in 或 final >0 @60s）且 norew 全为负；
     (ii) norew 的 v2 方向 frac_negative 不低于 intact；
     (iii) 节俭性：best_val_NLL(norew) ≤ intact + 0.05。
   - **执行侧全档判决**：若本地采纳 norew → 全档对跑
     `closure_cons ×3 vs closure_cons_norew ×3` + R1 双栈探针 +
     auditpack 双栈；三 seed 一致（方向全对且 val NLL 不差于 intact+0.05）则
     **侧A生产臂切换为 norew**，论文口径="最小闭包 SUPPORTED；再湿机制侧A
     不可辨识"（负结果声明，FMTS 明确欢迎）。否则保持③现状，v1 下游反号
     作为已知结构局限入论文"限制"节。
3. **门纪律**：判决完成前不改 R1 门、不改③主路径默认值（rewet_ablate
   默认 False）；R1 重跑只在全档判决后按所选栈执行一次。
4. **排期嵌入**：O1 重跑（890bd15 栈）与 v0.4 本地判决并行；若本地采纳
   norew，O1 与 T1-norew×3 在同一执行周期内完成（O1 基准 closure 切
   `conservative_norew` 由后续修正案在执行侧全档判决后决定——O1 本轮仍按
   现栈重跑以修复陈旧判决，此为恢复完整性的最低要求）。

## 已实现（本地）

- `contracts.py`: TransitionConfig.rewet_ablate（默认 False，保护冻结③语义）；
- `transition.py`: 消融冻结逻辑（raw=−30, requires_grad=False）；
- `training.py`: `_norew` 后缀解析（`removesuffix`——首版误用 `partition`
  取尾巴为空串，契约测试当场捕获，已修）；
- `matrix_spec.py`: t1_specs 增臂（指纹隔离已验）；
- 契约测试 `test_rewet_ablate_arm_freezes_gains`；
- 判决实验 `experiments/final_wm/v04_rewet_decision.py`（预注册规则即代码）。

## 风险与边界

- 本地判决档 10 epoch 的 aW 学习不充分风险：判决主证据是**结构性方向**
  （路径存在性，epoch 0 即现），val NLL 仅作节俭护栏；全档三 seed 才是终审。
- 若 norew 被采纳：③设计稿的导管机制不被否定——结论收窄为"侧A不可辨识"，
  侧B/AE 阶段再议辨识性（v1 在侧B是否活动待查）。

## 本地判决档结果（2026-08-22 15:xx，10ep/patience4/同预算对跑）

`results/final_wm/v04_rewet_decision_20260822.json`：

| 指标 | closure_cons | closure_cons_norew |
|---|---|---|
| best_val NLL | 3.545 | **3.494**（−1.4%） |
| H18 MAE | 5.52°C | 5.63°C（+0.11） |
| v1@60s sh1_out | −20.2 | −22.9 |
| v1@60s sh2_in | **+4.63（反号）** | **−1.78** |
| v1@60s sh2_out | **+4.53（反号）** | **−1.75** |
| v1@60s final | **+2.31（反号）** | **−0.64** |
| v1@60s sh1_in | −0.28 | 0.00（上游无通路，物理精确） |
| v1@600s mean / frac_neg | −0.113 / 1.00 | −0.411 / 1.00（更强） |
| v2@600s mean / frac_neg | −0.231 / 1.00 | −0.454 / 1.00（更强） |

三判据全真 → **本地判决：采纳 norew**。备注：intact 600s 长窗响应为负
（金属冷却最终主导），反号限于 60s 下游瞬态——控制相关视界（30-180s）
恰在此区间。终审（全档三 seed 双栈探针）runbook：
`results/final_wm/v04_full_tier_runbook_20260822.md`。

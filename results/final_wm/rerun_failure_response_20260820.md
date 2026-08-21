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

---

## 追加 2（2026-08-21 03:30）：并行 worker + tf32（Linux `b28bf25` 之上的加速审查）

### 本地侧对该 push 的补强

- **J1 并行防护**：`--arm-filter joint` 时 staged_boundary 的暖启动 checkpoint 属于另一
  worker，原来会 torch.load 崩溃；现在 fail-skip（聚合 pass 经 resume 补齐）；
- **`--tf32`**：tensor-core fp32 matmul（GRU/MLP），物理逐元素算子不受影响；
- **旗标入 ledger**：每条 final 记录 `flags={compile_substep, matmul_precision}`——
  **同一判决单元的所有臂必须旗标一致**，审计时检查均匀性（数值轨迹属旗标函数）。

### 并行 worker runbook（单 GB10，CPU 20 核空闲 + GPU 41% 有余量）

T1 四臂×3 种子 = 12 次训练，单进程串行 ≈ 15h；4 worker 并行（每 worker 单臂）+
`--compile --tf32` 预期 **3-5×**：

```bash
git pull
# 1) 冒烟（两档各 5 分钟，全过才继续）
python experiments/final_wm/run_matrix.py --phase dsyn --quick --compile --tf32 \
  --out artifacts/final_wm/_smoke && rm -rf artifacts/final_wm/_smoke
# 2) 4 个并行 worker（tmux/后台各一；同一 --out，run_id 互不冲突）
for ARM in physics_only closure_cons closure_steam latent4; do
  python experiments/final_wm/run_matrix.py --phase matrix \
    --record artifacts/final_wm/canonical_sideA.npz --side A \
    --units t1 --arm-filter $ARM --out artifacts/final_wm \
    --properties-npz artifacts/final_wm/iapws_surrogate.npz --compile --tf32 &
done; wait
# 3) R1 依赖 closure_cons 全部种子，放最后单 worker
python experiments/final_wm/run_matrix.py --phase matrix \
  --record artifacts/final_wm/canonical_sideA.npz --side A \
  --units t1,r1 --out artifacts/final_wm \
  --properties-npz artifacts/final_wm/iapws_surrogate.npz --compile --tf32
#    ↑ 聚合 pass：t1 全部 RESUMED，顺带产出判决；r1 全量
```

- 全程所有 worker 旗标必须一致（`--compile --tf32` 都带或都不带）；
- GPU 显存：batch32 小模型，4 worker 共存无压力；若 OOM 则降为 2-3 worker；
- tf32 数值平价门：冒烟时对比 `--tf32` 与不带的学生 val NLL，差 <1% 才放行。

---

## 追加 3（2026-08-21 10:50）：seed0 重跑审计（action_signal_analysis_20260821.md）

### 审计裁定

1. **leakage 门伪影说：接受**。打乱对照设计正确、数值自洽（r1_report blind 0.907 与报告
   §2.2 一致）。已协议化进 `leakage_probe`：新增 `aware_shuffled` 臂，判据改为
   **`leakage_delta = improvement(真) − improvement(打乱) > 5%`**。按新规则 seed0
   Δ=0.64% < 5% → 三门全过，R1 seed0 **暂定 PASS**（正式判决仍需 seed1/2）。
2. **时滞 caveat：部分接受**。本地用回传 checkpoint 直接复算：240 步（≈稳态）
   −0.194°C/2% vs 混合参考 −0.53~−1.48 → **稳态差距 2.7–7.6×**（报告瞬态口径 6–16×
   减半但未闭合）。R1 探针新增 `direction_steady`（240 步，证据位，不入判决门）。
3. **物理参数漂移记录**（closure_cons seed0）：tau_mix1 474s（5.9×锚定）、tau_mix2 183s、
   **th2 0.44×**（喷水增益被学弱）、M 3.5–7.4×、UA2 0.39×、tau_evap 0.55×。
   动作增益缺口与 th2 下调互相印证；可辨识性问题移交 roadmap §1（参数 MLP/锚定修订）。
4. **并行撤回**：4 路实测 0.9× 串行（19.4ks/臂争用），恢复单进程串行 runbook。
   `train_arm` 新增分段计时（data/step/eval 入 ledger `timing`）——下一跑直接定位
   19.4ks 的去向，不再猜。
5. 旗标均匀性已核：seed0 四臂 `compile_substep+tf32` 一致 ✓。

### 待用户裁定

- **T1 减臂**：接受"只训 closure_cons×3 seeds"（~4-5h，R1 链够用；T1 嵌套问题 v0.2 已判，
  修复批下的 T1 复判随修复批①再议）？还是四臂全保（~15h）？

---

## 追加 4（2026-08-21 11:05）：用户裁定 + 阀门非线性修正判读

### 裁定

**T1 减臂批准**：只训 `closure_cons`×3 seeds。执行命令（单进程串行，一次出 R1 三 seed 判决）：

```bash
python experiments/final_wm/run_matrix.py --phase matrix \
  --record artifacts/final_wm/canonical_sideA.npz --side A \
  --units t1,r1 --arm-filter closure_cons --out artifacts/final_wm \
  --properties-npz artifacts/final_wm/iapws_surrogate.npz --compile --tf32
```

预计 4.5-5.5h。arm-filter 下 T1 判决自动跳过（v0.2 已判），R1 判决正常产出。

### 增益缺口的判读修正（用户指出，phase1 证据支持）

1. mixing_reference 是**名义线性增益上界**（零延迟 + 阀位→流量线性 + 固定 Δh/cp 假设）；
2. 真实阀门非线性（等百分比特性），局部增益随开度点变化；phase1 执行机构证据
   （`actuator_identity_conclusion.json` D1：|Δcmd|>0.1% 仅 0.89% 步、>3% 零出现、
   自相关 0.998）表明真实数据只有窄区间微调激励——模型可辨识的是**工作点局部增益**；
3. 绝对阀位承载工作点信息（模型动作编码本就用绝对阀位 dsw=v×th）；
   `th2` 学到 0.44× 先验，可解读为拟合局部增益，而非单纯"学弱"。

**修正结论**：稳态 2.7–7.6× 缺口是对名义线性上界的距离，不等于对真实对象的失真。
R1 判决门维持三项（方向/leakage/blindness），量级不入判据。阀门非线性假设的可检验化
（按绝对开度分箱的局部增益曲线探针）列入可辨识性主线待办，不在本期。

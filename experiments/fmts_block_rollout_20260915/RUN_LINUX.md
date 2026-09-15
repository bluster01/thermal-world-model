# FMTS-BLOCK1 Linux执行单：180秒整块回填，无训练

先完整阅读 [注册](../../docs/fmts2026/PREREG_BLOCK_ROLLOUT_20260915.md)。本包只有固定checkpoint推理；严禁调用旧exp_048/054/058（其模块含历史训练/test代码），不执行原生H60延长、不采用stride1回填、不新增训练。

作者已于2026-09-15授权“push执行”。从origin/main取得本包后可按本执行单开始，不再等待额外训练授权；本授权仅覆盖9个固定权重推理单元及重放审计。领取时先检查是否已有本任务的claim或结果，避免重复执行。领取后在本包状态文件记录source commit、匿名worker标识和实际开始时间并回传；发布状态不等于领取或运行状态。

## 领取与预检

本包提交并发布后，Linux再 `git fetch origin main`、`git pull --ff-only origin main`。保留本地修改，禁止reset。继续使用生成v0.2/GNR1的原Python/PyTorch环境、9个原best.pt、canonical SideA v2.2数据与IAPWS grid；不升级包。

以下命令均从仓库根目录运行。

```bash
python -m pytest tests/final_wm/test_fmts_block_rollout.py -q
```

这些测试用确定性工程夹具，不运行优化器或真实电厂拟合，也不能当科学结果。正式推理入口自动核验所有冻结SHA、原256窗资格、三个arm×三seed和H18重放门。失败即保留并回报，不改变容差、筛选、权重、协议或样本。

## 正式推理与全重放

以下私有路径为新建输出目标，必须在Git仓库之外且尚不存在。如需换存储位置，只能改输出路径，不能改变协议或输入身份。

```bash
python -m experiments.fmts_block_rollout_20260915.run \
  --record /home/bluster/final_wm_v07_full_reissue_v1/inputs/canonical_sideA_v2.npz \
  --properties /home/bluster/final_wm_v07_full_reissue_v1/inputs/iapws_surrogate.npz \
  --private-out /home/bluster/fmts_block1_private_20260915 \
  --device cuda

python -m experiments.fmts_block_rollout_20260915.audit \
  --private-out /home/bluster/fmts_block1_private_20260915 \
  --record /home/bluster/final_wm_v07_full_reissue_v1/inputs/canonical_sideA_v2.npz \
  --properties /home/bluster/final_wm_v07_full_reissue_v1/inputs/iapws_surrogate.npz \
  --device cuda \
  --save /home/bluster/fmts_block1_private_20260915/audit.json \
  --public-receipt results/fmts_block_rollout_20260915/linux_block1_receipt
```

预期9/9推理单元，每个共同窗口调用7个18步预测块，最后只计12点，共120点。审计须 `complete=true`、`cells_replayed=9`、`training_updates=0`。只检查保存文件时 `complete=false` 是有意的；没有真实数据重放不能用“全审计通过”。若某单元失败，不补跑搜索或只回传成功种子。汇总排除原因可能重叠，不能将原因计数相加当排除总数。

## 回传内容与边界

1. 公共Git只提交 `results/fmts_block_rollout_20260915/linux_block1_receipt/`（汇总指标、哈希、审计状态）以及本包的 `experiment_state.json` / `RESULTS.md` 状态回执。不得提交原记录、checkpoint、新增真实历史或逐时预测，禁止 `git add -f *.npz`。
2. 私有输出目录完整交给作者：identity、manifest、256窗history_h18、eligibility、共同窗observed、9组prediction/报告、summary、progress、audit及任何failure。使用作者已有私有传输渠道；若渠道未就绪，明确标记 `private_traces_returned=false`，不要把Git汇总发布当作画图数据已回传。
3. 记录实际source commit、完成数、真实重放数、public receipt hash和私有bundle是否交付。只有两部分均到作者侧才设 `results_returned=true`；作者接收审计之前保持 `audited=false`。
4. 不修改正文、现有图、parent/GNR/M7/CORE1结果或旧实验状态；不作模型晋级或现场响应正确性的结论。

作者侧可先在无原记录时检查私有工件：

```bash
python -m experiments.fmts_block_rollout_20260915.audit \
  --private-out /path/outside/repository/returned_private_bundle \
  --save /path/outside/repository/returned_private_bundle/author_artifact_audit.json
```

此命令即使工件检查通过也退出1（`complete=false`，无真实重放）；结合Linux完整重放回执审核，不覆盖原audit.json。

## 后续画图（数据回传审计后）

- 原点左侧：96点实测主汽温（−950至0秒）；右侧：10至1,200秒真实值及三类模型三seed预测，标示每180秒回填边界。
- 固定row138、203不重新挑选；长程不合格的案例明确缺失，不按预测效果换图。
- 图注写明“仅主汽温整块反馈；其他温度与动作/边界使用同步记录上下文”。不要标为完全自主open-loop，也不要与原生H60混用。
- 双阀响应继续使用原H18配对差分证据，不画成有现场反事实真值的曲线。

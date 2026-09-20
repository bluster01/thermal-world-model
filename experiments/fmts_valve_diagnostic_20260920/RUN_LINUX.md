# FMTS-VD1 Linux执行单：固定权重阀门诊断

先读[注册](../../docs/fmts2026/PREREG_VALVE_DIAGNOSTIC_20260920.md)。作者2026-09-20授权诊断；本轮无训练许可。代码未发布前不要执行；发布后领取须记录实际source commit和时间，检查是否已有本任务的领取/输出，避免重复执行。保留工作区修改，禁止reset。

使用v0.2/GNR1原环境和原九个检查点。以下路径沿用此前BLOCK1实际记录，若不在该处，先按SHA找回同一输入，不能换另一版本数据、重训练或下载替代checkpoint。

## 1. 获取与工程预检

在包含本包的冻结提交上运行：

```bash
python -m pytest tests/final_wm/test_fmts_valve_diagnostic.py -q
```

这些测试使用确定性工程夹具，不是电厂实验。诊断包只实现no-grad推理；正式入口核验Linux、真实源身份、九个checkpoints和原窗口。不运行旧exp_025，不启动其他任务。

## 2. 一次性正式运行

私有目录必须尚不存在、位于Git仓库外；已有目录不得清空或覆盖。日志也放仓库外。可修改输出路径，不能改科学设置。

```bash
/usr/bin/time -v -o /home/bluster/fmts_vd1_20260920_resource.txt \
  timeout 2h python -u -m experiments.fmts_valve_diagnostic_20260920.run \
  --record /home/bluster/final_wm_v07_full_reissue_v1/inputs/canonical_sideA_v2.npz \
  --properties /home/bluster/final_wm_v07_full_reissue_v1/inputs/iapws_surrogate.npz \
  --private-out /home/bluster/fmts_vd1_private_20260920 \
  --device cuda \
  > /home/bluster/fmts_vd1_20260920_stdout.log 2> /home/bluster/fmts_vd1_20260920_stderr.log
```

先记录退出码。若非0，保存全部已生成目录/日志和失败回执，不重跑、不改门限、不切CPU或其他精度偷偷补齐。超时可能来不及写failure.json，非0退出码和缺少最终manifest本身即未完成。

预期：9个单元，每个原256窗预测重放+64窗响应重放，之后64窗×78场景；training_updates=0。诊断出现黑箱强响应、提前响应、正响应或融合模型弱响应都不是执行失败，全部照实保留。不得人为改动作使结果符合预设。

## 3. 保存数组审计

只有正式运行exit=0后执行，公共回执目标同样不得存在：

```bash
python -m experiments.fmts_valve_diagnostic_20260920.audit \
  --private-out /home/bluster/fmts_vd1_private_20260920 \
  --public-receipt results/fmts_valve_diagnostic_20260920/linux_vd1_receipt
```

预期`complete_artifact_audit=true`、9单元、702场景、18原协议重放检查、`new_probe_full_inference_replay=false`。最后一项必须保持false：这次重算保存数组的剂量、support和全部统计，不宣称第二轮完整新探针推理。

## 4. 回传

- 私有渠道：完整私有输出目录、stdout/stderr/resource、退出码。目录包含完整23通道历史、真实未来上下文、逐时预测，禁止放公共Git。不再只寄温度曲线，因为本轮需要复核动作和边界。
- 公共Git：只回传本任务的`linux_vd1_receipt`和`experiment_state.json`执行事实（source commit、开始/完成、退出码、9单元完成数、私有交付状态）。不更新旧结果、论文、选模结论或其他任务状态。
- 科学解释由作者侧审计；Linux不作“恢复真实响应”“黑箱失败”判决。直到公开回执与私有数组都收到才记results_returned；作者复核后才记audited。

## 作者侧要优先看什么

1. 各历史位置的signed/absolute响应，而非只看末点。
2. 未来相同脉冲的动作前变化与动作后60秒变化，开度三箱逐一看覆盖。
3. W-only是否敏感、joint是否非加性；不把人为W剂量当阀门真实喷水转换。
4. 原记录动作活跃窗口中，真实/预测主汽温的level和increment误差。它与阀门因果响应分开解释。
5. 所有种子和正反动作并列，不能只挑符合叙事的场景。

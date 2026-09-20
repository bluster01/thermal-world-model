# FMTS-VD1 本地验证（2026-09-20）

状态：`ready_for_linux_unpublished`。真实推理尚未开始，没有新增科学结论。

## 已完成

```text
python -m pytest tests/final_wm/test_fmts_valve_diagnostic.py -q
19 passed in 5.81s

python -m pytest tests/final_wm/test_fmts_valve_diagnostic.py \
  tests/final_wm/test_fmts_mainsteam_experiment.py \
  tests/final_wm/test_fmts_core.py \
  tests/final_wm/test_fmts_block_rollout.py -q
53 passed in 27.53s
```

- 19项新增测试覆盖78场景计数、history/future隔离、时刻与时长、截断实际剂量、W-only/joint、提前响应检测、异号抵消、日等权、空开度箱、增量误差、重复窗口、权重不变、Windows正式执行拒绝、跨平台源文本哈希，以及9单元/702场景的工程夹具端到端保存与审计、覆盖拒绝和篡改检测。
- 端到端测试明确mock数据来源与模型，使用确定性ToyRecord/ToyModel；不是模拟现场来补科学证据，不计为Linux执行或真实重放。
- 原BB/GRU/GNR三种子共9个正式检查点均经过原加载器SHA校验及`strict=True`加载；未向模型输入真实记录。
- 本地IAPWS文件SHA `9fd7a1dba96a5b968661f644fa185ac755266c85853d689d065432a2d41f6e92` 与父协议一致。
- 两个正式CLI的`--help`可用；`git diff --check`无报错。

## 尚未完成

- canonical Side A v2.2真实输入未在本地release树发现。BLOCK1私有zip的`history_h18.npz`只有主汽温history/target/timestamps，缺两阀、七边界和九扩展，不能用于新探针。
- 原256窗/64窗真实模型重放、新探针、真实开度箱覆盖、W输入依赖、模型时间一致性均等待Linux正式回传。
- 未改论文、旧实验结果、旧README/TODO；本包独立保存注册、计划、执行单和状态。
- 没有推送、领取或执行回执。不得把本地测试通过表述为阀门响应通过。

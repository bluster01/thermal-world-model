# 发布软件验证

日期：2026-09-23。此记录验证执行包的软件行为；工业训练结果由 Linux 执行后产生。

发布前本地合成检查 **45 / 45 通过**。测试环境为 Python 3.11.7、PyTorch 2.5.0 CPU、NumPy 1.26.4、pandas 2.1.4、pytest 7.4.0。

```bash
python -m pytest \
  experiments/unified_experiments_20260923/test_models.py \
  experiments/unified_experiments_20260923/test_runner.py \
  experiments/unified_experiments_20260923/test_prepare_linux.py \
  experiments/unified_experiments_20260923/test_summarize.py \
  experiments/unified_experiments_20260923/test_execution.py -q
```

| 检查文件 | 项数 | 覆盖范围 |
|---|---:|---|
| test_models.py | 10 | history-only、目标顺序、H60/H180前缀、梯度、局部储存收支/耗散及SCR正性 |
| test_runner.py | 7 | 逐目标损失、增量双端掩码、持续值基线、原单位指标、checkpoint与全矩阵评测门 |
| test_prepare_linux.py | 13 | 合成CSV重建、源身份、数组/标准化/读取配置篡改、Linux限制及私有目录 |
| test_summarize.py | 8 | 三种子均值/样本标准差、配对差值、完整矩阵、支撑标签与未定义指标 |
| test_execution.py | 7 | 中断恢复、不可变最佳权重引用、完整起点统计、产物提交顺序、重新验数据与进程锁 |

正式执行入口要求 Linux CUDA，逐次核对发布清单及实际私有数据身份。数据预运行、45次正式拟合、统一H60/H180评价和汇总均写入Linux私有目录。公开运行状态以远端实际领取及执行回执为准。

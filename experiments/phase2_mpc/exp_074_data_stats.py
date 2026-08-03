#!/usr/bin/env python3
"""exp_074_data_stats.py — 压线实验数据统计: 阀位范围/SP分布/571确认"""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
from experiments.phase1_dynamics.exp_025_unified_benchmark import test_raw, VALVE_IDX, SP_IDX, TARGET_IDX

v = test_raw[:, VALVE_IDX]
print('valve dims shape:', v.shape)
for j in range(2):
    col = v[:, j]
    print(f'valve{j}: min {col.min():.2f} max {col.max():.2f} p5 {np.percentile(col,5):.2f} '
          f'p50 {np.percentile(col,50):.2f} p95 {np.percentile(col,95):.2f} mean {col.mean():.2f}')
sp = test_raw[:, SP_IDX]
print(f'SP: min {sp.min():.2f} max {sp.max():.2f} p50 {np.percentile(sp,50):.2f} mean {sp.mean():.2f}')
T = test_raw[:, TARGET_IDX]
print(f'T:  min {T.min():.2f} max {T.max():.2f} p50 {np.percentile(T,50):.2f} mean {T.mean():.2f}')
print(f'SP-PV mean: {(sp - T).mean():.2f}')
for j in range(2):
    col = v[:, j]
    print(f'valve{j}: >95: {(col>95).mean()*100:.2f}%  >90: {(col>90).mean()*100:.2f}%  '
          f'<5: {(col<5).mean()*100:.2f}%  <10: {(col<10).mean()*100:.2f}%')
# 571 附近统计: SP 落在 [570,572] 的比例
in571 = np.mean((sp >= 570) & (sp <= 572)) * 100
print(f'SP in [570,572]: {in571:.1f}%')
print(f'SP p90 {np.percentile(sp,90):.2f} p95 {np.percentile(sp,95):.2f} p99 {np.percentile(sp,99):.2f}')
print(f'SP==571±0.5: {np.mean(np.abs(sp-571)<=0.5)*100:.1f}%')
print(f'SP==567±0.5: {np.mean(np.abs(sp-567)<=0.5)*100:.1f}%')
# 阀位安全带建议: 数据范围 + 余量
for j in range(2):
    col = v[:, j]
    print(f'valve{j} 建议安全带: [{np.percentile(col,2):.1f}, {np.percentile(col,98):.1f}]  '
          f'现运行区间 2-98% 宽 {np.percentile(col,98)-np.percentile(col,2):.1f}')

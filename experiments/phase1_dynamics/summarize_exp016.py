#!/usr/bin/env python3
"""汇总 exp_016 全矩阵 — 只读 results.json"""
import json
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

confs = ['L0_W1_l0.00','L0_W1_l0.01','L0_W1_l0.10','L0_W1_l1.00',
         'L0_W0_l0.10','L0_W2_l0.10','L3_W1_l0.10','L6_W1_l0.10']
base = '/home/bluster/projectA/thermal-world-model/results'

print("="*100)
print("ROLLOUT MAE (test, 主汽温 °C, H=18步@10s)")
print("="*100)
hdr = f"{'config':<16}" + "".join([f"t{i:<7}" for i in range(18)])
print(hdr)
for c in confs:
    d = json.load(open(f"{base}/exp_016_{c}/results.json"))
    m = d['rollout_mae']
    print(f"{c:<16}" + "".join([f"{v:<7.3f}" for v in m]))

for adim, name in [(1, '二级减温阀(主执行器)'), (0, '一级减温阀')]:
    print()
    print("="*100)
    print(f"SENSITIVITY — {name} 扰动±1/2/5/10 → ΔT@t1/t3/t8/t12")
    print("="*100)
    print(f"{'config':<16}" + "".join([f"{p:<24}" for p in ['+10','+5','+2','+1','-1','-2','-5','-10']]))
    for c in confs:
        d = json.load(open(f"{base}/exp_016_{c}/results.json"))
        s = d['sensitivity'][f'action_{adim}']
        row = f"{c:<16}"
        for delta in ['10.0','5.0','2.0','1.0','-1.0','-2.0','-5.0','-10.0']:
            vals = "|".join(f"{s[f'{delta}_{t}']:+.3f}" for t in [1,3,8,12])
            row += f"{vals:<24}"
        print(row)

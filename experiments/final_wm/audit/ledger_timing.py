"""转正版: 逻辑与 /tmp 运行版一致 (evidence_chain.md 数字来源)。路径改为仓内相对。"""
#!/usr/bin/env python3
"""ledger wall_seconds 分析: 每个 t1 run 的训练耗时与每 epoch 耗时"""
import json

rows = [json.loads(l) for l in open("/home/bluster/projectA/thermal-world-model/artifacts/final_wm/ledger.jsonl") if l.strip()]
groups = {}
for r in rows:
    rid = r.get("run_id")
    if rid and rid.startswith("t1") and "wall_seconds" in r and r.get("epoch") is not None:
        groups.setdefault(rid, []).append((r["epoch"], r["wall_seconds"]))

print("T1 各 run 训练耗时 (ledger wall_seconds, 含 eval):")
for rid in sorted(groups):
    es = sorted(groups[rid])
    e0, w0 = es[0]
    e1, w1 = es[-1]
    dur = (w1 - w0) / 60
    n = len(es)
    print(f"  {rid}: epoch{e0}->{e1}  {dur:.1f} min  ({n} 条记录, {(w1-w0)/max(n-1,1):.1f}s/epoch)")

# 跨 run 总墙钟: 用全局 min/max
allw = [r["wall_seconds"] for r in rows if "wall_seconds" in r and r.get("run_id", "").startswith("t1")]
print(f"\nT1 全局 wall_seconds 跨度: {(max(allw)-min(allw))/60:.1f} min")

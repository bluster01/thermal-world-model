import json

for tag, f in [
    ("intact", "artifacts/final_wm_sideB/auditpack_B.json"),
    ("norew", "artifacts/final_wm_sideB/auditpack_B_closure_cons_norew.json"),
]:
    a = json.load(open(f))
    pbg = a["position_binned_gain"]
    print("===", tag)
    for v in ["v1", "v2"]:
        blk = pbg[v]
        print(v, "obs_index", blk["obs_index"], "horizon", blk["horizon"])
        for b in blk["bins"]:
            d = b["data"]
            m = b["model"]
            dg = d.get("mean_gain")
            mg = m.get("mean_gain")
            print(
                "  [%.3f,%.3f] data n=%s gain=%s | model n=%s gain=%s"
                % (b["lo"], b["hi"], d["n"], ("%.3f" % dg) if dg is not None else None,
                   m["n"], ("%.3f" % mg) if mg is not None else None)
            )

import glob, json, os, re, csv
rows = []
for ds in ["euroc","run1","run2","run3","run4","run5","run6"]:
    f = glob.glob(f"comparisons/{ds}/dpvo_zoedepth_summary.json")
    ate = round(json.load(open(f[0]))["ate_rmse_m"], 1) if f else None
    for fde in ["zoedepth","depthanythingv2","fastdepth"]:
        log = f"/tmp/am_{ds}_{fde}.log"
        d = {"dataset": ds, "fde": fde, "ate_m": ate}
        if not os.path.exists(log):
            rows.append(d); continue
        t = open(log, errors="ignore").read().replace("\r", "\n")
        def g(p, cast=float):
            m = re.search(p, t, re.M)
            return cast(m.group(1)) if m else None
        d["auto_voxel_m"] = g(r"voxel_resolution = ([0-9.]+)")
        d["auto_max_depth_m"] = g(r"max_depth = ([0-9.]+)")
        d["occupied"] = g(r"^\s+(\d+) зайнятих", int)
        d["free"] = g(r"(\d+) впевнено", int)
        blk = re.search(r"^\[gt_compare\] adaptive:\n(.*)$", t, re.M)
        if blk:
            line = blk.group(1)
            # word boundary matters: "rec=" also occurs inside "prec="
            d["iou"] = g(r"IoU=([0-9.]+)"); 
            for key, pat in (("iou", r"IoU=([0-9.]+)"), ("fpvr", r"FPVR=([0-9.]+)"),
                             ("precision", r"\bprec=([0-9.]+)"), ("recall", r"(?<!p)rec=([0-9.]+)")):
                m = re.search(pat, line)
                d[key] = float(m.group(1)) if m else None
        d["note"] = "воксель завеликий" if "WARNING" in t else ("ok" if blk else "не побудувалась")
        rows.append(d)
cols = ["dataset","fde","ate_m","auto_voxel_m","auto_max_depth_m","occupied","free",
        "iou","fpvr","precision","recall","note"]
with open("all_maps_summary.csv","w",newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
hdr = f"{'набір':7s} {'FDE':16s} {'ATE':>6s} {'воксель':>8s} {'зайн':>7s} {'вільн':>6s} {'IoU':>7s} {'FPVR':>7s} {'prec':>6s} {'rec':>6s}  примітка"
print(hdr); print("-"*len(hdr))
for r in rows:
    f = lambda k, w=7, p=4: (f"{r[k]:{w}.{p}f}" if isinstance(r.get(k), float) else f"{'—':>{w}s}")
    i = lambda k, w=7: (f"{r[k]:{w}d}" if isinstance(r.get(k), int) else f"{'—':>{w}s}")
    print(f"{r['dataset']:7s} {r['fde']:16s} {f('ate_m',6,1)} {f('auto_voxel_m',8,2)} "
          f"{i('occupied')} {i('free',6)} {f('iou')} {f('fpvr')} {f('precision',6,3)} "
          f"{f('recall',6,3)}  {r.get('note','')}")

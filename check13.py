import json, sys
idx = sys.argv[1] if len(sys.argv) > 1 else "13"
r = json.load(open("confirm3d_c8/runs.json"))
rec = json.load(open("out_c8/optimization_checkpoint.json"))["all_raw"][int(idx)]
ref2d = {"ARO": "keff_core_bol", "ARI": "k_allre", "RE12": "k_re12"}
print(f"design {idx}: archive k_core {rec['keff_core_bol']:.5f}  k_ALLRE {rec['k_allre']:.5f}  k_RE12 {rec['k_re12']:.5f}")
print(f"{'state':5s} {'mode':5s} {'seed':>4s} {'keff':>9s} {'sd':>8s} {'vs archive':>11s} {'F':>7s} {'entropy':>8s} {'wall/min':>9s}")
bad = 0
for k in sorted(x for x in r if x.startswith(idx + "|")):
    p = k.split("|"); st, mode, seed = p[1], p[2], p[3]
    v = r[k]
    d_pcm = 1e5 * (v["keff"] - rec[ref2d[st]]) / rec[ref2d[st]]
    flag = ""
    if mode == "2D" and abs(d_pcm) > 300: flag = "  <-- CHECK"; bad += 1
    if mode == "3Dhw" and not (-4500 < d_pcm < -1500): flag = "  <-- CHECK"; bad += 1
    if not (0.5 < v["keff"] < 1.5) or v["sd"] > 0.001: flag = "  <-- CHECK"; bad += 1
    print(f"{st:5s} {mode:5s} {seed:>4s} {v['keff']:9.5f} {v['sd']:8.5f} {d_pcm:+10.0f}  "
          f"{v['fdh']:7.3f} {str(v['entropy_conv']):>8s} {v['wall_s']/60:9.1f}{flag}")
print("\nVERDICT:", "all entries consistent with the archive, keep them" if bad == 0
      else f"{bad} entry/entries flagged, delete the design {idx} keys and rerun all twelve")

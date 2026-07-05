"""Classify per-clip G8 failures into benign (jntMed-only, report-only model
drift) vs real (frame count / dropped player / fmatch / keypoint / position),
and surface the frame-loss tail. Reads a gate_gpu_parity JSON (pass the path,
e.g. g8_parity_m_full.json; default g8_parity_full.json)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

# gate thresholds (mirror gate_gpu_parity._verdict / G9)
FMATCH_MIN, MEDIAN_MAX, CONF_P90_MAX, POS_MED_MAX, JNT_MED_MAX = 0.85, 5.0, 12.0, 0.02, 0.03

src = sys.argv[1] if len(sys.argv) > 1 else "g8_parity_full.json"
rows = json.loads(Path(src).read_text())
print(f"source: {src}   clips: {len(rows)}")


def reasons(r):
    out = []
    if r["dF"] != 0:
        out.append(f"dF={r['dF']}")
    if r["rt_lt2"] > r["mm_lt2"]:
        out.append(f"DROPPED(rt<2={r['rt_lt2']}>mm<2={r['mm_lt2']})")
    if r["fmatch"] < FMATCH_MIN:
        out.append(f"fmatch={r['fmatch']:.3f}")
    if r["kp_med"] > MEDIAN_MAX:
        out.append(f"kpMed={r['kp_med']:.2f}")
    if r["kp_cp90"] > CONF_P90_MAX:
        out.append(f"kpP90={r['kp_cp90']:.2f}")
    if r["pos_med"] > POS_MED_MAX:
        out.append(f"posMed={r['pos_med']:.4f}")
    return out  # NB: jnt_med deliberately excluded -> "real" reasons only


fails = [r for r in rows if not r.get("ok", False)]
real, benign = [], []
for r in fails:
    (real if reasons(r) else benign).append(r)

print(f"\ng8-verdict fails: {len(fails)}   REAL: {len(real)}   jntMed-only(benign): {len(benign)}")

print("\n--- REAL fails (non-jntMed) ---")
for r in sorted(real, key=lambda r: r["stem"]):
    print(f"  {r['stem']:12s} {', '.join(reasons(r))}"
          f"   [jntMed={r['jnt_med']:.4f} failrateΔ={abs(r['rt_failrate']-r['mm_failrate'])*100:.2f}pp]")

print("\n--- benign (jntMed-only) ---")
print("  " + ", ".join(r["stem"] for r in sorted(benign, key=lambda r: r["stem"])))

# per-clip failed-rate tail (the 5pp gate)
print("\n--- per-clip failed-rate |Δ| > 5pp (frame-loss tail) ---")
tail = sorted(rows, key=lambda r: abs(r["rt_failrate"] - r["mm_failrate"]), reverse=True)
for r in tail:
    d = abs(r["rt_failrate"] - r["mm_failrate"]) * 100
    if d <= 5.0:
        break
    print(f"  {r['stem']:12s} |Δ|={d:5.2f}pp  rt={r['rt_failrate']*100:5.1f}% mm={r['mm_failrate']*100:5.1f}%"
          f"  rtOnlyFail={r.get('rt_only_fail',0):3d} mmOnlyFail={r.get('mm_only_fail',0):3d}"
          f"  rt<2={r['rt_lt2']} mm<2={r['mm_lt2']}")

# which video-ids are affected (cluster?)
print("\n--- REAL fails by video-id ---")
by_vid = defaultdict(list)
for r in real:
    by_vid[int(r["stem"].split("_", 1)[0])].append(r["stem"])
for vid in sorted(by_vid):
    print(f"  vid {vid:2d}: {len(by_vid[vid])} clip(s)  {by_vid[vid]}")

# coverage: which video-ids present, and per-video clip counts
vids = defaultdict(int)
for r in rows:
    vids[int(r["stem"].split("_", 1)[0])] += 1
print(f"\n--- coverage: {len(vids)} video-ids in the sample ---")
print(f"  vids present: {sorted(vids)}")
low = {v: n for v, n in vids.items() if n < 5}
if low:
    print(f"  under-5-clip vids (thin sampling): {dict(sorted(low.items()))}")

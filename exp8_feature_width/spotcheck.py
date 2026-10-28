"""Step 5: about 60 spot-check panels in spotcheck/ and spotcheck.csv with an empty verdict column.

Stratified by feature type, width class (<5, 5-12, >12 m) and multi-date detection outcome; seed fixed.
Capella (the truth) gets the purple outline; Sentinel-1 images get none (no detector output is drawn).
"""
import os
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from panel_roles import legend, outline  # noqa: E402

OUT = HERE / "spotcheck"
HALF = 120  # m


def chip(arr, gx0, gy1, res, cx, cy):
    j0 = int(round((cx - HALF - gx0) / res))
    i0 = int(round((gy1 - (cy + HALF)) / res))
    n = int(round(2 * HALF / res))
    return arr[max(i0, 0):i0 + n, max(j0, 0):j0 + n].astype("float32")


def show(ax, a, title):
    v = a[np.isfinite(a)]
    ax.imshow(a, cmap="gray", vmin=np.percentile(v, 2) if v.size else 0, vmax=np.percentile(v, 98) if v.size else 1,
              extent=(-HALF, HALF, -HALF, HALF), interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    OUT.mkdir(exist_ok=True)
    seg = pd.read_csv(HERE / "segments_all.csv")
    meta = pd.read_csv(HERE / "scene_meta.csv")
    good = meta[meta.osm_refined.astype(bool)].capella_id
    ok = seg[(seg.status == "ok") & seg.capella_id.isin(good)].copy()   # headline scenes only
    ok["wcls"] = pd.cut(ok.cap_width, [0, 5, 12, 100], labels=["narrow", "mid", "wide"])
    rng = np.random.default_rng(80)
    pick = []
    for (t, w, d), g in ok.groupby(["type", "wcls", "multi_either_det"], observed=True):
        pick += list(g.sample(min(2, len(g)), random_state=int(rng.integers(1e9))).index)
    rest = ok.drop(index=pick)
    if len(pick) < 60 and len(rest):
        pick += list(rest.sample(min(60 - len(pick), len(rest)), random_state=8).index)
    pick = pick[:60]
    sel = ok.loc[pick].sort_values(["type", "cap_width"])
    rows, cache = [], {}
    for k, r in enumerate(sel.itertuples()):
        d = HERE / "data" / r.capella_id
        if r.capella_id not in cache:
            cache.clear()
            cache[r.capella_id] = (dict(np.load(d / "arrays.npz")), np.load(d / "profiles.npz"))
        A, Pz = cache[r.capella_id]
        ids = list(Pz["ids"])
        prof, D = Pz["prof"][ids.index(r.seg_id)], Pz["D"]
        cx, cy = r.mid_x, r.mid_y
        dx, dy = float(A["dx"]), float(A["dy"])
        cap = chip(A["cap_db"], float(A["gxC"]), float(A["gyC"]), 1.0, cx + dx, cy + dy)
        fig, ax = plt.subplots(1, 5, figsize=(19, 4.3), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.5]})
        show(ax[0], cap, f"Capella {r.capella_id[8:11]} 1 m (truth)")
        outline(ax[0], "truth")
        for a_, key, title in [(ax[1], "single_vv", "Sentinel-1 VV single date"), (ax[2], "multi_vv", "Sentinel-1 VV multi-date mean"),
                               (ax[3], "multi_vh", "Sentinel-1 VH multi-date mean")]:
            show(a_, chip(A[key], float(A["gxS"]), float(A["gyS"]), 10.0, cx, cy), title)
        for a_ in ax[:4]:
            a_.plot([-HALF, -HALF + 30], [-HALF + 8] * 2, color="yellow", lw=2)
            a_.text(-HALF + 15, -HALF + 12, "30 m", color="yellow", fontsize=7, ha="center")
            a_.annotate("", xy=(0, 0), xytext=(0, 0))
            a_.plot(0, 0, marker="+", color="red", ms=10, mew=1)
        p = ax[4]
        p.plot(D, prof[0] - np.nanmedian(prof[0][np.abs(D) < 60]), color="#7e3fc0", lw=1, label="Capella (aligned)")
        for i, (lab, c, ls) in enumerate([("S1 VV single", "#d07a1e", "-"), ("S1 VH single", "#d07a1e", ":"),
                                          ("S1 VV multi", "#1f5fa8", "-"), ("S1 VH multi", "#1f5fa8", ":")]):
            y = prof[i + 1]
            p.plot(D, y - np.nanmedian(y[np.abs(D) < 60]), color=c, ls=ls, lw=1.4, label=lab)
        c0, w = r.cap_center, r.cap_width
        p.axvspan(c0 - w / 2, c0 + w / 2, color="#7e3fc0", alpha=0.15)
        p.set_xlim(-70, 70)
        p.set_xlabel("distance across the OSM line (m)")
        p.set_ylabel("dB relative to local median")
        p.legend(fontsize=7, loc="lower left")
        p.set_title(f"z single VV {r.single_vv_z:.1f} VH {r.single_vh_z:.1f} | multi VV {r.multi_vv_z:.1f} VH {r.multi_vh_z:.1f}",
                    fontsize=8)
        fig.suptitle(f"{r.type}  width {w:.1f} m  Capella contrast {r.cap_contrast:+.1f} dB  {r.country}  seg {r.seg_id}  "
                     f"(red + = segment midpoint; line runs {r.angle_to_flight:.0f} deg from S1 flight path)  "
                     f"detected: single {'yes' if r.single_either_det else 'no'}, multi {'yes' if r.multi_either_det else 'no'}",
                     fontsize=9)
        legend(fig)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        fn = f"{k:02d}_{r.type}_{r.seg_id}.png"
        fig.savefig(OUT / fn, dpi=85)
        plt.close(fig)
        rows.append({"panel": fn, "seg_id": r.seg_id, "capella_id": r.capella_id, "country": r.country, "type": r.type,
                     "cap_width_m": round(r.cap_width, 2), "cap_contrast_db": round(r.cap_contrast, 2),
                     "single_detected": r.single_either_det, "multi_detected": r.multi_either_det,
                     "question": "Is there a linear feature of about this width at the red + on Capella? Can you see it on Sentinel-1?",
                     "verdict": ""})
    pd.DataFrame(rows).to_csv(HERE / "spotcheck.csv", index=False)
    print("panels", len(rows), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)

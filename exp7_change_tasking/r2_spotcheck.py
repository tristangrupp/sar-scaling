"""Spot-check chips for round 2: is the Capella change truth real change?

Draws about 100 chips, fixed seed, at most 2 per site in each stratum:
  caught      Capella patches Sentinel-1 VV caught (loose rule), 7 per area band
  missed      Capella patches Sentinel-1 VV missed, 7 per area band
  false_alarm Sentinel-1 VV flag regions on Capella-stable ground, 30
Area bands (m^2): 8-32, 32-128, 128-512, 512-2048, >= 2048.

Each chip is a PNG panel in r2_spotcheck/ with: Capella before and after (inputs, no outline), the
Capella shift-tolerant change margin with the patch outlined (truth, purple frame), and the Sentinel-1
VV difference with its flags (detector output, green frame). r2_spotcheck/chips.csv lists every chip
with an empty "verdict" column. Suggested verdicts: real object change / real surface change /
not real (misregistration, layover, speckle) / can't tell.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
sys.path.insert(0, str(PROJ))
from panel_roles import legend, outline  # noqa: E402

DATA = ROOT / "data" / "r2"
OUT = ROOT / "r2_spotcheck"
BANDS = [8, 32, 128, 512, 2048, 1e9]
PER_BAND, N_FA, PER_SITE = 7, 30, 2
K, CRES = 5, 2.0


def land(r):
    if r.near_water and r.wc_class != 80:
        return "water edge"
    return {50: "urban", 40: "farm", 60: "bare/mine", 80: "water"}.get(int(r.wc_class), "vegetation/other")


def pick(df, n, rng):
    df = df.sample(frac=1, random_state=int(rng.integers(1e9)))
    df = df.groupby("site_id").head(PER_SITE)
    return df.head(n)


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    pr = pd.read_csv(ROOT / "r2_pair_results.csv")
    ok = pr[pr.status == "ok"]
    P = []
    for r in ok.itertuples():
        f = DATA / f"{r.tag}_patches.csv"
        if f.exists() and f.stat().st_size > 5:
            p = pd.read_csv(f)
            if len(p):
                p["tag"], p["site_id"] = r.tag, r.site_id
                P.append(p)
    P = pd.concat(P, ignore_index=True)
    P["band"] = np.digitize(P.area_m2, BANDS) - 1
    rows = []
    for stratum, cond in (("caught", P.vv_loose.astype(bool)), ("missed", ~P.vv_loose.astype(bool))):
        for b in range(len(BANDS) - 1):
            s = pick(P[cond & (P.band == b)], PER_BAND, rng)
            rows += [{"stratum": stratum, **r} for r in s.to_dict("records")]
    # false alarms: S1 VV flag regions lying wholly on Capella-stable ground
    fa = []
    for r in ok.itertuples():
        z = np.load(DATA / f"{r.tag}.npz")
        lab, n = ndimage.label(z["flag_vv"], structure=np.ones((3, 3)))
        if n == 0:
            continue
        idx = np.arange(1, n + 1)
        st = ndimage.minimum(z["stable10"].astype(int), lab, idx)
        size = ndimage.sum(np.ones_like(lab), lab, idx)
        com = ndimage.center_of_mass(np.ones_like(lab), lab, idx)
        for i in np.where(st == 1)[0]:
            fa.append({"stratum": "false_alarm", "tag": r.tag, "site_id": r.site_id, "s1_region": int(i + 1),
                       "area_m2": float(size[i] * 100), "row": int(com[i][0] * K), "col": int(com[i][1] * K)})
    fa = pd.DataFrame(fa)
    rows += pick(fa, N_FA, rng).to_dict("records")
    C = pd.DataFrame(rows).reset_index(drop=True)
    C["chip_id"] = [f"chip{i:03d}" for i in range(len(C))]
    meta = pr.set_index("tag")
    out = []
    cache = {}
    for c in C.itertuples():
        if c.tag not in cache:
            cache = {c.tag: np.load(DATA / f"{c.tag}.npz")}
        z = cache[c.tag]
        m = meta.loc[c.tag]
        lab = z["lab"]
        if c.stratum == "false_alarm":
            reg = ndimage.label(z["flag_vv"], structure=np.ones((3, 3)))[0] == c.s1_region
            sel = np.kron(reg, np.ones((K, K), bool))
        else:
            sel = lab == int(c.patch)
        rr, cc = np.where(sel)
        half = int(max(40, 0.5 * max(np.ptp(rr), np.ptp(cc)) + 25))
        half = min(half, 300)
        r0, c0 = int((rr.min() + rr.max()) / 2), int((cc.min() + cc.max()) / 2)
        r0, c0 = int(np.clip(r0, half, 1280 - half)), int(np.clip(c0, half, 1280 - half))
        sl = (slice(r0 - half, r0 + half), slice(c0 - half, c0 + half))
        sl10 = (slice((r0 - half) // K, (r0 + half) // K), slice((c0 - half) // K, (c0 + half) // K))
        a, b = z["a_db"].astype(float), z["b_db"].astype(float)
        use = z["use"]
        lo, hi = np.nanpercentile(np.r_[a[sl][np.isfinite(a[sl])], b[sl][np.isfinite(b[sl])]], [2, 99])
        fig, ax = plt.subplots(1, 4, figsize=(17, 4.9))
        ext = [0, 2 * half * CRES, 0, 2 * half * CRES]
        ax[0].imshow(a[sl], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest", extent=ext)
        ax[0].set_title(f"Capella before {m.x1} (2 m)", fontsize=9)
        ax[1].imshow(b[sl], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest", extent=ext)
        ax[1].set_title(f"Capella after {m.x2} (2 m)", fontsize=9)
        mg = np.where(use, z["mg"].astype(float), np.nan)
        ax[2].imshow(mg[sl], cmap="RdBu_r", vmin=-8, vmax=8, interpolation="nearest", extent=ext)
        ax[2].contour(np.flipud(lab[sl] > 0), [0.5], colors="k", linewidths=0.5, extent=ext)
        if c.stratum != "false_alarm":
            ax[2].contour(np.flipud(sel[sl]), [0.5], colors="#7e3fc0", linewidths=1.6, extent=ext)
        ax[2].set_title("Capella change (truth): this patch thick", fontsize=9)
        outline(ax[2], "truth")
        d = np.where(z["s1ok"], z["d_vv"].astype(float), np.nan)
        ext10 = [0, 2 * half * CRES, 0, 2 * half * CRES]
        ax[3].imshow(d[sl10], cmap="RdBu_r", vmin=-8, vmax=8, interpolation="nearest", extent=ext10)
        fl = z["flag_vv"][sl10]
        if fl.any():
            ax[3].contour(np.flipud(fl), [0.5], colors="k", linewidths=1.0, extent=ext10)
        if c.stratum == "false_alarm":
            ax[3].contour(np.flipud(reg[sl10]), [0.5], colors="#1f9d55", linewidths=1.8, extent=ext10)
        ax[3].set_title(f"Sentinel-1 VV {m.s1_before} to {m.s1_after}, flags", fontsize=9)
        outline(ax[3], "model")
        for x in ax:
            x.set_xticks([])
            x.set_yticks([])
        ax[0].set_ylabel(f"{2 * half * CRES:.0f} m across", fontsize=8)
        tr = Transformer.from_crs(int(z["epsg"]), 4326, always_xy=True)
        lon, lat = tr.transform(float(z["x0"]) + c0 * CRES, float(z["y1"]) - r0 * CRES)
        area = f"{c.area_m2:,.0f} m²"
        fig.suptitle(f"{c.chip_id}  {c.stratum}  site {c.site_id}  {area}  ({lat:.5f}, {lon:.5f})", fontsize=10)
        legend(fig)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        fn = f"{c.chip_id}_{c.stratum}.png"
        fig.savefig(OUT / fn, dpi=80)
        plt.close(fig)
        out.append({"chip_id": c.chip_id, "file": fn, "stratum": c.stratum, "site_id": c.site_id, "pair": c.tag,
                    "area_m2": round(c.area_m2), "sign": getattr(c, "sign", "") if c.stratum != "false_alarm" else "",
                    "land": land(c) if c.stratum != "false_alarm" else "",
                    "lat": round(lat, 5), "lon": round(lon, 5), "capella_before": m.x1, "capella_after": m.x2,
                    "s1_before": m.s1_before, "s1_after": m.s1_after, "verdict": "", "note": ""})
    pd.DataFrame(out).to_csv(OUT / "chips.csv", index=False)
    print(pd.DataFrame(out).groupby("stratum").size().to_dict(), "sites:", pd.DataFrame(out).site_id.nunique(), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)

"""Round 2: VH against VV under the pre-set rule and the amended rule (DESIGN.md, amendment 4).

Rules, all with the same windows, null lines and thresholds for both polarizations:
  original  the pre-set statistic (segments_all.csv): dB(mean linear center) minus mean flank dB, / own null sd
  amended   center minus mean of the flanks, every window averaged in dB, / own null sd (= rescore.py dB variant)
  linestep  secondary: amended line z or a flank-step z (left minus right flank, dB, / own null sd);
            one polarization |z| > 2.24 on either statistic, VV or VH > 2.49 on any of the four
Writes segments_vh_vv.csv, vh_vv_widths.csv, vh_vv_paired.csv, vh_vv.json. Does not read spotcheck.csv.
"""
import json
import os
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from measure import D, THR1, THR2, zscores  # noqa: E402
import analyze as A  # noqa: E402

NAMES = ["cap_db", "single_vv", "single_vh", "multi_vv", "multi_vh"]
THR_LS1, THR_LS2 = 2.24, 2.49
NB = 1000
RNG = np.random.default_rng(31)


def windows(c, w, r=10.0):
    hc = max(r / 2, w / 2)
    f0 = max(1.5 * r, w / 2 + r)
    return hc, f0, f0 + 15


def stats_db(pdb, c, w):
    """Line and step statistics (dB) at c and on the 16 null lines, all windows averaged in dB."""
    hc, f0, f1 = windows(c, w)

    def one(cc):
        dd = D - cc
        cen, L, R = pdb[np.abs(dd) <= hc], pdb[(dd >= -f1) & (dd <= -f0)], pdb[(dd >= f0) & (dd <= f1)]
        if np.isfinite(cen).mean() < 0.8 or np.isfinite(L).mean() < 0.8 or np.isfinite(R).mean() < 0.8:
            return np.nan, np.nan
        l, r = np.nanmean(L), np.nanmean(R)
        return np.nanmean(cen) - (l + r) / 2, l - r

    offs = f0 + 15 + w / 2 + 10 + 5 * np.arange(8)
    obs = one(c)
    nl = np.array([one(c + o) for o in offs] + [one(c - o) for o in offs])
    return obs, nl


def headline():
    seg = pd.read_csv(HERE / "segments_all.csv")
    meta = pd.read_csv(HERE / "scene_meta.csv")
    good = meta[meta.osm_refined.astype(bool)].capella_id
    ok = seg[(seg.status == "ok") & seg.capella_id.isin(good)].copy().reset_index(drop=True)
    for c in [c for c in ok.columns if c.endswith("_det")]:
        ok[c] = ok[c].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).astype(float)
    return ok


def score(ok):
    """Per segment: amended and line-or-step detections plus their false-alarm counts."""
    out = {}
    for cid, g in ok.groupby("capella_id"):
        z = np.load(HERE / "data" / cid / "profiles.npz")
        P = dict(zip(z["ids"], z["prof"]))
        for i, r in g.iterrows():
            pr = P[r.seg_id].astype(float)
            rec, zn = {}, {}
            for prod in ("single", "multi"):
                for q in ("vv", "vh"):
                    k = f"{prod}_{q}"
                    (oL, oS), nl = stats_db(pr[NAMES.index(k)], r.cap_center, r.cap_width)
                    zL, znL = zscores(oL, nl[:, 0])
                    zS, znS = zscores(oS, nl[:, 1])
                    rec[f"am_{k}_C"], rec[f"am_{k}_z"], rec[f"step_{k}_z"] = oL, zL, zS
                    rec[f"am_{k}_det"] = float(np.isfinite(zL) and abs(zL) > THR1)
                    rec[f"am_{k}_far_n"], rec[f"am_{k}_far_m"] = int(np.nansum(np.abs(znL) > THR1)), int(np.isfinite(znL).sum())
                    m = np.nanmax(np.abs([zL, zS])) if np.isfinite([zL, zS]).any() else np.nan
                    rec[f"ls_{k}_det"] = float(np.isfinite(m) and m > THR_LS1)
                    mn = np.fmax(np.abs(znL), np.abs(znS))
                    rec[f"ls_{k}_far_n"], rec[f"ls_{k}_far_m"] = int(np.nansum(mn > THR_LS1)), int(np.isfinite(mn).sum())
                    zn[k] = (znL, znS)
                zv, zh = rec[f"am_{prod}_vv_z"], rec[f"am_{prod}_vh_z"]
                rec[f"am_{prod}_either_det"] = float(np.isfinite([zv, zh]).any() and np.nanmax(np.abs([zv, zh])) > THR2)
                ze = np.fmax(np.abs(zn[f"{prod}_vv"][0]), np.abs(zn[f"{prod}_vh"][0]))
                rec[f"am_{prod}_either_far_n"], rec[f"am_{prod}_either_far_m"] = int(np.nansum(ze > THR2)), int(np.isfinite(ze).sum())
                four = [rec[f"am_{prod}_vv_z"], rec[f"step_{prod}_vv_z"], rec[f"am_{prod}_vh_z"], rec[f"step_{prod}_vh_z"]]
                rec[f"ls_{prod}_either_det"] = float(np.isfinite(four).any() and np.nanmax(np.abs(four)) > THR_LS2)
                zf = np.nanmax(np.abs(np.vstack([zn[f"{prod}_vv"][0], zn[f"{prod}_vv"][1], zn[f"{prod}_vh"][0], zn[f"{prod}_vh"][1]])), axis=0)
                rec[f"ls_{prod}_either_far_n"], rec[f"ls_{prod}_either_far_m"] = int(np.nansum(zf > THR_LS2)), int(np.isfinite(zf).sum())
            out[i] = rec
    return pd.concat([ok, pd.DataFrame.from_dict(out, orient="index")], axis=1)


def view(df, rule):
    """Columns renamed so that analyze.summarize sees {prod}_{pol}_det / _far_n / _far_m."""
    v = df[["capella_id", "seg_id", "cap_width", "type"]].copy()
    v["wbin"] = pd.cut(v.cap_width, A.EDGES, labels=A.BLAB, right=False)
    for prod in ("single", "multi"):
        for q in ("vv", "vh", "either"):
            src = f"{prod}_{q}" if rule == "original" else f"{rule}_{prod}_{q}"
            for s in ("det", "far_n", "far_m"):
                v[f"{prod}_{q}_{s}"] = df[f"{src}_{s}"].astype(float)
    return v


def paired(v, prod, scenes_groups):
    a, b = v[f"{prod}_vv_det"].astype(bool), v[f"{prod}_vh_det"].astype(bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    rec = {"product": prod, "both": int((a & b).sum()), "vv_only": n10, "vh_only": n01, "neither": int((~a & ~b).sum()),
           "vv_rate": round(float(a.mean()), 4), "vh_rate": round(float(b.mean()), 4),
           "mcnemar_exact_p": float(binomtest(n01, n01 + n10).pvalue) if n01 + n10 else np.nan}
    scenes = list(scenes_groups)
    diffs = []
    for _ in range(NB):
        pick = RNG.choice(scenes, len(scenes), replace=True)
        bb = pd.concat([scenes_groups[s] for s in pick])
        diffs.append(bb[f"{prod}_vh_det"].mean() - bb[f"{prod}_vv_det"].mean())
    rec["vh_minus_vv"] = round(float(b.mean() - a.mean()), 4)
    rec["vh_minus_vv_lo"], rec["vh_minus_vv_hi"] = (round(float(x), 4) for x in np.percentile(diffs, [2.5, 97.5]))
    return rec


def main():
    ok = headline()
    df = score(ok)
    # the amended rule must reproduce rescore.py's dB variant
    dbv = pd.read_csv(HERE / "segments_all_db.csv").set_index("seg_id")
    chk = np.nanmax(np.abs(df.set_index("seg_id")["am_multi_vh_z"] - dbv.loc[df.seg_id, "multi_vh_z"]))
    print(f"amended vs rescore.py dB variant, max |z diff| multi VH = {chk:.2e}", flush=True)
    df.to_csv(HERE / "segments_vh_vv.csv", index=False)

    rows, pr, far = [], [], {}
    for rule in ("original", "amended", "ls"):
        v = view(df, "am" if rule == "amended" else rule)
        groups = {s: g for s, g in v.groupby("capella_id")}
        for prod in ("single", "multi"):
            pr.append({"rule": rule, **paired(v, prod, groups)})
            for q in ("vv", "vh", "either"):
                f = A.far(v, prod, q)
                far[f"{rule}_{prod}_{q}"] = round(f, 4)
                rec, _ = A.summarize(v, "all", prod, q, f, nb=NB)
                rec["rule"] = rule
                rows.append(rec)
                print(rule, prod, q, rec["det_rate"], rec["false_alarm"], rec["w50_txt"], rec["w90_txt"], flush=True)
            print(rule, pr[-1], flush=True)
    wt = pd.DataFrame(rows)
    wt.to_csv(HERE / "vh_vv_widths.csv", index=False)
    pd.DataFrame(pr).to_csv(HERE / "vh_vv_paired.csv", index=False)
    (HERE / "vh_vv.json").write_text(json.dumps({"check_amended_vs_rescore": float(chk), "false_alarm": far,
                                                 "paired": pr}, indent=1, default=str))


if __name__ == "__main__":
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)

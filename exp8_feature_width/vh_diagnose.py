"""Round 2 diagnosis: does the pre-set test undervalue Sentinel-1 VH against VV? (DESIGN.md, amendment 4)

Runs only checks; changes no rule. Reads segments_all.csv, scene_meta.csv, data/<id>/arrays.npz and
profiles.npz. Writes vh_diagnose.json and prints a log. Does not read spotcheck.csv: the human verdicts
play no part in the diagnosis.
"""
import json
import os
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from measure import D, contrast_stat, zscores  # noqa: E402

NAMES = ["cap_db", "single_vv", "single_vh", "multi_vv", "multi_vh"]


def headline():
    seg = pd.read_csv(HERE / "segments_all.csv")
    meta = pd.read_csv(HERE / "scene_meta.csv")
    good = meta[meta.osm_refined.astype(bool)].capella_id
    ok = seg[(seg.status == "ok") & seg.capella_id.isin(good)].copy()
    for c in [c for c in ok.columns if c.endswith("_det")]:
        ok[c] = ok[c].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).astype(float)
    return ok, meta


def profiles(ok):
    out = {}
    for cid, g in ok.groupby("capella_id"):
        z = np.load(HERE / "data" / cid / "profiles.npz")
        P = dict(zip(z["ids"], z["prof"]))
        for sid in g.seg_id:
            out[sid] = P[sid].astype(float)
    return out


def main():
    ok, meta = headline()
    P = profiles(ok)
    R = {"n_segments": int(len(ok)), "n_scenes": int(ok.capella_id.nunique())}

    # 1. scene arrays: same grid, VH darker than VV everywhere (bands not swapped), nodata the same
    sc = []
    for cid in ok.capella_id.unique():
        A = np.load(HERE / "data" / cid / "arrays.npz")
        rec = {"capella_id": cid}
        for prod in ("single", "multi"):
            vv, vh = A[f"{prod}_vv"].astype(float), A[f"{prod}_vh"].astype(float)
            m = np.isfinite(vv) & np.isfinite(vh)
            rec[f"{prod}_shape_equal"] = vv.shape == vh.shape
            rec[f"{prod}_nan_vv"] = float(np.mean(~np.isfinite(vv)))
            rec[f"{prod}_nan_vh"] = float(np.mean(~np.isfinite(vh)))
            rec[f"{prod}_med_vv_db"] = float(np.median(vv[m]))
            rec[f"{prod}_med_vh_db"] = float(np.median(vh[m]))
            rec[f"{prod}_share_vh_below_vv"] = float(np.mean(vh[m] < vv[m]))
            rec[f"{prod}_share_vh_below_m22"] = float(np.mean(vh[m] < -22))
            rec[f"{prod}_corr_vv_vh"] = float(np.corrcoef(vv[m], vh[m])[0, 1])
        sc.append(rec)
    sc = pd.DataFrame(sc)
    sc.to_csv(HERE / "vh_diagnose_scenes.csv", index=False)
    R["arrays"] = {
        "all_shapes_equal": bool(sc[["single_shape_equal", "multi_shape_equal"]].all().all()),
        "max_nan_diff_multi": float((sc.multi_nan_vh - sc.multi_nan_vv).abs().max()),
        "median_vv_minus_vh_db_multi": float((sc.multi_med_vv_db - sc.multi_med_vh_db).median()),
        "range_vv_minus_vh_db_multi": [float((sc.multi_med_vv_db - sc.multi_med_vh_db).min()),
                                        float((sc.multi_med_vv_db - sc.multi_med_vh_db).max())],
        "min_share_vh_below_vv_multi": float(sc.multi_share_vh_below_vv.min()),
        "median_share_vh_below_m22_multi": float(sc.multi_share_vh_below_m22.median()),
        "median_corr_vv_vh_multi": float(sc.multi_corr_vv_vh.median()),
        "multi_dates_per_scene": ok.groupby("capella_id").s1_n_multi.first().describe().round(2).to_dict(),
    }

    # 2. units: shift VH by a constant in dB -> z unchanged? (both rules are differences in dB / own null sd)
    diffs = []
    for r in ok.head(80).itertuples():
        p = P[r.seg_id][NAMES.index("multi_vh")]
        o0, n0, _ = contrast_stat(10 ** (p / 10), r.cap_center, r.cap_width, 10.0)
        o1, n1, _ = contrast_stat(10 ** ((p + 7) / 10), r.cap_center, r.cap_width, 10.0)
        diffs.append(abs(zscores(o0, n0)[0] - zscores(o1, n1)[0]))
    R["z_invariant_to_7db_offset_maxdiff"] = float(np.nanmax(diffs))

    # 3. per-polarization contrast, null spread, z, detection
    pol = {}
    for prod in ("single", "multi"):
        for q in ("vv", "vh"):
            k = f"{prod}_{q}"
            pol[k] = {"det_rate": round(float(ok[f"{k}_det"].mean()), 3),
                      "median_abs_C_db": round(float(ok[f"{k}_C"].abs().median()), 3),
                      "median_null_sd_db": round(float(ok[f"{k}_null_sd"].median()), 3),
                      "median_abs_z": round(float(ok[f"{k}_z"].abs().median()), 3),
                      "false_alarm": round(float(ok[f"{k}_far_n"].sum() / ok[f"{k}_far_m"].sum()), 4)}
        pol[f"{prod}_null_sd_ratio_vh_over_vv_median"] = round(float((ok[f"{prod}_vh_null_sd"] / ok[f"{prod}_vv_null_sd"]).median()), 3)
        pol[f"{prod}_absC_ratio_vh_over_vv_median"] = round(float((ok[f"{prod}_vh_C"].abs() / ok[f"{prod}_vv_C"].abs()).median()), 3)
    R["per_pol"] = pol

    # 4. noise floor: VH background level and VH detection by level
    bg = {}
    for prod in ("single", "multi"):
        lv = []
        for r in ok.itertuples():
            p = P[r.seg_id][NAMES.index(f"{prod}_vh")]
            sel = (np.abs(D - r.cap_center) >= 15) & (np.abs(D - r.cap_center) <= 40)
            lv.append(np.nanmean(p[sel]))
        ok[f"{prod}_vh_bg_db"] = lv
        bins = pd.cut(ok[f"{prod}_vh_bg_db"], [-60, -24, -22, -20, -18, -15, 10])
        t = ok.groupby(bins, observed=True).agg(n=("seg_id", "size"), vh=(f"{prod}_vh_det", "mean"),
                                               vv=(f"{prod}_vv_det", "mean"))
        bg[prod] = {str(i): {k: round(float(v), 3) for k, v in row.items()} for i, row in t.iterrows()}
        bg[f"{prod}_share_bg_below_m22"] = round(float((ok[f"{prod}_vh_bg_db"] < -22).mean()), 3)
        # dark features (Capella-dark) whose VH center would sit near the floor
        cen = []
        for r in ok.itertuples():
            p = P[r.seg_id][NAMES.index(f"{prod}_vh")]
            cen.append(np.nanmean(p[np.abs(D - r.cap_center) <= max(5, r.cap_width / 2)]))
        ok[f"{prod}_vh_cen_db"] = cen
        bg[f"{prod}_share_center_below_m22"] = round(float((ok[f"{prod}_vh_cen_db"] < -22).mean()), 3)
    R["vh_level"] = bg

    # 5. where is the strongest deviation? offset of the profile extremum from the Capella center, per pol
    loc = {}
    for prod in ("multi",):
        for q in ("vv", "vh"):
            offs, wid = [], []
            for r in ok.itertuples():
                p = P[r.seg_id][NAMES.index(f"{prod}_{q}")]
                dd = D - r.cap_center
                base = np.nanmedian(p[(np.abs(dd) >= 20) & (np.abs(dd) <= 50)])
                near = np.abs(dd) <= 25
                dev = np.where(near, p - base, np.nan)
                if not np.isfinite(dev).any():
                    continue
                j = int(np.nanargmax(np.abs(dev)))
                offs.append(dd[j])
            offs = np.abs(np.array(offs))
            loc[f"{prod}_{q}"] = {"median_abs_offset_m": round(float(np.median(offs)), 2),
                                  "share_within_5m": round(float(np.mean(offs <= 5)), 3),
                                  "share_10_to_25m": round(float(np.mean(offs >= 10)), 3)}
    R["extremum_location"] = loc

    # 6. sign: share of z>0 among detections per pol, and against Capella sign
    sg = {}
    for prod in ("single", "multi"):
        for q in ("vv", "vh"):
            d = ok[ok[f"{prod}_{q}_det"] == 1]
            sg[f"{prod}_{q}"] = {"share_bright": round(float((d[f"{prod}_{q}_z"] > 0).mean()), 3),
                                 "agree_capella": round(float((np.sign(d[f"{prod}_{q}_C"]) == np.sign(d.cap_contrast)).mean()), 3)}
    R["sign"] = sg

    # 7. step edges: |left flank - right flank| per pol in units of the same pol's null spread
    st = {}
    for prod in ("multi",):
        for q in ("vv", "vh"):
            s = []
            for r in ok.itertuples():
                p = P[r.seg_id][NAMES.index(f"{prod}_{q}")]
                w = r.cap_width
                f0 = max(15, w / 2 + 10)
                dd = D - r.cap_center
                L = np.nanmean(p[(dd >= -f0 - 15) & (dd <= -f0)])
                Rr = np.nanmean(p[(dd >= f0) & (dd <= f0 + 15)])
                s.append(abs(L - Rr))
            ok[f"{prod}_{q}_step"] = s
            st[f"{prod}_{q}_median_step_db"] = round(float(np.nanmedian(s)), 3)
            st[f"{prod}_{q}_median_step_over_null_sd"] = round(float(np.nanmedian(np.array(s) / ok[f"{prod}_{q}_null_sd"])), 3)
    R["step_edges"] = st

    # 8. detection-rate gap vs width-at-50% gap: how flat are the curves?
    wt = pd.read_csv(HERE / "width_table.csv")
    a = wt[(wt.group == "all")][["product", "pol", "det_rate", "w50", "b"]]
    R["curve_slopes"] = a.round(3).to_dict(orient="records")
    bins = pd.read_csv(HERE / "curve_bins.csv")
    b = bins[(bins.group == "all") & (bins.pol.isin(["vv", "vh"]))][["product", "pol", "wbin", "n", "rate"]]
    R["bin_rates"] = b.to_dict(orient="records")

    # 9. paired agreement under the pre-set rule
    for prod in ("single", "multi"):
        v, h = ok[f"{prod}_vv_det"].astype(bool), ok[f"{prod}_vh_det"].astype(bool)
        R[f"paired_{prod}_preset"] = {"both": int((v & h).sum()), "vv_only": int((v & ~h).sum()),
                                      "vh_only": int((~v & h).sum()), "neither": int((~v & ~h).sum())}
    R["corr_z_vv_vh_multi"] = round(float(np.corrcoef(ok.multi_vv_z, ok.multi_vh_z)[0, 1]), 3)
    from scipy.stats import binomtest
    for prod in ("single", "multi"):
        d = R[f"paired_{prod}_preset"]
        R[f"paired_{prod}_preset"]["mcnemar_exact_p"] = round(float(binomtest(d["vh_only"], d["vh_only"] + d["vv_only"]).pvalue), 4)

    # 10. what a viewer sees: spot-check style 240 m chips, 2-98 percentile stretch per image (spotcheck.py)
    #     line contrast and flank step as a share of the displayed range, per polarization
    vis = {q: {"line": [], "step": [], "range": []} for q in ("vv", "vh")}
    for cid, g in ok.groupby("capella_id"):
        A = np.load(HERE / "data" / cid / "arrays.npz")
        gx, gy = float(A["gxS"]), float(A["gyS"])
        for r in g.itertuples():
            j0 = int(round((r.mid_x - 120 - gx) / 10))
            i0 = int(round((gy - (r.mid_y + 120)) / 10))
            for q in ("vv", "vh"):
                a = A[f"multi_{q}"][max(i0, 0):i0 + 24, max(j0, 0):j0 + 24].astype(float)
                v = a[np.isfinite(a)]
                if v.size < 100:
                    continue
                rng_ = np.percentile(v, 98) - np.percentile(v, 2)
                vis[q]["range"].append(rng_)
                vis[q]["line"].append(abs(getattr(r, f"multi_{q}_C")) / rng_)
                vis[q]["step"].append(getattr(r, f"multi_{q}_step") / rng_)
    R["display_stretch"] = {q: {"median_range_db": round(float(np.median(vis[q]["range"])), 2),
                                "median_line_share_of_range": round(float(np.median(vis[q]["line"])), 3),
                                "median_step_share_of_range": round(float(np.median(vis[q]["step"])), 3)} for q in vis}

    # 11. null bias: mean of the 16 null contrasts, per pol (thermal-noise residuals or banding would show here)
    nb = {}
    for q in ("vv", "vh"):
        m = []
        for r in ok.itertuples():
            p = P[r.seg_id][NAMES.index(f"multi_{q}")]
            _, nl, _ = contrast_stat(10 ** (p / 10), r.cap_center, r.cap_width, 10.0)
            m.append(np.nanmean(nl))
        nb[f"multi_{q}_median_null_mean_db"] = round(float(np.nanmedian(m)), 3)
        nb[f"multi_{q}_median_abs_null_mean_db"] = round(float(np.nanmedian(np.abs(m))), 3)
    R["null_bias"] = nb
    (HERE / "vh_diagnose.json").write_text(json.dumps(R, indent=1, default=str))
    print(json.dumps(R, indent=1, default=str), flush=True)


if __name__ == "__main__":
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main()
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)

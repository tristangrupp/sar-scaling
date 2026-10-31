"""Dataset facts for phase1.md: counts per split, rejection rates, resolution checks, bytes read. Writes
dataset_stats.json and prints a readable summary."""
from __future__ import annotations

import common  # noqa: F401
from common import CHIPS, HERE, RES_LIST, finish

import json
import traceback

import numpy as np
import pandas as pd


def main():
    ch = pd.read_csv(HERE / "chips.csv")
    al = pd.read_csv(HERE / "chips_all.csv")
    it = pd.read_csv(HERE / "items_built.csv")
    sp = pd.read_csv(HERE / "splits.csv")
    al = al.merge(sp[["site_id", "split"]], on="site_id", how="left")
    out = {}
    out["items"] = it.status.value_counts().to_dict()
    out["chips_status"] = al.status.value_counts().to_dict()
    out["by_split"] = {s: {"chips": int((ch.split == s).sum()), "sites": int(ch[ch.split == s].site_id.nunique()),
                           "items": int(ch[ch.split == s].capella_id.nunique()),
                           "countries": int(ch[ch.split == s].country.nunique())} for s in ("train", "val", "test")}
    out["align_reject_by_split"] = {s: int(((al.split == s) & (al.status == "align_reject")).sum())
                                    for s in ("train", "val", "test")}
    out["countries"] = int(ch.country.nunique())
    rej = al[al.status.isin(["ok", "align_reject"])]
    out["align_reject_rate_by_cover"] = rej.groupby("cover").status.apply(lambda s: round(float((s == "align_reject").mean()), 3)).to_dict()
    out["align_reject_rate_by_mode"] = rej.groupby("mode").status.apply(lambda s: round(float((s == "align_reject").mean()), 3)).to_dict()
    out["align_reject_rate"] = round(float((rej.status == "align_reject").mean()), 3)
    out["low_valid_rate"] = round(float((al.status == "low_valid").mean()), 3)
    out["cover_by_split"] = pd.crosstab(ch.cover, ch.split).to_dict()
    out["pol"] = ch.pol.value_counts().to_dict()
    out["pol_by_split"] = pd.crosstab(ch.pol, ch.split).to_dict()
    out["mode"] = ch["mode"].value_counts().to_dict()
    out["gap_abs_days"] = ch.s1_gap_d.abs().describe().round(2).to_dict()
    out["gap_le6_share"] = round(float((ch.s1_gap_d.abs() <= 6).mean()), 3)
    out["n_multi"] = ch.s1_n_multi.describe().round(2).to_dict()
    out["r_final"] = ch.r_final.describe().round(3).to_dict()
    out["shift_total_m"] = np.hypot(ch.align_dx, ch.align_dy).describe().round(1).to_dict()
    ok = it[it.status == "ok"]
    res = {}
    for R in RES_LIST:
        k = f"{R:g}"
        res[k] = {"fwhm_fine_median": round(float(ok[f"fwhm_fine_{k}"].median()), 2),
                  "fwhm_coarse_median": round(float(ok[f"fwhm_coarse_{k}"].median()), 2),
                  "fwhm_coarse_max": round(float(ok[f"fwhm_coarse_{k}"].max()), 2),
                  "looks_median": round(float(ok[f"looks_{k}"].median()), 1),
                  "looks_p10": round(float(ok[f"looks_{k}"].quantile(0.1)), 1),
                  "acf_median": round(float(ok[f"acf_fwhm_{k}"].median()), 2),
                  "acf_p90": round(float(ok[f"acf_fwhm_{k}"].quantile(0.9)), 2)}
    out["resolution"] = res
    out["bytes_build_gb"] = round(float(it.bytes.fillna(0).sum()) / 1e9, 2)
    for f in ("ceiling_pairs.csv", "ceiling_pairs_cross.csv"):
        if (HERE / f).exists():
            out[f"bytes_{f[:-4]}_gb"] = round(float(pd.read_csv(HERE / f).bytes.sum()) / 1e9, 2)
    out["stored_gb"] = round(sum(p.stat().st_size for p in CHIPS.glob("*")) / 1e9, 2)
    (HERE / "dataset_stats.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finish()

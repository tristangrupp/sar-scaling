"""Step 4: baseline scores on validation and test chips at each target resolution (DESIGN.md, "Metrics",
"Baselines"). CPU only.

Writes baselines_chips.csv (one row per chip x resolution x method), baselines_summary.csv (site-median of chip
scores, then median over sites, with 95% site-bootstrap intervals) and best_baselines.json (best baseline per
metric and resolution, chosen on validation sites).
Shuffle rows ("shuffle_<method>"): the baseline computed from a random chip of another site, scored against this
chip's target; they give the floor that the model's shuffled-input check is compared with.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import common  # noqa: F401
from common import (BASELINES, HERE, RES_LIST, SC, baseline_preds, bicubic_up, crop_c, finish, flat_mask, load_item,
                    scores, shuffle_partners)

import argparse
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

METRICS = ["psnr", "ssim", "detail_r", "hp_r", "edge_f1", "gms", "hf_ratio", "flat_ratio"]
SHUFFLED = ("bicubic_multi", "n2n_multi")


def masks_for(z, i, ri):
    up = np.kron(z["valid10"][i], np.ones((SC, SC), bool))
    m = z["tvalid"][i] & up & np.isfinite(z["tgt"][i, ri].astype("float32"))
    return m


def score_item(cid, chips, partners):
    """chips: rows of this item; partners: {chip_id: (capella_id, chip_id)} of the shuffle partner."""
    import torch
    torch.set_num_threads(1)
    z = load_item(cid)
    idx = {c: i for i, c in enumerate(z["chip_id"])}
    out = []
    pcache = {}
    for r in chips.itertuples():
        i = idx[r.chip_id]
        s1 = z["s1"][i].astype("float32")
        preds = baseline_preds(s1)
        pc, pchip = partners[r.chip_id]
        if pc not in pcache:
            pcache[pc] = load_item(pc)
        pz = pcache[pc]
        j = list(pz["chip_id"]).index(pchip)
        ps1 = pz["s1"][j].astype("float32")
        pv = np.kron(pz["valid10"][j], np.ones((SC, SC), bool))
        sh = {"shuffle_bicubic_multi": bicubic_up(ps1[2])}
        try:
            from common import n2n_apply, db
            sh["shuffle_n2n_multi"] = bicubic_up(db(n2n_apply(10 ** (ps1[2] / 10), 10 ** (ps1[3] / 10))))
        except Exception:
            traceback.print_exc()
        for ri, R in enumerate(RES_LIST):
            t = z["tgt"][i, ri].astype("float32")
            m = masks_for(z, i, ri)
            fl = flat_mask(np.where(m, t, np.nan), m)
            tc, mc, fc = crop_c(t), crop_c(m), crop_c(fl)
            for name, p in list(preds.items()) + list(sh.items()):
                mm = mc & crop_c(pv) if name.startswith("shuffle") else mc
                s = scores(crop_c(p), tc, mm, flat=fc & mm)
                if s is None:
                    continue
                out.append({"chip_id": r.chip_id, "capella_id": cid, "site_id": r.site_id, "split": r.split,
                            "cover": r.cover, "pol": r.pol, "R": R, "method": name, **s})
    return out


def site_boot(df, metric, n=1000, seed=0):
    """Median over sites of the site-median chip score, with a 95% bootstrap interval over sites."""
    s = df.groupby("site_id")[metric].median().dropna().values
    if len(s) == 0:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    bs = [np.median(rng.choice(s, len(s))) for _ in range(n)]
    return float(np.median(s)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(s)


def summarize(df, tag, out_csv):
    rows = []
    for (split, R, method), g in df.groupby(["split", "R", "method"]):
        rec = {"split": split, "R": R, "method": method, "n_chips": len(g), "n_sites": g.site_id.nunique()}
        for mt in METRICS:
            if mt in g:
                med, lo, hi, _ = site_boot(g, mt)
                rec[mt] = round(med, 4)
                rec[mt + "_lo"] = round(lo, 4)
                rec[mt + "_hi"] = round(hi, 4)
        rows.append(rec)
    s = pd.DataFrame(rows)
    s.to_csv(out_csv, index=False)
    print(f"--- {tag} (site medians)", flush=True)
    show = ["psnr", "ssim", "detail_r", "hp_r", "edge_f1", "gms", "flat_ratio"]
    for split in ("val", "test"):
        print(f"[{split}]", flush=True)
        print(s[s.split == split].set_index(["R", "method"])[show].round(3).to_string(), flush=True)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-per-site", type=int, default=0, help="cap chips per site (0 = all)")
    ap.add_argument("--rejects", action="store_true", help="score the align_reject chips instead (sensitivity)")
    a = ap.parse_args()
    if a.rejects:
        ch = pd.read_csv(HERE / "chips_all.csv")
        ch = ch[(ch.status == "align_reject") & (ch.valid_frac >= 0.8)].merge(pd.read_csv(HERE / "splits.csv")[["site_id", "split"]], on="site_id")
    else:
        ch = pd.read_csv(HERE / "chips.csv")
    ch = ch[ch.split.isin(["val", "test"])].copy()
    sfx = "_rej" if a.rejects else ""
    if a.max_per_site:
        ch = ch.groupby("site_id").head(a.max_per_site)
    partners = shuffle_partners(ch)
    print(f"scoring {len(ch)} chips at {ch.site_id.nunique()} sites", flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(score_item, cid, g, {c: partners[c] for c in g.chip_id}): cid for cid, g in ch.groupby("capella_id")}
        for i, f in enumerate(as_completed(futs)):
            try:
                rows += f.result()
            except Exception:
                traceback.print_exc()
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(futs)} items, {(time.time() - t0) / 60:.1f} min", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"baselines_chips{sfx}.csv", index=False)
    s = summarize(df, "baselines" + sfx, HERE / f"baselines_summary{sfx}.csv")
    if a.rejects:
        return
    best = {}
    v = s[(s.split == "val") & s.method.isin(BASELINES)]
    for R in RES_LIST:
        vr = v[v.R == R]
        best[str(R)] = {mt: vr.sort_values(mt, ascending=False).method.iloc[0] for mt in
                        ["psnr", "ssim", "detail_r", "hp_r", "edge_f1", "gms"] if len(vr)}
    (HERE / "best_baselines.json").write_text(json.dumps(best, indent=1))
    print("best baselines (val):", json.dumps(best), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
    finish()

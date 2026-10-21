"""Round 4 analysis of r4_scores_tiles.csv.

Variants: (architecture, start, training share). Exp 2's arm A checkpoints (esrt_a, cnn_a) are the scratch
seed 0 at 100 %. For each variant and group:
  mean over seeds of the per-seed group means, and the seed SD
  paired per-tile test against arm A of the same architecture: per tile, the variant's seed-averaged score
    minus arm A's; mean difference, share of tiles improved, Wilcoxon signed-rank p
Decision rule (per group, SSIM against the Lee reference, the arm A training objective): a pretrained
variant "beats arm A beyond seed noise" if its seed-mean exceeds the scratch seed-mean by more than twice
the larger of the two seed SDs AND its paired Wilcoxon p against arm A is below 0.05 with a positive mean.
Writes r4_variants.csv, r4_paired.csv, r4_decision.csv, r4_paper.csv.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
MET = ["ssim_lee", "psnr_lee", "ssim_raw", "psnr_raw", "mae_db_raw", "detail_r_raw", "hf_ratio_raw",
       "chamfer_p2t_m_raw", "chamfer_t2p_m_raw", "enl", "moi", "mor", "cx"]
HIGHER = {"ssim_lee": True, "psnr_lee": True, "ssim_raw": True, "psnr_raw": True, "mae_db_raw": False, "detail_r_raw": True,
          "chamfer_p2t_m_raw": False, "chamfer_t2p_m_raw": False}
GROUPS = ["houston (X-band holdout)", "owens (near)", "w_australia (far)", "paper test scenes", "all pilot holdouts"]
PAPER = {"ssim": 70.14, "psnr": 26.07, "enl": 201.51, "moi": 1.022, "mor": 1.027, "cx": 0.073}


def parse(m):
    if m in ("esrt_a", "cnn_a"):
        return m[:-2], "scratch", 1.0, 0
    r = re.match(r"r4_(esrt|cnn)_(scratch|x|xl)_f(1|25)_s(\d)", m)
    if not r:
        return None, None, None, None
    return r.group(1), r.group(2), 1.0 if r.group(3) == "1" else 0.25, int(r.group(4))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "r4_scores_tiles.csv"
    d = pd.read_csv(HERE / src)
    d = pd.concat([d, d[d.region != "amieva_test"].assign(group="all pilot holdouts")])
    d[["arch", "init", "frac", "seed"]] = d.model.apply(lambda m: pd.Series(parse(m)))
    gm = d.groupby(["group", "model", "arch", "init", "frac", "seed"], dropna=False)[MET].mean().reset_index()
    var = gm[gm.arch.notna()].groupby(["group", "arch", "init", "frac"])
    vs = var[MET].mean().reset_index()
    sd = var[MET].std().reset_index()
    ns = var.size().reset_index(name="n_seeds")
    vs = vs.merge(sd, on=["group", "arch", "init", "frac"], suffixes=("", "_sd")).merge(ns, on=["group", "arch", "init", "frac"])
    bic = gm[gm.model == "bicubic"].assign(arch="bicubic", init="-", frac=np.nan, n_seeds=1)
    vs = pd.concat([vs, bic[["group", "arch", "init", "frac", "n_seeds"] + MET]], ignore_index=True)
    vs.to_csv(HERE / "r4_variants.csv", index=False)

    rows, dec = [], []
    for arch in ["esrt", "cnn"]:
        ref = d[d.model == f"{arch}_a"].set_index(["group", "tile_id"])
        for (init, frac), g in d[(d.arch == arch)].groupby(["init", "frac"]):
            avg = g.groupby(["group", "tile_id"])[MET].mean()
            for grp in GROUPS:
                if grp not in avg.index.get_level_values(0):
                    continue
                a, r = avg.loc[grp], ref.loc[grp].loc[avg.loc[grp].index]
                rec = dict(arch=arch, init=init, frac=frac, group=grp, n_tiles=len(a))
                for m in ["ssim_lee", "psnr_lee", "mae_db_raw", "detail_r_raw", "chamfer_p2t_m_raw", "chamfer_t2p_m_raw"]:
                    diff = (a[m] - r[m]).dropna()
                    better = diff > 0 if HIGHER[m] else diff < 0
                    try:
                        p = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
                    except ValueError:
                        p = np.nan
                    rec.update({f"{m}_diff": diff.mean(), f"{m}_better": better.mean(), f"{m}_p": p})
                rows.append(rec)
        # decision: pretrained vs scratch at the same share, on ssim_lee
        for frac in [1.0, 0.25]:
            sc = vs[(vs.arch == arch) & (vs.init == "scratch") & (vs.frac == frac)].set_index("group")
            for init in ["x", "xl"]:
                pv = vs[(vs.arch == arch) & (vs.init == init) & (vs.frac == frac)].set_index("group")
                for grp in GROUPS:
                    if grp not in pv.index or grp not in sc.index:
                        continue
                    pr = [r for r in rows if r["arch"] == arch and r["init"] == init and r["frac"] == frac and r["group"] == grp]
                    noise = 2 * max(pv.loc[grp, "ssim_lee_sd"] or 0, sc.loc[grp, "ssim_lee_sd"] or 0)
                    gain = pv.loc[grp, "ssim_lee"] - sc.loc[grp, "ssim_lee"]
                    p = pr[0]["ssim_lee_p"] if pr else np.nan
                    da = pr[0]["ssim_lee_diff"] if pr else np.nan
                    dec.append(dict(arch=arch, init=init, frac=frac, group=grp, gain_vs_scratch=gain, two_seed_sd=noise,
                                    diff_vs_armA=da, p_vs_armA=p,
                                    beats=bool(gain > noise and da > 0 and p < 0.05)))
    pd.DataFrame(rows).to_csv(HERE / "r4_paired.csv", index=False)
    pd.DataFrame(dec).to_csv(HERE / "r4_decision.csv", index=False)

    pp = vs[(vs.group == "paper test scenes")][["arch", "init", "frac", "n_seeds", "ssim_lee", "psnr_lee", "ssim_raw", "psnr_raw", "enl", "moi", "mor", "cx"]]
    pp.to_csv(HERE / "r4_paper.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(dec).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()

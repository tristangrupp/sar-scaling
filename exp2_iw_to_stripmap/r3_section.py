"""Write results_r3_section.html: round 3, ESRT trained on the expanded (v2) set, with and without
the change mask, against bicubic and the round 1 ESRT on old and new holdouts; per WorldCover class;
paired per-tile tests; transfer panels for the new holdouts. make_results.py inserts it.
Also writes r3_paired.csv.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
from html import escape

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import ROOT

WC = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up", 60: "Bare / sparse",
      70: "Snow / ice", 80: "Water", 90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss / lichen"}
GROUPS = ["houston (X-band holdout)", "owens (near)", "w_australia (far)", "paper test scenes", "all pilot holdouts",
          "yemen_west (near, inland)", "bangladesh (far, inland)", "all inland holdouts"]
OLD = ["houston", "owens", "w_australia", "amieva_test"]
NEW = ["yemen_west", "bangladesh"]


def img64(p, width=1200):
    from PIL import Image
    im = Image.open(p).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def tbl(head, rows):
    th = "".join(f"<th{' class=n' if i else ''}>{escape(h)}</th>" for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f"<td{' class=n' if i else ''}>{v}</td>" for i, v in enumerate(r)) + "</tr>" for r in rows)
    return f"<div class='tw'><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>"


def auto_notes(pr, win):
    """Data-driven summary used until a hand-written r3_notes.html exists."""
    def line(comp, grp):
        r = pr[(pr.comparison == comp) & (pr.group == grp)]
        if r.empty:
            return "n/a"
        r = r.iloc[0]
        sig = "significant" if r.p < 0.05 else "not significant"
        return f"{r.mean_dssim:+.2f} SSIM points ({r.win_pct:.0f}% of {int(r.n)} tiles better, p = {r.p:.2g}, {sig})"
    v = "v2 (best) vs round 1 ESRT"
    m = "change mask vs no mask (seed 0)"
    return (f"<p>ESRT trained on the v2 set (307 training tiles, against 200 in round 1; paper loss: SSIM + histogram "
            f"matching, 8,000 steps, batch 16) is compared with the round 1 ESRT. Against Lee-filtered Stripmap the v2 model "
            f"(the {'masked' if win == 'mask' else 'unmasked'} variant, which won on validation) changes SSIM by "
            f"{line(v, 'new inland holdouts')} on the new inland holdouts and by {line(v, 'old holdouts')} on the old "
            f"holdouts. The change mask (7.2% of training loss pixels removed) changes SSIM by "
            f"{line(m, 'old holdouts')} on the old holdouts and {line(m, 'new inland holdouts')} on the inland ones.</p>")


def main():
    win = (ROOT / "train_r3_winner.txt").read_text().split()[-1]
    models = ["bicubic", "lee_bicubic", "esrt_paper", "esrt_v2_mask_s0", "esrt_v2_nomask_s0", f"esrt_v2_{win}_s1"]
    label = {"bicubic": "Bicubic IW", "lee_bicubic": "Lee IW + bicubic", "esrt_paper": "ESRT, round 1 set (v1)",
             "esrt_v2_mask_s0": "ESRT, v2 + change mask", "esrt_v2_nomask_s0": "ESRT, v2, no mask",
             f"esrt_v2_{win}_s1": f"ESRT, v2{' + change mask' if win == 'mask' else ', no mask'}, seed 1"}
    s = pd.read_csv(ROOT / "scores_summary_r3.csv")
    t = pd.read_csv(ROOT / "scores_tiles_r3.csv")
    val = {m: json.loads((ROOT / "ckpt" / f"{m}.json").read_text())["best_val_ssim"]
           for m in models[2:] if (ROOT / "ckpt" / f"{m}.json").exists()}
    # ---- per group table
    blocks = ""
    for g in GROUPS:
        x = s[(s.group == g) & s.model.isin(models)].set_index("model").reindex([m for m in models if m in set(s.model)])
        if x.empty:
            continue
        rows = [[label[m], f"{r.ssim_lee:.1f}", f"{r.psnr_lee:.2f}", f"{r.ssim_raw:.1f}", f"{r.detail_r_raw:.3f}",
                 f"{r.hf_ratio_raw:.2f}", f"{r.enl:.0f}"] for m, r in x.iterrows()]
        blocks += f"<h3>{escape(g)}</h3>" + tbl(["Model", "SSIM vs Lee", "PSNR vs Lee", "SSIM vs raw", "Detail r",
                                                "HF ratio", "ENL"], rows)
    # ---- paired tests
    piv = t.pivot_table(index="tile_id", columns="model", values="ssim_lee")
    reg = t.drop_duplicates("tile_id").set_index("tile_id").region
    pairs = [(f"esrt_v2_{win}_s0", "esrt_paper", "v2 (best) vs round 1 ESRT"),
             ("esrt_v2_mask_s0", "esrt_v2_nomask_s0", "change mask vs no mask (seed 0)")]
    if f"esrt_v2_{win}_s1" in piv:
        pairs.append((f"esrt_v2_{win}_s1", "esrt_paper", "v2 (best, seed 1) vs round 1 ESRT"))
    prow, out = [], []
    for a, b, name in pairs:
        if a not in piv or b not in piv:
            continue
        for gname, regs in [("old holdouts", OLD), ("houston", ["houston"]), ("owens", ["owens"]),
                            ("w_australia", ["w_australia"]), ("paper test", ["amieva_test"]),
                            ("new inland holdouts", NEW), ("yemen_west", ["yemen_west"]), ("bangladesh", ["bangladesh"])]:
            idx = reg[reg.isin(regs)].index
            d = (piv.loc[idx, a] - piv.loc[idx, b]).dropna()
            if len(d) < 5:
                continue
            p = wilcoxon(d).pvalue
            out.append({"comparison": name, "group": gname, "n": len(d), "mean_dssim": d.mean(), "win_pct": 100 * (d > 0).mean(), "p": p})
            prow.append([escape(name), gname, len(d), f"{d.mean():+.2f}", f"{100 * (d > 0).mean():.0f}%", f"{p:.3g}"])
    pd.DataFrame(out).to_csv(ROOT / "r3_paired.csv", index=False)
    ptab = tbl(["Comparison", "Holdout", "Tiles", "Mean ΔSSIM (Lee)", "Tiles better", "Wilcoxon p"], prow)
    # ---- per class (pixel-weighted SSIM vs Lee), old vs new holdouts
    ct = pd.read_csv(ROOT / "scores_class_tiles_r3.csv")
    ct = ct[(ct.ref == "lee") & ct.model.isin(models)]
    ct["set"] = np.where(ct.tile_id.map(reg).isin(NEW), "inland holdouts", "old holdouts")
    agg = ct.groupby(["set", "cls", "model"])[["ssim_sum", "px"]].sum().reset_index()
    agg["ssim"] = 100 * agg.ssim_sum / agg.px
    crow = []
    for (st, c), g in agg.groupby(["set", "cls"]):
        px = g.px.iloc[0]
        if px < 20000:
            continue
        v = g.set_index("model").ssim
        crow.append([st, WC.get(c, c), f"{px / 1e6:.2f} M"] + [f"{v.get(m, np.nan):.1f}" for m in models])
    ctab = tbl(["Holdouts", "WorldCover class", "Pixels"] + [label[m] for m in models], crow)
    vtxt = ", ".join(f"{label[m]} {v:.4f}" for m, v in val.items())
    notes = (ROOT / "r3_notes.html").read_text(encoding="utf-8") if (ROOT / "r3_notes.html").exists() else auto_notes(pd.DataFrame(out), win)
    figs = "".join(f"<figure class='card'><img src='{img64(p)}' alt='{p.stem}'><figcaption>{escape(p.stem.replace('_', ' '))}</figcaption></figure>"
                   for p in sorted((ROOT / "panels").glob("r3_neighbors_*.png")))
    html = f"""<section class='card' style='border-left:3px solid var(--accent)'><h2>Round 3: training on the expanded set</h2>
{notes}<p>Validation SSIM (Lee, 63 v2 validation tiles): {vtxt}.</p>
<h3>Paired per-tile tests</h3>{ptab}
{blocks}
<h3>By land-cover class (pixel-weighted SSIM vs Lee-filtered Stripmap)</h3>{ctab}
<p class='lede' style='font-size:.85rem'>All models score in one common normalization (the round 1 model's min-max), each
predicting in its own; bicubic and Lee + bicubic as before. Classes with under 20,000 test pixels are omitted.</p>
<div class='grid' style='grid-template-columns:1fr'>{figs}</div></section>"""
    (ROOT / "results_r3_section.html").write_text(html, encoding="utf-8")
    print("wrote results_r3_section.html", flush=True)
    print(pd.DataFrame(out).round(3).to_string(), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)

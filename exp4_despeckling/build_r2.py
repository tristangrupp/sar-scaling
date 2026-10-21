"""Round 2 section of the Experiment 4 page (appended after round 1 by build_page.main)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_page import FIG, fmt, img64

ROOT = Path(__file__).resolve().parent
LAB = {"raw": "Raw", "lee": "Lee 7x7", "refined_lee": "Refined Lee", "gamma_map": "Gamma-MAP",
       "cnn_sup": "R1 CNN, supervised", "cnn_n2n": "R1 Noise2Noise (8 sites)",
       "r2_sup_cond_vv": "R2 supervised + ENL conditioning", "r2_n2n_big_vv": "R2 Noise2Noise, 24 sites",
       "r2_n2n_big_cond_vv": "R2 Noise2Noise, 24 sites + conditioning",
       "r2_sup_cond_vvvh": "R2 supervised + cond., VV+VH", "r2_n2n_big_cond_vvvh": "R2 Noise2Noise + cond., VV+VH",
       "temporal_mean": "Temporal mean", "ref": "Temporal mean"}
ORDER = ["raw", "lee", "refined_lee", "gamma_map", "cnn_sup", "cnn_n2n", "r2_sup_cond_vv", "r2_n2n_big_vv",
         "r2_n2n_big_cond_vv", "r2_sup_cond_vvvh", "r2_n2n_big_cond_vvvh", "temporal_mean", "ref"]


def _t(head, rows):
    return ('<div class="tw"><table><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>" +
            "".join("<tr>" + "".join(rows_i) + "</tr>" for rows_i in rows) + "</table></div>")


def med_table(d, cols, heads, digits):
    ps = d.groupby(["method", "site"]).median(numeric_only=True).reset_index()
    md = ps.groupby("method").median(numeric_only=True)
    rows = []
    for m in [m for m in ORDER if m in md.index]:
        rows.append([f"<td>{LAB[m]}</td>"] + [f"<td class='n'>{fmt(md.loc[m, c], g)}</td>" for c, g in zip(cols, digits)])
    return _t(["Method"] + heads, rows), ps


def site_table(ps, metric, digits=2):
    sites = sorted(ps.site.unique())
    rows = []
    for m in [m for m in ORDER if m in ps.method.unique()]:
        s = ps[ps.method == m].set_index("site")[metric]
        rows.append([f"<td>{LAB[m]}</td>"] + [f"<td class='n'>{fmt(s.get(x, np.nan), digits)}</td>" for x in sites])
    return _t(["Method"] + [s.replace("_", " ") for s in sites], rows)


def section(fill):
    f = ROOT / "metrics_r2.csv"
    if not f.exists():
        return ""
    d = pd.read_csv(f).replace([np.inf, -np.inf], np.nan)
    out = ['<section class="card"><h2>Round 2</h2>', fill.get("R2_INTRO", "")]
    cols = ["psnr", "ssim", "mae", "bias_db", "enl", "mean_ratio", "ratio_enl", "grad_corr", "epd_h"]
    heads = ["PSNR dB", "SSIM", "MAE dB", "Bias dB", "ENL homog.", "Mean out/in", "Ratio ENL", "Gradient corr", "EPD-ROA H"]
    dg = [2, 3, 2, 2, 0, 3, 1, 3, 3]
    vv1 = d[(d.pol == "vv") & (d.k == 1)]
    tab, ps = med_table(vv1, cols, heads, dg)
    out += ["<h3>1. Single-date VV, held-out medians</h3>", tab,
            "<h3>PSNR by site (dB), single date VV</h3>", site_table(ps, "psnr"),
            "<h3>Bias by site (dB), single date VV</h3>", site_table(ps, "bias_db"), fill.get("R2_FIX", "")]
    # multi-look inputs
    rows = []
    for k in (1, 2, 4):
        dk = d[(d.pol == "vv") & (d.k == k)]
        pk = dk.groupby(["method", "site"]).median(numeric_only=True).reset_index()
        for m in [m for m in ORDER if m in pk.method.unique()]:
            a = pk[pk.method == m]
            at = a[a.site == "atacama_desert"]
            rows.append([f"<td class='n'>{k}</td>", f"<td>{LAB[m]}</td>", f"<td class='n'>{a.psnr.median():.2f}</td>",
                         f"<td class='n'>{a.bias_db.median():+.2f}</td>", f"<td class='n'>{a.grad_corr.median():.3f}</td>",
                         f"<td class='n'>{at.psnr.median():.2f}</td>", f"<td class='n'>{at.bias_db.median():+.2f}</td>"])
    out += ["<h3>Inputs that are already cleaner: mean of k dates</h3>",
            _t(["k dates", "Method", "PSNR, median of sites", "Bias dB", "Gradient corr", "PSNR, Atacama", "Bias, Atacama"], rows),
            fill.get("R2_K", "")]
    # VH
    vh = d[(d.pol == "vh") & (d.k == 1)]
    if len(vh):
        tab, psh = med_table(vh, cols, heads, dg)
        out += ["<h3>3. VH (cross-pol), single date, held-out medians</h3>", tab,
                "<h3>VH PSNR by site (dB)</h3>", site_table(psh, "psnr"), fill.get("R2_VH", "")]
    # panels
    figs = [p for p in sorted(FIG.glob("r2_panel_*.png"))]
    if figs:
        out.append("<h3>Panels</h3>")
        for p in figs:
            out.append(f'<figure><img src="{img64(p)}" alt="{p.stem}" style="width:100%;height:auto"><figcaption>'
                       f'{p.stem.replace("r2_panel_", "").replace("_", " ")}: 2.56 km crop at 10 m; top row VV, bottom row VH '
                       f'(VV-only methods leave the VH cell blank). Each row shares the stretch of its temporal mean.</figcaption></figure>')
    # downstream
    wc = ROOT / "downstream_r2_wc.csv"
    if wc.exists():
        c = pd.read_csv(wc)
        rows = []
        for m in [m for m in ORDER if m in c.method.unique()]:
            g = c[c.method == m]
            v, b = g[g.features == "vv"], g[g.features == "vvvh"]
            rows.append([f"<td>{LAB[m]}</td>", f"<td class='n'>{v.oa.mean():.3f}</td>", f"<td class='n'>{v.macro_f1.mean():.3f}</td>",
                         f"<td class='n'>{fmt(b.oa.mean() if len(b) else np.nan, 3)}</td>",
                         f"<td class='n'>{fmt(b.macro_f1.mean() if len(b) else np.nan, 3)}</td>"])
        out += ["<h3>4a. WorldCover land cover, held-out sites (texture features, mean of 3 dates)</h3>",
                _t(["Input", "OA, VV", "Macro F1, VV", "OA, VV+VH", "Macro F1, VV+VH"], rows), fill.get("R2_WC", "")]
    ft = ROOT / "downstream_r2_ftw.csv"
    if ft.exists():
        c = pd.read_csv(ft)
        rows = []
        for m in [m for m in ORDER if m in c.method.unique()]:
            g = c[c.method == m]
            v, b = g[g.features == "vv"], g[g.features == "vvvh"]
            rows.append([f"<td>{LAB[m]}</td>", f"<td class='n'>{v.f1_boundary.mean():.3f}</td>",
                         f"<td class='n'>{v.f1_field.mean():.3f}</td>", f"<td class='n'>{v.macro_f1.mean():.3f}</td>",
                         f"<td class='n'>{fmt(b.f1_boundary.mean() if len(b) else np.nan, 3)}</td>",
                         f"<td class='n'>{fmt(b.macro_f1.mean() if len(b) else np.nan, 3)}</td>",
                         f"<td class='n'>{v.edge_f1.mean():.3f}</td>"])
        out += ["<h3>4b. Field boundaries, Fields of The World labels (20 test chips)</h3>",
                _t(["Input", "Boundary F1, VV", "Field F1, VV", "Macro F1, VV", "Boundary F1, VV+VH", "Macro F1, VV+VH",
                    "Canny edge F1, VV"], rows), fill.get("R2_FTW", "")]
    out += [fill.get("R2_CAVEATS", ""), fill.get("R2_VERDICT", ""), "</section>"]
    return "\n".join(x for x in out if x)

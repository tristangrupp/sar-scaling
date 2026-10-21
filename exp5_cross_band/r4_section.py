"""Round 4 section: matched-recipe headline test against Experiment 2's arm A. Tables come from the
e5r4_analyze.py outputs; prose from r4_prose.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
GROUPS = ["owens (near)", "w_australia (far)", "houston (X-band holdout)", "paper test scenes", "all pilot holdouts"]
INIT = {"scratch": "from scratch", "x": "X-band pretrained", "xl": "X+L pretrained", "-": ""}
ARCH = {"esrt": "ESRT", "cnn": "CNN", "bicubic": "Bicubic"}


def f(v, d=2):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def variants_table(vs, grp):
    d = vs[vs.group == grp].copy()
    d["o"] = d.arch.map({"bicubic": 0, "esrt": 1, "cnn": 2}) * 10 + d.init.map({"-": 0, "scratch": 1, "x": 2, "xl": 3}) + (1 - d.frac.fillna(1)) * 5
    d = d.sort_values("o")
    cols = [("ssim_lee", "SSIM vs Lee ref.", 2), ("psnr_lee", "PSNR vs Lee ref.", 2), ("mae_db_raw", "MAE dB", 3),
            ("detail_r_raw", "detail_r", 3), ("chamfer_p2t_m_raw", "Edges out→ref m", 1), ("chamfer_t2p_m_raw", "Edges ref→out m", 1),
            ("enl", "ENL", 0)]
    h = ("<div class='tw'><table><tr><th>Model</th><th>Start</th><th>Share</th><th>Seeds</th>" +
         "".join(f"<th>{lbl}</th>" for _, lbl, _ in cols) + "</tr>")
    for _, r in d.iterrows():
        h += (f"<tr><td>{ARCH[r.arch]}</td><td>{INIT[r.init]}</td><td class='n'>{'' if pd.isna(r.frac) else f'{r.frac:.0%}'}</td>"
              f"<td class='n'>{int(r.n_seeds)}</td>")
        for c, _, dg in cols:
            sd = r.get(c + "_sd", np.nan)
            h += f"<td class='n'>{f(r[c], dg)}{'' if pd.isna(sd) else ' ± ' + f(sd, dg)}</td>"
        h += "</tr>"
    return h + "</table></div>"


def decision_table(dec):
    h = ("<div class='tw'><table><tr><th>Model</th><th>Start</th><th>Share</th><th>Group</th><th>SSIM gain vs scratch seed-mean</th>"
         "<th>2 x seed SD</th><th>Paired diff vs arm A (per tile)</th><th>Wilcoxon p</th><th>Beyond seed noise?</th></tr>")
    for _, r in dec.iterrows():
        pill = "<span class='pill done'>yes</span>" if r.beats else "<span class='pill neg'>no</span>"
        h += (f"<tr><td>{ARCH[r.arch]}</td><td>{INIT[r.init]}</td><td class='n'>{r.frac:.0%}</td><td>{r.group}</td>"
              f"<td class='n'>{r.gain_vs_scratch:+.2f}</td><td class='n'>{r.two_seed_sd:.2f}</td>"
              f"<td class='n'>{r.diff_vs_armA:+.2f}</td><td class='n'>{r.p_vs_armA:.3g}</td><td>{pill}</td></tr>")
    return h + "</table></div>"


def paper_table(pp):
    h = ("<div class='tw'><table><tr><th>Model</th><th>Start</th><th>Share</th><th>SSIM (Lee ref.)</th><th>PSNR (Lee ref.)</th>"
         "<th>SSIM (raw ref.)</th><th>PSNR (raw ref.)</th><th>ENL</th><th>MoI</th><th>MoR</th><th>Cx</th></tr>"
         "<tr><td><b>Amieva et al. 2026, published</b></td><td>ESRT, SSIM + HM</td><td></td><td class='n'>70.14</td><td class='n'>26.07</td>"
         "<td class='n'>–</td><td class='n'>–</td><td class='n'>201.51</td><td class='n'>1.022</td><td class='n'>1.027</td><td class='n'>0.073</td></tr>")
    pp = pp[(pp.frac.isna()) | (pp.frac == 1.0)]
    for _, r in pp.iterrows():
        h += (f"<tr><td>{ARCH[r.arch]}</td><td>{INIT[r.init]}</td><td class='n'>{'' if pd.isna(r.frac) else f'{r.frac:.0%}'}</td>"
              f"<td class='n'>{f(r.ssim_lee)}</td><td class='n'>{f(r.psnr_lee)}</td><td class='n'>{f(r.ssim_raw)}</td><td class='n'>{f(r.psnr_raw)}</td>"
              f"<td class='n'>{f(r.enl, 1)}</td><td class='n'>{f(r.moi, 3)}</td><td class='n'>{f(r.mor, 3)}</td><td class='n'>{f(r.cx, 3)}</td></tr>")
    return h + "</table></div>"


def section(c):
    if not (ROOT / "r4_decision.csv").exists() or not (ROOT / "r4_scores_tiles.csv").exists():
        return ""
    vs = pd.read_csv(ROOT / "r4_variants.csv")
    dec = pd.read_csv(ROOT / "r4_decision.csv")
    pp = pd.read_csv(ROOT / "r4_paper.csv")
    T = {g: variants_table(vs, g) for g in GROUPS}
    import r4_prose
    return r4_prose.render(c, vs, dec, T, decision_table(dec), paper_table(pp))

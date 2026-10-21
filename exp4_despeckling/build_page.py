"""Write exp4_despeckling/page_body.html (body fragment for build_site.py) from the CSVs and figures."""
from __future__ import annotations

import base64
import html
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figs"

LABEL = {"raw": "Raw single date", "box3": "Boxcar 3x3", "box5": "Boxcar 5x5", "lee": "Lee 7x7",
         "refined_lee": "Refined Lee 7x7", "gamma_map": "Gamma-MAP 7x7", "exp1v1_l1": "Exp 1 CNN v1 (block-avg to 10 m)",
         "exp1v1_gan": "Exp 1 CNN+GAN v1", "exp1_l1": "Exp 1 CNN, Capella retrain", "exp1_gan": "Exp 1 CNN+GAN, Capella retrain", "cnn_sup": "New CNN, supervised", "cnn_n2n": "New CNN, Noise2Noise",
         "cnn_gan": "New CNN + GAN fine-tune", "ref_loo": "Temporal mean (reference)", "oracle": "Oracle (input / reference)"}
ORDER = ["raw", "box3", "box5", "lee", "refined_lee", "gamma_map", "exp1v1_l1", "exp1v1_gan", "exp1_l1", "exp1_gan", "cnn_sup", "cnn_n2n", "cnn_gan"]


def img64(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


def site_medians(df):
    """Median over dates within each site, then the per-site values."""
    return df.groupby(["method", "site"]).median(numeric_only=True).reset_index()


def fmt(v, d=2):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "&mdash;"
    return f"{v:.{d}f}"


def dotplot(per_site, metric, title, better, ideal=None, d=2, methods=None):
    """Inline SVG: one row per method, small dots = held-out sites, bar mark = median across sites."""
    methods = [m for m in (methods or ORDER) if m in per_site.method.unique()]
    sub = per_site[per_site.method.isin(methods)][["method", "site", metric]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return ""
    lo, hi = sub[metric].min(), sub[metric].max()
    if ideal is not None:
        lo, hi = min(lo, ideal), max(hi, ideal)
    pad = (hi - lo) * 0.06 or 1
    lo, hi = lo - pad, hi + pad
    W, left, right, rh, top = 560, 190, 20, 20, 26
    H = top + rh * len(methods) + 26
    X = lambda v: left + (v - lo) / (hi - lo) * (W - left - right)  # noqa: E731
    s = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;max-width:{W}px;height:auto;font:11px var(--f-body)" role="img" '
         f'aria-label="{html.escape(title)}">',
         f'<text x="0" y="14" fill="var(--fg)" style="font-weight:600">{html.escape(title)}</text>',
         f'<text x="{W - right}" y="14" fill="var(--muted)" text-anchor="end">{html.escape(better)}</text>']
    for t in np.linspace(lo + pad, hi - pad, 5):
        s.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top - 4}" y2="{H - 22}" stroke="var(--line)"/>'
                 f'<text x="{X(t):.1f}" y="{H - 8}" fill="var(--muted)" text-anchor="middle">{t:.{d}f}</text>')
    if ideal is not None:
        s.append(f'<line x1="{X(ideal):.1f}" x2="{X(ideal):.1f}" y1="{top - 4}" y2="{H - 22}" stroke="var(--good)" '
                 f'stroke-dasharray="3 3"/>')
    for i, m in enumerate(methods):
        y = top + rh * i + rh / 2
        col = "var(--accent)" if m.startswith("cnn") else ("var(--a2)" if m.startswith("exp1") else "var(--a3)")
        if m == "raw":
            col = "var(--muted)"
        s.append(f'<text x="{left - 8}" y="{y + 4:.1f}" fill="var(--fg)" text-anchor="end">{html.escape(LABEL[m])}</text>')
        v = sub[sub.method == m][metric].values
        for x in v:
            s.append(f'<circle cx="{X(x):.1f}" cy="{y:.1f}" r="3" fill="{col}" fill-opacity="0.35"/>')
        if len(v):
            med = np.median(v)
            s.append(f'<rect x="{X(med) - 1.5:.1f}" y="{y - 7:.1f}" width="3" height="14" fill="{col}"><title>'
                     f'{LABEL[m]}: median {med:.3f}</title></rect>')
    s.append("</svg>")
    return "".join(s)


def logdot(per_site, metric, title, better, ideal=None, methods=None):
    p = per_site.copy()
    p[metric] = np.log10(p[metric].replace([np.inf, -np.inf], np.nan))
    svg = dotplot(p, metric, title + " (log10)", better, None if ideal is None else math.log10(ideal), d=1, methods=methods)
    return svg


def table(df, cols, heads, digits, first="Method"):
    h = "".join(f"<th>{x}</th>" for x in [first] + heads)
    rows = []
    for _, r in df.iterrows():
        tds = "".join(f'<td class="n">{fmt(r[c], dg)}</td>' for c, dg in zip(cols, digits))
        rows.append(f"<tr><td>{r['label']}</td>{tds}</tr>")
    return f'<div class="tw"><table><tr>{h}</tr>{"".join(rows)}</table></div>'


def main():
    from text import TEXT  # prose lives in text.py so numbers and words stay separate
    stack = pd.read_csv(ROOT / "stack_summary.csv")
    d = pd.read_csv(ROOT / "metrics_dates.csv").replace([np.inf, -np.inf], np.nan)
    ps = site_medians(d)
    med = ps.groupby("method").median(numeric_only=True).reset_index()
    methods = [m for m in ORDER if m in med.method.values]
    med = med.set_index("method").loc[methods + ["oracle"]].reset_index()
    med["label"] = med.method.map(LABEL)
    orc = ps[ps.method == "oracle"]
    oracle_enl_ratio = float(orc.ratio_enl.median())
    oracle_enl_ref = float(orc.enl.median())
    T = TEXT(locals())

    out = [T["header"], T["goal"]]

    # ---- design
    srows = "".join(
        f"<tr><td>{r.site.replace('_', ' ')}</td><td>{r.landcover}</td><td>{'<span class=\"pill done\">train</span>' if r.split == 'train' else '<span class=\"pill run\">held out</span>'}</td>"
        f"<td class='n'>{r.n_dates}</td><td>{r.first} to {r.last}</td><td>{r.orbit.strip('()').replace(chr(39), '')}</td>"
        f"<td class='n'>{r.L_single:.1f}</td><td class='n'>{r.cv_thr99:.2f}</td><td class='n'>{100 * r.stable_frac:.0f}%</td></tr>"
        for r in stack.itertuples())
    out.append(T["design"].replace("{SITES}", '<div class="tw"><table><tr><th>Site</th><th>Land cover</th><th>Split</th>'
                                    '<th>Dates</th><th>Window</th><th>Relative orbit</th><th>Single-date ENL</th>'
                                    '<th>CV threshold</th><th>Stable pixels</th></tr>' + srows + "</table></div>"))

    # ---- results table
    cols = ["psnr", "ssim", "mae", "bias_db", "enl", "mean_ratio", "ratio_enl", "ratio_struct", "grad_corr", "epd_h", "epd_v"]
    heads = ["PSNR dB", "SSIM", "MAE dB", "Bias dB", "ENL homog.", "Mean out/in", "Ratio ENL", "Ratio-edge corr",
             "Gradient corr", "EPD-ROA H", "EPD-ROA V"]
    tab = table(med, cols, heads, [2, 3, 2, 2, 0, 3, 1, 3, 3, 3, 3])
    charts = [dotplot(ps, "psnr", "PSNR vs temporal mean (dB)", "higher is better"),
              dotplot(ps, "ssim", "SSIM vs temporal mean", "higher is better", d=2),
              dotplot(ps, "mae", "MAE vs temporal mean (dB)", "lower is better"),
              logdot(ps, "enl", "ENL in homogeneous areas", f"reference {oracle_enl_ref:.0f}", ideal=oracle_enl_ref),
              dotplot(ps, "mean_ratio", "Mean preservation (out / in)", "ideal 1", ideal=1.0, d=2),
              logdot(ps, "ratio_enl", "Ratio-image ENL", f"pure speckle {oracle_enl_ratio:.1f}", ideal=oracle_enl_ratio,
                     methods=[m for m in ORDER if m != "raw"]),
              dotplot(ps, "grad_corr", "Gradient correlation with reference", "higher is better", d=2),
              dotplot(ps, "epd_h", "EPD-ROA (horizontal) vs reference", "ideal 1", ideal=1.0, d=2)]
    persite = ps[ps.method.isin(methods)].pivot(index="method", columns="site", values="psnr").loc[methods]
    psh = "".join(f"<th>{c.replace('_', ' ')}</th>" for c in persite.columns)
    psr = "".join(f"<tr><td>{LABEL[m]}</td>" + "".join(f"<td class='n'>{v:.2f}</td>" for v in persite.loc[m]) + "</tr>"
                  for m in methods)
    out.append(T["results"].replace("{TABLE}", tab).replace("{CHARTS}", '<div class="grid">' + "".join(
        f"<div>{c}</div>" for c in charts) + "</div>").replace(
        "{PERSITE}", f'<div class="tw"><table><tr><th>Method</th>{psh}</tr>{psr}</table></div>'))

    # ---- panels
    figs = []
    for r in stack[stack.split == "test"].itertuples():
        p = FIG / f"panel_{r.site}.png"
        if p.exists():
            figs.append(f'<figure><img src="{img64(p)}" alt="Despeckling panel, {r.site}" style="width:100%;height:auto">'
                        f'<figcaption>{r.site.replace("_", " ")} ({r.landcover}, held out). Top: the 2.56 km crop with the '
                        f'most structure, at native 10 m. Bottom: the orange box, 640 m. All panels share the dB stretch of '
                        f'the temporal mean. Dark areas in the single-date panels that are bright in the reference changed '
                        f'on this date (for example water) and fall outside the scored mask.</figcaption></figure>')
    out.append(T["panels"].replace("{FIGS}", "".join(figs)))

    # ---- downstream
    if (ROOT / "downstream_cls.csv").exists():
        c = pd.read_csv(ROOT / "downstream_cls.csv")
        e = pd.read_csv(ROOT / "downstream_edges.csv")
        dm = [m for m in ORDER + ["ref_loo"] if m in c.method.unique()]
        rows = []
        for m in dm:
            g = c[c.method == m]
            px, tx = g[g.features == "pixel"], g[g.features == "texture"]
            ee = e[e.method == m].groupby("date_idx").f1.mean()
            rows.append(f"<tr><td>{LABEL[m]}</td><td class='n'>{px.oa.mean():.3f}</td><td class='n'>{px.macro_f1.mean():.3f}</td>"
                        f"<td class='n'>{tx.oa.mean():.3f}</td><td class='n'>{tx.macro_f1.mean():.3f}</td>"
                        f"<td class='n'>{ee.mean():.3f}</td></tr>")
        dtab = ('<div class="tw"><table><tr><th>Input</th><th>OA, pixel only</th><th>Macro F1, pixel only</th>'
                '<th>OA, texture</th><th>Macro F1, texture</th><th>Boundary F1</th></tr>' + "".join(rows) + "</table></div>")
        out.append(T["downstream"].replace("{DTABLE}", dtab).replace("{CLASSES}", c.classes.iloc[0]))
    out += [T["gan"], T["caveats"], T["verdict"]]
    from build_r2 import section
    from findings_r2 import FILL_R2
    out.append(section(FILL_R2))
    (ROOT / "page_body.html").write_text("\n".join(out), encoding="utf-8")
    print("page_body.html", (ROOT / "page_body.html").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()

"""Round 2 results section (rendered once r2_nisar_scores.csv exists)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MET = ["mae", "detail_r", "hf_ratio", "chamfer_m", "edgeF_10m", "edgeF_20m"]
LBL = {"mae": "MAE (std units)", "detail_r": "detail_r", "hf_ratio": "hf_ratio", "chamfer_m": "Edge chamfer (m)",
       "edgeF_10m": "Edge F @10 m", "edgeF_20m": "Edge F @20 m", "ssim": "SSIM", "n": "Tiles",
       "edgeF_10m_s10": "Edge F @10 m, outline", "edgeF_20m_s10": "Edge F @20 m, outline", "cross": "Cross index"}


def parse(name):
    m = re.match(r"ft_(scratch|x2_unc|x2_aug)_f([\d.]+)_(s\d+|3k|b300)", name)
    return (m.group(1), float(m.group(2)), m.group(3)) if m else (None, None, None)


def tab(df, cols, first, names=None, digits=None):
    digits = digits or {}
    h = "<div class='tw'><table><tr>" + "".join(f"<th>{x}</th>" for x in first[1]) + "".join(f"<th>{LBL.get(c, c)}</th>" for c in cols) + "</tr>"
    for _, r in df.iterrows():
        h += "<tr>" + "".join(f"<td>{(names or {}).get(r[k], r[k]) if not isinstance(r[k], float) else f'{r[k]:.2f}'}</td>" for k in first[0])
        h += "".join(f"<td class='n'>{r[c]:.{digits.get(c, 3)}f}</td>" if pd.notna(r[c]) else "<td class='n'>–</td>" for c in cols) + "</tr>"
    return h + "</table></div>"


def curve_data():
    d = pd.read_csv(ROOT / "r2_nisar_scores.csv")
    d["role2"] = "all"
    rows = []
    for role in ["near", "far", "all"]:
        dd = d if role == "all" else d[d.role == role]
        med = dd.groupby(["model"])[MET].median().reset_index()
        for _, r in med.iterrows():
            init, frac, seed = parse(r.model)
            rows.append(dict(role=role, model=r.model, init=init, frac=frac, seed=seed, **{m: float(r[m]) for m in MET}))
    return pd.DataFrame(rows), d


def section(c):
    cd, d = curve_data()
    ntrain = 600
    ft = cd[cd.init.notna() & cd.seed.isin(["s0", "s1"])]
    # mean over seeds of the per-model medians
    lc = ft.groupby(["role", "init", "frac"])[MET].mean().reset_index()
    ref = cd[cd.init.isna()].set_index(["role", "model"])
    R2 = dict(curve=lc.to_dict("records"), ref=cd[cd.init.isna()].to_dict("records"))
    c["r2_json"] = json.dumps(R2, default=float)

    def L(role, init, frac, m):
        x = lc[(lc.role == role) & (lc.init == init) & (lc.frac == frac)]
        return float(x[m].iloc[0]) if len(x) else np.nan

    def Rf(role, model, m):
        try:
            return float(ref.loc[(role, model), m])
        except KeyError:
            return np.nan

    fracs = sorted(lc.frac.unique())
    rows = ""
    for role in ["near", "far"]:
        for f in fracs:
            rows += f"<tr><td>{role}</td><td class='n'>{f:.0%}</td><td class='n'>{int(round(f * ntrain))}</td>"
            for m in ["mae", "detail_r", "edgeF_10m"]:
                a, b = L(role, "scratch", f, m), L(role, "x2_unc", f, m)
                better = (b < a) if m == "mae" else (b > a)
                rows += f"<td class='n'>{a:.3f}</td><td class='n'>{'<b>' if better else ''}{b:.3f}{'</b>' if better else ''}</td>"
            rows += "</tr>"
    lc_tab = ("<div class='tw'><table><tr><th>Holdout</th><th>Fine-tune share</th><th>Tiles</th>"
              "<th>MAE, scratch</th><th>MAE, X-pretrained</th><th>detail_r, scratch</th><th>detail_r, X-pretrained</th>"
              "<th>Edge F @10 m, scratch</th><th>Edge F @10 m, X-pretrained</th></tr>" + rows + "</table></div>")

    # joint vs per-band on NISAR
    names = {"bicubic": "Bicubic", "zeroshot_x2_unc": "X-band 2x, no fine-tune (zero-shot)",
             "zeroshot_x2_cond": "X-band 2x cond., zero-shot", "zeroshot_x2_aug": "X-band 2x band-aug, zero-shot",
             "ft_scratch_f1.0_3k": "NISAR only, 100 %, 3000 steps (per-band model)",
             "ft_scratch_f1.0_s0": "NISAR only, 100 %, 1500 steps", "ft_x2_unc_f1.0_s0": "X-pretrained + NISAR 100 %, 1500 steps",
             "joint": "Joint shared trunk (X + NISAR), 3000 steps", "joint_noembed": "Joint, no embeddings"}
    keep = [k for k in names if (("all", k) in ref.index) or k in set(cd.model)]
    jt = cd[(cd.role.isin(["near", "far"])) & cd.model.isin(keep)].copy()
    jt["o"] = jt.model.map({k: i for i, k in enumerate(names)})
    jt = jt.sort_values(["role", "o"])
    nis_tab = tab(jt, ["mae", "detail_r", "hf_ratio", "chamfer_m", "edgeF_10m", "edgeF_20m"],
                  (["role", "model"], ["Holdout", "Model"]), names, {"chamfer_m": 1})

    # X-band side
    xs = ROOT / "r2_x_scores.csv"
    x_tab, cross_tab = "", ""
    if xs.exists():
        x = pd.read_csv(xs)
        x = x[x.grid == "5m"]
        xm = x.groupby(["input", "model"])[["mae", "detail_r", "hf_ratio", "chamfer_m", "edgeF_10m", "edgeF_20m_s10"]].median().reset_index()
        xn = {"bicubic": "Bicubic", "x2_unc": "X-band 2x (round 1, 8000 steps)", "x2_cond": "X-band 2x cond. (round 1)",
              "x2_aug": "X-band 2x band-aug", "joint (target: X spotlight)": "Joint, target code X spotlight",
              "joint (target: NISAR style)": "Joint, target code NISAR", "joint_noembed": "Joint, no embeddings"}
        x_tab = tab(xm, ["mae", "detail_r", "hf_ratio", "chamfer_m", "edgeF_10m", "edgeF_20m_s10"],
                    (["input", "model"], ["Input", "Model"]), xn, {"chamfer_m": 1})
        cr = ROOT / "r2_cross.csv"
        if cr.exists():
            cc = pd.read_csv(cr).groupby("model").cross.median().reset_index()
            cross_tab = tab(cc, ["cross"], (["model"], ["Model"]), xn)
    c["r2_parts"] = dict(lc_tab=lc_tab, nis_tab=nis_tab, x_tab=x_tab, cross_tab=cross_tab, L=L, Rf=Rf, fracs=fracs)
    import r2_prose
    return r2_prose.render(c)


SCRIPT = r"""
(function(){
  const el = document.getElementById('r2curve'); if(!el || typeof R2==='undefined') return;
  const mets = [['mae','MAE (lower is better)'],['detail_r','detail_r (higher is better)'],['edgeF_10m','Edge F within 10 m']];
  const roles = ['near','far'];
  const W = Math.min(el.clientWidth || 900, 980);
  const w = Math.max(260, Math.floor((W - 20) / 3)), h = 210, m = {l:46, r:10, t:26, b:34};
  roles.forEach(role=>{
    const row = d3.select(el).append('div').style('display','flex').style('flex-wrap','wrap').style('gap','8px');
    mets.forEach(([k,label])=>{
      const data = R2.curve.filter(d=>d.role===role);
      const refs = R2.ref.filter(d=>d.role===role && (d.model==='bicubic'));
      const vals = data.map(d=>d[k]).concat(refs.map(d=>d[k]));
      const x = d3.scaleLog().domain(d3.extent(data, d=>d.frac)).range([m.l, w-m.r]);
      const y = d3.scaleLinear().domain(d3.extent(vals)).nice().range([h-m.b, m.t]);
      const svg = row.append('svg').attr('width',w).attr('height',h).attr('role','img').attr('aria-label', `${label}, ${role} holdout`);
      svg.append('text').attr('x',m.l).attr('y',14).attr('fill','var(--fg)').style('font-size','12px').style('font-weight',600).text(`${label}: ${role} holdout`);
      svg.append('g').attr('transform',`translate(0,${h-m.b})`).call(d3.axisBottom(x).tickValues([...new Set(data.map(d=>d.frac))]).tickFormat(d3.format('.0%')))
        .call(g=>g.selectAll('text').attr('fill','var(--muted)')).call(g=>g.selectAll('line,path').attr('stroke','var(--line)'));
      svg.append('g').attr('transform',`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5))
        .call(g=>g.selectAll('text').attr('fill','var(--muted)')).call(g=>g.selectAll('line,path').attr('stroke','var(--line)'));
      refs.forEach(r=>{ svg.append('line').attr('x1',m.l).attr('x2',w-m.r).attr('y1',y(r[k])).attr('y2',y(r[k]))
        .attr('stroke','var(--muted)').attr('stroke-dasharray','4 3');
        svg.append('text').attr('x',w-m.r).attr('y',y(r[k])-4).attr('text-anchor','end').attr('fill','var(--muted)').style('font-size','10px').text('bicubic'); });
      [['scratch','var(--a2)','from scratch'],['x2_unc','var(--accent)','X-band pretrained']].forEach(([init,col,nm])=>{
        const s = data.filter(d=>d.init===init).sort((a,b)=>a.frac-b.frac);
        svg.append('path').datum(s).attr('fill','none').attr('stroke',col).attr('stroke-width',2)
          .attr('d', d3.line().x(d=>x(d.frac)).y(d=>y(d[k])));
        svg.selectAll(null).data(s).join('circle').attr('cx',d=>x(d.frac)).attr('cy',d=>y(d[k])).attr('r',3.5).attr('fill',col)
          .append('title').text(d=>`${nm}, ${(d.frac*100).toFixed(0)} %: ${d[k].toFixed(3)}`);
      });
    });
  });
  d3.select(el).append('p').style('color','var(--muted)').style('font-size','0.85rem')
    .html('<span style="color:var(--accent)">&#9679;</span> X-band pretrained, then fine-tuned on NISAR &nbsp; <span style="color:var(--a2)">&#9679;</span> NISAR only, from scratch. Same steps and batch for every point; x axis = share of the 600 NISAR training tiles (log scale).');
})();
"""

"""Round 3 section: Stripmap anchor, scored with Experiment 2's own scorer next to its arm A."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXP2 = ROOT.parent / "exp2_iw_to_stripmap"
MET = ["ssim_lee", "psnr_lee", "mae_db_raw", "detail_r_raw", "hf_ratio_raw", "chamfer_p2t_m_raw", "chamfer_t2p_m_raw", "enl", "moi"]
LBL = {"ssim_lee": "SSIM vs Lee ref. (x100)", "psnr_lee": "PSNR vs Lee ref.", "mae_db_raw": "MAE (dB)",
       "detail_r_raw": "detail_r", "hf_ratio_raw": "hf_ratio", "chamfer_p2t_m_raw": "Edges out→ref (m)",
       "chamfer_t2p_m_raw": "Edges ref→out (m)", "enl": "ENL", "moi": "Mean of intensity ratio"}
NAMES = {"bicubic": "Bicubic", "cnn_a": "Exp 2 arm A, CNN (8000 steps)", "esrt_a": "Exp 2 arm A, ESRT (8000 steps)",
         "zeroshot_x2_unc": "X-band 2x, zero-shot (round 1)", "zeroshot_x2_cond": "X-band 2x cond., zero-shot",
         "zeroshot_joint_r2_target_NISAR": "X+NISAR joint (round 2), zero-shot, NISAR code",
         "zeroshot_joint_r2_target_X_spotlight": "X+NISAR joint (round 2), zero-shot, X code",
         "sm_scratch_f1.0_s0": "Stripmap only, 100 %, 3000 steps", "sm_x2_unc_f1.0_s0": "X-pretrained + Stripmap 100 %, 3000 steps",
         "joint3_target_stripmap": "Joint X+L+C, Stripmap target code", "joint3_target_NISAR": "Joint X+L+C, NISAR target code",
         "joint3_target_X_spotlight": "Joint X+L+C, X spotlight target code"}
ORDER = list(NAMES)


def parse(m):
    r = re.match(r"sm_(scratch|x2_unc)_f([\d.]+)_s(\d)", m)
    return (r.group(1), float(r.group(2)), int(r.group(3))) if r else (None, None, None)


def load():
    s = pd.read_csv(ROOT / "r3_scores_summary.csv")
    e = pd.read_csv(EXP2 / "scores_summary.csv")
    e = e[e.model.isin(["cnn_a", "esrt_a"])]
    # exp 2 summary has no "all pilot holdouts" rows for every file version; rebuild them from its tiles if absent
    return pd.concat([s, e[[c for c in s.columns if c in e.columns]]], ignore_index=True)


def table(d, cols, digits=None):
    digits = digits or {}
    d = d.copy()
    d["o"] = d.model.map({k: i for i, k in enumerate(ORDER)}).fillna(99)
    d = d.sort_values("o")
    h = "<div class='tw'><table><tr><th>Model</th>" + "".join(f"<th>{LBL[c]}</th>" for c in cols) + "</tr>"
    for _, r in d.iterrows():
        h += f"<tr><td>{NAMES.get(r.model, r.model)}</td>" + "".join(
            f"<td class='n'>{r[c]:.{digits.get(c, 3)}f}</td>" if pd.notna(r[c]) else "<td class='n'>–</td>" for c in cols) + "</tr>"
    return h + "</table></div>"


def section(c):
    if not (ROOT / "r3_scores_summary.csv").exists():
        return ""
    s = load()
    lc = s[s.model.str.startswith("sm_")].copy()
    lc[["init", "frac", "seed"]] = lc.model.apply(lambda m: pd.Series(parse(m)))
    lcm = lc.groupby(["group", "init", "frac"])[MET].mean().reset_index()
    refs = s[s.model.isin(["bicubic", "cnn_a", "esrt_a", "zeroshot_x2_unc"])]
    c["r3_json"] = json.dumps(dict(curve=lcm.to_dict("records"), ref=refs[["group", "model"] + MET].to_dict("records")), default=float)
    import r3_prose
    return r3_prose.render(c, s, lcm, table)


SCRIPT = r"""
(function(){
  const el = document.getElementById('r3curve'); if(!el || typeof R3==='undefined') return;
  const mets = [['ssim_lee','SSIM vs Lee-filtered reference (x100)'],['detail_r_raw','detail_r'],['chamfer_p2t_m_raw','Edges output→reference (m, lower better)']];
  const groups = ['all pilot holdouts','paper test scenes'];
  const W = Math.min(el.clientWidth || 900, 980);
  const w = Math.max(260, Math.floor((W - 20) / 3)), h = 220, m = {l:46, r:10, t:26, b:34};
  const RC = {bicubic:['var(--muted)','bicubic'], cnn_a:['var(--a3)','exp 2 CNN'], esrt_a:['var(--good)','exp 2 ESRT'], zeroshot_x2_unc:['var(--warn)','X-band zero-shot']};
  groups.forEach(gp=>{
    const row = d3.select(el).append('div').style('display','flex').style('flex-wrap','wrap').style('gap','8px');
    mets.forEach(([k,label])=>{
      const data = R3.curve.filter(d=>d.group===gp);
      const refs = R3.ref.filter(d=>d.group===gp);
      const vals = data.map(d=>d[k]).concat(refs.map(d=>d[k])).filter(v=>v!=null && !isNaN(v));
      const x = d3.scaleLog().domain(d3.extent(data, d=>d.frac)).range([m.l, w-m.r]);
      const y = d3.scaleLinear().domain(d3.extent(vals)).nice().range([h-m.b, m.t]);
      const svg = row.append('svg').attr('width',w).attr('height',h).attr('role','img').attr('aria-label', `${label}, ${gp}`);
      svg.append('text').attr('x',m.l).attr('y',14).attr('fill','var(--fg)').style('font-size','12px').style('font-weight',600).text(`${label}: ${gp}`);
      svg.append('g').attr('transform',`translate(0,${h-m.b})`).call(d3.axisBottom(x).tickValues([...new Set(data.map(d=>d.frac))]).tickFormat(d3.format('.0%')))
        .call(g=>g.selectAll('text').attr('fill','var(--muted)')).call(g=>g.selectAll('line,path').attr('stroke','var(--line)'));
      svg.append('g').attr('transform',`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5))
        .call(g=>g.selectAll('text').attr('fill','var(--muted)')).call(g=>g.selectAll('line,path').attr('stroke','var(--line)'));
      refs.forEach(r=>{ if(!(r.model in RC) || r[k]==null) return; const [col,nm]=RC[r.model];
        svg.append('line').attr('x1',m.l).attr('x2',w-m.r).attr('y1',y(r[k])).attr('y2',y(r[k])).attr('stroke',col).attr('stroke-dasharray','4 3');
        svg.append('text').attr('x',w-m.r).attr('y',y(r[k])-3).attr('text-anchor','end').attr('fill',col).style('font-size','10px').text(nm); });
      [['scratch','var(--a2)','from scratch'],['x2_unc','var(--accent)','X-band pretrained']].forEach(([init,col,nm])=>{
        const s = data.filter(d=>d.init===init).sort((a,b)=>a.frac-b.frac);
        svg.append('path').datum(s).attr('fill','none').attr('stroke',col).attr('stroke-width',2).attr('d', d3.line().x(d=>x(d.frac)).y(d=>y(d[k])));
        svg.selectAll(null).data(s).join('circle').attr('cx',d=>x(d.frac)).attr('cy',d=>y(d[k])).attr('r',3.5).attr('fill',col)
          .append('title').text(d=>`${nm}, ${(d.frac*100).toFixed(0)} %: ${d[k].toFixed(3)}`);
      });
    });
  });
  d3.select(el).append('p').style('color','var(--muted)').style('font-size','0.85rem')
    .html('<span style="color:var(--accent)">&#9679;</span> X-band pretrained, then fine-tuned on Stripmap &nbsp; <span style="color:var(--a2)">&#9679;</span> Stripmap only, from scratch. 3000 steps each; x axis = share of exp 2\'s 200 training tiles (log scale). Dashed lines: references.');
})();
"""

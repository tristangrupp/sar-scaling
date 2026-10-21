"""Build the project site: one folder of linked, self-contained HTML pages.

  site/index.html      overview, experiment log, status, links
  site/overlap.html    X-band x Sentinel-1 Stripmap overlap atlas
  site/spike.html      experiment 1: S1 IW -> X-band GAN spike
  site/ceiling.html    experiment 1b: real X-band vs real X-band ceiling check
  site/exp2.html       experiment 2: S1 IW -> Stripmap, X-band as geometry check

Every page carries the same top navigation. Pages work offline (libraries and images inlined);
only Google Fonts need a connection. Rerun this script after any stage finishes.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import markdown
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
SPIKE = ROOT / "exp1_gan_spike"
EXP0 = ROOT / "exp0_overlap_atlas"          # overlap atlas, plus d3 / topojson / land-110m used by every map
EXP2 = ROOT / "exp2_iw_to_stripmap"
CEIL = SPIKE / "ceiling"

PAGES = [("index.html", "Overview"), ("related.html", "Related work"), ("overlap.html", "Overlap atlas"), ("spike.html", "Exp 1: GAN spike"),
         ("ceiling.html", "Exp 1b: ceiling"), ("exp2.html", "Exp 2: IW → Stripmap"), ("exp3.html", "Exp 3: other sensors"), ("exp4.html", "Exp 4: despeckling"), ("exp5.html", "Exp 5: cross-band transfer")]

FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700'
         '&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">')

CSS = """
:root { --bg:#f3f5f7; --panel:#fff; --fg:#18212b; --muted:#5b6875; --line:#d5dbe1; --land:#dde3e8;
  --accent:#1f6fd1; --a2:#c2410c; --a3:#0f8a6a; --good:#1d7a3a; --warn:#a15c00; --bad:#b42318;
  --f-display:"Archivo","Helvetica Neue",Arial,sans-serif; --f-body:"Source Sans 3","Segoe UI",system-ui,sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,Consolas,monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#0f151b; --panel:#161e26; --fg:#e3e9ef;
  --muted:#93a1af; --line:#2a3541; --land:#26313c; --accent:#6aa8ff; --a2:#ff8a4c; --a3:#3fd0a5; --good:#5fd27d;
  --warn:#f0a640; --bad:#ff6b5e; color-scheme:dark; } }
:root[data-theme="dark"] { --bg:#0f151b; --panel:#161e26; --fg:#e3e9ef; --muted:#93a1af; --line:#2a3541; --land:#26313c;
  --accent:#6aa8ff; --a2:#ff8a4c; --a3:#3fd0a5; --good:#5fd27d; --warn:#f0a640; --bad:#ff6b5e; color-scheme:dark; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 var(--f-body); }
.sitenav { position:sticky; top:0; z-index:20; background:var(--panel); border-bottom:1px solid var(--line); }
.sitenav .in { max-width:1200px; margin:0 auto; padding:8px 16px; display:flex; flex-wrap:wrap; gap:4px 18px; align-items:center; }
.sitenav b { font:700 0.95rem var(--f-display); margin-right:8px; }
.sitenav a { color:var(--muted); text-decoration:none; font-size:0.9rem; padding:4px 0; border-bottom:2px solid transparent; }
.sitenav a:hover { color:var(--fg); } .sitenav a.on { color:var(--fg); border-bottom-color:var(--accent); }
.wrap { max-width:1200px; margin:0 auto; padding:24px 16px 56px; display:grid; gap:22px; }
h1 { font:700 1.7rem/1.2 var(--f-display); margin:0; text-wrap:balance; }
h2 { font:600 0.78rem/1.2 var(--f-display); text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:0 0 10px; }
h3 { font:700 1.05rem var(--f-display); margin:14px 0 6px; }
p, li { max-width:78ch; } .lede { color:var(--muted); margin:6px 0 0; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:14px 16px; min-width:0; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:16px; }
.tw { overflow-x:auto; } table { border-collapse:collapse; width:100%; font-size:0.9rem; }
th,td { padding:6px 10px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }
th { font:600 0.74rem var(--f-display); text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
td.n, th.n { text-align:right; font-family:var(--f-mono); font-variant-numeric:tabular-nums; }
code { font-family:var(--f-mono); font-size:0.85em; word-break:break-all; }
.pill { display:inline-block; padding:1px 9px; border-radius:999px; font-size:0.78rem; border:1px solid currentColor; }
.pill.done { color:var(--good); } .pill.run { color:var(--accent); } .pill.blocked { color:var(--warn); } .pill.neg { color:var(--bad); }
.exp { display:grid; grid-template-columns:auto 1fr; gap:4px 16px; align-items:start; }
.exp .k { font:700 1rem var(--f-display); } .exp a { color:var(--accent); }
figure { margin:0; } figure img { width:100%; height:auto; border-radius:4px; background:#fff; }
figcaption { font-size:0.85rem; color:var(--muted); margin-top:4px; }
svg text { fill:var(--muted); font:11px var(--f-body); }
.md h1 { display:none; } .md h2 { font:700 1.1rem var(--f-display); text-transform:none; letter-spacing:0; color:var(--fg); margin:20px 0 6px; }
.md ul { padding-left:20px; } .md li { margin-bottom:4px; }
a { color:var(--accent); }
"""


def nav(active):
    links = "".join(f'<a href="{f}" class="{"on" if f == active else ""}">{t}</a>' for f, t in PAGES)
    return f'<nav class="sitenav"><div class="in"><b>SAR super-resolution</b>{links}</div></nav>'


def page(fname, title, body, scripts=""):
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>{FONTS}'
            f'<style>{CSS}</style></head><body>{nav(fname)}<main class="wrap">{body}</main>{scripts}</body></html>\n')


def inject_nav(src_html, fname):
    """Add the site nav and its CSS to a page built elsewhere."""
    navcss = "<style>" + "\n".join(l for l in CSS.splitlines() if l.startswith((".sitenav", ":root", "@media", "  ")))
    navcss += "</style>"
    nav_css_only = ("<style>.sitenav{position:sticky;top:0;z-index:20;background:var(--panel,#fff);border-bottom:1px solid var(--line,#d5dbe1)}"
                    ".sitenav .in{max-width:1200px;margin:0 auto;padding:8px 16px;display:flex;flex-wrap:wrap;gap:4px 18px;align-items:center;font-family:'Source Sans 3','Segoe UI',system-ui,sans-serif}"
                    ".sitenav b{font:700 .95rem 'Archivo',Arial,sans-serif;margin-right:8px;color:var(--fg,#18212b)}"
                    ".sitenav a{color:var(--muted,#5b6875);text-decoration:none;font-size:.9rem;padding:4px 0;border-bottom:2px solid transparent}"
                    ".sitenav a:hover{color:var(--fg,#18212b)}.sitenav a.on{color:var(--fg,#18212b);border-bottom-color:#1f6fd1}</style>")
    html = src_html.replace("</head>", nav_css_only + "</head>", 1)
    return re.sub(r"<body([^>]*)>", lambda m: f"<body{m.group(1)}>{nav(fname)}", html, count=1)


def img64(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def lib(name):
    return (EXP0 / name).read_text(encoding="utf-8")


# ------------------------------------------------------------------ status
def status():
    s = {}
    spike = SPIKE / "eval" / "summary.csv"
    s["spike"] = pd.read_csv(spike).set_index("model") if spike.exists() else None
    s["ceiling"] = pd.read_csv(CEIL / "summary.csv") if (CEIL / "summary.csv").exists() else None
    s["pilot_start"] = (EXP2 / "pilot_start.log").read_text().strip() if (EXP2 / "pilot_start.log").exists() else ""
    fl = EXP2 / "data" / "fetch_log.csv"
    # the fetch log also holds the Houston test download; count it only once the pilot has started
    s["fetch"] = pd.read_csv(fl) if fl.exists() and "starting pilot" in s["pilot_start"] else None
    s["n_pilot"] = len(pd.read_csv(EXP2 / "pilot_scenes.csv")) if (EXP2 / "pilot_scenes.csv").exists() else 0
    s["houston_tile"] = (EXP2 / "sm_tiles" / "test_houston.tif").exists()
    return s


# ------------------------------------------------------------------ index
def frag_pill(slug):
    f = ROOT / slug / "status.txt"
    state = f.read_text().strip() if f.exists() else "running"
    cls = {"done": "done", "negative": "neg", "blocked": "blocked"}.get(state, "run")
    return f'<span class="pill {cls}">{state}</span>'


def build_index(st):
    sp = st["spike"]
    spike_line = (f"Detail match {sp.loc['l1', 'detail_r']:.2f} (near random); fine-detail energy "
                  f"{sp.loc['l1', 'hf_ratio']:.0%} of the real X-band. The models denoise; they do not sharpen."
                  if sp is not None else "Pending.")
    ceil = st["ceiling"]
    ceil_line = "Running: scoring pairs of real X-band images of the same place against each other."
    ceil_pill = '<span class="pill run">running</span>'
    if ceil is not None:
        ceil_line = "Done. See the ceiling page for how high the detail metric can go between two real X-band images."
        ceil_pill = '<span class="pill done">done</span>'
    exp2_line = ("Arm A (the published ESRT recipe, reproduced) beats bicubic on every held-out group, mostly by removing "
                 "speckle. Arm B, which adds an X-band edge loss, scores slightly worse on held-out Stripmap (SSIM −0.21, "
                 "p &lt; 0.001), so X-band stays a validation reference only. Adding 25 inland scenes and a change mask "
                 "(round 3) barely moves held-out scores; gains over bicubic come from despeckling.")
    exp2_pill = '<span class="pill neg">negative for X-band</span>'
    body = f"""
<header><h1>Can open SAR sharpen Sentinel-1?</h1>
<p class="lede">A running log of experiments on super-resolving Sentinel-1 radar using open X-band satellite data
(Capella, Umbra, ICEYE) and Sentinel-1's own Stripmap mode. Each page below is self-contained and rebuilt from the
project's data with <code>build_site.py</code>.</p></header>

<section class="card"><h2>Experiment log</h2><div class="exp">
 <span class="k">0</span><div><a href="overlap.html">Overlap atlas</a> <span class="pill done">done</span>
   <p>Where open X-band collects overlap Sentinel-1 Stripmap scenes. Umbra: 1,069 usable pairs at 12 sites;
   ICEYE: 211 at 7; Capella: 110 at 4. Capella is mostly HH, so it rarely matches Sentinel-1 IW over land.</p></div>
 <span class="k">1</span><div><a href="spike.html">GAN spike: Sentinel-1 IW → X-band at 2.5 m</a> <span class="pill neg">negative</span>
   <p>{spike_line}</p></div>
 <span class="k">1b</span><div><a href="ceiling.html">Ceiling check: real X-band vs real X-band</a> {ceil_pill}
   <p>{ceil_line}</p></div>
 <span class="k">2</span><div><a href="exp2.html">Sentinel-1 IW → Stripmap at 5 m, X-band for geometry only</a> {exp2_pill}
   <p>{exp2_line}</p></div>
 <span class="k">3</span><div><a href="exp3.html">Is there a better open sensor than Sentinel-1?</a> {frag_pill("exp3_sensor_survey")}
   <p>NISAR L-band is the best open training source: same-instant pairs at two bandwidths. A tuned model beats
   bicubic on all five far-holdout metrics; an X-band edge loss again adds nothing.</p></div>
 <span class="k">4</span><div><a href="exp4.html">Despeckling: cleaner, more interpretable Sentinel-1</a> {frag_pill("exp4_despeckling")}
   <p>A VV+VH Noise2Noise CNN trained on 24 sites, with no clean target, is the best single-date despeckler
   (about 0.7 dB above Lee) and lifts land-cover accuracy from 0.770 to 0.806. It still trails Lee in textured
   desert and does not help field boundaries at 10 m.</p></div>
 <span class="k">5</span><div><a href="exp5.html">Cross-band transfer: does an X-band-trained model work on C-band?</a> {frag_pill("exp5_cross_band")}
   <p>An X-band-only model beats bicubic on real Stripmap zero-shot, and pretraining helps short L1 runs. But matched to
   the published recipe (round 4: ESRT and CNN, SSIM loss, 8,000 steps, 3 seeds), neither X-band nor X + L pretraining
   beats training from scratch beyond seed noise: SSIM 41.6–41.8 for every variant, 0 of 30 tests pass.</p></div>
</div></section>

<section class="card"><h2>Where things stand</h2>
<p>X-band fails as a direct training target: as a brightness target (experiment 1) and as an edge loss
(experiments 2 and 3) it adds nothing or makes held-out Stripmap worse. Real Stripmap sits no closer to X-band edges
than bicubic IW does, so at 5 m X-band edges are a weak reference for C-band detail. As pretraining, X-band transfers
real knowledge (zero-shot it beats bicubic, and it speeds up short training), but with the full published recipe and
200 real pairs it does not improve the final Stripmap model (experiment 5, round 4). So the open question this project
set out to test, whether Capella-style X-band beats the published IW to Stripmap baseline, comes back no at this data
size. What works is same-sensor supervision (IW to Stripmap, NISAR bandwidth pairs) and despeckling, where learned
models beat classic filters and help a downstream map.</p></section>

<section class="grid">
 <div class="card"><h2>What changed after experiment 1</h2>
 <p>Experiment 1 asked a 10 m (about 20 m true resolution) C-band input to produce 2.5 m X-band. That is an 8x
 resolution jump across two wavelengths, scored against images taken from different look directions. The models
 learned to remove speckle and blur; detail they added did not match real detail.</p>
 <p>Experiment 2 keeps the band fixed: Sentinel-1 IW to Sentinel-1 Stripmap, about 4x in true resolution. X-band
 only checks where edges sit, never how bright things are.</p></div>
 <div class="card"><h2>Files</h2><ul>
 <li>Project folder: <code>E:\\SAR_superresolution</code> (see <code>README.md</code>)</li>
 <li>One folder per experiment: <code>exp0_overlap_atlas</code>, <code>exp1_gan_spike</code>,
   <code>exp2_iw_to_stripmap</code>, <code>exp3_sensor_survey</code>, <code>exp4_despeckling</code>,
   <code>exp5_cross_band</code></li>
 <li>Large data sits in each experiment's <code>data\\</code> subfolder (Stripmap SLC: <code>exp2_iw_to_stripmap\\data</code>)</li>
 <li>Papers and contacts: <code>related_work\\</code>, <code>CONTACTS.md</code></li>
 <li>Rebuild this site: <code>python build_site.py</code> (ftwtools env)</li></ul></div>
</section>"""
    (SITE / "index.html").write_text(page("index.html", "SAR Super-resolution Log", body), encoding="utf-8")


# ------------------------------------------------------------------ overlap + spike
def build_overlap():
    src = EXP0 / "sar_overlap_atlas.html"
    if src.exists():
        (SITE / "overlap.html").write_text(inject_nav(src.read_text(encoding="utf-8"), "overlap.html"), encoding="utf-8")


def build_spike():
    src = SPIKE / "gan_spike_report.html"
    if src.exists():
        html = src.read_text(encoding="utf-8")
        extra = ('<section class="card" style="border-left:3px solid #c2410c"><h2>Was the design fair?</h2>'
                 '<p>Not fully. The target was physically very hard (8x in true resolution, across bands), and the setup '
                 'made success hard to see: X-band look direction and incidence were not matched or given to the model, '
                 'tiles were aligned with a single 10 m shift, and no ceiling was measured for the detail metric. See the '
                 '<a href="ceiling.html">ceiling check</a> for how high two real X-band images score against each other.</p></section>')
        html = html.replace('<section class="grid">', extra + '<section class="grid">', 1)
        (SITE / "spike.html").write_text(inject_nav(html, "spike.html"), encoding="utf-8")


# ------------------------------------------------------------------ ceiling
def build_ceiling(st):
    c = st["ceiling"]
    if c is None:
        body = ('<header><h1>Ceiling check</h1><p class="lede">Scores pairs of real X-band images of the same place '
                'against each other with the spike\'s metrics. If two real images only agree weakly, the spike\'s 0.05 '
                'detail match could never have looked good.</p></header>'
                '<section class="card"><h2>Status</h2><p>Running. This page fills in when the check finishes.</p></section>')
    else:
        pngs = sorted((CEIL / "png").glob("*.png"))[:11]
        figs = "".join(f'<figure class="card"><img src="{img64(p)}" alt="{p.stem}"><figcaption>{p.stem}</figcaption></figure>'
                       for p in pngs)
        keep = {"group (medians)": "Group", "n": "Pairs", "detail_r": "Detail match", "hf_ratio": "Detail energy",
                "ssim": "SSIM", "deg_detail_r": "Detail match, B blurred to 10 m", "self1px_detail_r": "Self, 2.5 m shift",
                "self2px_detail_r": "Self, 5 m shift", "raw_off_m": "Raw offset (m)"}
        cc = c[[k for k in keep if k in c.columns]].rename(columns=keep)
        cc["Pairs"] = cc["Pairs"].astype(int)
        tbl = cc.to_html(index=False, float_format=lambda v: f"{v:.2f}", border=0)
        notes = (CEIL / "notes.html").read_text(encoding="utf-8") if (CEIL / "notes.html").exists() else ""
        spike = st["spike"]
        ref = (f"<p>Spike models for reference: detail match {spike.loc['l1', 'detail_r']:.3f}, fine-detail energy "
               f"{spike.loc['l1', 'hf_ratio']:.3f}, SSIM {spike.loc['l1', 'ssim']:.3f}.</p>" if spike is not None else "")
        body = (f'<header><h1>Ceiling check</h1><p class="lede">Real X-band vs real X-band, same place, scored with the '
                f'spike\'s metrics after per-tile normalization.</p></header>'
                f'<section class="card"><h2>Summary</h2><div class="tw">{tbl}</div>{ref}{notes}</section>'
                f'<section><h2>Example pairs</h2><div class="grid" style="grid-template-columns:1fr">{figs}</div></section>')
    (SITE / "ceiling.html").write_text(page("ceiling.html", "X-band Ceiling Check", body), encoding="utf-8")


# ------------------------------------------------------------------ exp2
def build_exp2(st):
    design = markdown.markdown((EXP2 / "DESIGN.md").read_text(encoding="utf-8"), extensions=["tables"])
    cells = pd.read_csv(EXP2 / "cells.csv")
    pilot = pd.read_csv(EXP2 / "pilot_scenes.csv") if (EXP2 / "pilot_scenes.csv").exists() else pd.DataFrame()
    land = json.loads((EXP0 / "land-110m.json").read_text())
    roles = pilot.groupby("cell").role.first().to_dict() if len(pilot) else {}
    names = pilot.groupby("cell").region.first().to_dict() if len(pilot) else {}
    cdata = [{"cell": r.cell, "lon": r.lon, "lat": r.lat, "n_sm": int(r.n_sm),
              "pairs": None if pd.isna(r.n_pair_iw6_samedir) else int(r.n_pair_iw6_samedir),
              "km": round(r.km_to_xband), "role": roles.get(r.cell), "name": names.get(r.cell)}
             for r in cells.itertuples()]
    top = cells.dropna(subset=["n_pair_iw6_samedir"]).copy()
    top["name"] = top.cell.map(names)
    top["role"] = top.cell.map(roles)
    top = top[top.name.notna()].sort_values("n_pair_iw6_samedir", ascending=False)
    bars = [{"name": r.name, "role": r.role, "v": int(r.n_pair_iw6_samedir), "km": round(r.km_to_xband)}
            for r in top.itertuples()]

    fetch = st["fetch"]
    if fetch is not None:
        fetch_ok = fetch[fetch.status == "ok"].granule
        pilot["fetched"] = pilot.granule.isin(fetch_ok)
    else:
        pilot["fetched"] = False
    ptbl = "".join(f"<tr><td>{r.region}</td><td>{r.role}</td><td class='n'>{r.start[:10]}</td><td>{r.beam}</td>"
                   f"<td>{r.orbit_dir.lower()}</td><td class='n'>{r.iw_dt_days}</td>"
                   f"<td>{'✓' if r.fetched else ''}</td></tr>" for r in pilot.itertuples())

    houston = ""
    q = EXP2 / "sm_tiles" / "test_houston.png"
    if q.exists():
        houston = (f'<figure class="card"><h2>Processing test: Houston Stripmap SLC, terrain-corrected to 5 m</h2>'
                   f'<img src="{img64(q)}" alt="Houston Stripmap test tile"><figcaption>{st["pilot_start"]}</figcaption></figure>')

    steps = [("Design written", "done"), ("Stripmap inventory and region ranking", "done"),
             ("61 pilot scenes picked (same-direction IW within 6 days)", "done"),
             ("SLC downloader (VV only, ~2.3 GB per scene)", "done"),
             ("SLC → 5 m gamma0 processing (sarsen)", "done" if st["houston_tile"] else "run"),
             ("Pilot downloads to E:", "run" if fetch is not None else "wait"),
             ("Pair tiles: Stripmap 5 m ↔ IW 10 m, change and layover masks", "wait"),
             ("Arm A: IW → Stripmap", "wait"), ("Stage 1: X-band geometry validation", "wait"),
             ("Arm B: + X-band edge loss", "wait"), ("Score both arms, near vs far holdout", "wait"),
             ("Transfer panels: neighbors of held-out tiles and unseen landscapes", "wait")]
    prog = EXP2 / "progress.json"          # written by the experiment; overrides the list above
    if prog.exists():
        steps = list(json.loads(prog.read_text(encoding="utf-8")).items())
    results = (EXP2 / "results_body.html").read_text(encoding="utf-8") if (EXP2 / "results_body.html").exists() else ""
    lab = {"done": ("done", "done"), "run": ("run", "running"), "wait": ("blocked", "waiting")}
    steplist = "".join(f'<li>{t} <span class="pill {lab[s][0]}">{lab[s][1]}</span></li>' for t, s in steps)

    body = f"""
<header><h1>Experiment 2: Sentinel-1 IW to Stripmap</h1>
<p class="lede">Train IW (about 20 m true resolution) to match Stripmap (about 5 m) from the same sensor, then test
whether X-band edges add real detail. Design, region inventory, pilot scene list and progress.</p></header>

<section class="grid">
 <div class="card"><h2>Progress</h2><ol style="margin:0;padding-left:20px">{steplist}</ol></div>
 <div class="card"><h2>Pilot regions: Stripmap scenes with same-direction IW within 6 days</h2>
 <svg id="bars" viewBox="0 0 420 300" width="100%" role="img" aria-label="Bar chart of pairs per region"></svg>
 <p class="lede" style="font-size:.85rem">Color shows role: training, held out with local X-band, held out near, held out far.
 Distance is to the nearest open X-band collect.</p></div>
</section>

<section class="card"><h2>Worldwide Stripmap VV since 2023 (1° cells)</h2>
<svg id="map" viewBox="0 0 960 470" width="100%" role="img" aria-label="World map of Stripmap inventory cells"></svg>
<p class="lede" style="font-size:.85rem">Dot area ∝ number of Stripmap scenes. Ringed cells are the pilot regions. Many busy
cells in the Gulf of Guinea are open ocean and were skipped.</p></section>

{houston}

{results}

<section class="card"><h2>Pilot scenes ({len(pilot)})</h2><div class="tw"><table>
<thead><tr><th>Region</th><th>Role</th><th class="n">Date</th><th>Beam</th><th>Orbit</th><th class="n">IW gap (d)</th><th>Fetched</th></tr></thead>
<tbody>{ptbl}</tbody></table></div></section>

<section class="card md"><h2 style="font:600 .78rem var(--f-display);text-transform:uppercase;letter-spacing:.08em;color:var(--muted)">Design</h2>{design}</section>
"""
    js = f"""
<script>{lib('d3.min.js')}</script><script>{lib('topojson.min.js')}</script>
<script>
const CELLS={json.dumps(cdata)}, BARS={json.dumps(bars)}, LAND={json.dumps(land)};
const COL={{train:'var(--accent)',holdout_xband:'var(--a2)',holdout_near:'var(--a3)',holdout_far:'var(--warn)'}};
(function(){{
 const svg=d3.select('#bars'), m={{l:92,r:56,t:6,b:20}}, W=420, H=300;
 const x=d3.scaleLinear().domain([0,d3.max(BARS,d=>d.v)]).nice().range([m.l,W-m.r]);
 const y=d3.scaleBand().domain(BARS.map(d=>d.name)).range([m.t,H-m.b]).padding(.25);
 svg.append('g').selectAll('line').data(x.ticks(4)).join('line').attr('x1',x).attr('x2',x).attr('y1',m.t).attr('y2',H-m.b).attr('stroke','var(--line)');
 svg.append('g').selectAll('text').data(x.ticks(4)).join('text').attr('x',x).attr('y',H-4).attr('text-anchor','middle').text(d=>d);
 svg.append('g').selectAll('rect').data(BARS).join('rect').attr('x',m.l).attr('y',d=>y(d.name)).attr('height',y.bandwidth())
   .attr('width',d=>x(d.v)-m.l).attr('fill',d=>COL[d.role]).attr('rx',2);
 svg.append('g').selectAll('text').data(BARS).join('text').attr('x',m.l-6).attr('y',d=>y(d.name)+y.bandwidth()/2+4).attr('text-anchor','end').text(d=>d.name);
 svg.append('g').selectAll('text').data(BARS).join('text').attr('x',d=>x(d.v)+4).attr('y',d=>y(d.name)+y.bandwidth()/2+4).text(d=>d.v+' · '+d.km+' km');
}})();
(function(){{
 const svg=d3.select('#map'), proj=d3.geoEqualEarth().fitExtent([[6,6],[954,464]],{{type:'Sphere'}}), path=d3.geoPath(proj);
 svg.append('path').attr('d',path(d3.geoGraticule10())).attr('fill','none').attr('stroke','var(--line)').attr('stroke-width',.5);
 svg.append('path').attr('d',path(topojson.feature(LAND,LAND.objects.land))).attr('fill','var(--land)');
 const r=d3.scaleSqrt().domain([1,d3.max(CELLS,d=>d.n_sm)]).range([1.5,10]);
 const c=[...CELLS].sort((a,b)=>(a.role?1:0)-(b.role?1:0));
 svg.append('g').selectAll('circle').data(c).join('circle').attr('transform',d=>`translate(${{proj([d.lon,d.lat])}})`)
   .attr('r',d=>r(d.n_sm)).attr('fill',d=>d.role?COL[d.role]:'var(--muted)').attr('fill-opacity',d=>d.role?.85:.35)
   .attr('stroke',d=>d.role?'var(--fg)':'none').attr('stroke-width',1.2)
   .append('title').text(d=>`${{d.name||d.cell}}: ${{d.n_sm}} Stripmap scenes`+(d.pairs!=null?`, ${{d.pairs}} same-direction IW pairs`:'')+`, ${{d.km}} km to X-band`);
}})();
</script>"""
    (SITE / "exp2.html").write_text(page("exp2.html", "IW to Stripmap Experiment", body, js), encoding="utf-8")


def build_exp3():
    """Exp 3 page is written by exp3_sensor_survey/build_page.py (body HTML fragment); wrap it in the site shell."""
    frag = ROOT / "exp3_sensor_survey" / "page_body.html"
    body = frag.read_text(encoding="utf-8") if frag.exists() else (
        "<header><h1>Experiment 3: is there a better open sensor?</h1><p class='lede'>Survey of open, global SAR "
        "sensors as alternatives to Sentinel-1 for super-resolution, then the experiment 2 design adapted to the best "
        "one.</p></header><section class='card'><h2>Status</h2><p>Running. This page fills in as the work lands.</p></section>")
    scripts = (ROOT / "exp3_sensor_survey" / "page_scripts.html").read_text(encoding="utf-8") if (ROOT / "exp3_sensor_survey" / "page_scripts.html").exists() else ""
    (SITE / "exp3.html").write_text(page("exp3.html", "Open SAR Sensor Survey", body, scripts), encoding="utf-8")


def build_fragment_page(slug, folder, title, placeholder):
    """Pages written by an experiment folder as body/scripts fragments, wrapped in the site shell."""
    frag, scr = ROOT / folder / "page_body.html", ROOT / folder / "page_scripts.html"
    body = frag.read_text(encoding="utf-8") if frag.exists() else placeholder
    scripts = scr.read_text(encoding="utf-8") if scr.exists() else ""
    (SITE / f"{slug}.html").write_text(page(f"{slug}.html", title, body, scripts), encoding="utf-8")


def build_related():
    notes = markdown.markdown((ROOT / "related_work" / "NOTES.md").read_text(encoding="utf-8"))
    contacts = markdown.markdown((ROOT / "CONTACTS.md").read_text(encoding="utf-8"))
    body = ("<header><h1>Related work and contacts</h1><p class='lede'>What has been published on Sentinel-1 and "
            "Capella super-resolution, the gap this project tests, and people at Capella to talk to. Papers and "
            "extracted text are in the <code>related_work</code> folder; contacts in <code>CONTACTS.md</code>.</p></header>"
            f"<section class='card md'>{notes}</section><section class='card md'>{contacts}</section>")
    (SITE / "related.html").write_text(page("related.html", "Related Work and Contacts", body), encoding="utf-8")


def main():
    SITE.mkdir(exist_ok=True)
    st = status()
    build_index(st)
    build_related()
    build_overlap()
    build_spike()
    build_ceiling(st)
    build_exp2(st)
    build_exp3()
    build_fragment_page("exp4", "exp4_despeckling", "Sentinel-1 Despeckling", "<header><h1>Experiment 4: cleaner, more interpretable Sentinel-1</h1><p class='lede'>Can a CNN or GAN remove speckle from "
        "Sentinel-1 while keeping real structure, and does the cleaner image help a real mapping task?</p></header>"
        "<section class='card'><h2>Status</h2><p>Running. This page fills in as the work lands.</p></section>")
    build_fragment_page("exp5", "exp5_cross_band", "Cross-band Radar Transfer",
        "<header><h1>Experiment 5: does radar knowledge transfer across bands?</h1><p class='lede'>Train sharpening and "
        "despeckling on X-band alone, apply it to Sentinel-1 C-band unchanged, and compare with a C-band-only model.</p>"
        "</header><section class='card'><h2>Status</h2><p>Running. This page fills in as the work lands.</p></section>")
    print("site:", SITE / "index.html")


if __name__ == "__main__":
    main()

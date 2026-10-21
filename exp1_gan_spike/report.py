"""Write gan_spike_report.html: metrics, loss curves and side-by-side panels, all embedded."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
E = ROOT / "eval"

summary = pd.read_csv(E / "summary.csv")
by_region = pd.read_csv(E / "by_region.csv")
by_provider = pd.read_csv(E / "by_provider.csv") if (E / "by_provider.csv").exists() else None
xpol = pd.read_csv(E / "xpol_summary.csv") if (E / "xpol_summary.csv").exists() else None


def pngs(folder):
    return [(f.stem, base64.b64encode(f.read_bytes()).decode()) for f in sorted((E / folder).glob("*.png"))]         if (E / folder).exists() else []


neighbors, contexts = pngs("neighbors"), pngs("contexts")
man = pd.read_csv(ROOT / "manifest.csv")
ok = man[man.status == "ok"]
used = ok[ok.corr10 >= 0.3]
logs = {s: pd.read_csv(ROOT / f"ckpt/{s}_log.csv").to_dict("list") for s in ["l1", "gan"]
        if (ROOT / f"ckpt/{s}_log.csv").exists()}
sm = json.loads((E / "sm_check.json").read_text()) if (E / "sm_check.json").exists() else None
panels = [{**p, "img": base64.b64encode((E / "panels" / f"{p['id']}.png").read_bytes()).decode()}
          for p in json.loads((E / "panels.json").read_text())]

NAMES = {"bicubic": "Bicubic", "l1": "CNN, pixel loss", "gan": "CNN + GAN"}


def table(df, cols):
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join("<tr>" + "".join(
        f"<td class='n'>{r[c]:.3f}</td>" if isinstance(r[c], float) else f"<td>{NAMES.get(r[c], r[c])}</td>"
        for c in cols) + "</tr>" for _, r in df.iterrows())
    return f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


data = {
    "counts": {"selected": int(len(man)), "ok": int(len(ok)), "used_train": int((used.split == "train").sum()),
               "used_test": int((used.split == "test").sum()),
               "no_s1": int((man.status == "no_s1").sum()), "error": int((man.status == "error").sum())},
    "logs": logs,
}

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentinel-1 to X-band Spike</title>
<style>
:root {{ --bg:#f4f6f8; --panel:#fff; --fg:#17202a; --muted:#5a6774; --line:#d6dce2; --l1:#1f6fd1; --gan:#c2410c; --good:#1d7a3a;
  --f:"Segoe UI",system-ui,sans-serif; --m:ui-monospace,Consolas,monospace; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0f151b; --panel:#161e26; --fg:#e3e9ef; --muted:#93a1af; --line:#2a3541;
  --l1:#6aa8ff; --gan:#ff8a4c; --good:#5fd27d; color-scheme:dark; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 var(--f); }}
.wrap {{ max-width:1200px; margin:0 auto; padding:28px 16px 60px; display:grid; gap:22px; }}
h1 {{ margin:0; font-size:1.6rem; }} h2 {{ margin:0 0 8px; font-size:0.8rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }}
p {{ max-width:75ch; margin:0 0 8px; }} .muted {{ color:var(--muted); }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:14px 16px; min-width:0; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }}
.tw {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; font-size:.92rem; }}
th,td {{ padding:6px 10px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
th {{ font-size:.75rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }}
td.n {{ text-align:right; font-family:var(--m); font-variant-numeric:tabular-nums; }}
dl {{ display:grid; grid-template-columns:auto auto; gap:2px 16px; margin:0; justify-content:start; }} dt {{ color:var(--muted); }} dd {{ margin:0; font-family:var(--m); }}
.panel img {{ width:100%; height:auto; background:#fff; border-radius:4px; }}
.panel figcaption {{ font-size:.85rem; color:var(--muted); margin-top:4px; }}
svg text {{ fill:var(--muted); font-size:11px; font-family:var(--f); }}
ul {{ margin:0; padding-left:20px; max-width:80ch; }} li {{ margin-bottom:5px; }}
</style></head><body><div class="wrap">
<header><h1>Can a model sharpen Sentinel-1 into X-band?</h1>
<p class="muted">Throwaway feasibility test, {pd.Timestamp.now():%Y-%m-%d}. Input: Sentinel-1 IW VV, terrain-corrected, 10 m pixels (about 20 m true resolution).
Target: Umbra and ICEYE VV spotlight, averaged to 2.5 m. Test sites (Houston, Newark/NY, Tokyo, Antwerp, Rotterdam) were never seen in training.</p></header>

<section class="card"><h2>Verdict</h2><div id="verdict">VERDICT_PLACEHOLDER</div></section>

<section class="grid">
 <div class="card"><h2>Scores on held-out tiles (median)</h2>
 {table(summary, ["model", "mae", "ssim", "detail_r", "hf_ratio"])}
 <p class="muted" style="margin-top:8px">mae: error in per-tile standardized dB (lower is better). ssim: structural similarity (higher is better).
 detail_r: does the added fine detail land where the real detail is (0 = random, 1 = perfect). hf_ratio: fine-detail energy versus the real X-band (1 = as sharp as truth).</p></div>
 <div class="card"><h2>Data</h2><dl>
  <dt>Collects selected</dt><dd>{data['counts']['selected']}</dd>
  <dt>Tiles built</dt><dd>{data['counts']['ok']}</dd>
  <dt>No S1 within 30 days</dt><dd>{data['counts']['no_s1']}</dd>
  <dt>Download errors (404)</dt><dd>{data['counts']['error']}</dd>
  <dt>Train tiles used (r ≥ 0.3)</dt><dd>{data['counts']['used_train']}</dd>
  <dt>Test tiles used (r ≥ 0.3)</dt><dd>{data['counts']['used_test']}</dd></dl>
  <p class="muted" style="margin-top:8px">Tiles were kept only where X-band averaged to 10 m correlates with Sentinel-1 at r ≥ 0.3. The rest were mostly sea (wind flips the contrast between dates) or hills (different look angles shift features).</p>
  <h2 style="margin-top:12px">Training loss (L1)</h2><svg id="loss" viewBox="0 0 320 120" width="100%"></svg></div>
</section>

<section class="card"><h2>By test region (median)</h2>{table(by_region, ["region", "model", "mae", "ssim", "detail_r"])}</section>

SM_PLACEHOLDER

EXTRA_PLACEHOLDER

<section class="card"><h2>How to read the panels</h2>
<p>Each row is one 640 m × 640 m crop from a held-out test tile, a place the models never saw in training. All five frames
share one gray scale, set from the real X-band's 2nd–98th percentile, so brightness compares directly across the row.</p>
<ol>
<li><b>Sentinel-1 IW (10 m)</b>: the model input, Sentinel-1 VV backscatter. Each 10 m pixel is drawn as a 4 × 4 block of 2.5 m pixels, so the frames line up.</li>
<li><b>Bicubic</b>: the same input smoothly interpolated to 2.5 m. No learning; it is the floor any model has to beat.</li>
<li><b>CNN, pixel loss</b>: output of model 1. A residual convolutional network (EDSR-style: 16 residual blocks at 10 m, then two 2x upsampling steps) trained only to minimize the per-pixel difference (L1) from the real X-band. Pixel loss rewards the average of all plausible answers, so an unsure model blurs.</li>
<li><b>CNN + GAN</b>: output of model 2, the only GAN output. The same network, started from model 1 and trained further with a discriminator that tries to tell its output from real X-band; the generator is rewarded for fooling it. That push is meant to add realistic texture. Here the discriminator won from the start and the generator ignored it, so this frame looks the same as model 1.</li>
<li><b>Real X-band (2.5 m)</b>: the answer. Umbra or ICEYE spotlight imagery averaged from about 0.5 m to 2.5 m. For test tiles it was used only for scoring.</li>
</ol>
<p>Earlier versions of this page called the network a U-Net. It is a residual CNN without U-Net skip connections.</p></section>

<section><h2>Side by side, one 640 m crop per test tile</h2><div class="grid" style="grid-template-columns:1fr">
{"".join(f'<figure class="panel card" style="margin:0"><img src="data:image/png;base64,{p["img"]}" alt="panel {p["id"]}"><figcaption>{p["region"]} · {p.get("split", "test")} · {p["id"]}</figcaption></figure>' for p in panels)}
</div></section>

<section class="card"><h2>Caveats</h2><ul>
<li>Small data: a few hundred 3.2 km tiles; test metrics rest on {data['counts']['used_test']} tiles in 4–5 cities.</li>
<li>Per-tile normalization: each tile's S1 and X-band are standardized on their own, so the model never has to learn absolute calibration. Umbra and ICEYE amplitudes are on different scales, so this was necessary, but it means the outputs are relative brightness, not calibrated backscatter.</li>
<li>Different dates (median 2 days apart) and very different viewing geometry: real changes and look-angle effects count as error.</li>
<li>Co-registration is a single shift per tile at 10 m; residual misalignment of a few X-band pixels penalizes every sharp model.</li>
</ul></section>
</div>
<script>
const L = {json.dumps(logs)};
const svg = document.getElementById('loss'); const W=320,H=120,p=24;
const all = [...(L.l1?L.l1.l1:[]), ...(L.gan?L.gan.l1:[])];
if (all.length) {{
  const ymin=Math.min(...all), ymax=Math.max(...all), n=all.length;
  const X=i=>p+(W-p-6)*i/(n-1), Y=v=>H-18-(H-28)*(v-ymin)/(ymax-ymin||1);
  const nl=(L.l1?L.l1.l1.length:0);
  const path=(arr,off)=>arr.map((v,i)=>(i?'L':'M')+X(i+off).toFixed(1)+','+Y(v).toFixed(1)).join('');
  let s='';
  if (L.l1) s+=`<path d="${{path(L.l1.l1,0)}}" fill="none" stroke="var(--l1)" stroke-width="1.5"/>`;
  if (L.gan) s+=`<path d="${{path(L.gan.l1,nl)}}" fill="none" stroke="var(--gan)" stroke-width="1.5"/>`;
  s+=`<text x="${{p}}" y="10">${{ymax.toFixed(2)}}</text><text x="${{p}}" y="${{H-4}}">${{ymin.toFixed(2)}}</text>`;
  s+=`<text x="${{W-150}}" y="10" style="fill:var(--l1)">pixel-loss stage</text><text x="${{W-60}}" y="10" style="fill:var(--gan)">GAN stage</text>`;
  svg.innerHTML=s;
}}
</script></body></html>"""

if sm:
    rows = "".join(f"<tr><td>{k}</td><td class='n'>{v:.3f}</td></tr>" for k, v in sm["detail_r_vs_sm"].items())
    sm_html = (f"<section class='card'><h2>Stripmap honesty check ({sm['site']})</h2>"
               f"<p>Each image blurred to 5 m and compared with real Sentinel-1 Stripmap SLC (same C-band, ~5 m). "
               f"Correlation of the 5 m detail layer: detail that matches Stripmap is real; detail that does not is invented.</p>"
               f"<div class='tw'><table><thead><tr><th>Image</th><th>detail r vs SM</th></tr></thead><tbody>{rows}</tbody></table></div>"
               f"<p class='muted'>{sm.get('note', '')}</p></section>")
else:
    sm_html = ("<section class='card'><h2>Stripmap honesty check</h2><p>Not run: ASF refused the Stripmap SLC download "
               "until the Earthdata account approves the ASF data agreement.</p></section>")
html = html.replace("SM_PLACEHOLDER", sm_html)

extra = ""
if by_provider is not None:
    extra += (f"<section class='card'><h2>By X-band provider (held-out tiles, median)</h2>"
              f"{table(by_provider, ['provider', 'model', 'mae', 'ssim', 'detail_r', 'hf_ratio'])}"
              f"<p class='muted'>Capella open data is mostly HH, and Sentinel-1 IW over land is VV, so only Capella's 197 VV "
              f"collects can be co-polarized training pairs. Capella VV test tiles come from 1-degree cells held out "
              f"entirely; none of them fall in the original five test cities.</p></section>")
if xpol is not None and len(xpol):
    extra += (f"<section class='card'><h2>Capella HH in the test cities (cross-pol, median)</h2>"
              f"{table(xpol, ['model', 'mae', 'ssim', 'detail_r', 'hf_ratio'])}"
              f"<p class='muted'>Capella HH collects over Houston, Newark/NY, Tokyo, Antwerp and Rotterdam, scored against the "
              f"VV-trained models. Polarization differs between input and truth, so these are scored apart from the main "
              f"test set; panels marked test_xpol below.</p></section>")
if neighbors or contexts:
    fig = lambda items: "".join(f'<figure class="panel card" style="margin:0"><img src="data:image/png;base64,{b}" alt="{n}">'
                                f'<figcaption>{n.replace("_", " ")}</figcaption></figure>' for n, b in items)
    extra += (f"<section class='card'><h2>Beyond the footprints: did the models learn anything that transfers?</h2>"
              f"<p>The models were run on Sentinel-1 around each held-out tile (9.6 km, the X-band tile outlined in orange) "
              f"and on landscapes the training set barely covers: farmland, forest, mountains, desert and villages. No "
              f"X-band truth exists in these areas, so this is a visual check only.</p>"
              f"<p>What transfers is a speckle filter. Field edges, roads, canals and building blocks visible at 10 m come "
              f"through cleaner and smoother, everywhere. No structure finer than the 10 m input appears, and contrast is "
              f"compressed toward gray because the models learned to hedge. The GAN output again matches the pixel-loss "
              f"output.</p></section>"
              f"<section><h2>Neighbors of held-out tiles</h2><div class='grid' style='grid-template-columns:1fr'>{fig(neighbors)}</div></section>"
              f"<section><h2>Other landscapes</h2><div class='grid' style='grid-template-columns:1fr'>{fig(contexts)}</div></section>")
html = html.replace("EXTRA_PLACEHOLDER", extra)
verdict = (ROOT / "verdict.html").read_text(encoding="utf-8") if (ROOT / "verdict.html").exists() else "<p>Pending.</p>"
html = html.replace("VERDICT_PLACEHOLDER", verdict)
(ROOT / "gan_spike_report.html").write_text(html, encoding="utf-8")
print("wrote", ROOT / "gan_spike_report.html")

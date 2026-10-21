"""Prose for the Experiment 5 page. body(ctx) returns the HTML body fragment; SCRIPT is the d3 chart code."""
from __future__ import annotations

import json

import numpy as np

V = {}          # verdict text, filled by RESULTS below once numbers are in


def g(agg, inp, grid, model, col):
    d = agg[(agg.input == inp) & (agg.grid == grid) & (agg.model == model)]
    return float(d[col].iloc[0]) if len(d) else float("nan")


def body(c):
    T, agg, cal = c["T"], c["agg"], c["cal"]
    f = lambda inp, grid, m, col, d=3: f"{g(agg, inp, grid, m, col):.{d}f}"
    gap = json.loads(open(c["gap_json"]).read()) if "gap_json" in c else None
    h = []
    h.append(f"""<header><h1>Experiment 5: does radar knowledge learned on X-band transfer to Sentinel-1 C-band?</h1>
<p class="lede">{c['lede']}</p></header>""")

    h.append("""<section class="card"><h2>Hypothesis</h2>
<p>A super-resolution model for radar learns two kinds of knowledge. The first kind is geometric: where edges run,
how building outlines look, how layover pushes tall objects toward the sensor and shadow trails behind them, and how
speckle behaves statistically. Radar geometry depends on the imaging geometry (incidence angle and look direction),
not on wavelength, and fully developed speckle follows the same multiplicative statistics at every band. That kind
should transfer. The second kind is radiometric: how bright each surface appears. Brightness depends on wavelength
through surface roughness relative to the wavelength, penetration into vegetation and dry soil, and the size of the
scatterers that dominate a resolution cell. X-band (3 cm) sees fine roughness and leaf-scale scatterers; C-band
(5.6 cm) sees through more of the canopy and finds the same field smoother. That kind should not transfer.</p>
<div class="grid">
<div><h3>Should transfer</h3><ul>
<li>Edge and outline geometry of buildings, roads, shorelines, field boundaries</li>
<li>Layover and shadow patterns, given the look direction and incidence angle</li>
<li>Speckle statistics: gamma-distributed intensity, correlation set by the point-spread function</li>
<li>Point-target shapes (corner reflectors, masts, ships)</li></ul></div>
<div><h3>Should not transfer</h3><ul>
<li>Absolute and relative brightness of vegetation, soil, water under wind</li>
<li>Penetration: X-band stops in the canopy top, C-band reaches branches and sometimes ground</li>
<li>Roughness response: a field rough at 3 cm can be smooth at 5.6 cm</li>
<li>Texture density: how many bright scatterers populate a 20 m cell</li></ul></div></div>
<p>The test: train a model only on X-band, degraded to look like Sentinel-1, then run it on real Sentinel-1 with no
retraining. If the geometry priors transfer, it should place edges better than bicubic interpolation on real
Sentinel-1, even if its brightness is wrong.</p></section>""")

    h.append(f"""<section class="card"><h2>Design</h2>
<h3>Training data: X-band only</h3>
<p>{c['n_tiles_txt']} The X-band tiles are the spike's 3.2 km tiles at 2.5 m (Umbra, ICEYE and Capella, Capella in both
HH and VV), plus collects from the ceiling check that sit in the training area. The spike's held-out regions (Houston,
Newark-New York, Tokyo, Antwerp, Rotterdam and twelve Capella test cells) stay held out: no exp 5 model sees any of
them, and extra collects within 0.3 degrees of any test tile were dropped.</p>
<h3>Degradation to Sentinel-1 IW</h3>
<p>Each training batch builds a fresh Sentinel-1-like input on the GPU from the X-band power image <code>I</code>:
for each of K looks, draw complex white Gaussian noise <code>n</code>, form <code>h * (sqrt(I) n)</code> with an
anisotropic Gaussian amplitude point-spread function <code>h</code> (intensity full width at half maximum 20 m in
range and 22 m in azimuth, the IW product's true resolution), take the squared magnitude, average the K looks, then
average 4 x 4 blocks to 10 m and convert to dB. The range axis points along the X-band collect's own look direction,
since the layover in the X-band image comes from that geometry. This produces speckle with the right distribution
and the right spatial correlation in one step, because speckle and blur come from the same convolution. The 5 m
target is the 2 x 2 power average of the 2.5 m X-band.</p>
<p>We calibrated K against real Sentinel-1 rather than taking the nominal ENL of 4.4. On Experiment 4's 14
single-orbit stacks, we measured single-date ENL with Experiment 4's definition (9 x 9 windows whose temporal-mean
coefficient of variation is in the lowest 5 %). Real Sentinel-1 RTC at 10 m measures ENL {cal['real_enl_median']:.1f}
(range {cal['real_enl_range'][0]:.1f} to {cal['real_enl_range'][1]:.1f}). The 10 m resampling raises it above
the nominal 4.4. Synthetic inputs with K = 5 and 6 bracket that value, and the lag-1 autocorrelation of the speckle
matches as well ({cal['real_lag1_median']:.2f} real against 0.74 synthetic), which checks the point-spread function
width independently. Training draws K uniformly from 4 to 7 to cover the spread between sites.</p>
{T['calib']}
<h3>Models</h3>
<p>All models use the spike's residual CNN (<code>Gen</code>: a 16-block residual trunk at 10 m, pixel-shuffle
upsampling, bicubic skip), generalized to 4x (10 to 2.5 m) and 2x (10 to 5 m). The conditioned versions add FiLM
layers: a small network maps a 7-number metadata vector (look azimuth as cosine and sine, incidence angle,
polarization VV or HH, band X or C) to a per-channel scale and shift in every residual block. Rotation and flip
augmentation rotates the look azimuth with the image. Umbra azimuths come from the STAC <code>view:azimuth</code>;
ICEYE and most Capella azimuths come from orbit direction and look side. All models train with a masked L1 loss for
8,000 steps (batch 16, mixed precision; the GPU was shared with three other experiments). We did not train a GAN
version: the spike's discriminator overpowered its generator, and the L1 models already answer the transfer
question.</p>
<div class="tw"><table><tr><th>Model</th><th>Trained on</th><th>Scale</th><th>Metadata</th><th>Role</th></tr>
<tr><td>X-synthetic 4x, conditioned</td><td>X-band, synthetic S1-like inputs</td><td class="n">4x</td><td>yes</td><td>main transfer test</td></tr>
<tr><td>X-synthetic 4x, unconditioned</td><td>same</td><td class="n">4x</td><td>no</td><td>does metadata matter</td></tr>
<tr><td>X-synthetic 4x, band-augmented</td><td>same, plus random spectral and contrast perturbation of the input</td><td class="n">4x</td><td>yes</td><td>band-agnostic training</td></tr>
<tr><td>X-synthetic 2x, conditioned / unconditioned</td><td>same, 5 m target</td><td class="n">2x</td><td>yes / no</td><td>Stripmap-scale target</td></tr>
<tr><td>Exp 1 CNN</td><td>real S1 IW paired with real X-band ({c['exp1_which']})</td><td class="n">4x</td><td>no</td><td>supervised cross-band baseline</td></tr>
<tr><td>C-band-only 4x / 2x</td><td>Experiment 4 temporal-mean S1 VV at 10 m, 8 training sites, degraded the same way in pixel units</td><td class="n">4x / 2x</td><td>no</td><td>same-band control</td></tr>
<tr><td>Bicubic</td><td>nothing</td><td class="n">4x / 2x</td><td>–</td><td>floor</td></tr></table></div>
<p>The C-band-only control had to fall back. Experiment 2's Stripmap tiles were not ready (one Houston test tile
exists, see below), so we used Experiment 4's temporal means: 20 to 27 dates averaged, close to speckle-free, at 10 m.
They were degraded with the same pipeline in pixel units (a 10 m pixel stands in for 2.5 m, so the point-spread
function becomes 80 x 88 m). That is a scale-shift assumption, and a weak control: it learns C-band texture at 40 m
and applies it at 10 m.</p>
<h3>Evaluation</h3>
<ul>
<li><b>Synthetic held-out X-band</b> (the reverse sanity check): held-out tiles degraded with a fixed seed and K = 6. The X-trained models should work here.</li>
<li><b>Real Sentinel-1 against real X-band</b> on the same held-out tiles. The spike co-registered each pair at 10 m; we
then fine-register at 2.5 m (phase correlation against the bicubic upsample, bounded at 10 m). Brightness differs
by band, so the geometry metrics carry the transfer question.</li>
<li><b>Ceiling and chance</b>: a second real X-band collect of the same place (the ceiling check's pairs in held-out
regions) shows how well two real X-band images agree. Bicubic of an unrelated tile's Sentinel-1 shows the chance level.</li>
<li><b>Despeckling</b> on Experiment 4's six held-out sites, against each site's temporal mean, with Experiment 4's ENL definition.</li>
<li><b>Transfer panels</b> on 9.6 km Sentinel-1 windows around held-out sites and on unseen landscapes.</li>
<li><b>Stripmap</b> (same sensor, 5 m) wherever Experiment 2 has tiles.</li></ul>
<p>Metrics follow the spike: MAE, SSIM, <code>detail_r</code> (correlation of the fine-detail layer, the image minus
its 10 m block mean) and <code>hf_ratio</code> (fine-detail energy over the truth's). Because the ceiling check showed
<code>detail_r</code> collapses with 2.5 m misregistration, we lean on a shift-tolerant edge metric: Canny edges with
quantile thresholds, so every image gets the same edge density, then the mean distance from each output edge to
the nearest truth edge (invented edges raise it), the reverse (missed edges raise it), their mean (chamfer), and an
edge F-score within 5, 10 or 20 m. We compute edges at two scales: a 5 m Gaussian, which includes texture, and a 10 m
Gaussian, which keeps outlines.</p></section>""")

    h.append(c["results_html"])
    return "\n".join(h)


SCRIPT = r"""
(function(){
  const el = document.getElementById('gapchart'); if(!el) return;
  const metrics = [['edgeF_10m','Edge F within 10 m (5 m edge scale)'],['detail_r','detail_r']];
  const keys = ['bicubic4','exp1_cnn','x4_cond','x4_unc','x4_aug','c4'];
  const names = {}; GAP.forEach(d=>names[d.key]=d.model);
  const W = Math.min(el.clientWidth || 900, 980), rowH = 26, padL = 230, padR = 20, padT = 34;
  metrics.forEach(([m,label])=>{
    const data = keys.map(k=>{
      const s = GAP.find(d=>d.key===k && d.input==='synthetic' && d.grid==='2.5m');
      const r = GAP.find(d=>d.key===k && d.input==='real_s1' && d.grid==='2.5m');
      return s && r ? {k, name:names[k], syn:s[m], real:r[m]} : null}).filter(Boolean);
    const H = padT + rowH*data.length + 30;
    const svg = d3.select(el).append('svg').attr('width',W).attr('height',H).attr('role','img')
      .attr('aria-label', label + ': synthetic versus real Sentinel-1 input, per model');
    const vals = data.flatMap(d=>[d.syn,d.real]);
    const x = d3.scaleLinear().domain([Math.min(0,d3.min(vals)), d3.max(vals)*1.08]).range([padL, W-padR]);
    svg.append('text').attr('x',padL).attr('y',16).attr('fill','var(--fg)').style('font-weight',600).text(label);
    svg.append('g').attr('transform',`translate(0,${H-24})`).call(d3.axisBottom(x).ticks(6))
      .call(g=>g.selectAll('text').attr('fill','var(--muted)')).call(g=>g.selectAll('line,path').attr('stroke','var(--line)'));
    const g = svg.selectAll('g.r').data(data).join('g').attr('transform',(d,i)=>`translate(0,${padT+i*rowH})`);
    g.append('text').attr('x',padL-8).attr('y',4).attr('text-anchor','end').attr('fill','var(--fg)').style('font-size','12px').text(d=>d.name);
    g.append('line').attr('x1',d=>x(d.syn)).attr('x2',d=>x(d.real)).attr('stroke','var(--line)').attr('stroke-width',3);
    g.append('circle').attr('cx',d=>x(d.syn)).attr('r',6).attr('fill','var(--accent)').append('title').text(d=>`synthetic: ${d.syn.toFixed(3)}`);
    g.append('circle').attr('cx',d=>x(d.real)).attr('r',6).attr('fill','var(--a2)').append('title').text(d=>`real S1: ${d.real.toFixed(3)}`);
  });
  const lg = d3.select(el).append('p').style('color','var(--muted)').style('font-size','0.85rem');
  lg.html('<span style="color:var(--accent)">&#9679;</span> synthetic Sentinel-1-like input made from held-out X-band &nbsp; '+
          '<span style="color:var(--a2)">&#9679;</span> real Sentinel-1 input, same tiles. Truth is the real X-band at 2.5 m in both cases.');
})();
"""

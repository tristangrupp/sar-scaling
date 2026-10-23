"""Prose for the Experiment 4 page. TEXT(ctx) returns HTML sections; ctx is build_page.main's locals()."""
from __future__ import annotations


def TEXT(ctx):
    L = ctx.get("oracle_enl_ratio", float("nan"))
    S = {}
    S["header"] = """<header><h1>Experiment 4: Sentinel-1 despeckling</h1>
<p class="lede">We tested whether learned filters can reduce speckle in a single Sentinel-1 image while preserving structure and improving mapping. Standard filters provide the baseline. Evaluation uses six excluded sites and temporal means from 22–27 same-orbit dates, with changing pixels masked. The later round adds VH and trains Noise2Noise models on 24 sites.</p></header>"""

    S["goal"] = """<section class="card"><h2>Purpose</h2>
<p>Speckle obscures features in a single radar image. Spatial averaging reduces it but blurs edges; temporal averaging requires repeated observations and a stable scene. This experiment tests filters that need only one image at inference.</p>
<p>Experiment 1 produced smoother images, but visual smoothness did not establish that it preserved useful structure. Here we measure that tradeoff against same-sensor references and compare the transferred model with purpose-trained despecklers.</p></section>"""

    S["design"] = """<section class="card"><h2>Round 1: design</h2>
<h3>Sites and reference</h3>
<p>Fourteen sites span cropland, forest, urban, mountain, desert and wetland on five continents. Eight training
sites cover 10.24 km each; six held-out sites cover 5.12 km each, one per land cover, and no model or setting ever
saw them. For each site we read every Sentinel-1 RTC VV date (Planetary Computer <code>sentinel-1-rtc</code>, 10 m)
from one relative orbit onto a UTM grid through <code>WarpedVRT</code>. We started from a season of 2025 and widened
the window by two months each side until the orbit held at least 20 dates. Outside Europe only Sentinel-1A flies, at
12-day repeat, so most windows span 9 to 13 months. For Louisiana we forced orbit 63, because orbit 165 clips the
site at its swath edge.</p>
<p>The reference for a date <i>t</i> is the <b>leave-one-out temporal mean</b>: the mean in linear power of every
other date, so the input never enters its own target. We score only <b>stable</b> pixels. A pixel counts as changed
when its temporal coefficient of variation exceeds the 99th percentile that pure speckle would give. We get that
percentile by Monte Carlo from N independent gamma(L) draws, with the site's own date count N and single-date ENL L.
We measure L in the 5 % of 9x9 windows whose temporal mean is most uniform. Cropland keeps only 25 to 37 % of its
pixels, because crops grow and fields get tilled; forest, desert and wetland keep 85 to 97 %.</p>
{SITES}
<h3>Methods, all on one single date</h3>
<ul>
<li><b>Raw</b> single date.</li>
<li><b>Boxcar</b> 3x3 and 5x5 mean in linear power.</li>
<li><b>Lee</b> (Lee 1980), 7x7: <code>k = clip(var_x/v, 0, 1)</code>, <code>var_x = (v - m²Cu²)/(1 + Cu²)</code>,
<code>out = m + k(I - m)</code>, with <code>Cu = 1/√L</code>.</li>
<li><b>Refined Lee</b> (Lee 1981), 7x7: the same estimator on one of eight edge-aligned half-windows, chosen from the
3x3 sub-means (strongest of four directional gradients, then the side closer to the centre).</li>
<li><b>Gamma-MAP</b> (Lopes et al. 1990), 7x7: <code>out = m</code> when <code>Ci ≤ Cu</code>; <code>out = I</code> when
<code>Ci ≥ √2·Cu</code>; otherwise <code>out = (b·m + √(m²b² + 4αL·I·m)) / 2α</code>, with <code>α = (1 + Cu²)/(Ci² - Cu²)</code>
and <code>b = α - L - 1</code>.</li>
<li><b>Experiment 1 CNN and CNN+GAN</b>, in two versions: the v1 weights (trained without Capella) and the weights
retrained with Capella data, which finished while this experiment ran. They output X-band-like 2.5 m images in standardized units. We map them back to dB with the input's own
median and standard deviation, then block-average 4x4 in linear power to 10 m.</li>
<li><b>New CNN, supervised</b>: Experiment 1's residual trunk (16 blocks, 64 channels) at 1x, predicting a
correction added to the input. It maps a single date to its leave-one-out temporal mean, with L1 loss in standardized
dB on stable pixels only. Training uses 96 px patches, batch 16, 8,000 steps, Adam 2e-4 with cosine decay, and
rotations and flips.</li>
<li><b>New CNN, Noise2Noise</b>: same network, date A to a random date B of the same orbit, L1 in dB, no clean
target and no change mask. L1 in dB targets the median of log-speckle, which sits below the log of the mean, so we
add one constant offset fitted on training sites.</li>
<li><b>New CNN + GAN</b>: the supervised network fine-tuned for 3,000 steps with a conditional PatchGAN. In
Experiment 1 the discriminator overpowered the generator (its loss fell to 0.001), so here it gets spectral norm on
every layer, a relativistic-average loss, a learning rate one quarter of the generator's, and adversarial weight
0.005.</li>
</ul>
<p>The classic filters all use one ENL, L = 6.7: the median single-date ENL of the training sites. Bilinear
resampling onto the UTM grid raises it above the nominal 4.4 of IW GRD.</p>
<h3>Metrics</h3>
<p>We score every date of every held-out site (148 site-dates), take the median over dates within a site, then the
median over the six sites. Against the reference, in dB on stable pixels: PSNR (data range = the reference's
0.5 to 99.5 percentile span), SSIM (Gaussian, σ = 1.5), MAE and bias. Without a reference: ENL in homogeneous areas
(the lowest-CV 5 % of 9x9 windows of the temporal mean), and mean preservation, the ratio of output to input mean in
linear power. For structure: the <b>ratio image</b> input/output, which should be pure speckle. Its ENL should match
the oracle ratio input/reference. A lower value means the filter removed real structure; a higher value means it left
speckle in. We also correlate the 5x5-smoothed absolute dB ratio with the reference's edge strength; this should be
near zero. Edge preservation: the correlation of Sobel gradient magnitude with the reference's, and EPD-ROA against the
reference at its strongest 10 % of edges (1 = edges kept, below 1 = blurred, above 1 = noisy).</p>
<h3>Downstream task</h3>
<p>Land cover from ESA WorldCover 2021 (10 m, Planetary Computer <code>esa-worldcover</code>) on the six held-out
sites. One random forest (200 trees, balanced class weights, fixed seed), pooled over the sites, refitted for each
input. Each site is cut into 64 px blocks in a checkerboard: the forest trains on one colour and is tested on the
other, with 8 px dropped along every block border. Every input uses the same pixels. We run two feature sets: the dB
value alone (how readable one pixel is), and a texture set (value, 3, 7 and 15 px means, 7 px standard deviation,
gradient). We repeat this on three dates per site. A second task compares Canny edges with the boundaries of a 5x5
majority-filtered WorldCover. Each input marks the same share of pixels as edges, and we score F1 with a 1 px
tolerance.</p></section>"""

    S["results"] = """<section class="card"><h2>Results on the held-out sites</h2>
<p>Medians over the six held-out sites; each site value is itself the median over its 22 to 27 dates. The oracle row
holds the ideal values: the temporal mean's own ENL and the ratio of input to reference.</p>
{TABLE}
<p style="color:var(--muted);font-size:0.9em">Dots: one per held-out site. Bars: median across sites. Green dashed
line: ideal value. Raw has no ratio ENL, since its ratio image is constant.</p>
{CHARTS}
<h3>PSNR by site (dB)</h3>
{PERSITE}
{FINDINGS}
</section>"""
    S["panels"] = """<section class="card"><h2>Panels</h2>{FIGS}</section>"""
    S["downstream"] = """<section class="card"><h2>Downstream: land cover and boundaries</h2>
<p>Classes, each at least 0.5 % of held-out pixels: {CLASSES}. The table gives means over three dates.</p>
{DTABLE}
{DFINDINGS}
</section>"""
    S["gan"] = """<section class="card"><h2>The GAN fine-tune</h2>{GAN}</section>"""
    S["caveats"] = """<section class="card"><h2>Caveats</h2>{CAVEATS}</section>"""
    S["verdict"] = """<section class="card"><h2>Round 1 conclusion</h2>{VERDICT}</section>"""
    for k, v in FILL.items():
        for s in S:
            S[s] = S[s].replace("{" + k + "}", v)
    return S


from findings import FILL  # noqa: E402  (results prose, written after the runs)

"""Round 2 prose. Numbers from metrics_r2.csv, downstream_r2_wc.csv, downstream_r2_ftw.csv, ckpt/r2_*.json."""

FILL_R2 = {
"R2_INTRO": """<p>Round 2 tests speckle-level conditioning, a larger Noise2Noise training set and VV+VH inputs on the same six test sites. It then evaluates WorldCover classification and field-boundary prediction with Fields of The World (FTW) labels.</p>
<h3>What changed</h3>
<ul>
<li><b>Speckle-level conditioning.</b> Each training input is the mean of k = 1 to 4 random dates, so the network also
sees cleaner inputs. One extra constant channel per polarization carries log(ENL̂) of the input. ENL̂ is the 90th
percentile of local 7x7 ENL on a stride-7 grid; it needs no reference, and test images get the same estimator. On the
round-1 stacks it rises about 3x from one to four dates over farmland, forest and wetland. It saturates where texture
dominates (Atacama 10.7 to 14, Madrid 4.8 to 10.7).</li>
<li><b>Noise2Noise on 24 sites.</b> We added 16 sites with only 5 to 12 dates each and VV+VH: Namib, Sonoran, Gobi and
Thar deserts; Tokyo, Cairo, São Paulo and Lagos; Okavango, Everglades and Sundarbans wetlands; Andes, Norway and
Taiwan mountains; Ukraine and Kansas cropland. Pampas had no RTC items and Borneo only one valid date. None lies near
a held-out site, and none needs a stable stack. The per-polarization log-bias offset (+0.34 to +0.52 dB) is still
fitted on the round-1 training sites.</li>
<li><b>VH.</b> We downloaded VH for every date of the 14 round-1 sites and built VH temporal means and change masks
the same way as for VV. The dual-pol models take VV and VH (each standardized on its own) and output both.</li>
<li>Five models, all with Experiment 1's residual trunk. Supervised + conditioning, VV (6,000 steps). Noise2Noise on 24
sites, VV, unconditioned (6,000 steps). Noise2Noise on 24 sites + conditioning, VV (4,000 steps). Supervised +
conditioning, VV+VH (4,000 steps). Noise2Noise on 24 sites + conditioning, VV+VH (4,000 steps). We cut the last three
to 4,000 steps because four experiments shared the GPU; round 1's loss had flattened by about 3,500 steps.</li>
</ul>""",

"R2_FIX": """<p><b>Conditioning fixes the Atacama failure, but only for the VV-only supervised model.</b> The conditioned
supervised CNN scores 20.9 dB in the Atacama, level with Lee (20.9) and up from 18.6 in round 1; its bias falls from
+0.67 to +0.32 dB, and its gradient correlation there (0.57) is the best of any method. The cost is less smoothing
elsewhere. Its ENL in flat areas falls from 100 to 42, its ratio ENL rises to 13.8 (more speckle left), and its median
PSNR slips from 21.5 to 21.3 dB, mostly at Madrid. The other CNNs still fail in the Atacama: 19.3 to 19.7 dB for the
Noise2Noise models and 18.5 dB for the dual-pol supervised model, whose VH there is biased by +1.25 dB. Conditioning
moves the bias in the right direction, but a scalar speckle estimate cannot tell texture from speckle. On a surface
that is both textured and less speckled than the training sites, the networks still smooth it and brighten it.</p>
<p>The VV+VH Noise2Noise model trained on 24 sites gives the highest VV scores in this test: PSNR 21.85 dB, SSIM 0.54, MAE 0.90 dB and gradient correlation 0.59, with +0.06 dB bias. The supervised VV+VH model reaches 21.78 dB. Both exceed Lee by about 0.7 dB and refined Lee by about 1.2 dB. The roughly 0.1 dB difference between the top CNNs is too small to establish a ranking without repeated seeds.</p>""",

"R2_K": """<p><b>Without conditioning, CNNs make already-clean inputs worse.</b> When the input is the mean of 2 or 4
dates, the round-1 CNN and the unconditioned Noise2Noise over-brighten by +0.2 to +0.4 dB. At k = 4 they score below
Lee and even below the untouched input (22.6 to 22.9 dB vs 22.8 raw and 24.3 Lee). The conditioned models stay
unbiased (bias +0.03 to +0.06 dB at k = 4 for the supervised and dual-pol Noise2Noise models) and match or beat Lee
(24.6 to 24.7 dB). Conditioning on Noise2Noise with VV alone fixes only part of this (bias +0.18 to +0.23 dB). In the
Atacama at k = 4, the conditioned supervised models score 25.5 dB, against 22.9 for the round-1 CNN.</p>""",

"R2_VH": """<p><b>VH behaves like VV.</b> Single-date VH has the same speckle (ENL about 7). Lee lifts it from 17.1 to
20.9 dB and refined Lee to 19.7. The dual-pol CNNs reach 21.0 to 21.2 dB, with the best SSIM (0.54 to 0.56) and gradient
correlation (0.59 to 0.60). They lose in the Atacama (17.2 to 18.9 dB vs 21.9 for Lee), where desert VH sits near the
noise floor and the networks brighten it by +0.6 to +1.3 dB. VH's real value shows downstream.</p>""",

"R2_WC": """<p><b>VH matters more than the despeckler.</b> Adding VH texture features lifts every input by about 12
points of overall accuracy: raw from 0.649 to 0.770, the temporal mean from 0.733 to 0.830. With VV+VH, the dual-pol
Noise2Noise CNN reaches 0.806 (macro F1 0.60) and the dual-pol supervised CNN 0.799, against 0.770 raw, 0.775 refined Lee
and 0.779 Lee. Classic filters add about one point; the CNNs add three, closing about 60 % of the gap between a raw
date and a full stack. The spread across three dates is about ±0.01 with VV+VH, so this ordering holds. On VV alone, the
round-2 models (0.67 to 0.69) do not beat round 1's Noise2Noise (0.70); conditioning makes them smooth less, and
pooled texture features reward smoothing.</p>""",

"R2_FTW": """<p>Despeckling did not improve field-boundary detection in this test. We used 20 FTW test chips: four each from France, Spain, the Netherlands, South Africa and Vietnam. Chips were regridded to 10 m; a cell counted as boundary when FTW boundary labels covered at least a quarter of it. Each chip had a 24-date VV+VH stack from its label year (2018–2022). Training used two chips per country and testing used the other two, then the folds were swapped.</p>
<p>Boundary-class F1 is 0.08 to 0.12 for every single-date input and 0.11 (VV) or 0.13 (VV+VH) for the 24-date
temporal mean. Field F1 is 0.62 to 0.67. Despeckling does not move either beyond the spread across dates (±0.02). FTW
boundaries are one to two pixels wide at 10 m, and many fields in Spain and Vietnam are a few pixels across. C-band
backscatter often differs little between neighbouring fields of the same crop. The Canny test is not informative:
in Vietnam boundaries cover 15 to 40 % of the labelled area, so edge F1 sits near chance, and the raw image scores
highest (0.46) because its noise edges are spread evenly.</p>
<p>The earlier boundary F1 of 0.99 was invalid: a fold-construction bug put the same chips in training and test. The corrected script creates the folds explicitly and asserts that they do not overlap. The results above use the corrected split.</p>""",

"R2_CAVEATS": """<h3>Round 2 caveats</h3>
<ul>
<li>One training run per model, with 4,000 to 6,000 steps on a GPU shared with three other experiments. Differences
of 0.1 to 0.3 dB between CNNs are not firm.</li>
<li>The Noise2Noise bias offset still comes from round-1 stacks. A deployment with no stacks at all would need another
calibration, for example the digamma value for the estimated ENL.</li>
<li>The FTW chips are only 1.0 to 1.5 km across and come from five countries. Their labels date from 2018 to 2022
and need not match the field layout on every Sentinel-1 date of that year.</li>
<li>The ENL estimate is one number per image, so it cannot separate texture from speckle inside a scene. A per-pixel
noise map, or training on more textured deserts with clean targets, is the next thing to try for the Atacama
case.</li>
</ul>""",

"R2_VERDICT": """<h3>Round 2 conclusion</h3>
<p><span class="pill done">Yes, with VH</span> The best single-date despeckler is now a dual-pol Noise2Noise CNN.
It trains on date pairs from 24 sites with no clean target, and is conditioned on the speckle level. It beats refined
Lee by about 1.2 dB PSNR, matches or beats the supervised CNN, stays unbiased on inputs that are already multi-looked,
and lifts land-cover accuracy from 0.779 (Lee, VV+VH) to 0.806. A 24-date stack reaches 0.830.</p>
<p><span class="pill run">Partly</span> Conditioning fixes over-smoothing of already-clean inputs on the averaged inputs, but it fixes
the Atacama only for the VV-only supervised model. On textured, low-backscatter desert, Lee is still the safer
filter.</p>
<p><span class="pill neg">No</span> Despeckling does not make 10 m Sentinel-1 good enough for field boundaries. Every
input, the 24-date mean included, gets a boundary F1 near 0.1 on FTW labels.</p>""",
}

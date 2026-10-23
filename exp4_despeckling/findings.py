"""Results prose for the Experiment 4 page (numbers from metrics_dates.csv, downstream_*.csv, ckpt/*_log.csv)."""

FILL = {
"FINDINGS": """<h3>Results</h3>
<p><b>The supervised CNN is the most faithful single-date filter, by a modest margin.</b> Its held-out median PSNR is
21.5 dB, against 21.1 for Lee, 20.6 for refined Lee, 20.7 for Gamma-MAP and 18.1 for the raw image. It also has the best
SSIM (0.52 vs 0.49 for Lee), the lowest MAE (0.98 dB, tied with Lee), the best gradient correlation with the reference
(0.56 vs 0.51), no measurable bias (+0.04 dB), and no edge structure in its ratio image (correlation -0.01, the same as
the oracle). It beats refined Lee by 1.3 to 1.8 dB at five of six held-out sites. Against plain Lee it wins by 0.3 to
0.9 dB at four sites and ties in Gabon.</p>
<p>The original supervised CNN performs poorly in the Atacama. It smooths stable desert texture and introduces a +0.67 dB bias, scoring 18.6 dB PSNR versus 20.9 for Lee and 22.9 for a 3×3 boxcar. This site has less speckle than the training data (single-date ENL 9.5) and substantial real texture (reference ENL 21).</p>
<p><b>Every CNN smooths flat areas more than the reference does, and blurs edges a little.</b> The supervised CNN's
ENL in homogeneous areas is 100, against 61 for the 22 to 27 date mean itself. Its EPD-ROA of 0.86 means edge contrast
drops by about 14 %, like Lee (0.88) and worse than refined Lee (0.97) and Gamma-MAP (1.00). Those two keep edges by
design but leave more speckle (ratio ENL 12 to 13, against 5.8 for pure speckle). By eye, the new CNNs keep a fine
grain that makes them look less clean than Lee, even though they sit closer to the reference.</p>
<p>The transferred Experiment 1 model scores only 18.5–18.7 dB PSNR at 10 m, about 2.4 dB below Lee. It removes texture, loses about 18% of mean power and compresses contrast. Its ratio-image ENL is 1.3–1.4, versus 5.8 for the reference speckle ratio. The Capella retraining changes PSNR by about 0.1 dB. The smooth appearance mainly reflects excessive filtering.</p>
<p>Noise2Noise reaches 20.7 dB PSNR, SSIM 0.50 and gradient correlation 0.56 without a temporal-mean training target. It smooths more than the supervised model (ENL 162) and uses a +0.38 dB bias correction fitted on training sites. Training needs pairs of same-orbit dates rather than a stable 20-date stack.</p>""",

"DFINDINGS": """<p><b>With texture features, every CNN helps a little and the classic filters do not help at all.</b>
Overall accuracy is 0.649 on the raw image and 0.65 to 0.66 after any classic filter. The window features already
average speckle, so a 7x7 filter adds nothing. The CNNs reach 0.67 to 0.70: Noise2Noise 0.702, Experiment 1 0.69 to 0.70,
supervised 0.681, GAN 0.669. The 22 to 27 date temporal mean reaches 0.733, so a single-date CNN closes a third to a
half of the gap between raw and a full stack. Macro F1 follows the same order (0.45 raw, 0.51 Noise2Noise, 0.55 stack).
Accuracy varies by about ±0.02 across the three dates, so the CNN-vs-classic gap holds but the order among the CNNs
is not firm.</p>
<p>Pixel-only classification favors heavily smoothed images: Experiment 1 reaches 0.57 accuracy, above the temporal mean’s 0.48. Smoothing makes each pixel represent its neighborhood, which can agree with spatially smooth WorldCover labels. This score alone is a poor measure of image fidelity.</p>
<p>The WorldCover boundary test gives similar F1 for all single-date methods (0.37–0.40); the temporal mean reaches 0.44. These optically derived boundaries need not produce a radar edge. Atacama and Gabon have too few boundary pixels to score. Round 2 uses FTW field labels for a separate boundary test.</p>""",

"GAN": """<p>The fine-tune ran 3,000 steps from the supervised weights. The stabilizers worked: the discriminator
never ran away. Its loss settled at 0.09 rather than Experiment 1's 0.001; it scored real targets at 0.72 and generated
ones at 0.21, and the generator's L1 stayed at 0.32 throughout. It changed almost nothing. Against the supervised CNN it
gives slightly lower PSNR (21.4 vs 21.5 dB) and SSIM (0.50 vs 0.52), a slightly higher gradient correlation (0.57 vs
0.56), and a ratio ENL closer to pure speckle (8.8 vs 9.5). Downstream accuracy is slightly lower (0.669 vs 0.681). By
eye the panels match the supervised CNN. With a stable discriminator, the adversarial loss adds nothing measurable here.</p>""",

"CAVEATS": """<ul>
<li><b>The reference is not truth.</b> A 22 to 27 date mean has its own residual speckle (ENL about 60 in flat areas)
and averages over seasons. The change mask removes pixels that vary more than speckle can explain. That drops 63 to 75 %
of cropland pixels, so the cropland scores describe the stable parts of farm landscapes: roads, trees, settlements and
fallow ground.</li>
<li><b>Training targets come from the same kind of stack</b>, so the supervised CNN learns to predict a multi-month
mean. That suits stable features. For a field that changed on the input date, the network's prior is the long-term
average, the wrong answer for change detection.</li>
<li><b>Six held-out sites</b>, one per land cover, at 5 km. Medians over sites hide real variation: the Atacama
result shows the CNN failing outside its training distribution. More sites, especially deserts and textured terrain,
would firm up the ranking.</li>
<li><b>Experiment 1's outputs are mapped back to Sentinel-1 dB</b> with the input's own median and standard deviation,
the inverse of their training normalization. They were trained toward X-band statistics, so part of their contrast loss
and mean bias comes from that mismatch, not from despeckling itself.</li>
<li><b>One ENL for the classic filters</b> (6.7). Per-image tuning would help them at the low-ENL (Madrid 4.6) and
high-ENL (Atacama 9.5) sites. We kept it fixed because the CNNs get no per-site tuning either.</li>
<li><b>One training run per model</b>, no repeated seeds. The differences among the three new CNNs (0.1 to 0.8 dB)
are within what another seed might reorder.</li>
<li>The European stacks mix Sentinel-1A and 1C dates; elsewhere only 1A flies. VV only.</li>
</ul>""",

"VERDICT": """<p><span class="pill done">Modest yes</span> A CNN trained for despeckling, on a temporal-mean target
or on Noise2Noise date pairs, beats refined Lee and the other classic filters on held-out sites. It gains about 1 dB
PSNR over refined Lee and 0.4 dB over Lee, with better SSIM and edge correlation and no bias. It also lifts land-cover
accuracy by 2 to 5 points, where the classic filters lift it by none. It does not reach a multi-date mean, and it fails
on a textured desert unlike its training sites.</p>
<p><span class="pill neg">No</span> for Experiment 1's network as a despeckler. Its clean look comes from removing
real texture and mean power, and it scores barely above the raw image against the reference. A GAN fine-tune, even a
stable one, adds nothing measurable.</p>
<p>Round 2 below tests more training sites, speckle-level conditioning and VV+VH inputs. It also repeats the mapping evaluation and adds FTW field-boundary labels.</p>""",
}

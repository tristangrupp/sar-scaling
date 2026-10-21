"""Results prose for the Experiment 4 page (numbers from metrics_dates.csv, downstream_*.csv, ckpt/*_log.csv)."""

FILL = {
"FINDINGS": """<h3>What the numbers say</h3>
<p><b>The supervised CNN is the most faithful single-date filter, by a modest margin.</b> Its held-out median PSNR is
21.5 dB, against 21.1 for Lee, 20.6 for refined Lee, 20.7 for Gamma-MAP and 18.1 for the raw image. It also has the best
SSIM (0.52 vs 0.49 for Lee), the lowest MAE (0.98 dB, tied with Lee), the best gradient correlation with the reference
(0.56 vs 0.51), no measurable bias (+0.04 dB), and no edge structure in its ratio image (correlation -0.01, the same as
the oracle). It beats refined Lee by 1.3 to 1.8 dB at five of six held-out sites. Against plain Lee it wins by 0.3 to
0.9 dB at four sites and ties in Gabon.</p>
<p><b>It fails in the Atacama.</b> The desert surface has real, stable 10 m texture: the temporal mean there has an
ENL of only 21. The input is also less speckled than anything in training (ENL 9.5). The CNN smooths that texture away
and over-brightens by 0.67 dB, scoring 18.6 dB against 20.9 for Lee and 22.9 for a 3x3 boxcar. The network learned a
speckle level and a log-bias correction from the training sites and applies them whatever the input looks like.</p>
<p><b>Every CNN smooths flat areas more than the reference does, and blurs edges a little.</b> The supervised CNN's
ENL in homogeneous areas is 100, against 61 for the 22 to 27 date mean itself. Its EPD-ROA of 0.86 means edge contrast
drops by about 14 %, like Lee (0.88) and worse than refined Lee (0.97) and Gamma-MAP (1.00). Those two keep edges by
design but leave more speckle (ratio ENL 12 to 13, against 5.8 for pure speckle). By eye, the new CNNs keep a fine
grain that makes them look less clean than Lee, even though they sit closer to the reference.</p>
<p><b>Experiment 1's network is not a good despeckler once you measure it.</b> Back at 10 m it scores 18.5 to 18.7 dB
PSNR, barely above the raw image and 2.4 dB below Lee. It wipes out fine texture (ENL about 800 to 900), loses 18 % of
the mean power, and compresses contrast. Its ratio image has an ENL of 1.3 to 1.4, far below the 5.8 of pure speckle, so
it removed real signal along with the speckle. The Capella retrain changes almost nothing (+0.1 dB). The clean look in
Experiment 1's context panels was mostly oversmoothing.</p>
<p><b>Noise2Noise nearly matches the supervised CNN without any clean target</b> (20.7 dB, SSIM 0.50, gradient
correlation 0.56). It smooths more (ENL 162) and needed a +0.38 dB offset, fitted on training sites, to undo the
log-domain bias. This matters in practice: it needs only two dates of the same orbit, not a stable 20-date stack.</p>""",

"DFINDINGS": """<p><b>With texture features, every CNN helps a little and the classic filters do not help at all.</b>
Overall accuracy is 0.649 on the raw image and 0.65 to 0.66 after any classic filter. The window features already
average speckle, so a 7x7 filter adds nothing. The CNNs reach 0.67 to 0.70: Noise2Noise 0.702, Experiment 1 0.69 to 0.70,
supervised 0.681, GAN 0.669. The 22 to 27 date temporal mean reaches 0.733, so a single-date CNN closes a third to a
half of the gap between raw and a full stack. Macro F1 follows the same order (0.45 raw, 0.51 Noise2Noise, 0.55 stack).
Accuracy varies by about ±0.02 across the three dates, so the CNN-vs-classic gap holds but the order among the CNNs
is not firm.</p>
<p><b>Pixel-only accuracy rewards smoothing, not fidelity.</b> The heavily oversmoothed Experiment 1 outputs score highest
(0.57), above even the temporal mean (0.48). One pixel of a very smooth image carries its neighbourhood's average, and
WorldCover labels are spatially smooth. Read this column as a measure of how much each method averages, not of
quality.</p>
<p><b>The boundary test does not separate the methods.</b> Every single-date input lands between 0.37 and 0.40 F1; the
temporal mean reaches 0.44. WorldCover boundaries come from optical imagery and mark land-cover changes that C-band
often does not see, and even after a 5x5 majority filter 10 to 19 % of pixels sit on a boundary. Atacama and Gabon
have too few boundary pixels and are skipped.</p>""",

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
<p>Next steps, if this continues: train on many more sites, including deserts and rough terrain; feed the network the
input's ENL so it stops assuming one speckle level; and prefer Noise2Noise, which needs only two dates and so scales to
any orbit without a stable stack.</p>""",
}

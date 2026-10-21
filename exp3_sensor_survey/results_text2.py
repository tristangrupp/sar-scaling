"""Write results_text2.json: prose for the Round 2 section of the page."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
T = {}
T["lede"] = """<p>Round 2 ran the four next steps from round 1 in order. For tuning, it held out three training frames as a
<b>validation split</b>: Iowa farmland (4,22), West Africa (152,84) and northern Italy (8,66), 120 tiles. Every model in
round 2 trained on the other 600 tiles. The near and far holdouts were scored only after a setting had been chosen on
validation. Round 2 runs are shorter than round 1's (3,000 steps; 1,500 for step 4) because six or more jobs shared
the GPU.</p>"""
T["sweep"] = """<h2>Step 1: input-consistency weight</h2>
<p>Six settings: weight 0, 0.05, 0.2 and 0.5 with the consistency measured on the 20 m input grid, and 0.5 and 2.0
measured on a 40 m grid, so the output can vary freely inside each 40 m block. The rule, fixed before the results came in:
beat bicubic on as many of MAE, SSIM, detail correlation and both chamfer directions as possible; break ties by the
lowest grid ratio. Two settings beat bicubic on all five validation metrics. <b>Weight 0.5 at 40 m won</b> on
grid ratio (2.28 against 2.50 for weight 0.2 at 20 m).</p>"""
T["sweep_hold"] = """<p><b>On the far holdout the chosen setting beats bicubic on all five metrics</b>: MAE 1.719 vs 1.760 dB, SSIM
0.492 vs 0.487, detail correlation 0.327 vs 0.273, invented-edge chamfer 10.7 vs 11.3 m, missed-edge chamfer
10.9 vs 12.1 m. Round 1's arm A did not beat bicubic on MAE or invented edges. The grid imprint falls from 3.06
(round 1 arm A) to 2.41, though it stays above bicubic's 1.38. Near tiles are mixed: MAE, detail correlation and
invented edges beat bicubic, while SSIM (0.342 vs 0.350) and missed edges (20.8 vs 20.0 m) fall slightly short.
Measuring consistency at 40 m fixes the over-constraint only in part.</p>"""
T["multi"] = """<h2>Step 2: a cleaner target from several dates</h2>
<p>For every tile we read the same 10 m window from up to six other 12-day repeats of the same track and frame (40 MHz,
remote reads). Per 80 m block, a date counted only if it sat within 1.5 dB of the input date (the change mask), and the
kept dates were averaged in linear power. The median tile kept 6 extra dates. Melbourne had no repeats in the archive
and Shandong (82,20) had one, so their tiles drop out of the multi-date scoring (mask: at least 3 dates).</p>
<p><b>A cleaner target changes the verdict on the old models more than it improves new ones.</b> Scored against the
multi-date mean, the single-date-trained model's detail correlation rises to 0.34 (far) and 0.29 (near), against
0.33 and 0.15 when scored against one noisy date. Its gains were partly hidden by target speckle. The model
<i>trained</i> on the multi-date target cuts MAE sharply (far 1.40 dB vs 1.49 for the single-date model and 1.54 for
bicubic; near 0.98 vs 1.30) and invents fewer edges. It has the same detail correlation, but it is smoother: lower
SSIM, fine-detail energy 0.40 vs 0.45, and many more missed edges, especially on near tiles (33 m vs 21 m). L1 against a
smooth target buys accuracy and costs sharpness.</p>"""
T["g2"] = """<h2>Step 3: 10 m GCOV to 5 m GSLC</h2>
<p>The 40 MHz GSLC carries the full 5 m x 5 m resolution. A 1024 x 1024 window reads remotely in about 5 s. A
single-date GSLC is about one look, so the target is again a multi-date mean (up to 5 extra dates, change-masked). It
is calibrated to gamma0 with the product's LUT and anchored at about 80 m to the RTC'd 10 m GCOV of the input date,
so that only the fine structure comes from GSLC. Some GSLC files returned 404 at ASF, which cut the scored set to 82
tiles (32 validation, 22 near, 28 far).</p>
<p><b>Result: the same pattern one level finer.</b> The CNN beats bicubic on MAE (far 2.16 vs 2.21 dB; near 1.75 vs
1.98), detail correlation (0.223 vs 0.193 far; 0.153 vs 0.118 near) and invented edges. It loses on SSIM and missed
edges and is smoother. The target is still speckled with so few dates.</p>"""
T["armb"] = """<h2>Step 4: arm B, the X-band edge loss</h2>
<p>Training frames hold little open X-band over built-up land: most open Capella collects sit on non-urban sites.
So we added 19 X-band-rich training frames, still excluding every X-band site inside any holdout frame. We split
sites 60/20/20 into train, edge-weight tuning (xval) and test (xtest). That gave 60 training windows (36 Umbra, 19
Capella, 5 ICEYE), 15 tuning and 11 test windows. The loss weights output gradients by their distance beyond 10 m
from the nearest X-band edge (capped at 50 m). It counts only infrastructure pixels: WorldCover built-up, X-band
valid, and not in DEM-derived layover or shadow, because NISAR's static layers are not yet published. Arm A′ trains
on the same X-band windows with the edge weight at 0. Weights 0.05 and 0.2 were tried; all runs used 1,500 steps.</p>
<p><b>Arm B does not beat arm A′.</b> On the near and far holdouts all four models (with and without the X-band
windows, both edge weights) agree to within 0.003 on SSIM and detail correlation and 0.3 m on both chamfers. On
X-band they agree to within about 1 m. Far Umbra x2p: A′ 22.7 m, B 23.2 to 23.3 m, truth 22.5 m. xtest Capella: 31.0 vs 30.9 to 31.5 m. xtest
Umbra: 30.3 vs 29.6 to 30.8 m. The edge loss fell during training (0.53 to 0.39 at weight 0.2) without moving any
held-out metric. By the design's decision rule, <b>X-band stays a validation reference and leaves training</b>.
At 10 m, L-band edges already sit where X-band puts them (round 1); there is little left for the loss to correct.</p>"""
T["caveats"] = """<ul><li>Round 2 models trained for 3,000 steps (1,500 for step 4), against 8,000 in round 1, on 600 tiles; compare
round 2 settings with each other and with bicubic, not with the round 1 numbers.</li>
<li>One seed per setting; differences under about 0.005 in SSIM or detail correlation are within run-to-run noise.</li>
<li>Multi-date targets mix up to seven dates over four months; the change mask removes blocks that changed by more
than 1.5 dB, but gradual change below that remains.</li>
<li>The GSLC arm scored only 82 tiles because some GSLC files are missing at ASF.</li>
<li>Arm B used DEM layover and shadow from a nominal 40 degree incidence, not NISAR's static layers; X-band test
counts remain small (2 Capella, 2 ICEYE, 7 Umbra in xtest; 30 held-out Umbra and 2 ICEYE collects).</li></ul>"""
(ROOT / "results_text2.json").write_text(json.dumps(T, indent=1), encoding="utf-8")
print("ok")

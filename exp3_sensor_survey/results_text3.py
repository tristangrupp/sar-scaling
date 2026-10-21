"""Write results_text3.json: prose for the Round 3 section (placed at the top of the page)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
T = {}
T["lede"] = """<p>Round 3 followed up round 2's leads: a deeper 5 m GSLC target, a sharper loss for the multi-date target,
and longer runs. Experiment 5 had priority on the GPU, so round 3 ran one small job at a time (batch 16). A memory
gate held each job until at least 2 GB was free, and all scoring ran on the CPU. Settings were chosen on the
validation frames from round 2. Holdouts were scored afterwards.</p>
<p><b>In short:</b> the deeper GSLC index fixed every round 2 download failure and raised the median tile from 5 to 8
extra dates. Gradient-L1 is a small, repeatable gain in accuracy but not in sharpness. Training to round 1 length
(8,000 steps) helps a little. Nothing in round 3 changes the round 2 conclusions.</p>"""
T["gslc"] = """<h2>Step 1: more GSLC dates per tile</h2>
<p>Round 2 guessed each GSLC file name from its GCOV twin and stopped at 5 extra dates. Round 3 lists every 40 MHz
GSLC granule per track and frame from NASA CMR (full and partial frames) and reads up to 10 extra dates per tile,
retrying transient errors. <b>The 404s are gone</b> (0 of 4,044 reads failed); they came from the guessed names, not
from missing data. The median tile now averages 8 extra dates (7 looks per pixel, up from 5). Tiles with at least 3
dates over 90% of their area rose from 292 to 351 of 468. Coverage is still uneven. Melbourne (1 GSLC in the
archive) and Shandong (2) get no extra dates, and on Johannesburg, north Taiwan and São Paulo many repeats are
partial frames that miss the tile ("outside window" below).</p>
<p>The round 2 model scored against the deeper target gives the same picture as before. Retrained on it
(r3_g2v2), it changes little: MAE ties on far tiles (2.065 dB) and improves slightly on near and validation tiles. The
output gets smoother still (fine-detail energy 0.28 vs 0.30 far), so more looks in the target do not make L1 output
sharper.</p>"""
T["loss"] = """<h2>Step 2: gradient-L1 for the multi-date target</h2>
<p>We added an L1 term on Sobel x and y gradients (prediction against target, valid pixels only) to the multi-date
model's loss, at weights 0.5 and 1.0. The rule, fixed in advance: on validation against the multi-date target, count wins over plain L1 on MAE,
SSIM, detail correlation and both chamfers, and break ties with the higher fine-detail energy (sharper). Both weights
won 3 of 5 (MAE, detail correlation, invented edges); <b>weight 0.5 won</b> on sharpness. The winner and plain L1 then got a
second seed each.</p>
<p><b>Across two seeds the effect is consistent but small, and it is not the one we wanted.</b> Against the
multi-date target, gradient-L1 lowers MAE (far 1.373 to 1.374 vs 1.397 to 1.398 dB; near 0.95 vs 0.98) and invents
fewer edges (12.6 vs 13.3 m far). It also lowers the grid imprint (2.10 vs 2.22) and nudges detail correlation up
(0.343 to 0.344 vs 0.339 to 0.340). But it is <i>not</i> sharper: fine-detail energy falls slightly (0.39 vs 0.40),
SSIM drops (0.493 vs 0.499 to 0.500) and missed edges rise (14.5 to 14.6 vs 13.8 to 13.9 m). The seed spread is at
most 0.004 on the scores and 0.6 m on chamfer, so these differences are real but small. Matching target gradients
in L1 still rewards the conditional median: gradients that L1 cannot place are averaged away. An adversarial term
would be the next lever. We did not run it, given the GPU budget.</p>"""
T["long"] = """<h2>Step 3: round 1 length (8,000 steps)</h2>
<p>We reran the round 2 winner (consistency 0.5 at 40 m) and the multi-date model at 8,000 steps, against 3,000 in
round 2. <b>Both improve a little.</b> Against the single-date target on far tiles, the winner's MAE goes from 1.719
to 1.707 dB, detail correlation from 0.327 to 0.332 and grid imprint from 2.41 to 2.22. It still beats bicubic on all
five far metrics. On near tiles its grid imprint also falls (4.36 to 3.76), while missed edges rise slightly. The
multi-date model's grid imprint drops from 2.08 to 1.91 and its detail correlation rises from 0.322 to 0.326. The
rankings between models do not change, so round 2's shorter runs did not distort its conclusions.</p>"""
T["caveats"] = """<ul><li>Round 3 shared the GPU with Experiment 5 under a memory gate. One run (r3_g1_s0) was killed during a
memory spill and rerun from scratch; no reported number comes from an interrupted run.</li>
<li>Two seeds is the minimum for a seed check. The step 2 differences exceed the seed spread, but only by a few
hundredths of a dB.</li>
<li>The deeper GSLC target is still uneven across frames. Holdout 5 m scores cover 29 far and 29 near tiles,
from Buenos Aires, Johannesburg, Abu Dhabi and north Taiwan; Melbourne has no GSLC repeats.</li>
<li>The plain-L1 seed 0 run is round 2's m_multi (same settings, unseeded), reused to save GPU time.</li></ul>"""
(ROOT / "results_text3.json").write_text(json.dumps(T, indent=1), encoding="utf-8")
print("ok")
